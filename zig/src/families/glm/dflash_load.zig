// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! DFlash2's checkpoint onto the device: the config checked, linears tiled for glm_mm, scratch carved from one buffer.
const std = @import("std");
const cuda = @import("cuda");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const quant = @import("cuda_quant.zig");
const taps_mod = @import("cuda_taps.zig");
const df = @import("dflash.zig");
const Drafter = df.Drafter;

pub const CHUNK: usize = 64 * 1024 * 1024;
pub const PACK_ROWS: usize = 128;

/// Row ranges used by a linear loader. Whole rows preserve q4's 128-row packed layout.
pub const RowChunk = struct { first: usize, rows: usize };

/// Quantize and pack one host slice of a chunked linear into its tensor-wide Q4 layout.
/// `host_rows` contains `count` rows beginning at `first`; `chunk_rows` is the staging
/// capacity and must preserve q4's 128-row packed layout.
pub fn quantizeLinearChunked(
    host_rows: []const u16,
    rows: usize,
    k: usize,
    first: usize,
    chunk_rows: usize,
    stream: cuda.Stream,
    qk: *const quant.Kernels,
    pk: *const q4.Kernels,
    input: *cuda.DeviceBuffer,
    raw: *cuda.DeviceBuffer,
    staged: q4.Q4,
    dest: q4.Q4,
) !void {
    if (k == 0 or k % 64 != 0 or host_rows.len % k != 0) return error.BadChunkShape;
    const count = host_rows.len / k;
    if (rows == 0 or first > rows or count == 0 or first + count > rows) return error.BadChunkShape;
    if (chunk_rows < PACK_ROWS or chunk_rows % PACK_ROWS != 0 or first % PACK_ROWS != 0 or count > chunk_rows) return error.BadChunkSize;
    if (dest.n != rows or dest.k != k or dest.npad != q4.npadOf(rows) or staged.n != count or staged.k != k or staged.npad != q4.npadOf(count)) return error.BadChunkShape;
    const scales = raw.ptr + std.mem.alignForward(u64, count * k / 2, 256);
    const biases = scales + std.mem.alignForward(u64, count * k / 32, 256);
    if (input.len < host_rows.len * 2 or raw.len < rawBytes(count, k)) return error.BadChunkShape;

    // Explicitly initialize the tensor-wide affine buffers, including the padded tail.
    if (first == 0) {
        const dw = cuda.DeviceBuffer{ .d = input.d, .ptr = dest.w, .len = dest.npad * k / 2, .borrowed = true };
        const ds = cuda.DeviceBuffer{ .d = input.d, .ptr = dest.s, .len = dest.npad * (k / 64) * 2, .borrowed = true };
        const db = cuda.DeviceBuffer{ .d = input.d, .ptr = dest.b, .len = dest.npad * (k / 64) * 2, .borrowed = true };
        try dw.fill8(0, stream.handle);
        try ds.fill8(0, stream.handle);
        try db.fill8(0, stream.handle);
    }
    try input.upload(0, std.mem.sliceAsBytes(host_rows));
    try quant.q4(qk, stream, input.ptr, count, k, raw.ptr, scales, biases);
    const chunk_sizes = q4.bytes(count, k);
    if (staged.w == 0 or staged.s == 0 or staged.b == 0) return error.BadChunkShape;
    try q4.pack(pk, stream, raw.ptr, scales, biases, staged);
    if (staged.w != dest.w + first * k / 2) {
        const dw = cuda.DeviceBuffer{ .d = input.d, .ptr = dest.w, .len = dest.npad * k / 2, .borrowed = true };
        try dw.copyFrom(first * k / 2, staged.w, chunk_sizes[0], stream.handle);
    }

    const kg = k / 64;
    const tensor_npad = dest.npad;
    const chunk_npad = staged.npad;
    for (0..kg) |g| {
        const dst_off = (g * tensor_npad + first) * 2;
        const src_off = g * chunk_npad * 2;
        const copy_bytes = count * 2;
        try dsCopy(input, dest.s, tensor_npad * kg * 2, dst_off, staged.s + src_off, copy_bytes, stream);
        try dsCopy(input, dest.b, tensor_npad * kg * 2, dst_off, staged.b + src_off, copy_bytes, stream);
    }
    try stream.synchronize();
}

// A Q4 stores raw device addresses; use the staging buffer's driver for its borrowed views/copies.
fn dsCopy(input: *cuda.DeviceBuffer, dst: u64, len: usize, off: usize, src: u64, n: usize, stream: cuda.Stream) !void {
    const view = cuda.DeviceBuffer{ .d = input.d, .ptr = dst, .len = len, .borrowed = true };
    try view.copyFrom(off, src, n, stream.handle);
}

pub fn planRows(allocator: std.mem.Allocator, rows: usize, row_bytes: usize, limit: usize) ![]RowChunk {
    if (row_bytes == 0 or limit < row_bytes * PACK_ROWS) return error.BadChunkSize;
    const per_chunk: usize = @max(PACK_ROWS, (limit / row_bytes) / PACK_ROWS * PACK_ROWS);
    var chunks: std.ArrayList(RowChunk) = .empty;
    errdefer chunks.deinit(allocator);
    var first: usize = 0;
    while (first < rows) {
        const count: usize = @min(per_chunk, rows - first);
        try chunks.append(allocator, .{ .first = first, .rows = count });
        first += count;
    }
    return chunks.toOwnedSlice(allocator);
}

/// Bytes of quantizeLinearChunked's raw layout for `count` rows of `k`: MLX words, then scales, then biases, each
/// start 256-aligned.
pub fn rawBytes(count: usize, k: usize) usize {
    return std.mem.alignForward(usize, count * k / 2, 256) + std.mem.alignForward(usize, count * k / 32, 256) + count * k / 32;
}

/// Device staging for one 4-bit row chunk. A chunk holds at most chunk_bytes / 2 bf16 weights whatever its k, and
/// npadOf(rows) is at most planRows' per-chunk row count (a multiple of PACK_ROWS): size raw and qscale for that,
/// not for k = hidden. Sizing raw
/// as (chunk_bytes / 2 / hidden) * hidden weights broke the GLM-5.3 load: an o_proj chunk (k 8192, 4096 rows) holds
/// 2048 weights more and failed quantizeLinearChunked's raw check (BadChunkShape) on both ranks.
pub fn stagingBytes(chunk_bytes: usize) struct { raw: usize, qscale: usize } {
    const elems = chunk_bytes / 2;
    // words, scales and biases, each start 256-aligned (quantizeLinearChunked) | scales + biases, bf16, k / 64 a row
    return .{ .raw = elems / 2 + 2 * elems / 32 + 1024, .qscale = 2 * (elems / 64) * 2 };
}

test "dflash row planner stays within the host byte limit and covers each row once" {
    const chunks = try planRows(std.testing.allocator, 40000, 2048, CHUNK);
    defer std.testing.allocator.free(chunks);
    var next: usize = 0;
    for (chunks) |chunk| {
        try std.testing.expectEqual(next, chunk.first);
        try std.testing.expect(chunk.rows * 2048 <= CHUNK);
        next += chunk.rows;
    }
    try std.testing.expectEqual(@as(usize, 40000), next);
}

const Ent = struct { off: u64, n: u64 };

pub const TensorInfo = struct { name: []u8, off: u64, n: u64 };
pub const SafetensorsHeader = struct {
    allocator: std.mem.Allocator,
    entries: []TensorInfo,
    pub fn deinit(h: *SafetensorsHeader) void {
        for (h.entries) |e| h.allocator.free(e.name);
        h.allocator.free(h.entries);
    }
};

/// Parse a BF16 safetensors header. Test fixtures use this exact function without model weights or a CUDA device.
pub fn parseSafetensorsHeader(gpa: std.mem.Allocator, text: []const u8, tensor_data_offset: u64) !SafetensorsHeader {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.BadDrafterHeader;
    var entries: std.ArrayList(TensorInfo) = .empty;
    errdefer {
        for (entries.items) |e| gpa.free(e.name);
        entries.deinit(gpa);
    }
    var it = parsed.value.object.iterator();
    while (it.next()) |x| {
        if (std.mem.eql(u8, x.key_ptr.*, "__metadata__")) continue;
        const ent = x.value_ptr.*.object;
        if (!std.mem.eql(u8, ent.get("dtype").?.string, "BF16")) return error.DrafterNotBf16;
        const shape = ent.get("shape").?.array.items;
        if (shape.len == 0) return error.BadDrafterHeader;
        var elements: u64 = 1;
        for (shape) |dim| {
            if (dim != .integer or dim.integer <= 0) return error.BadDrafterHeader;
            elements = std.math.mul(u64, elements, @intCast(dim.integer)) catch return error.BadDrafterHeader;
        }
        const offs = ent.get("data_offsets").?.array.items;
        if (offs.len != 2 or offs[0] != .integer or offs[1] != .integer or offs[0].integer < 0 or offs[1].integer < offs[0].integer) return error.BadDrafterHeader;
        const n: u64 = @intCast(offs[1].integer - offs[0].integer);
        const expected_bytes = std.math.mul(u64, elements, 2) catch return error.BadDrafterHeader;
        if (n != expected_bytes) return error.BadDrafterHeader;
        try entries.append(gpa, .{ .name = try gpa.dupe(u8, x.key_ptr.*), .off = tensor_data_offset + @as(u64, @intCast(offs[0].integer)), .n = n });
    }
    return .{ .allocator = gpa, .entries = try entries.toOwnedSlice(gpa) };
}

const Loader = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    ents: *const std.StringHashMap(Ent),
    used: *std.StringHashMap(void),
    buf: std.ArrayList(u8) = .empty,
    chunk_bytes: usize,
    w: *cuda.DeviceBuffer,
    tmp: *cuda.DeviceBuffer,
    mk: *const mm.Kernels,
    qk: *const quant.Kernels,
    pk: *const q4.Kernels,
    qraw: *cuda.DeviceBuffer, // a 4-bit linear's MLX words, scales and biases before tiling
    qscale: *cuda.DeviceBuffer, // chunk-local packed scales and biases before scattering into tensor layout
    bits: u8,
    stream: cuda.Stream,
    at: u64 = 0,

    fn take(l: *Loader, n: u64) !u64 {
        if (l.at + n > l.w.len) { // the size estimate is short: never write past it
            std.log.err("dflash loader: a {d} B tensor at {d} B does not fit the {d} B weights buffer", .{ n, l.at, l.w.len });
            return error.DrafterLayout;
        }
        const p = l.at;
        l.at += std.mem.alignForward(u64, n, 256);
        return l.w.ptr + p;
    }

    /// A small bf16 tensor as stored.
    fn raw(l: *Loader, name: []const u8, n: u64) !u64 {
        const p = try l.take(n);
        const e = l.require(name, n) catch |err| return err;
        var off: u64 = 0;
        while (off < n) {
            const count: usize = @intCast(@min(@as(u64, l.chunk_bytes), n - off));
            try l.buf.resize(l.gpa, count);
            _ = try l.file.readPositionalAll(l.io, l.buf.items, e.off + off);
            try l.w.upload(p - l.w.ptr + off, l.buf.items);
            off += count;
        }
        return p;
    }

    fn readHost(l: *Loader, name: []const u8, dst: []u16) !void {
        const bytes = std.mem.sliceAsBytes(dst);
        const e = try l.require(name, @intCast(bytes.len));
        _ = try l.file.readPositionalAll(l.io, bytes, e.off);
    }

    fn require(l: *Loader, name: []const u8, want: u64) !Ent {
        const entry = l.ents.getEntry(name) orelse {
            std.log.err("dflash2: no {s} in the drafter's checkpoint", .{name});
            return error.DrafterTensorMissing;
        };
        const e = entry.value_ptr.*;
        if (e.n != want) {
            std.log.err("dflash2: {s} holds {d} bytes, not {d}", .{ name, e.n, want });
            return error.DrafterTensorShape;
        }
        // Key on the header's own name: `name` often points into a reused layerName buffer, and a key whose bytes
        // change turns the map's rehash into duplicate keys.
        try l.used.put(entry.key_ptr.*, {});
        return e;
    }

    /// `names` stacked as linear(), at the drafter's bits: quantize4 then q4's tiling, or glm_mm's.
    fn linear(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize, part: ?[4]usize) !df.Lin {
        if (l.bits == 0) return .{ .bf16 = try l.linearB(names, rows, kk, part) };
        const k = if (part) |p| p[3] else kk;
        const total_rows = try stackedRows(names, rows, part);
        const sz = q4.bytes(total_rows, k);
        const q: q4.Q4 = .{ .w = try l.take(sz[0]), .s = try l.take(sz[1]), .b = try l.take(sz[2]), .n = total_rows, .k = k, .npad = q4.npadOf(total_rows) };
        const plan = try planRows(l.gpa, total_rows, k * 2, l.chunk_bytes);
        defer l.gpa.free(plan);
        const chunk_capacity = @max(PACK_ROWS, (l.chunk_bytes / (2 * k)) / PACK_ROWS * PACK_ROWS);
        for (plan) |chunk| {
            try l.stackChunk(names, rows, kk, part, chunk.first, chunk.rows);
            const n = chunk.rows;
            const chunk_sizes = q4.bytes(n, k);
            if (2 * chunk_sizes[1] > l.qscale.len) return error.BadChunkShape; // the chunk's packed scales then biases
            const cq: q4.Q4 = .{ .w = q.w + chunk.first * k / 2, .s = l.qscale.ptr, .b = l.qscale.ptr + chunk_sizes[1], .n = n, .k = k, .npad = q4.npadOf(n) };
            try quantizeLinearChunked(
                @alignCast(std.mem.bytesAsSlice(u16, l.buf.items)),
                total_rows,
                k,
                chunk.first,
                chunk_capacity,
                l.stream,
                l.qk,
                l.pk,
                l.tmp,
                l.qraw,
                cq,
                q,
            );
        }
        return .{ .q4 = q };
    }

    /// Rows row0.. (rn of each tensor's `rows`) and columns col0.. (cn of `kk`) of `names`, stacked and tiled bf16.
    fn linearB(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize, part: ?[4]usize) !mm.Tiled {
        const n = try stackedRows(names, rows, part);
        const cn = if (part) |p| p[3] else kk;
        const dst = try l.take(n * cn * 2);
        const plan = try planRows(l.gpa, n, cn * 2, l.chunk_bytes);
        defer l.gpa.free(plan);
        for (plan) |chunk| {
            try l.stackChunk(names, rows, kk, part, chunk.first, chunk.rows);
            try mm.tile(l.mk, l.stream, l.tmp.ptr, dst + chunk.first * cn * 2, chunk.rows, cn);
            try l.stream.synchronize();
        }
        return .{ .w = dst, .n = n, .k = cn };
    }

    fn stackChunk(l: *Loader, names: []const []const u8, rows: []const usize, kk: usize, part: ?[4]usize, first: usize, count: usize) !void {
        const p: [4]usize = part orelse .{ 0, 0, 0, kk };
        const cn = p[3];
        try l.buf.resize(l.gpa, count * cn * 2);
        var global: usize = 0;
        var outrow: usize = 0;
        for (names, rows) |name, r| {
            const rn = if (part != null and p[1] > 0) p[1] else r;
            if (part != null and (p[0] + rn > r or p[2] + cn > kk)) return error.DrafterTensorShape;
            const from = @max(first, global);
            const until: usize = @min(first + count, global + rn);
            if (from < until) {
                const local_first = from - global;
                const take_rows = until - from;
                const e = try l.require(name, @intCast(r * kk * 2));
                if (part == null) {
                    const byte_count = take_rows * cn * 2;
                    const start = outrow * cn * 2;
                    _ = try l.file.readPositionalAll(l.io, l.buf.items[start..][0..byte_count], e.off + @as(u64, @intCast(local_first * kk * 2)));
                } else {
                    for (0..take_rows) |ri| {
                        const byte_count = cn * 2;
                        const start = (outrow + ri) * cn * 2;
                        _ = try l.file.readPositionalAll(l.io, l.buf.items[start..][0..byte_count], e.off + @as(u64, @intCast((p[0] + local_first + ri) * kk * 2 + p[2] * 2)));
                    }
                }
                outrow += take_rows;
            }
            global += rn;
        }
        if (outrow != count) return error.DrafterTensorShape;
        try l.tmp.upload(0, l.buf.items);
    }
};

fn stackedRows(names: []const []const u8, rows: []const usize, part: ?[4]usize) !usize {
    if (names.len != rows.len) return error.DrafterTensorShape;
    var total: usize = 0;
    for (rows) |r| total += if (part) |p| (if (p[1] > 0) p[1] else r) else r;
    return total;
}

fn layerName(b: *[96]u8, li: usize, s: []const u8) ![]const u8 {
    return std.fmt.bufPrint(b, "layers.{d}.{s}", .{ li, s });
}

/// The drafter from `dir` (config.json and model.safetensors, bf16) for rank `rank` of `world`, on engine `e`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize, bits: u8) !*Drafter {
    return loadWithHead(gpa, io, e, dir, rank, world, bits, .{}, null, false);
}

/// TF_GLM_DRAFTER_MEMLOG=1: one info line with MemAvailable at a drafter load step (diagnostic; a failed read prints
/// nothing and never changes the load).
fn memMark(gpa: std.mem.Allocator, io: std.Io, what: []const u8) void {
    const on = std.c.getenv("TF_GLM_DRAFTER_MEMLOG") orelse return;
    if (!std.mem.eql(u8, std.mem.span(on), "1")) return;
    var file = std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{}) catch return;
    defer file.close(io);
    var scratch: [4096]u8 = undefined;
    var reader = file.readerStreaming(io, &scratch);
    const text = reader.interface.allocRemaining(gpa, .limited(1 << 16)) catch return;
    defer gpa.free(text);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| if (std.mem.startsWith(u8, line, "MemAvailable:")) {
        var it = std.mem.tokenizeAny(u8, line["MemAvailable:".len..], " \tkB");
        const kb = std.fmt.parseInt(u64, it.next() orelse return, 10) catch return;
        std.log.info("glm dflash load: {s}: MemAvailable {d:.3} GiB", .{ what, @as(f64, @floatFromInt(kb)) / (1024.0 * 1024.0) });
        return;
    };
}

/// Shared loader entry for target families that synthesize the rank-local draft head at load time.
/// Geometry is supplied from this drafter's config; no MTP tensors are required.
pub fn loadWithHead(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize, bits: u8, geometry: df.Geometry, draft_head: ?q4.Q4, share_scratch: bool) !*Drafter {
    return loadWithHeadChunkedShared(gpa, io, e, dir, rank, world, bits, geometry, draft_head, CHUNK, share_scratch);
}

/// Test seam: a limit at least as large as the largest linear restores whole-linear staging.
pub fn loadWithHeadChunked(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize, bits: u8, geometry: df.Geometry, draft_head: ?q4.Q4, chunk_limit: usize) !*Drafter {
    return loadWithHeadChunkedShared(gpa, io, e, dir, rank, world, bits, geometry, draft_head, chunk_limit, false);
}

fn loadWithHeadChunkedShared(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize, bits: u8, geometry: df.Geometry, draft_head: ?q4.Q4, chunk_limit: usize, share_scratch: bool) !*Drafter {
    return loadWithHeadChunkedSharedInner(gpa, io, e, dir, rank, world, bits, geometry, draft_head, chunk_limit, share_scratch) catch |err| {
        std.log.err("glm dflash load failed: {s}", .{@errorName(err)});
        return err;
    };
}

fn loadWithHeadChunkedSharedInner(gpa: std.mem.Allocator, io: std.Io, e: anytype, dir: []const u8, rank: usize, world: usize, bits: u8, geometry: df.Geometry, draft_head: ?q4.Q4, chunk_limit: usize, share_scratch: bool) !*Drafter {
    if (bits != 0 and bits != 4) return error.BadDrafterBits;
    try checkConfig(gpa, io, dir, geometry);
    if (world == 0 or world > 3 or rank >= world) return error.BadDrafterWorld;
    if (!geometry.flashKernelAbi() and geometry.kind != .dspark) _ = try df.kernelShape(geometry);
    if (geometry.layers == 0 or geometry.hidden == 0 or geometry.block_size == 0 or geometry.block_size > df.BLOCK or geometry.selector_rank == 0 or geometry.selector_rank > df.RANK or geometry.selector_top_k == 0 or geometry.selector_top_k > df.MAX_TOPK or geometry.head_dim != (if (geometry.kind == .dspark) 64 else df.HD) or geometry.kv_heads == 0 or (geometry.conv and (geometry.conv_group == 0 or geometry.conv_kernel == 0 or geometry.conv_kernel > 2 or geometry.hidden % geometry.conv_group != 0)) or geometry.hidden % 1024 != 0 or geometry.ff % 1024 != 0 or geometry.heads % geometry.kv_heads != 0 or geometry.heads / geometry.kv_heads > 8) return error.DrafterGeometryUnsupported;
    const D = geometry.hidden;
    const D64: u64 = @intCast(D);
    const KVW = geometry.kvWidth();
    const KVW64: u64 = @intCast(KVW);
    const QW = geometry.queryWidth();
    const QW64: u64 = @intCast(QW);
    const CONVW = if (geometry.conv) geometry.convWidth() else 0;
    const FF = geometry.ff;
    const QKV = geometry.qkvWidth();
    const QKV64: u64 = @intCast(QKV);
    const RANK = geometry.selector_rank;
    const MAX_M = df.MAX_S * geometry.block_size;
    const d = e.d;
    const head = draft_head orelse (e.w.mtp orelse return error.DrafterNeedsDraftHead).draft;
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
    defer gpa.free(path);
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    // The pages read leave the page cache when the load returns: on unified memory they would take the device's
    // (a 4.9 GB BF16 drafter kept MemFree under the 2 GiB guard at 262k), as weights.zig dropShardPages does.
    defer if (@import("builtin").os.tag == .linux) {
        _ = std.os.linux.fadvise(file.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
    };
    var lenb: [8]u8 = undefined;
    _ = try file.readPositionalAll(io, &lenb, 0);
    const hlen = std.mem.readInt(u64, &lenb, .little);
    if (hlen > 1 << 20) return error.BadDrafterHeader;
    const text = try gpa.alloc(u8, hlen);
    defer gpa.free(text);
    _ = try file.readPositionalAll(io, text, 8);
    var ents = std.StringHashMap(Ent).init(gpa);
    defer ents.deinit();
    var header = try parseSafetensorsHeader(gpa, text, 8 + hlen);
    defer header.deinit();
    for (header.entries) |entry| try ents.put(entry.name, .{ .off = entry.off, .n = entry.n });
    var used = std.StringHashMap(void).init(gpa);
    defer used.deinit();
    // Two ranks always split the MLP; three ranks do with TF_GLM53_DRAFT_SPLIT3=1 (each a third of the width).
    const split = world == 2 or (world == 3 and split3Switch());
    const ff: usize = if (split) FF / world else FF;
    if (split and FF % (world * 128) != 0) return error.BadDrafterSplit;
    const small: u64 = (if (geometry.conv) @as(u64, 6) * D64 * 2 else @as(u64, 2) * D64 * 2) + 2 * @as(u64, @intCast(geometry.head_dim)) * 2 + (if (geometry.conv) @as(u64, 2) * @as(u64, @intCast(geometry.conv_kernel)) * 2 * D64 * 2 + @as(u64, @intCast(geometry.conv_group)) * 256 else @as(u64, 0));
    const per: u64 = if (bits == 4) 1 else 4; // quarter-bytes a weight: 4 bits (plus scales and biases below) or bf16
    const lin_elems: u64 = QKV64 * D64 + 2 * KVW64 * D64 + QW64 * D64 + 2 * @as(u64, @intCast(ff)) * D64 + (if (geometry.conv) @as(u64, 2) * @as(u64, @intCast(CONVW)) * D64 else @as(u64, 0)) + @as(u64, @intCast(geometry.taps.len)) * D64 * D64 / @as(u64, @intCast(geometry.layers));
    const estimate: u64 = @as(u64, @intCast(geometry.layers)) * (lin_elems * per / 2 + lin_elems / 16 + D64 * @as(u64, @intCast(ff)) * 2 + small + 32 * 256) + (if (geometry.kind == .dspark) @as(u64, 0) else @as(u64, RANK) * D64 * 2) + 2 * D64 * 2 + 64 * 256;
    // 32 x 256 a layer and 64 x 256 overall are take()'s 256-byte alignment slack for every drafter (not a conv term).
    // DSpark adds 2% headroom.
    const total: u64 = if (geometry.kind == .dspark) estimate + estimate / 50 else estimate;
    memMark(gpa, io, "start");
    var wbuf = try cuda.DeviceBuffer.alloc(d, total);
    errdefer wbuf.free();
    memMark(gpa, io, "weights buffer allocated");
    const max_linear = @max(2 * FF * D, @max(QKV * D, geometry.taps.len * D * D));
    const max_k = @max(FF, @max(D, @max(QW, geometry.taps.len * D)));
    const min_chunk = PACK_ROWS * 2 * max_k;
    const chunk_bytes: usize = @intCast(std.math.clamp(@as(u64, @intCast(chunk_limit)), @as(u64, @intCast(min_chunk)), @as(u64, @intCast(max_linear * 2))));
    const staging = stagingBytes(chunk_bytes);
    var tmp = try cuda.DeviceBuffer.alloc(d, chunk_bytes); // one row chunk of bf16
    defer tmp.free();
    var raw = try cuda.DeviceBuffer.alloc(d, if (bits == 4) staging.raw else 256);
    defer raw.free();
    var qscale = try cuda.DeviceBuffer.alloc(d, if (bits == 4) staging.qscale else 256);
    defer qscale.free();
    memMark(gpa, io, "load staging (tmp, raw) allocated");
    var module = try cuda.Module.load(d, cuda.kernels.glm_dflash);
    memMark(gpa, io, "drafter module loaded");
    errdefer module.unload();
    var l: Loader = .{ .gpa = gpa, .io = io, .file = file, .ents = &ents, .used = &used, .w = &wbuf, .tmp = &tmp, .mk = &e.m.mm, .qk = &e.m.quant, .pk = &e.m.q4, .qraw = &raw, .qscale = &qscale, .bits = bits, .stream = e.stream, .chunk_bytes = chunk_bytes };
    defer {
        l.buf.deinit(gpa);
    }
    const owned_taps = try gpa.dupe(usize, geometry.taps);
    errdefer gpa.free(owned_taps);
    const layer_weights = try gpa.alloc(df.LayerW, geometry.layers);
    errdefer gpa.free(layer_weights);
    const dr = try gpa.create(Drafter);
    errdefer gpa.destroy(dr);
    var stored_geometry = geometry;
    stored_geometry.taps = owned_taps;
    dr.* = .{ .gpa = gpa, .d = d, .module = module, .k = try df.Kernels.load(module, stored_geometry), .weights = wbuf, .geometry = stored_geometry, .layers = layer_weights, .split = split, .ff = ff, .rank = rank, .world = world, .head = head, .share_scratch = share_scratch };
    defer @import("cuda_load_owner.zig").drain(e.stream, d);
    var nb: [4][96]u8 = undefined;
    for (dr.layers, 0..) |*lw, i| {
        lw.in_norm = try l.raw(try layerName(&nb[0], i, "input_layernorm.weight"), D * 2);
        lw.post_norm = try l.raw(try layerName(&nb[0], i, "post_attention_layernorm.weight"), D * 2);
        lw.qn = try l.raw(try layerName(&nb[0], i, "self_attn.q_norm.weight"), geometry.head_dim * 2);
        lw.kn = try l.raw(try layerName(&nb[0], i, "self_attn.k_norm.weight"), geometry.head_dim * 2);
        if (geometry.conv) {
            lw.a_base = try l.raw(try layerName(&nb[0], i, "attention_conv.base_kernel"), geometry.conv_kernel * 2 * D * 2);
            lw.m_base = try l.raw(try layerName(&nb[0], i, "mlp_conv.base_kernel"), geometry.conv_kernel * 2 * D * 2);
        } else {
            lw.a_base = 0;
            lw.m_base = 0;
        }
        const q = try layerName(&nb[1], i, "self_attn.q_proj.weight");
        const kq = try layerName(&nb[2], i, "self_attn.k_proj.weight");
        const v = try layerName(&nb[3], i, "self_attn.v_proj.weight");
        lw.qkv = try l.linear(&.{ q, kq, v }, &.{ geometry.heads * geometry.head_dim, KVW, KVW }, D, null);
        lw.kv = try l.linear(&.{ kq, v }, &.{ KVW, KVW }, D, null);
        lw.o = try l.linear(&.{try layerName(&nb[1], i, "self_attn.o_proj.weight")}, &.{D}, QW, null);
        const gate = try layerName(&nb[1], i, "mlp.gate_proj.weight");
        const up = try layerName(&nb[2], i, "mlp.up_proj.weight");
        const down = try layerName(&nb[3], i, "mlp.down_proj.weight");
        if (split) { // this rank's share of the MLP width: gate and up rows, down columns
            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D, .{ rank * ff, ff, 0, D });
            lw.down = try l.linearB(&.{down}, &.{D}, FF, .{ 0, 0, rank * ff, ff });
        } else {
            lw.gu = try l.linear(&.{ gate, up }, &.{ FF, FF }, D, null);
            lw.down = try l.linearB(&.{down}, &.{D}, FF, null);
        }
        if (geometry.conv) {
            lw.a_proj = try l.linear(&.{try layerName(&nb[1], i, "attention_conv.kernel_projection.weight")}, &.{CONVW}, D, null);
            lw.m_proj = try l.linear(&.{try layerName(&nb[1], i, "mlp_conv.kernel_projection.weight")}, &.{CONVW}, D, null);
        } else {
            lw.a_proj = undefined;
            lw.m_proj = undefined;
        }
    }
    dr.fc = try l.linear(&.{"fc.weight"}, &.{D}, geometry.taps.len * D, null);
    if (geometry.kind == .dspark) dr.hproj = undefined else dr.hproj = try l.linearB(&.{"candidate_selector.hidden_projection.weight"}, &.{RANK}, D, null);
    dr.hnorm = try l.raw("hidden_norm.weight", D * 2);
    dr.norm = try l.raw("norm.weight", D * 2);
    if (l.at > total) {
        std.log.err("dflash loader: weights need {d} B, the buffer holds {d} B", .{ l.at, total });
        return error.DrafterLayout;
    }
    var tail: Tail = .{ .loader = &l, .head_n = head.n };
    const candidates = @import("dflash_candidates.zig").Layout.init(world, MAX_M, df.cw(geometry));
    try @import("dflash_load_owner.zig").acquire(gpa, dr, &tail, Tail, .{
        .book = geometry.vocab * RANK,
        .ctx = geometry.window,
        .cand = candidates.host,
        .hidden = D,
        .gh = MAX_M * RANK,
        .kind = geometry.kind,
        .confidence = geometry.confidence,
        .block_size = geometry.block_size,
    });
    if (used.count() != header.entries.len) {
        for (header.entries) |entry| if (!used.contains(entry.name)) std.log.err("dflash loader: unused checkpoint tensor {s}", .{entry.name});
        return error.DrafterTensorUnused;
    }
    if (std.c.getenv("TF_GLM_DFLASH2_LOG")) |v| dr.log = std.fmt.parseInt(u64, std.mem.span(v), 10) catch 0;
    if (std.c.getenv("TF_GLM_DFLASH2_PROF") != null) dr.prof = io;
    std.log.info("dflash: drafter kind={s} from {s}, linears at {d} bits, {d:.2} GiB on the device", .{ @tagName(geometry.kind), dir, if (bits == 0) 16 else bits, @as(f64, @floatFromInt(l.at)) / (1 << 30) });
    memMark(gpa, io, "end of load (staging still live until return)");
    return dr;
}

const Tail = struct {
    loader: *Loader,
    head_n: usize,

    pub fn readBook(t: *Tail, i: usize, dst: []u16) !void {
        const name = if (i == 0) "candidate_selector.predecessor_codebook" else "candidate_selector.successor_codebook";
        const e = try t.loader.require(name, @intCast(dst.len * 2));
        var off: usize = 0;
        while (off < dst.len * 2) {
            const count: usize = @min(t.loader.chunk_bytes, dst.len * 2 - off);
            _ = try t.loader.file.readPositionalAll(t.loader.io, std.mem.sliceAsBytes(dst)[off..][0..count], e.off + @as(u64, @intCast(off)));
            off += count;
        }
    }

    pub fn readDspark(t: *Tail, dr: *Drafter) !void {
        try t.loader.readHost("markov_head.markov_w1.weight", dr.markov_w1);
        try t.loader.readHost("markov_head.markov_w2.weight", dr.markov_w2);
        try t.loader.readHost("confidence_head.proj.weight", dr.conf_w);
        var bias: [2]u8 = undefined;
        const e = try t.loader.require("confidence_head.proj.bias", 2);
        _ = try t.loader.file.readPositionalAll(t.loader.io, &bias, e.off);
        const b = std.mem.readInt(u16, &bias, .little);
        dr.conf_b = @bitCast(@as(u32, b) << 16);
    }

    pub fn scratch(t: *Tail, dr: *Drafter) !void {
        memMark(t.loader.gpa, t.loader.io, "weights loaded, before scratch carve");
        try carve(dr, t.loader.w.d, t.head_n);
        memMark(t.loader.gpa, t.loader.io, "scratch carved");
    }

    pub fn freeScratch(_: *Tail, dr: *Drafter) void {
        dr.scratch.free();
    }
};

/// The drafter's scratch, every field one carved piece of one buffer.
/// TF_GLM53_DRAFT_SPLIT3=1: with three ranks, split the MLP like two ranks do (gate and up by rows, down by K, the
/// fp32 partials gathered and summed by df_sum3) instead of every rank reading the whole MLP each pass.
pub fn split3Switch() bool {
    const v = std.c.getenv("TF_GLM53_DRAFT_SPLIT3") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn carve(dr: *Drafter, d: *const cuda.Driver, head_n: usize) !void {
    const g = dr.geometry;
    const D: u64 = @intCast(g.hidden);
    const KVW: u64 = @intCast(g.kvWidth());
    const QW: u64 = @intCast(g.queryWidth());
    const CONVW: u64 = if (g.conv) @intCast(g.convWidth()) else 0; // drafts.zig memoryPlan counts no conv scratch without conv
    const FF: u64 = @intCast(g.ff);
    const M: u64 = @intCast(df.MAX_S * g.block_size);
    const C: u64 = @intCast(g.window);
    const TAP_ROW: u64 = @intCast(g.tapRowBytes());
    const RANK: u64 = @intCast(g.selector_rank);
    const QH: u64 = @intCast(g.heads);
    const TOPK: u64 = @intCast(g.selector_top_k);
    const chunks: u64 = (head_n + 4095) / 4096;
    var part: u64 = 16;
    for (dr.layers) |l| {
        for ([_]df.Lin{ l.qkv, l.kv, l.o, l.gu }) |w| if (w == .bf16) {
            part = @max(part, mm.partBytes(64, w.bf16));
        };
        if (g.conv) for ([_]df.Lin{ l.a_proj, l.m_proj }) |w| if (w == .bf16) {
            part = @max(part, mm.partBytes(64, w.bf16));
        };
        part = @max(part, mm.partBytes(64, l.down));
    }
    if (dr.fc == .bf16) part = @max(part, mm.partBytes(64, dr.fc.bf16));
    if (g.kind != .dspark) part = @max(part, mm.partBytes(64, dr.hproj));
    const fields = .{
        .{ "x", M * D * 2 },                                                                     .{ "y", M * D * 2 },                          .{ "y2", M * D * 2 },             .{ "dyn", M * CONVW * 2 },
        .{ "qkv", M * g.qkvWidth() * 2 },                                                        .{ "q", M * QW * 2 },                         .{ "kb", M * KVW * 2 },           .{ "vb", M * KVW * 2 },
        .{ "att", M * QW * 2 },                                                                  .{ "a", M * D * 2 },                          .{ "gu", M * 2 * FF * 2 },        .{ "act", M * FF * 2 },
        .{ "hn", M * D * 2 },                                                                    .{ "xs", M * (D / 64) * 4 },                  .{ "g", M * RANK * 4 },           .{ "logits", M * head_n * 2 },
        .{ "part", part },                                                                       .{ "gs", C * g.taps.len * (D / 64) * 4 },     .{ "sums", @as(u64, if (dr.split) dr.world else 2) * M * D * 4 },      .{ "cin", C * TAP_ROW },
        .{ "c1", C * D * 2 },                                                                    .{ "ctx", C * D * 2 },                        .{ "ckv", C * 2 * KVW * 2 },      .{ "taps_step", df.STEP_ROWS * TAP_ROW },
        .{ "taps_prefill", C * TAP_ROW },                                                        .{ "ids", M * 4 },                            .{ "bpos", M * 4 },               .{ "rings", df.MAX_S * 8 },
        .{ "starts", df.MAX_S * 8 },                                                             .{ "froms", df.MAX_S * 8 },                   .{ "cpos", C * 8 },               .{ "crings", C * 8 },
        .{ "gtk", @import("dflash_candidates.zig").Layout.init(dr.world, M, df.cw(g)).device * 4 }, .{ "pv", M * chunks * TOPK * 4 },             .{ "pi", M * chunks * TOPK * 4 }, .{ "pm", M * QH * df.CH * 4 },
        .{ "pl", M * QH * df.CH * 4 },                                                           .{ "pacc", M * QH * df.CH * g.head_dim * 4 },
    };
    const share = dr.share_scratch;
    var at: u64 = 0;
    var shared_at: u64 = 0;
    inline for (fields) |f| {
        if (share and sharedField(f[0])) {
            @field(dr, f[0]) = shared_at;
            shared_at += std.mem.alignForward(u64, f[1], 256);
        } else {
            @field(dr, f[0]) = at;
            at += std.mem.alignForward(u64, f[1], 256);
        }
    }
    dr.shared_scratch_bytes = shared_at;
    dr.scratch = if (at == 0) .{ .d = d, .ptr = 0, .len = 0 } else try cuda.DeviceBuffer.alloc(d, at);
    errdefer if (at != 0) dr.scratch.free();
    inline for (fields) |f| {
        if (!(share and sharedField(f[0]))) @field(dr, f[0]) += dr.scratch.ptr;
    }
    if (at != 0) try d.check(d.api.cuMemsetD8_v2(dr.scratch.ptr, 0, at), "cuMemsetD8");
}

fn sharedField(comptime name: []const u8) bool {
    // Lifetime table (stream launch/record order):
    //   drafter hook between forwards | share x/y/y2/dyn/qkv/q/kb/vb/att/a/gu/act/hn/xs/g/logits/part/gs/sums/cin/c1/ctx/ckv/ids/bpos/starts/froms/cpos/crings/gtk/pv/pi/pm/pl/pacc
    //   forward pass                 | keep taps_step and taps_prefill private; the pass writes them and the next hook reads them
    //   across forward passes        | keep each slot's ring allocation private; its context survives verify and draft calls
    // Both drafter hooks and forward launches use the same ordered stream; hook scratch writes begin after the forward's last read.
    inline for (.{"x", "y", "y2", "dyn", "qkv", "q", "kb", "vb", "att", "a", "gu", "act", "hn", "xs", "g", "logits", "part", "gs", "sums", "cin", "c1", "ctx", "ckv", "ids", "bpos", "starts", "froms", "cpos", "crings", "gtk", "pv", "pi", "pm", "pl", "pacc"}) |field| {
        if (std.mem.eql(u8, name, field)) return true;
    }
    return false;
}

/// Device bytes the drafter's scratch takes beside its weights (the memory budget's share, before loading).
pub fn checkConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, g: df.Geometry) !void {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    return checkConfigBytes(gpa, text, g);
}

pub fn checkConfigBytes(gpa: std.mem.Allocator, text: []const u8, g: df.Geometry) !void {
    const p = try std.json.parseFromSlice(std.json.Value, gpa, stripHeader(text), .{});
    defer p.deinit();
    if (p.value != .object) return error.DrafterConfigUnexpected;
    const o = p.value.object;
    if (g.kind == .dspark) return checkDsparkConfig(o, g);
    if (!checkBaseConfig(o, g)) return error.DrafterConfigUnexpected;
    // Flash's real config predates these checks. Preserve precisely the field set
    // the base loader consumed; the additional geometry is GLM-5.3-specific.
    if (g.hidden == df.D) return;
    const arch = arrayField(o, "architectures") orelse return error.DrafterConfigUnexpected;
    if (arch.len != 1 or arch[0] != .string or !std.mem.eql(u8, arch[0].string, "DFlash2DraftModel") or !jsonStringEq(o, "model_type", "qwen3")) return error.DrafterConfigUnexpected;
    const dc_value = o.get("dflash_config") orelse return error.NotDFlash2;
    if (dc_value != .object) return error.DrafterConfigUnexpected;
    const dc = dc_value.object;
    if (intField(o, "head_dim") != g.head_dim or intField(o, "sliding_window") != g.window or
        !floatFieldEq(o, "rms_norm_eps", g.eps) or !boolFieldEq(o, "is_causal", false) or
        !boolFieldEq(o, "tie_word_embeddings", false) or !jsonStringEq(o, "hidden_act", "silu") or
        @as(f32, @floatCast(configRopeTheta(o))) != g.rope_theta) return error.DrafterConfigUnexpected;
    const taps = arrayField(dc, "target_layer_ids") orelse return error.DrafterConfigUnexpected;
    if (taps.len != g.taps.len) return error.DrafterConfigUnexpected;
    for (taps, g.taps) |t, w| if (t != .integer or t.integer < 0 or t.integer != w) return error.DrafterConfigUnexpected;
    const layer_types = arrayField(o, "layer_types") orelse return error.DrafterConfigUnexpected;
    if (layer_types.len != g.layers) return error.DrafterConfigUnexpected;
    for (layer_types) |kind| if (kind != .string or !std.mem.eql(u8, kind.string, "sliding_attention")) return error.DrafterConfigUnexpected;
}

fn checkDsparkConfig(o: std.json.ObjectMap, g: df.Geometry) !void {
    if (!jsonStringEq(o, "speculators_model_type", "dspark") or !boolFieldEq(o, "enable_confidence_head", true) or
        !boolFieldEq(o, "confidence_head_with_markov", true) or !boolFieldEq(o, "sample_from_anchor", true) or
        !jsonStringEq(o, "markov_head_type", "vanilla") or !intFieldEq(o, "block_size", 8) or
        !intFieldEq(o, "markov_rank", g.selector_rank) or !intFieldEq(o, "mask_token_id", g.mask)) return error.UnsupportedDspark;
    const tc_value = o.get("transformer_layer_config") orelse return error.UnsupportedDspark;
    if (tc_value != .object) return error.UnsupportedDspark;
    const tc = tc_value.object;
    if (!intFieldEq(tc, "hidden_size", g.hidden) or !intFieldEq(tc, "num_hidden_layers", g.layers) or
        !intFieldEq(tc, "num_attention_heads", g.heads) or !intFieldEq(tc, "num_key_value_heads", g.kv_heads) or
        !intFieldEq(tc, "head_dim", g.head_dim) or !intFieldEq(tc, "intermediate_size", g.ff) or
        !intFieldEq(tc, "vocab_size", g.vocab) or !intFieldEq(tc, "sliding_window", g.window) or
        !floatFieldEq(tc, "rms_norm_eps", g.eps) or !jsonStringEq(tc, "model_type", "qwen3") or
        !jsonStringEq(tc, "hidden_act", "silu")) return error.UnsupportedDspark;
    const rope_v = tc.get("rope_parameters") orelse return error.UnsupportedDspark;
    if (rope_v != .object or @as(f32, @floatCast(configRopeTheta(tc))) != g.rope_theta) return error.UnsupportedDspark;
    const layer_types = arrayField(tc, "layer_types") orelse return error.UnsupportedDspark;
    if (layer_types.len != g.layers) return error.UnsupportedDspark;
    for (layer_types) |kind| if (kind != .string or !std.mem.eql(u8, kind.string, "sliding_attention")) return error.UnsupportedDspark;
    const aux = arrayField(o, "aux_hidden_state_layer_ids") orelse return error.UnsupportedDspark;
    const expected = [_]usize{ 2, 20, 39, 58, 75 };
    if (aux.len != expected.len or g.taps.len != expected.len) return error.UnsupportedDspark;
    for (aux, expected, g.taps) |v, source, target| if (v != .integer or v.integer != source or target != source - 1) return error.UnsupportedDspark;
}

fn checkBaseConfig(o: std.json.ObjectMap, g: df.Geometry) bool {
    const dc_value = o.get("dflash_config") orelse return false;
    if (dc_value != .object) return false;
    const dc = dc_value.object;
    if (!intFieldEq(o, "hidden_size", g.hidden) or !intFieldEq(o, "num_hidden_layers", g.layers) or
        !intFieldEq(o, "num_attention_heads", g.heads) or !intFieldEq(o, "num_key_value_heads", g.kv_heads) or
        !intFieldEq(o, "intermediate_size", g.ff) or !intFieldEq(o, "vocab_size", g.vocab) or
        !intFieldEq(dc, "block_size", g.block_size) or !intFieldEq(dc, "mask_token_id", g.mask) or
        !intFieldEq(dc, "selector_top_k", g.selector_top_k) or !intFieldEq(dc, "selector_rank", g.selector_rank) or
        !intFieldEq(dc, "conv_kernel_size", g.conv_kernel) or !intFieldEq(dc, "conv_group_size", g.conv_group)) return false;
    const taps = arrayField(dc, "target_layer_ids") orelse return false;
    if (taps.len != g.taps.len) return false;
    for (taps, g.taps) |t, expected| if (t != .integer or t.integer < 0 or t.integer != expected) return false;
    return true;
}

fn intFieldEq(o: std.json.ObjectMap, key: []const u8, expected: usize) bool {
    const v = o.get(key) orelse return false;
    return v == .integer and v.integer >= 0 and v.integer == expected;
}

fn intField(o: std.json.ObjectMap, key: []const u8) usize {
    const v = o.get(key) orelse return 0;
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else 0;
}

fn boolFieldEq(o: std.json.ObjectMap, key: []const u8, expected: bool) bool {
    const v = o.get(key) orelse return false;
    return v == .bool and v.bool == expected;
}

fn floatFieldEq(o: std.json.ObjectMap, key: []const u8, expected: f32) bool {
    const v = o.get(key) orelse return false;
    return v == .integer and @as(f32, @floatFromInt(v.integer)) == expected or v == .float and @as(f32, @floatCast(v.float)) == expected;
}

fn arrayField(o: std.json.ObjectMap, key: []const u8) ?[]std.json.Value {
    const v = o.get(key) orelse return null;
    return if (v == .array) v.array.items else null;
}

fn jsonStringEq(o: std.json.ObjectMap, key: []const u8, expected: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, expected);
}

fn stripHeader(text: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, text, "//")) return text;
    return if (std.mem.indexOfScalar(u8, text, '\n')) |nl| text[nl + 1 ..] else text;
}

fn jsonNumber(v: std.json.Value) f64 {
    return if (v == .float) v.float else if (v == .integer) @floatFromInt(v.integer) else std.math.nan(f64);
}

fn configRopeTheta(o: std.json.ObjectMap) f64 {
    if (o.get("rope_theta")) |v| return jsonNumber(v);
    const rope = o.get("rope_parameters") orelse return std.math.nan(f64);
    if (rope != .object) return std.math.nan(f64);
    const theta = rope.object.get("rope_theta") orelse return std.math.nan(f64);
    return jsonNumber(theta);
}
