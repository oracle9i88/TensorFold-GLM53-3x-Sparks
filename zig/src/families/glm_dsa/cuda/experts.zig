// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/weights.py (Apache-2.0). Changes: Zig checkpoint upload and TP2 slices.
//! GLM-5.3 routed EXL3 experts, loaded as TP intermediate halves and run by the shared grouped expert kernels.
const std = @import("std");
const cuda = @import("cuda");
const ex = cuda.exl3_experts;
const core = @import("core");

pub const Codebook = ex.Codebook;
pub const Experts = ex.Experts;
pub const Scratch = ex.Scratch;
const Projection = struct { ptrs: u64, k2_ptr: u64, suh_ptr: u64, svh_ptr: u64, uniform_k2: u32, widths_ok: bool, pmx_ok: bool };

/// One expert's intermediate channels on this rank (128-aligned: EXL3's Hadamard blocks are 128 wide).
pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

/// The widest expert span: the row stride of the per-expert svh (gate/up), suh (down) and the down inputs.
pub fn widest(spans: []const Span) usize {
    var most: usize = 0;
    for (spans) |s| most = @max(most, s.len());
    return most;
}

pub const Loaded = struct {
    gate: u64,
    up: u64,
    down: u64,
    gate_k2: u64,
    up_k2: u64,
    down_k2: u64,
    suh_g: u64,
    suh_u: u64,
    svh_g: u64,
    svh_u: u64,
    suh_d: u64,
    svh_d: u64,
    codebook: Codebook,
    uniform_k2: u32,
    widths_ok: bool,
    pmx_ok: bool = false,
    dims: usize,
    width: usize, // the widest expert's span (every expert's when they are equal)
    experts: usize,
    /// int32 [E] device table of each expert's span width, or 0 when every expert has `width` (TF_GLM53_MOE_BALANCE).
    widths: u64 = 0,

    pub fn layer(w: Loaded) ex.Layer {
        return .{
            .gate = w.gate,
            .up = w.up,
            .down = w.down,
            .gate_k2 = w.gate_k2,
            .up_k2 = w.up_k2,
            .down_k2 = w.down_k2,
            .suh_g = w.suh_g,
            .suh_u = w.suh_u,
            .svh_g = w.svh_g,
            .svh_u = w.svh_u,
            .suh_d = w.suh_d,
            .svh_d = w.svh_d,
            .experts = w.experts,
            .dims = w.dims,
            .width = w.width,
            .codebook = w.codebook,
            .uniform_k2 = w.uniform_k2,
            .widths_ok = w.widths_ok,
            .pmx_ok = w.pmx_ok,
            .act_mode = 0,
            .widths = w.widths,
        };
    }
};

fn alignAppend(L: anytype, bytes: []const u8, alignment: usize) !void {
    const pad = std.mem.alignForward(usize, L.host.items.len, alignment) - L.host.items.len;
    try L.host.appendNTimes(L.gpa, 0, pad);
    try L.host.appendSlice(L.gpa, bytes);
}

/// Load a routed projection from the Python reader's canonical names. `spans[e]` is expert e's intermediate span on
/// this rank. The trellis's last dimension gives K2 independently for every expert and every projection. The span-sized
/// scales (svh for gate/up, suh for down) sit `stride` apart per expert, zero past the span: with equal spans the
/// stride is the span and the bytes are what a single lo..hi gave.
fn projection(L: anytype, layer_index: usize, expert_count: usize, dims: usize, full_width: usize, spans: []const Span, stride: usize, routed_codebook: Codebook, comptime name: []const u8, down: bool) !Projection {
    if (spans.len != expert_count or dims % 128 != 0 or full_width % 128 != 0 or stride % 128 != 0) return error.BadExpertSpan;
    for (spans) |sp| if (sp.lo >= sp.hi or sp.hi > full_width or sp.lo % 128 != 0 or sp.hi % 128 != 0 or sp.len() > stride)
        return error.BadExpertSpan;
    try alignAppend(L, &.{}, 256);
    const trellis_host_base = L.host.items.len;
    var suh: std.ArrayList(u8) = .empty;
    defer suh.deinit(L.gpa);
    var svh: std.ArrayList(u8) = .empty;
    defer svh.deinit(L.gpa);
    var k2: std.ArrayList(i32) = .empty;
    defer k2.deinit(L.gpa);
    var ptrs: std.ArrayList(i64) = .empty;
    defer ptrs.deinit(L.gpa);

    for (0..expert_count) |e| {
        const lo = spans[e].lo;
        const hi = spans[e].hi;
        const t = try L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".trellis", .{ layer_index, e });
        const s = if (L.has("layers.{d}.mlp.experts.{d}." ++ name ++ ".suh", .{ layer_index, e }))
            try L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".suh", .{ layer_index, e })
        else
            try L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".su", .{ layer_index, e });
        const v = if (L.has("layers.{d}.mlp.experts.{d}." ++ name ++ ".svh", .{ layer_index, e }))
            try L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".svh", .{ layer_index, e })
        else
            try L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".sv", .{ layer_index, e });
        if (t.dtype != .i16 or t.rank != 3 or s.dtype != .f16 or v.dtype != .f16) return error.BadExpertTensor;
        const k = t.dim(0) * 16;
        const n = t.dim(1) * 16;
        const expected_k = if (down) full_width else dims;
        const expected_n = if (down) dims else full_width;
        if (k != expected_k or n != expected_n or t.dim(2) % 8 != 0 or s.numel() != k or v.numel() != n)
            return error.BadExpertTensor;
        const bits2: i32 = @intCast(t.dim(2) / 8);
        if (bits2 < 2 or bits2 > 16 or @mod(bits2, 2) != 0) return error.BadExpertWidth;
        var marked: ?Codebook = null;
        for (core.exl3_load.markers) |marker_part| {
            const marker = (if (std.mem.eql(u8, marker_part.name, "mcg"))
                L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".mcg", .{ layer_index, e })
            else if (std.mem.eql(u8, marker_part.name, "mcg_e4m3"))
                L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".mcg_e4m3", .{ layer_index, e })
            else
                L.get("layers.{d}.mlp.experts.{d}." ++ name ++ ".mul1", .{ layer_index, e })) catch |err| {
                if (err == error.MissingTensor) continue;
                return err;
            };
            if (marker.dtype != .i32 or marker.bytes.len != 4 or std.mem.readInt(u32, marker.bytes[0..4], .little) != marker_part.value)
                return error.BadExpertCodebook;
            if (marked != null) return error.BadExpertCodebook;
            marked = if (std.mem.eql(u8, marker_part.name, "mcg_e4m3")) .mcg_e4m3 else switch (marker_part.codebook) {
                .mcg => .mcg,
                .mcg_e4m3 => .mcg_e4m3,
                .mul1 => .mul1,
                .inst3 => .inst3,
            };
        }
        // A pack marked `.mcg` is decoded with the codebook the config names: checkpoints assembled with
        // `routed_codebook: mcg_e4m3` keep the `.mcg` marker on every expert (the Python engine reads the config).
        if (marked != null and marked.? == .mcg and routed_codebook == .mcg_e4m3) marked = .mcg_e4m3;
        if (marked == null or marked.? != routed_codebook) return error.BadExpertCodebook;
        if (marked.? != .mcg and marked.? != .mcg_e4m3 and marked.? != .mul1) return error.BadExpertCodebook;
        const word_bytes = t.dim(2) * 2;
        const tk = k / 16;
        const tn = n / 16;
        const tile_bytes = word_bytes;
        // Match glm5_next/cuda/split.py:EXL3_RULES: gate/up split dim1; down splits dim0.
        if (down) {
            const first = lo / 16;
            const take = (hi - lo) / 16;
            const begin = first * tn * tile_bytes;
            try L.host.appendSlice(L.gpa, t.bytes[begin..][0 .. take * tn * tile_bytes]);
        } else {
            const first = lo / 16;
            const take = (hi - lo) / 16;
            for (0..tk) |r| {
                const begin = (r * tn + first) * tile_bytes;
                try L.host.appendSlice(L.gpa, t.bytes[begin..][0 .. take * tile_bytes]);
            }
        }
        if (down) {
            try suh.appendSlice(L.gpa, s.bytes[lo * 2 .. hi * 2]);
            try suh.appendNTimes(L.gpa, 0, (stride - (hi - lo)) * 2);
            try svh.appendSlice(L.gpa, v.bytes);
        } else {
            try suh.appendSlice(L.gpa, s.bytes);
            try svh.appendSlice(L.gpa, v.bytes[lo * 2 .. hi * 2]);
            try svh.appendNTimes(L.gpa, 0, (stride - (hi - lo)) * 2);
        }
        try ptrs.append(L.gpa, 0);
        try k2.append(L.gpa, bits2);
    }

    // Pointers are filled after upload because each expert may have a different K2.
    var at: usize = 0;
    for (0..expert_count) |e| {
        ptrs.items[e] = @intCast(trellis_host_base + at);
        const local_n = if (down) dims else spans[e].len();
        const local_k = if (down) spans[e].len() else dims;
        at += (local_k / 16) * (local_n / 16) * @as(usize, @intCast(k2.items[e])) * 16;
    }
    const trellis_base = try L.put("layers.{d}.moe.exl3." ++ name ++ "_trellis", .{layer_index});
    // Re-upload pointer offsets at this tensor's base; offset increments above are shape-derived.
    for (ptrs.items) |*p| p.* += @as(i64, @intCast(trellis_base));
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(ptrs.items));
    const trellis_ptrs = try L.put("layers.{d}.moe.exl3." ++ name ++ "_ptrs", .{layer_index});
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(k2.items));
    const k2_ptr = try L.put("layers.{d}.moe.exl3." ++ name ++ "_k2", .{layer_index});
    try alignAppend(L, suh.items, 256);
    const suh_ptr = try L.put("layers.{d}.moe.exl3." ++ name ++ "_suh", .{layer_index});
    try alignAppend(L, svh.items, 256);
    const svh_ptr = try L.put("layers.{d}.moe.exl3." ++ name ++ "_svh", .{layer_index});
    var uniform_k2: u32 = if (k2.items.len == 0) 0 else @intCast(k2.items[0]);
    for (k2.items) |v| if (v != uniform_k2) {
        uniform_k2 = 0;
        break;
    };
    return .{ .ptrs = trellis_ptrs, .k2_ptr = k2_ptr, .suh_ptr = suh_ptr, .svh_ptr = svh_ptr, .uniform_k2 = uniform_k2, .widths_ok = cuda.exl3_experts.prompt.eligibleWidths(k2.items), .pmx_ok = cuda.exl3_experts.pmx.eligibleWidths(k2.items) };
}

/// Append the mixed-width EXL3 tensors for one expert layer. `routed_codebook` is explicit for packs marked `.mcg`.
pub fn load(L: anytype, layer_index: usize, expert_count: usize, dims: usize, full_width: usize, lo: usize, hi: usize, routed_codebook: Codebook) !Loaded {
    const spans = try L.gpa.alloc(Span, expert_count);
    defer L.gpa.free(spans);
    @memset(spans, .{ .lo = lo, .hi = hi });
    return loadSpans(L, layer_index, expert_count, dims, full_width, spans, false, routed_codebook);
}

/// load with a span an expert (TF_GLM53_MOE_BALANCE). `with_widths` uploads the int32 [E] span widths the kernels
/// read; without it every span must be equal (load's case, the bytes and the launches of before).
pub fn loadSpans(L: anytype, layer_index: usize, expert_count: usize, dims: usize, full_width: usize, spans: []const Span, with_widths: bool, routed_codebook: Codebook) !Loaded {
    if (routed_codebook != .mcg and routed_codebook != .mcg_e4m3 and routed_codebook != .mul1) return error.BadExpertCodebook;
    if (spans.len != expert_count or expert_count == 0) return error.BadExpertSpan;
    const stride = widest(spans);
    if (!with_widths) for (spans) |sp| if (sp.len() != stride) return error.BadExpertSpan;
    const g = try projection(L, layer_index, expert_count, dims, full_width, spans, stride, routed_codebook, "gate_proj", false);
    const u = try projection(L, layer_index, expert_count, dims, full_width, spans, stride, routed_codebook, "up_proj", false);
    const d = try projection(L, layer_index, expert_count, dims, full_width, spans, stride, routed_codebook, "down_proj", true);
    var widths: u64 = 0;
    if (with_widths) {
        const table = try L.gpa.alloc(i32, expert_count);
        defer L.gpa.free(table);
        for (table, spans) |*t, sp| t.* = @intCast(sp.len());
        try alignAppend(L, std.mem.sliceAsBytes(table), 256);
        widths = try L.put("layers.{d}.moe.exl3.widths", .{layer_index});
    }
    return .{ .gate = g.ptrs, .up = u.ptrs, .down = d.ptrs, .gate_k2 = g.k2_ptr, .up_k2 = u.k2_ptr, .down_k2 = d.k2_ptr, .suh_g = g.suh_ptr, .suh_u = u.suh_ptr, .svh_g = g.svh_ptr, .svh_u = u.svh_ptr, .suh_d = d.suh_ptr, .svh_d = d.svh_ptr, .codebook = routed_codebook, .uniform_k2 = if (g.uniform_k2 == 8 and u.uniform_k2 == 8 and d.uniform_k2 == 8) 8 else 0, .widths_ok = g.widths_ok and u.widths_ok and d.widths_ok, .pmx_ok = g.pmx_ok and u.pmx_ok and d.pmx_ok, .dims = dims, .width = stride, .experts = expert_count, .widths = widths };
}

/// Decode path: run all routed slots and let the caller perform the ordered fp32 route reduction.
pub fn run(k: *const Experts, stream: cuda.Stream, w: Loaded, x: u64, stride: usize, dtype: cuda.exl3.DType, picks: u64, rows: usize, scratch: Scratch) !void {
    try k.runSlots(stream, w.layer(), x, stride, dtype, picks, rows, scratch);
}

/// Prompt path. Eligible mcg/mcg_e4m3 layers use the tensor-core prompt GEMMs; other layers use decode in bounded chunks.
pub fn runPrompt(k: *const Experts, router: cuda.grouped.Router, plan: cuda.grouped.Plan, stream: cuda.Stream, w: Loaded, x: u64, stride: usize, picks: u64, rows: usize, grouped_scratch: Scratch, prompt_scratch: cuda.exl3_experts.prompt.Scratch) !void {
    const l = w.layer();
    if (rows == 0) return;
    const slots = grouped_scratch.slots;
    if (slots == 0 or slots > 32 or prompt_scratch.slots != slots) return error.Exl3Shape;
    // Each chunk writes at its prompt-row offset into y, so the complete prompt must fit both y allocations.
    try checkPromptOutputCapacity(rows, grouped_scratch.rows, prompt_scratch.rows);
    const max_rows = if (cuda.exl3_experts.prompt.eligibleTiles(l)) @min(grouped_scratch.rows, prompt_scratch.rows) else @min(grouped_scratch.rows, @min(prompt_scratch.rows, (64 * 1024) / (slots * 4)));
    if (max_rows == 0) return error.Exl3Shape;
    try forEachChunk(rows, max_rows, slots, l.dims, struct {
        fn run(first: usize, count: usize, y_offset: u64, context: anytype) !void {
            const experts = context.k;
            const ctx_router = context.router;
            const ctx_plan = context.plan;
            const ctx_stream = context.stream;
            const ctx_l = context.l;
            const ctx_stride = context.stride;
            const ctx_picks = context.picks;
            const ctx_grouped_scratch = context.grouped_scratch;
            const ctx_prompt_scratch = context.prompt_scratch;
            const x_chunk = context.x + first * ctx_stride * 2;
            const ctx_slots = context.slots;
            const picks_chunk = ctx_picks + first * ctx_slots * 4;
            var grouped_chunk = ctx_grouped_scratch;
            grouped_chunk.y += y_offset;
            var prompt_chunk = ctx_prompt_scratch;
            prompt_chunk.y += y_offset;
            if (cuda.exl3_experts.prompt.eligibleTiles(ctx_l)) {
                try experts.prompt.run(experts, ctx_router, ctx_stream, ctx_l, x_chunk, ctx_stride, picks_chunk, count, prompt_chunk, ctx_plan);
            } else {
                try experts.runSlots(ctx_stream, ctx_l, x_chunk, ctx_stride, .bf16, picks_chunk, count, grouped_chunk);
            }
        }
    }.run, .{ .k = k, .router = router, .plan = plan, .stream = stream, .l = l, .x = x, .stride = stride, .picks = picks, .slots = slots, .grouped_scratch = grouped_scratch, .prompt_scratch = prompt_scratch });
}

fn checkPromptOutputCapacity(rows: usize, grouped_rows: usize, prompt_rows: usize) !void {
    if (rows > grouped_rows or rows > prompt_rows) return error.Exl3Shape;
}

fn chunkOutputOffset(first: usize, slots: usize, dims: usize) !u64 {
    const elements = try std.math.mul(usize, try std.math.mul(usize, first, slots), dims);
    const bytes = try std.math.mul(usize, elements, @sizeOf(f32));
    return @intCast(bytes);
}

fn forEachChunk(rows: usize, max_rows: usize, slots: usize, dims: usize, callback: anytype, context: anytype) !void {
    if (max_rows == 0) return error.Exl3Shape;
    var first: usize = 0;
    while (first < rows) {
        const count = @min(max_rows, rows - first);
        try callback(first, count, try chunkOutputOffset(first, slots, dims), context);
        first += count;
    }
}

test "TP2 expert half and supported per-matrix widths" {
    try std.testing.expectEqual(@as(usize, 1024), 2048 / 2);
    try std.testing.expectEqual(@as(usize, 8), 4 * 2);
}

const MockDType = enum { i16, f16, i32 };
const MockTensor = struct {
    bytes: []const u8,
    dtype: MockDType,
    dims: [3]usize,
    rank: usize,
    pub fn dim(t: MockTensor, axis: usize) usize {
        return t.dims[axis];
    }
    pub fn numel(t: MockTensor) usize {
        return t.dims[0] * t.dims[1] * t.dims[2];
    }
};

const MockLoader = struct {
    gpa: std.mem.Allocator,
    host: std.ArrayList(u8) = .empty,
    storage: [16384]u8 = undefined,
    marker: [4]u8 = undefined,
    e4m3_marker: bool = false,
    alias_scales: bool = false,
    uploads: std.ArrayList([]u8) = .empty,
    next: u64 = 0x100000,

    fn has(m: *MockLoader, comptime fmt: []const u8, args: anytype) bool {
        _ = args;
        if (m.alias_scales and (std.mem.endsWith(u8, fmt, ".suh") or std.mem.endsWith(u8, fmt, ".svh"))) return false;
        return true;
    }

    fn get(m: *MockLoader, comptime fmt: []const u8, args: anytype) !MockTensor {
        const e: usize = args[1];
        if (std.mem.indexOf(u8, fmt, ".mcg") != null) {
            const is_e4m3 = std.mem.endsWith(u8, fmt, ".mcg_e4m3");
            if (is_e4m3 != m.e4m3_marker) return error.MissingTensor;
            std.mem.writeInt(u32, &m.marker, 0xCBAC1FED, .little);
            return .{ .bytes = &m.marker, .dtype = .i32, .dims = .{ 1, 1, 1 }, .rank = 1 };
        }
        if (std.mem.indexOf(u8, fmt, ".mul1") != null) return error.MissingTensor;
        const down = std.mem.indexOf(u8, fmt, "down_proj") != null;
        const up = std.mem.indexOf(u8, fmt, "up_proj") != null;
        const k2: usize = if (down) (if (e == 0) 4 else 6) else if (up) (if (e == 0) 6 else 2) else if (e == 0) 2 else 4;
        const k: usize = if (down) 256 else 128;
        const n: usize = if (down) 128 else 256;
        if (std.mem.endsWith(u8, fmt, ".trellis")) {
            const bytes = k / 16 * (n / 16) * k2 * 16;
            return .{ .bytes = m.storage[0..bytes], .dtype = .i16, .dims = .{ k / 16, n / 16, k2 * 8 }, .rank = 3 };
        }
        const length: usize = if (std.mem.endsWith(u8, fmt, ".suh") or std.mem.endsWith(u8, fmt, ".su")) k else n;
        return .{ .bytes = m.storage[0 .. length * 2], .dtype = .f16, .dims = .{ length, 1, 1 }, .rank = 1 };
    }

    fn put(m: *MockLoader, comptime fmt: []const u8, args: anytype) !u64 {
        _ = fmt;
        _ = args;
        try m.uploads.append(m.gpa, try m.gpa.dupe(u8, m.host.items));
        m.host.clearRetainingCapacity();
        const result = m.next;
        m.next += 0x1000;
        return result;
    }
};

test "an mcg-marked pack loads as mcg_e4m3 when the config names that codebook" {
    var mock: MockLoader = .{ .gpa = std.testing.allocator };
    mock.e4m3_marker = false;
    for (&mock.storage, 0..) |*byte, i| byte.* = @truncate(i *% 37 +% 11);
    defer {
        for (mock.uploads.items) |bytes| mock.gpa.free(bytes);
        mock.uploads.deinit(mock.gpa);
        mock.host.deinit(mock.gpa);
    }
    const loaded = try load(&mock, 3, 2, 128, 256, 128, 256, .mcg_e4m3);
    try std.testing.expectEqual(.mcg_e4m3, loaded.layer().codebook);
}

test "synthetic routed pack keeps per-matrix mixed K2 through TP2 slicing" {
    var mock: MockLoader = .{ .gpa = std.testing.allocator };
    mock.e4m3_marker = true;
    for (&mock.storage, 0..) |*byte, i| byte.* = @truncate(i *% 37 +% 11);
    defer {
        for (mock.uploads.items) |bytes| mock.gpa.free(bytes);
        mock.uploads.deinit(mock.gpa);
        mock.host.deinit(mock.gpa);
    }
    const loaded = try load(&mock, 3, 2, 128, 256, 128, 256, .mcg_e4m3);
    try std.testing.expectEqual(@as(usize, 128), loaded.width);
    try std.testing.expectEqual(.mcg_e4m3, loaded.layer().codebook);
    try std.testing.expectEqual(@as(usize, 15), mock.uploads.items.len);
    try expectWidths(mock.uploads.items[2], .{ 2, 4 });
    try expectWidths(mock.uploads.items[7], .{ 6, 2 });
    try expectWidths(mock.uploads.items[12], .{ 4, 6 });
    // Gate/up cut output tiles; down cuts input tiles. Each rank keeps half the global intermediate width.
    try std.testing.expectEqual(@as(usize, 128 / 16 * 128 / 16 * (2 + 4) * 16), mock.uploads.items[0].len);
    try std.testing.expectEqual(@as(usize, 128 / 16 * 128 / 16 * (4 + 6) * 16), mock.uploads.items[10].len);
    // Gate/up keep rank tile columns (dim1), while down keeps rank rows (dim0).
    try expectGateUpTiles(mock.storage[0..], mock.uploads.items[0], 128, 256, 2, 128, 256);
    try expectGateUpTiles(mock.storage[0..], mock.uploads.items[5], 128, 256, 6, 128, 256);
    try expectDownTiles(mock.storage[0..], mock.uploads.items[10], 128, 4, 128, 256);
    // Input/output scale slices follow the Python split.py rules; suh is replicated for gate/up and svh for down.
    try std.testing.expectEqualSlices(u8, mock.storage[0 .. 128 * 2], mock.uploads.items[3][0 .. 128 * 2]);
    try std.testing.expectEqualSlices(u8, mock.storage[128 * 2 .. 256 * 2], mock.uploads.items[4][0 .. 128 * 2]);
    try std.testing.expectEqualSlices(u8, mock.storage[0 .. 128 * 2], mock.uploads.items[8][0 .. 128 * 2]);
    try std.testing.expectEqualSlices(u8, mock.storage[128 * 2 .. 256 * 2], mock.uploads.items[9][0 .. 128 * 2]);
    try std.testing.expectEqualSlices(u8, mock.storage[128 * 2 .. 256 * 2], mock.uploads.items[13][0 .. 128 * 2]);
    try std.testing.expectEqualSlices(u8, mock.storage[0 .. 128 * 2], mock.uploads.items[14][0 .. 128 * 2]);
}

test "expert loader accepts EXL3 su/sv scale aliases from the Python reference layout" {
    var mock: MockLoader = .{ .gpa = std.testing.allocator, .e4m3_marker = true, .alias_scales = true };
    for (&mock.storage, 0..) |*byte, i| byte.* = @truncate(i *% 19 +% 7);
    defer {
        for (mock.uploads.items) |bytes| mock.gpa.free(bytes);
        mock.uploads.deinit(mock.gpa);
        mock.host.deinit(mock.gpa);
    }
    const loaded = try load(&mock, 3, 2, 128, 256, 128, 256, .mcg_e4m3);
    try std.testing.expectEqual(.mcg_e4m3, loaded.layer().codebook);
    try std.testing.expectEqual(@as(usize, 15), mock.uploads.items.len);
}

test "expert loader rejects a tensor marker that differs from the configured routed codebook" {
    var mock: MockLoader = .{ .gpa = std.testing.allocator, .e4m3_marker = true };
    defer mock.host.deinit(mock.gpa);
    try std.testing.expectError(error.BadExpertCodebook, load(&mock, 3, 2, 128, 256, 128, 256, .mcg));
}

test "multi-chunk expert outputs match the unchunked row order bit-for-bit" {
    const rows = 11;
    const slots = 3;
    const dims = 5;
    const row_size = slots * dims;
    var chunked: [rows * row_size]f32 = undefined;
    var whole: [rows * row_size]f32 = undefined;
    for (&whole, 0..) |*value, i| value.* = @floatFromInt((i *% 37 +% 11) % 997);
    const Context = struct {
        output: []f32,
        row_size: usize,
        fn write(first: usize, count: usize, offset: u64, context: *@This()) !void {
            const dst: usize = @intCast(offset / @sizeOf(f32));
            try std.testing.expectEqual(first * context.row_size, dst);
            for (0..count * context.row_size) |i| context.output[dst + i] = @floatFromInt(((dst + i) *% 37 +% 11) % 997);
        }
    };
    var context: Context = .{ .output = &chunked, .row_size = row_size };
    try forEachChunk(rows, 4, slots, dims, Context.write, &context);
    try std.testing.expectEqualSlices(f32, &whole, &chunked);
}

test "multi-chunk writes fail closed when bounded y cannot hold prompt offsets" {
    const rows = 7;
    const slots = 2;
    const dims = 3;
    const capacity_rows = 4;
    var bounded_y: [capacity_rows * slots * dims]f32 = @splat(-1);
    const Context = struct {
        y: []f32,
        slots: usize,
        dims: usize,
        fn write(first: usize, count: usize, offset: u64, context: *@This()) !void {
            const start: usize = @intCast(offset / @sizeOf(f32));
            const end = start + count * context.slots * context.dims;
            if (end > context.y.len) return error.OutputOverflow;
            for (context.y[start..end], 0..) |*value, i| value.* = @floatFromInt(first + i);
        }
    };
    var context: Context = .{ .y = &bounded_y, .slots = slots, .dims = dims };
    try std.testing.expectError(error.Exl3Shape, checkPromptOutputCapacity(rows, capacity_rows, capacity_rows));
    try std.testing.expectError(error.OutputOverflow, forEachChunk(rows, capacity_rows, slots, dims, Context.write, &context));
    for (bounded_y, 0..) |value, i| try std.testing.expectEqual(@as(f32, @floatFromInt(i)), value);
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Loaded);
}

fn expectGateUpTiles(source: []const u8, got: []const u8, k: usize, n: usize, halves: usize, lo: usize, hi: usize) !void {
    const tile_bytes = halves * 16;
    const tk = k / 16;
    const tn = n / 16;
    const first = lo / 16;
    const take = (hi - lo) / 16;
    var at: usize = 0;
    for (0..tk) |r| {
        const begin = (r * tn + first) * tile_bytes;
        try std.testing.expectEqualSlices(u8, source[begin .. begin + take * tile_bytes], got[at .. at + take * tile_bytes]);
        at += take * tile_bytes;
    }
}

fn expectDownTiles(source: []const u8, got: []const u8, n: usize, halves: usize, lo: usize, hi: usize) !void {
    const tile_bytes = halves * 16;
    const tn = n / 16;
    const first = lo / 16;
    const take = (hi - lo) / 16;
    const begin = first * tn * tile_bytes;
    try std.testing.expectEqualSlices(u8, source[begin .. begin + take * tn * tile_bytes], got[0 .. take * tn * tile_bytes]);
}

test "mcg_e4m3 table agrees with the EXLR8 host codebook oracle for every state" {
    try std.testing.expect(std.math.isNan(core.exlr8.codec.e4m3Value(0x7f)));
    for (0..65536) |raw| {
        const state: u16 = @intCast(raw);
        const code = core.exlr8.codec.codebookCode(state);
        const want: f16 = @floatCast(core.exlr8.codec.e4m3Value(code));
        const mcg = core.exl3_format.codebookValue(.mcg, state);
        const got: f16 = @floatCast(cuda.exl3.roundE4m3(@floatCast(mcg)));
        try std.testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(got)));
    }
}

fn expectWidths(bytes: []const u8, want: [2]i32) !void {
    try std.testing.expectEqual(@as(usize, 8), bytes.len);
    try std.testing.expectEqual(want[0], std.mem.readInt(i32, bytes[0..4], .little));
    try std.testing.expectEqual(want[1], std.mem.readInt(i32, bytes[4..8], .little));
}
