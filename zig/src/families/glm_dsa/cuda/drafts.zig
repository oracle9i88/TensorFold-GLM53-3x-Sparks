// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! GLM-5.3 DFlash2 config admission and per-rank memory sizing.
const std = @import("std");
const cuda = @import("cuda");
pub const Geometry = @import("geometry").Geometry;
const glm = @import("glm");

const EmbedRow = struct { owner: i32, local: i32 };
fn embedRowFor(token: u32, rank: usize, vocab_local: usize) EmbedRow {
    if (vocab_local == 51712 or vocab_local == 51584) {
        const o: usize = if (token < 51712) 0 else if (token < 103296) 1 else 2;
        const offset: u32 = if (o == 0) 0 else if (o == 1) 51712 else 103296;
        return .{ .owner = @intCast(o), .local = if (o == rank) @intCast(token - offset) else 0 };
    }
    const o = token / vocab_local;
    return .{ .owner = @intCast(o), .local = if (o == rank) @intCast(token - o * vocab_local) else 0 };
}

pub const head_chunk_rows: usize = 8192;

pub fn shareScratchEnabled() bool {
    const raw = std.c.getenv("TF_GLM53_DRAFTER_SHARE") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "1");
}

pub fn parseConfTemp(value: ?[]const u8) !f64 {
    const v = value orelse return 1.0;
    const parsed = std.fmt.parseFloat(f64, v) catch return error.BadDraftConfTemp;
    if (!std.math.isFinite(parsed) or parsed < 0.05 or parsed > 20.0) return error.BadDraftConfTemp;
    return parsed;
}

pub fn confTempSwitch() !f64 {
    const raw = std.c.getenv("TF_GLM53_DRAFT_CONF_TEMP") orelse return 1.0;
    return parseConfTemp(std.mem.span(raw));
}

/// TF_GLM53_DRAFT_CHANCE_TEMP: the DFlash2 selector's softmax temperature for the chances (1: as trained, same range
/// as the DSpark confidence temperature). DSpark never reads it.
pub fn chanceTempSwitch() !f64 {
    const raw = std.c.getenv("TF_GLM53_DRAFT_CHANCE_TEMP") orelse return 1.0;
    return parseConfTemp(std.mem.span(raw));
}

/// Host mirror of tf_glm53_unpack_bf16_rows' inverse qmmf packing map.
pub fn packedIndex(row: usize, col: usize, k: usize) usize {
    const rr = row % 64;
    const cc = col % 64;
    const q = cc % 16;
    const rest = q % 8;
    const lane = (rr % 8) * 4 + rest / 2;
    const e = (q / 8) * 2 + rest % 2;
    const tg = (row / 64) * (k / 64) + col / 64;
    return tg * 4096 + (rr / 8) * 512 + (cc / 16) * 128 + lane * 4 + e;
}

pub fn unpackRows(f: cuda.Function, s: cuda.Stream, packed_w: u64, out: u64, k: usize, row0: usize, rows: usize) !void {
    if (k % 64 != 0) return error.KNotMultipleOf64;
    if (row0 % 64 != 0 or rows == 0) return error.BadUnpackRows;
    var a: cuda.Args = .{};
    a.add(packed_w);
    a.add(out);
    a.add(@as(i64, @intCast(k)));
    a.add(@as(i64, @intCast(row0)));
    a.add(@as(i64, @intCast(rows)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * k + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn quantizePacked(unpack: cuda.Function, quant: *const glm.quant.Kernels, s: cuda.Stream, packed_w: u64, n: usize, k: usize, chunk_rows: usize, tmp: u64, words: u64, scales: u64, biases: u64) !void {
    if (chunk_rows == 0 or chunk_rows % 64 != 0) return error.BadChunkRows;
    var r0: usize = 0;
    while (r0 < n) : (r0 += chunk_rows) {
        const rows = @min(chunk_rows, n - r0);
        try unpackRows(unpack, s, packed_w, tmp, k, r0, rows);
        try glm.quant.q4(quant, s, tmp, rows, k, words + r0 * k / 2, scales + r0 * k / 32, biases + r0 * k / 32);
    }
}

pub fn packerTransientBytes(g: Geometry, world: usize) u64 {
    const n = if (world == 3) 51712 else g.vocab / world;
    const k = g.hidden;
    return @intCast(n * k / 2 + 2 * (n * k / 32) + head_chunk_rows * k * 2);
}

test "DFlash2 packed head transient matches TP2 plan" {
    try std.testing.expectEqual(@as(u64, 368_295_936), packerTransientBytes(.{}, 2));
}

/// GLM-5.3 DFlash2 entry point. Read incoai/GLM-5.3-DFlash2 weights unmodified (CC BY-NC-ND 4.0); quantization stays in memory.
/// Spark parity hook after WP9: on 50 fixed prompts, compare final greedy IDs from drafted and plain decode; report
/// acceptance as `matched_draft_tokens / verified_rows` from the same runs.
pub const Drafter = struct {
    pub fn load(gpa: std.mem.Allocator, io: std.Io, h: anytype, dir: []const u8, rank: usize, world: usize, bits: u8, geometry: Geometry, share_scratch: bool) !*glm.dflash.Drafter {
        var target_head = try packTargetHead(h, geometry, rank, world);
        var own_head = true;
        defer if (own_head) target_head.buffer.free();
        const inner_geometry = runtimeGeometry(geometry);
        const dr = try glm.dflash_load.loadWithHead(gpa, io, h, dir, rank, world, bits, inner_geometry, target_head.head, share_scratch);
        dr.head_storage = target_head.buffer;
        own_head = false;
        return dr;
    }
};

pub const Host = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    stream: cuda.Stream,
    group: ?*const cuda.tp_group.Group,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    spans: [3]glm.shard.Span,
    share: struct { rank: usize, vocab: glm.shard.Span },
    m: struct { glue: glm.glue.Kernels, mm: glm.mm.Kernels, q4: glm.q4.Kernels, quant: glm.quant.Kernels },
    w: struct { embed: u64, head: ?cuda.qmmf.Weight, mtp: ?struct { draft: glm.q4.Q4 } = null },
    modules: [4]cuda.Module,
    unpack: cuda.Function,
    embed_pick: cuda.Function,
    scratch: cuda.DeviceBuffer,
    h_local: [glm.dflash.MAX_M]i32 = undefined,
    h_owner: [glm.dflash.MAX_M]i32 = undefined,
    dr: ?*glm.dflash.Drafter = null,

    pub fn init(h: *Host, gpa: std.mem.Allocator, ctx: *const cuda.Context, stream: cuda.Stream, group: *const cuda.tp_group.Group, glm_glue: cuda.Module, glm53_glue: cuda.Module, embed: u64, head: ?cuda.qmmf.Weight, vocab: glm.shard.Span, rank: usize) !void {
        var modules: [4]cuda.Module = undefined;
        var loaded: usize = 0;
        errdefer {
            while (loaded > 0) {
                loaded -= 1;
                modules[loaded].unload();
            }
        }
        modules[0] = try cuda.Module.load(ctx.d, cuda.kernels.glm_mm);
        loaded += 1;
        modules[1] = try cuda.Module.load(ctx.d, cuda.kernels.qmm_group);
        loaded += 1;
        modules[2] = try cuda.Module.load(ctx.d, cuda.kernels.affine4_pack);
        loaded += 1;
        modules[3] = try cuda.Module.load(ctx.d, cuda.kernels.glm_quant);
        loaded += 1;
        const glue = try glm.glue.Kernels.load(glm_glue);
        const mm = try glm.mm.Kernels.load(modules[0]);
        const q4 = try glm.q4.Kernels.load(modules[1], modules[2], ctx);
        const quant = try glm.quant.Kernels.load(modules[3]);
        const unpack = try glm53_glue.function("tf_glm53_unpack_bf16_rows");
        const embed_pick = try glm53_glue.function("tf_glm53_embed_pick");
        const scratch_len = group.world * glm.dflash.MAX_M * 6144 * 2 + 2 * glm.dflash.MAX_M * 4;
        var scratch = try cuda.DeviceBuffer.alloc(ctx.d, scratch_len);
        errdefer scratch.free();
        const vl = vocab.len();
        h.* = .{
            .gpa = gpa,
            .d = ctx.d,
            .stream = stream,
            .group = group,
            .lib = &group.lib,
            .comm = group.comm,
            .spans = if (group.world == 3)
                .{ .{ .lo = 0, .hi = 51712 }, .{ .lo = 51712, .hi = 103296 }, .{ .lo = 103296, .hi = 154880 } }
            else
                .{ .{ .lo = 0, .hi = vl }, .{ .lo = vl, .hi = 2 * vl }, .{ .lo = 0, .hi = 0 } },
            .share = .{ .rank = rank, .vocab = vocab },
            .m = .{ .glue = glue, .mm = mm, .q4 = q4, .quant = quant },
            .w = .{ .embed = embed, .head = head },
            .modules = modules,
            .unpack = unpack,
            .embed_pick = embed_pick,
            .scratch = scratch,
        };
    }

    pub fn load(h: *Host, io: std.Io, dir: []const u8, bits: u8, share_scratch: bool) !void {
        if (bits != 4) return error.BadDrafterBits;
        h.dr = try Drafter.load(h.gpa, io, h, dir, h.share.rank, h.group.?.world, bits, try checkConfig(h.gpa, io, dir), share_scratch);
        const conf_temp = try confTempSwitch();
        h.dr.?.conf_temp = conf_temp;
        if (h.share.rank == 0) std.log.info("glm53 draft conf temp: {d:.3}", .{conf_temp});
        // DFlash2's chance is the selector's softmax over its own 16 candidates, which runs hot next to the target's
        // acceptance; dividing the scores by this temperature cools the chances only (the drafts stay the argmax).
        const chance_temp = try chanceTempSwitch();
        h.dr.?.temp = chance_temp;
        if (h.share.rank == 0 and chance_temp != 1.0) std.log.info("TF_GLM53_DRAFT_CHANCE_TEMP={d:.3}: DFlash2 selector chances use this softmax temperature (drafts unchanged)", .{chance_temp});
        // Make slot 0 (its ring) now, on both ranks, before the after-load MemAvailable reading: a lazy allocation
        // during serving could fail on one rank only and leave the other in a collective.
        try h.dr.?.restart(0, 0);
    }

    pub fn deinit(h: *Host) void {
        if (h.dr) |dr| dr.deinit();
        h.scratch.free();
        var i = h.modules.len;
        while (i > 0) {
            i -= 1;
            h.modules[i].unload();
        }
        h.* = undefined;
    }

    pub fn embedDraft(h: *Host, ids: u64, out: u64, rows: usize, width: usize) !void {
        _ = ids;
        const vl = h.share.vocab.len();
        const gathered = h.scratch.ptr;
        const gathered_cap = h.group.?.world * glm.dflash.MAX_M * 6144 * 2;
        const local_dev = gathered + gathered_cap;
        const owners_dev = local_dev + glm.dflash.MAX_M * 4;
        for (h.dr.?.h_ids[0..rows], 0..) |id, r| {
            if (h.group.?.world == 3) {
                const er = embedRowFor(@intCast(id), h.share.rank, vl);
                h.h_owner[r] = er.owner;
                h.h_local[r] = er.local;
            } else {
                const owner = @divTrunc(id, @as(i32, @intCast(vl))); // ids are non-negative token ids
                h.h_owner[r] = owner;
                h.h_local[r] = if (owner == @as(i32, @intCast(h.share.rank))) id - owner * @as(i32, @intCast(vl)) else 0;
            }
        }
        try h.d.check(h.d.api.cuMemcpyHtoDAsync_v2(local_dev, @ptrCast(&h.h_local), rows * @sizeOf(i32), h.stream.handle), "cuMemcpyHtoDAsync");
        try h.d.check(h.d.api.cuMemcpyHtoDAsync_v2(owners_dev, @ptrCast(&h.h_owner), rows * @sizeOf(i32), h.stream.handle), "cuMemcpyHtoDAsync");
        const mine = gathered + h.share.rank * rows * width * 2;
        try glm.glue.embedB16(.{ .glue = &h.m.glue }, h.stream, local_dev, h.w.embed, mine, rows, width, 1);
        try h.group.?.allGather(h.stream, mine, gathered, rows * width * 2);
        var a: cuda.Args = .{};
        a.add(gathered);
        a.add(owners_dev);
        a.add(out);
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(width)));
        a.add(@as(c_int, @intCast(h.group.?.world)));
        try cuda.launch.launch(h.embed_pick, .{ .grid = .{ .x = @intCast((rows * width + 255) / 256) }, .block = .{ .x = 256 } }, h.stream, &a);
    }

    pub fn restartHook(ctx: *anyopaque, from: usize) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        try h.dr.?.restart(0, from);
    }

    pub fn promptHook(ctx: *anyopaque, start: usize, rows: usize) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        try h.dr.?.prompt(h, 0, start, rows, start + rows);
    }

    pub fn verifyHook(ctx: *anyopaque, rows: usize, pos0: usize) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        try h.dr.?.noteVerify(0, 0, rows, pos0, true);
    }

    pub fn draftHook(ctx: *anyopaque, pos: usize, n: usize, anchor: u32, out: ?[]u32, probs: ?[]f64) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        try h.dr.?.draft(h, &[1]glm.dflash.Req{.{ .slot = 0, .pos = @intCast(pos), .n = n, .anchor = anchor, .out = out, .probs = probs }});
    }

    /// TF_GLM53_SLOTS: one request slot's hooks. Each slot's engine Drafter carries one of these as its ctx, so the
    /// slot's prompt, verify rows and drafts go to its own ring (slot `slot` of the one DFlash2 drafter).
    pub const Lane = struct {
        host: *Host,
        slot: u32,

        pub fn restartHook(ctx: *anyopaque, from: usize) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.restart(l.slot, from);
        }

        pub fn promptHook(ctx: *anyopaque, start: usize, rows: usize) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.prompt(l.host, l.slot, start, rows, start + rows);
        }

        pub fn verifyHook(ctx: *anyopaque, rows: usize, pos0: usize) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.noteVerify(l.slot, 0, rows, pos0, true);
        }

        /// A shared window's part: this slot's `rows` tap rows sit at row `row0` of taps_step; `first` counts the
        /// window's one verify (its first part only), so every part's rows stay current until the next verify.
        pub fn verifyAtHook(ctx: *anyopaque, row0: usize, rows: usize, pos0: usize, first: bool) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.noteVerify(l.slot, row0, rows, pos0, first);
        }

        pub fn draftHook(ctx: *anyopaque, pos: usize, n: usize, anchor: u32, out: ?[]u32, probs: ?[]f64) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.draft(l.host, &[1]glm.dflash.Req{.{ .slot = l.slot, .pos = @intCast(pos), .n = n, .anchor = anchor, .out = out, .probs = probs }});
        }

        /// A depth-0 round in a shared window: the `n` kept rows of the slot's last verify into its ring, no block
        /// pass. Its context stays current, so the stream drafts from its first round alone.
        pub fn absorbHook(ctx: *anyopaque, pos: usize, n: usize) anyerror!void {
            const l: *Lane = @ptrCast(@alignCast(ctx));
            try l.host.dr.?.absorb(l.host, l.slot, @intCast(pos), n);
        }
    };

    /// Export the preallocated DFlash2 ring through host memory; this does not allocate device memory.
    pub fn stateHook(ctx: *anyopaque, dst: []u8) anyerror!i64 {
        const h: *Host = @ptrCast(@alignCast(ctx));
        const dr = h.dr orelse return error.NoDrafter;
        if (dr.slots.items.len == 0 or dr.slots.items[0].ring == null or dst.len != dr.geometry.ringBytes()) return error.BadDrafterState;
        try h.stream.synchronize();
        const s = &dr.slots.items[0];
        try h.d.check(h.d.api.cuMemcpyDtoH_v2(dst.ptr, s.ring.?.ptr, dst.len), "DFlash2 ring export");
        return s.from;
    }

    pub const RingView = struct { ptr: u64, len: usize, from: i64 };

    /// Device-direct v2 cache hook. Caller orders access with CUDA events; no host-sized ring copy is made.
    pub fn ringView(ctx: *anyopaque) anyerror!RingView {
        const h: *Host = @ptrCast(@alignCast(ctx));
        const dr = h.dr orelse return error.NoDrafter;
        if (dr.slots.items.len == 0 or dr.slots.items[0].ring == null) return error.BadDrafterState;
        const s = &dr.slots.items[0];
        return .{ .ptr = s.ring.?.ptr, .len = dr.geometry.ringBytes(), .from = s.from };
    }

    /// Adopt a ring restored directly into its device allocation.
    pub fn ringAdopt(ctx: *anyopaque, from: i64) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        const dr = h.dr orelse return error.NoDrafter;
        if (dr.slots.items.len == 0 or dr.slots.items[0].ring == null or from < 0) return error.BadDrafterState;
        const s = &dr.slots.items[0];
        s.from = from;
        s.v_rows = 0;
    }

    /// Import into the ring allocated when the drafter was admitted at load time.
    pub fn adoptHook(ctx: *anyopaque, src: []const u8, from: i64) anyerror!void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        const dr = h.dr orelse return error.NoDrafter;
        if (dr.slots.items.len == 0 or dr.slots.items[0].ring == null or src.len != dr.geometry.ringBytes() or from < 0) return error.BadDrafterState;
        const s = &dr.slots.items[0];
        try h.d.check(h.d.api.cuMemcpyHtoDAsync_v2(s.ring.?.ptr, src.ptr, src.len, h.stream.handle), "DFlash2 ring import");
        try h.stream.synchronize();
        s.from = from;
        s.v_rows = 0;
    }
};

pub fn runtimeGeometry(geometry: Geometry) glm.dflash.Geometry {
    return .{
        .kind = if (geometry.kind == .dspark) .dspark else .dflash2,
        .conv = geometry.conv,
        .markov_top_k = geometry.markov_top_k,
        .confidence = geometry.confidence,
        .sample_from_anchor = geometry.sample_from_anchor,
        .hidden = geometry.hidden,
        .layers = geometry.layers,
        .heads = geometry.heads,
        .kv_heads = geometry.kv_heads,
        .head_dim = geometry.head_dim,
        .ff = geometry.intermediate,
        .rope_theta = geometry.rope_theta,
        .taps = geometry.target_layers,
        .window = geometry.window,
        .block_size = geometry.block,
        .selector_rank = geometry.selector_rank,
        .selector_top_k = geometry.selector_top_k,
        .conv_group = geometry.conv_group_size,
        .conv_kernel = geometry.conv_kernel_size,
        .vocab = geometry.vocab,
        .mask = geometry.mask,
        .eps = geometry.rms_norm_eps,
        .tap_streams = 1,
    };
}

const PackedHead = struct { buffer: @import("cuda").DeviceBuffer, head: glm.q4.Q4 };

fn packTargetHead(h: anytype, geometry: Geometry, rank: usize, world: usize) !PackedHead {
    if (world == 0 or world > 3 or rank >= world or h.share.rank != rank) return error.BadDrafterWorld;
    if (world != 3 and geometry.vocab % world != 0) return error.BadDrafterWorld;
    const n = if (world == 3) (if (rank == 0) @as(usize, 51712) else @as(usize, 51584)) else geometry.vocab / world;
    const k = geometry.hidden;
    const w = h.w.head orelse return error.DrafterNeedsTargetHead;
    if (w.mode != .bf16 or @as(usize, w.n) != n or @as(usize, w.k) != k or w.codes == 0) return error.DraftHeadShape;
    const raw_words: u64 = @intCast(n * k / 2);
    const raw_group: u64 = @intCast(n * k / 32);
    var raw = try cuda.DeviceBuffer.alloc(h.d, raw_words + 2 * raw_group);
    defer raw.free();
    const words = raw.ptr;
    const scales = words + raw_words;
    const biases = scales + raw_group;
    var tmp = try cuda.DeviceBuffer.alloc(h.d, head_chunk_rows * k * 2);
    defer tmp.free();
    try quantizePacked(h.unpack, &h.m.quant, h.stream, w.codes, n, k, head_chunk_rows, tmp.ptr, words, scales, biases);
    const sizes = glm.q4.bytes(n, k);
    const woff: u64 = 0;
    const soff = std.mem.alignForward(u64, sizes[0], 256);
    const boff = soff + std.mem.alignForward(u64, sizes[1], 256);
    const total = boff + sizes[2];
    var head_storage = try cuda.DeviceBuffer.alloc(h.d, total);
    errdefer head_storage.free();
    const head: glm.q4.Q4 = .{ .w = head_storage.ptr + woff, .s = head_storage.ptr + soff, .b = head_storage.ptr + boff, .n = n, .k = k, .npad = glm.q4.npadOf(n) };
    try glm.q4.pack(&h.m.q4, h.stream, words, scales, biases, head);
    try h.stream.synchronize();
    return .{ .buffer = head_storage, .head = head };
}

pub const Config = struct {
    geometry: Geometry,
    allocator: std.mem.Allocator,

    pub fn deinit(c: *Config) void {
        c.allocator.free(c.geometry.target_layers);
        c.* = undefined;
    }
};

/// Parse only the public geometry required by the DFlash2 CUDA kernels and weights loader.
pub fn parse(gpa: std.mem.Allocator, bytes: []const u8) !Config {
    const json = try std.json.parseFromSlice(std.json.Value, gpa, stripHeader(bytes), .{});
    defer json.deinit();
    const o = try object(json.value);
    if (stringFieldEq(o, "speculators_model_type", "dspark")) return parseDspark(gpa, o);
    const architectures = try arrayField(o, "architectures");
    if (architectures.len != 1 or !stringEq(architectures[0], "DFlash2DraftModel")) return error.DrafterConfigUnexpected;
    if (!stringFieldEq(o, "model_type", "qwen3")) return error.DrafterConfigUnexpected;
    const dc = try object(o.get("dflash_config") orelse return error.NotDFlash2);
    const hidden = try intField(o, "hidden_size");
    const layers = try intField(o, "num_hidden_layers");
    const heads = try intField(o, "num_attention_heads");
    const kv_heads = try intField(o, "num_key_value_heads");
    const head_dim = try intField(o, "head_dim");
    const intermediate = try intField(o, "intermediate_size");
    const vocab = try intField(o, "vocab_size");
    const window = try intField(o, "sliding_window");
    if (hidden != 6144 or layers != 6 or heads != 64 or kv_heads != 8 or head_dim != 128 or intermediate != 12288 or vocab != 154880 or window != 2048) return error.DrafterConfigUnexpected;
    if (try intField(o, "num_target_layers") != 78 or !boolFieldEq(o, "is_causal", false) or !boolFieldEq(o, "tie_word_embeddings", false)) return error.DrafterConfigUnexpected;
    if (!stringFieldEq(o, "hidden_act", "silu")) return error.DrafterConfigUnexpected;
    if (!floatFieldEq(o, "rms_norm_eps", 1e-5)) return error.DrafterConfigUnexpected;
    const rope_theta = if (o.get("rope_theta")) |v| try number(v) else blk: {
        const rope = try object(o.get("rope_parameters") orelse return error.DrafterConfigUnexpected);
        break :blk try number(rope.get("rope_theta") orelse return error.DrafterConfigUnexpected);
    };
    if (rope_theta != 1e6) return error.DrafterConfigUnexpected;
    const layer_types = try arrayField(o, "layer_types");
    if (layer_types.len != 6) return error.DrafterConfigUnexpected;
    for (layer_types) |kind| if (!stringEq(kind, "sliding_attention")) return error.DrafterConfigUnexpected;
    if (try intField(dc, "block_size") != 8 or try intField(dc, "conv_group_size") != 16 or try intField(dc, "conv_kernel_size") != 2 or try intField(dc, "mask_token_id") != 154856 or try intField(dc, "selector_rank") != 256 or try intField(dc, "selector_top_k") != 16) return error.DrafterConfigUnexpected;
    const source_taps = try arrayField(dc, "target_layer_ids");
    if (source_taps.len != 6) return error.DrafterConfigUnexpected;
    const taps = try gpa.alloc(usize, source_taps.len);
    errdefer gpa.free(taps);
    const expected_taps = [_]usize{ 5, 19, 33, 47, 61, 75 };
    for (source_taps, 0..) |v, i| {
        const tap = try intValue(v);
        if (tap != expected_taps[i]) return error.DrafterConfigUnexpected;
        taps[i] = tap;
    }
    const geom: Geometry = .{
        .hidden = hidden,
        .layers = layers,
        .heads = heads,
        .kv_heads = kv_heads,
        .head_dim = head_dim,
        .intermediate = intermediate,
        .vocab = vocab,
        .block = 8,
        .window = window,
        .mask = 154856,
        .selector_rank = 256,
        .selector_top_k = 16,
        .target_layers = taps,
        .conv_group_size = 16,
        .conv_kernel_size = 2,
        .rope_theta = @floatCast(rope_theta),
        .rms_norm_eps = 1e-5,
    };
    return .{ .geometry = geom, .allocator = gpa };
}

fn parseDspark(gpa: std.mem.Allocator, o: std.json.ObjectMap) !Config {
    const tc = try object(o.get("transformer_layer_config") orelse return error.UnsupportedDspark);
    if (!boolFieldEq(o, "enable_confidence_head", true) or !boolFieldEq(o, "confidence_head_with_markov", true) or
        !boolFieldEq(o, "sample_from_anchor", true) or !stringFieldEq(o, "markov_head_type", "vanilla") or
        try intField(o, "block_size") != 8 or try intField(o, "markov_rank") != 256 or
        try intField(o, "mask_token_id") != 154856) return error.UnsupportedDspark;
    const source_taps = try arrayField(o, "aux_hidden_state_layer_ids");
    const expected = [_]usize{ 2, 20, 39, 58, 75 };
    if (source_taps.len != expected.len) return error.UnsupportedDspark;
    const taps = try gpa.alloc(usize, expected.len);
    errdefer gpa.free(taps);
    for (source_taps, 0..) |v, i| {
        const tap = intValue(v) catch return error.UnsupportedDspark;
        if (tap != expected[i]) return error.UnsupportedDspark;
        taps[i] = tap - 1;
    }
    const hidden = try intField(tc, "hidden_size");
    const layers = try intField(tc, "num_hidden_layers");
    const heads = try intField(tc, "num_attention_heads");
    const kv_heads = try intField(tc, "num_key_value_heads");
    const head_dim = try intField(tc, "head_dim");
    const intermediate = try intField(tc, "intermediate_size");
    const vocab = try intField(tc, "vocab_size");
    const window = try intField(tc, "sliding_window");
    const rope = try object(tc.get("rope_parameters") orelse return error.UnsupportedDspark);
    const layer_types = try arrayField(tc, "layer_types");
    if (layer_types.len != layers) return error.UnsupportedDspark;
    for (layer_types) |kind| if (!stringEq(kind, "sliding_attention")) return error.UnsupportedDspark;
    if (hidden != 6144 or layers != 3 or heads != 64 or kv_heads != 64 or head_dim != 64 or intermediate != 12288 or vocab != 154880 or window != 2048 or
        !stringFieldEq(tc, "model_type", "qwen3") or !stringFieldEq(tc, "hidden_act", "silu") or
        !floatFieldEq(tc, "rms_norm_eps", 1e-5) or !floatFieldEq(rope, "rope_theta", 8e6)) return error.UnsupportedDspark;
    const geom: Geometry = .{ .kind = .dspark, .conv = false, .markov_top_k = 64, .confidence = true, .sample_from_anchor = true,
        .hidden = hidden, .layers = layers, .heads = heads, .kv_heads = kv_heads, .head_dim = head_dim,
        .intermediate = intermediate, .vocab = vocab, .block = 8, .window = window, .mask = 154856,
        .selector_rank = 256, .selector_top_k = 64, .target_layers = taps, .conv_group_size = 16,
        .conv_kernel_size = 2, .rope_theta = 8e6, .rms_norm_eps = 1e-5 };
    return .{ .geometry = geom, .allocator = gpa };
}

pub fn checkConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Geometry {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(bytes);
    var c = try parse(gpa, bytes);
    defer c.deinit();
    var result = c.geometry;
    result.target_layers = if (result.kind == .dspark) &.{ 1, 19, 38, 57, 74 } else &.{ 5, 19, 33, 47, 61, 75 };
    return result;
}

/// Estimate drafter resident bytes per rank, excluding the separately reported target-derived draft head.
/// Four-bit linears use one nibble plus affine metadata per 64 weights; eight-bit is a byte per weight.
pub fn weightBytes(g: Geometry, bits: u8, world: usize) !u64 {
    if (bits != 4 and bits != 8) return error.BadDrafterBits;
    if (world == 0 or world > 3) return error.BadDrafterWorld;
    const d: u64 = @intCast(g.hidden);
    const kv: u64 = @intCast(g.kvWidth());
    const ff: u64 = @intCast(if (world == 2) g.intermediate / 2 else if (world == 3 and glm.dflash_load.split3Switch()) g.intermediate / 3 else g.intermediate);
    const qkv: u64 = @intCast(g.qkvWidth());
    const conv: u64 = @intCast(g.convWidth());
    const qw: u64 = @intCast(g.queryWidth());
    const per_layer = qkv * d + 2 * kv * d + qw * d + 2 * ff * d + (if (g.conv) @as(u64, 2) * conv * d else @as(u64, 0)) + @as(u64, @intCast(g.target_layers.len)) * d * d / @as(u64, @intCast(g.layers));
    const qweight = @as(u64, @intCast(g.layers)) * per_layer;
    const quantized = if (bits == 4) (qweight / 2 + qweight / 16) else qweight;
    const small_per_layer = (if (g.conv) @as(u64, 6) * d * 2 else @as(u64, 2) * d * 2) + 2 * @as(u64, @intCast(g.head_dim)) * 2 + (if (g.conv) @as(u64, 2) * @as(u64, @intCast(g.conv_kernel_size)) * 2 * d * 2 + @as(u64, @intCast(g.conv_group_size)) * 256 else @as(u64, 0));
    const meta = @as(u64, @intCast(g.layers)) * (small_per_layer + (if (g.conv) @as(u64, 32 * 256) else @as(u64, 0)));
    const down = @as(u64, @intCast(g.layers)) * d * ff * 2;
    const selector = if (g.kind == .dspark) @as(u64, 0) else @as(u64, @intCast(g.selector_rank)) * d * 2;
    return quantized + meta + down + selector + 2 * d * 2 + (if (g.conv) @as(u64, 64 * 256) else @as(u64, 0));
}

pub fn ringBytes(g: Geometry) u64 {
    return g.ringBytes();
}

/// Device and host residency for one rank. Codebooks and candidate readback are host allocations.
pub const MemoryPlan = struct {
    weights: u64,
    draft_head: u64,
    scratch: u64,
    shared_scratch: u64 = 0,
    rings: u64,
    host_books: u64,
    host_tables: u64,

    pub fn deviceBytes(p: MemoryPlan) u64 {
        return p.weights + p.draft_head + p.scratch + p.rings;
    }
    pub fn deviceBytesWithSharedScratch(p: MemoryPlan) u64 {
        return p.weights + p.draft_head + (p.scratch - p.shared_scratch) + p.rings;
    }
    pub fn hostBytes(p: MemoryPlan) u64 {
        return p.host_books + p.host_tables;
    }
};

pub fn memoryPlan(g: Geometry, bits: u8, world: usize, streams: usize) !MemoryPlan {
    if (world == 0 or world > 3) return error.BadDrafterWorld;
    if (world == 2 and g.vocab % 2 != 0) return error.BadDrafterWorld;
    if (streams == 0 or streams > 16) return error.BadDrafterStreams;
    const d: u64 = @intCast(g.hidden);
    const ff: u64 = @intCast(if (world == 2) g.intermediate / 2 else if (world == 3 and glm.dflash_load.split3Switch()) g.intermediate / 3 else g.intermediate);
    const qkv: u64 = @intCast(g.qkvWidth());
    const kv: u64 = @intCast(g.kvWidth());
    const qw: u64 = @intCast(g.queryWidth());
    const conv: u64 = if (g.conv) @intCast(g.convWidth()) else @as(u64, 0);
    const m: u64 = 16 * g.block;
    const c: u64 = @intCast(g.window);
    const rank: u64 = @intCast(g.selector_rank);
    const head_n: u64 = if (world == 3) 51712 else @intCast(g.vocab / world);
    const chunks = (head_n + 4095) / 4096;
    const mm = @import("glm").mm;
    var part: u64 = 16;
    if (bits == 8) {
        for ([_]mm.Tiled{
            .{ .w = 0, .s = 0, .bits = 16, .n = qkv, .k = d },
            .{ .w = 0, .s = 0, .bits = 16, .n = 2 * kv, .k = d },
            .{ .w = 0, .s = 0, .bits = 16, .n = qw, .k = d },
            .{ .w = 0, .s = 0, .bits = 16, .n = 2 * ff, .k = d },
            .{ .w = 0, .s = 0, .bits = 16, .n = 2 * conv, .k = d },
        }) |w| part = @max(part, mm.partBytes(64, w));
    }
    part = @max(part, mm.partBytes(64, .{ .w = 0, .s = 0, .bits = 16, .n = rank, .k = d }));
    part = @max(part, mm.partBytes(64, .{ .w = 0, .s = 0, .bits = 16, .n = d, .k = ff }));
    const fields = [_]u64{
        m * d * 2,                     m * d * 2,                              m * d * 2,                                      m * conv * 2,         m * qkv * 2,          m * qw * 2,                        m * kv * 2,     m * kv * 2,
        m * qw * 2,                    m * d * 2,                              m * 2 * @as(u64, @intCast(g.intermediate)) * 2, m * @as(u64, @intCast(g.intermediate)) * 2,           m * d * 2,            m * (d / 64) * 4,                  m * rank * 4,   m * head_n * 2,
        part,                          c * g.target_layers.len * (d / 64) * 4, (if (world == 3 and glm.dflash_load.split3Switch()) @as(u64, 3) else 2) * m * d * 4,                                 c * g.tapRowBytes(),  c * d * 2,            c * d * 2,                         c * 2 * kv * 2, 32 * g.tapRowBytes(),
        c * g.tapRowBytes(),           m * 4,                                  m * 4,                                          16 * 8,               16 * 8,               16 * 8,                            c * 8,          c * 8,
        world * m * (2 * g.selector_top_k) * 4, m * chunks * g.selector_top_k * 4, m * chunks * g.selector_top_k * 4, m * g.heads * 16 * 4, m * g.heads * 16 * 4, m * g.heads * 16 * g.head_dim * 4,
    };
    var scratch: u64 = 0;
    var shared_scratch: u64 = 0;
    for (fields, 0..) |n, i| {
        const aligned = std.mem.alignForward(u64, n, 256);
        scratch += aligned;
        // Ephemeral work buffers are only touched inside prompt/draft hooks, between forward launches.
        // Keep taps_step (23), taps_prefill (24), and the stateful ring table (27) outside the shared set.
        if (i != 23 and i != 24 and i != 27) shared_scratch += aligned;
    }
    const host_books = 2 * @as(u64, @intCast(g.vocab)) * rank * 2 + (if (g.confidence) @as(u64, 6400 * 2 + 4) else 0);
    const host_tables = c * 16 + world * m * (2 * g.selector_top_k) * 4 + m * rank * 4 + (if (g.confidence) m * d * 2 else 0);
    return .{
        .weights = try weightBytes(g, bits, world),
        .draft_head = try draftHeadBytes(g, world),
        .scratch = scratch,
        .shared_scratch = shared_scratch,
        .rings = ringBytes(g) * streams,
        .host_books = host_books,
        .host_tables = host_tables,
    };
}

test "drafter scratch admission counts the shared overlay once" {
    const p = try memoryPlan(.{}, 4, 2, 1);
    try std.testing.expect(p.shared_scratch > 0);
    try std.testing.expectEqual(p.deviceBytes() - p.shared_scratch, p.deviceBytesWithSharedScratch());
}

test "fake driver launch and record order keeps shared drafter writes outside forward lifetimes" {
    const Event = enum { forward_first_write, forward_last_read, drafter_write, drafter_read };
    const FakeDriver = struct {
        events: [8]Event = undefined,
        used: usize = 0,
        fn recordForward(self: *@This()) void {
            self.events[self.used] = .forward_first_write;
            self.used += 1;
            self.events[self.used] = .forward_last_read;
            self.used += 1;
        }
        fn launchDrafter(self: *@This()) void {
            self.events[self.used] = .drafter_write;
            self.used += 1;
            self.events[self.used] = .drafter_read;
            self.used += 1;
        }
    };
    var fake: FakeDriver = .{};
    fake.recordForward(); // prompt pass launch/record sequence
    fake.launchDrafter(); // promptHook
    fake.recordForward(); // verify pass launch/record sequence
    fake.launchDrafter(); // draftHook
    var forward_live = false;
    for (fake.events[0..fake.used]) |event| switch (event) {
        .forward_first_write => {
            try std.testing.expect(!forward_live);
            forward_live = true;
        },
        .forward_last_read => {
            try std.testing.expect(forward_live);
            forward_live = false;
        },
        .drafter_write => try std.testing.expect(!forward_live),
        .drafter_read => try std.testing.expect(!forward_live),
    };
    try std.testing.expect(!forward_live);
}

/// The target-derived Q4 draft head for this rank's vocabulary rows.
pub fn draftHeadBytes(g: Geometry, world: usize) !u64 {
    if (world == 0 or world > 3) return error.BadDrafterWorld;
    if (world == 2 and g.vocab % 2 != 0) return error.BadDrafterWorld;
    const hn: u64 = if (world == 3) 51712 else @intCast(g.vocab / world);
    const d: u64 = @intCast(g.hidden);
    return @intCast(hn * d / 2 + hn * (d / 64) * 4);
}

fn stripHeader(bytes: []const u8) []const u8 {
    if (!std.mem.startsWith(u8, bytes, "//")) return bytes;
    return if (std.mem.indexOfScalar(u8, bytes, '\n')) |n| bytes[n + 1 ..] else bytes;
}
fn object(v: std.json.Value) !std.json.ObjectMap {
    return if (v == .object) v.object else error.DrafterConfigUnexpected;
}
fn arrayField(o: std.json.ObjectMap, key: []const u8) ![]std.json.Value {
    const v = o.get(key) orelse return error.DrafterConfigUnexpected;
    return if (v == .array) v.array.items else error.DrafterConfigUnexpected;
}
fn stringEq(v: std.json.Value, s: []const u8) bool {
    return v == .string and std.mem.eql(u8, v.string, s);
}
fn stringFieldEq(o: std.json.ObjectMap, key: []const u8, s: []const u8) bool {
    return if (o.get(key)) |v| stringEq(v, s) else false;
}
fn boolFieldEq(o: std.json.ObjectMap, key: []const u8, expected: bool) bool {
    return if (o.get(key)) |v| v == .bool and v.bool == expected else false;
}
fn intValue(v: std.json.Value) !usize {
    return if (v == .integer and v.integer >= 0) @intCast(v.integer) else error.DrafterConfigUnexpected;
}
fn intField(o: std.json.ObjectMap, key: []const u8) !usize {
    return intValue(o.get(key) orelse return error.DrafterConfigUnexpected);
}
fn number(v: std.json.Value) !f64 {
    return switch (v) {
        .float => v.float,
        .integer => @floatFromInt(v.integer),
        else => error.DrafterConfigUnexpected,
    };
}
fn floatFieldEq(o: std.json.ObjectMap, key: []const u8, expected: f64) bool {
    const v = o.get(key) orelse return false;
    return (if (v == .float) v.float else if (v == .integer) @as(f64, @floatFromInt(v.integer)) else return false) == expected;
}

test "glm dsa dflash config geometry and rank ring census" {
    const fixture = "{\"architectures\":[\"DFlash2DraftModel\"],\"model_type\":\"qwen3\",\"hidden_size\":6144,\"num_hidden_layers\":6,\"num_attention_heads\":64,\"num_key_value_heads\":8,\"head_dim\":128,\"intermediate_size\":12288,\"vocab_size\":154880,\"sliding_window\":2048,\"num_target_layers\":78,\"is_causal\":false,\"tie_word_embeddings\":false,\"hidden_act\":\"silu\",\"rms_norm_eps\":0.00001,\"rope_theta\":1000000,\"layer_types\":[\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"],\"dflash_config\":{\"block_size\":8,\"conv_group_size\":16,\"conv_kernel_size\":2,\"mask_token_id\":154856,\"selector_rank\":256,\"selector_top_k\":16,\"target_layer_ids\":[5,19,33,47,61,75]}}";
    var c = try parse(std.testing.allocator, fixture);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 6144), c.geometry.hidden);
    try std.testing.expectEqual(@as(usize, 10240), c.geometry.qkvWidth());
    try std.testing.expectEqual(@as(usize, 73728), c.geometry.tapRowBytes());
    try std.testing.expectEqual(@as(u64, 50_331_648), ringBytes(c.geometry));
    try std.testing.expectEqual(Geometry.Kind.dflash2, c.geometry.kind);
    const flash_abi: glm.dflash.Geometry = .{};
    try std.testing.expect(flash_abi.flashKernelAbi());
    try std.testing.expectEqual(@as(u64, 1_327_860_736), try weightBytes(c.geometry, 4, 2));
    try std.testing.expectEqual(@as(u64, 2_004_978_688), try weightBytes(c.geometry, 8, 2));
    try std.testing.expectEqual(@as(u64, 267_632_640), try draftHeadBytes(c.geometry, 2));
}

test "dspark config maps residual input taps to engine output taps" {
    const fixture = "{\"architectures\":[\"DSparkDraftModel\"],\"speculators_model_type\":\"dspark\",\"aux_hidden_state_layer_ids\":[2,20,39,58,75],\"block_size\":8,\"confidence_head_with_markov\":true,\"enable_confidence_head\":true,\"markov_head_type\":\"vanilla\",\"markov_rank\":256,\"mask_token_id\":154856,\"sample_from_anchor\":true,\"transformer_layer_config\":{\"hidden_size\":6144,\"num_hidden_layers\":3,\"num_attention_heads\":64,\"num_key_value_heads\":64,\"head_dim\":64,\"intermediate_size\":12288,\"vocab_size\":154880,\"sliding_window\":2048,\"model_type\":\"qwen3\",\"hidden_act\":\"silu\",\"rms_norm_eps\":0.00001,\"layer_types\":[\"sliding_attention\",\"sliding_attention\",\"sliding_attention\"],\"rope_parameters\":{\"rope_theta\":8000000}}}";
    var c = try parse(std.testing.allocator, fixture);
    defer c.deinit();
    const g = c.geometry;
    try std.testing.expectEqual(Geometry.Kind.dspark, g.kind);
    try std.testing.expect(!g.conv and g.confidence and g.sample_from_anchor);
    try std.testing.expectEqual(@as(usize, 6144), g.hidden);
    try std.testing.expectEqual(@as(usize, 3), g.layers);
    try std.testing.expectEqual(@as(usize, 4096), g.kvWidth());
    try std.testing.expectEqual(@as(usize, 12288), g.qkvWidth());
    try std.testing.expectEqualSlices(usize, &.{ 1, 19, 38, 57, 74 }, g.target_layers);
    try std.testing.expectEqual(@as(usize, 64), g.selector_top_k);
    try std.testing.expectEqual(@as(u64, 100_663_296), ringBytes(g));
    try std.testing.expectEqual(@as(u64, 714_965_760), try weightBytes(g, 4, 2));
    const plan = try memoryPlan(g, 4, 2, 1);
    try std.testing.expectEqual(@as(u64, 267_632_640), plan.draft_head);
    try std.testing.expectEqual(@as(u64, 100_663_296), plan.rings);
    try std.testing.expectEqual(@as(u64, 158_609_924), plan.host_books);
    try std.testing.expectEqual(@as(u64, 1_867_776), plan.host_tables);
}

test "dspark rejects unsupported sampler and selector variants" {
    const unsupported = "{\"speculators_model_type\":\"dspark\",\"block_size\":8,\"confidence_head_with_markov\":false,\"enable_confidence_head\":true,\"markov_head_type\":\"gated\",\"markov_rank\":256,\"mask_token_id\":154856,\"sample_from_anchor\":false,\"aux_hidden_state_layer_ids\":[2,20,39,58,75],\"transformer_layer_config\":{}}";
    try std.testing.expectError(error.UnsupportedDspark, parse(std.testing.allocator, unsupported));
}

test "dspark checkpoint tensor map is the exact 40-key contract" {
    const keys = [_][]const u8{
        "confidence_head.proj.bias", "confidence_head.proj.weight", "fc.weight", "hidden_norm.weight",
        "layers.0.input_layernorm.weight", "layers.0.mlp.down_proj.weight", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight", "layers.0.post_attention_layernorm.weight", "layers.0.self_attn.k_norm.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.v_proj.weight",
        "layers.1.input_layernorm.weight", "layers.1.mlp.down_proj.weight", "layers.1.mlp.gate_proj.weight", "layers.1.mlp.up_proj.weight", "layers.1.post_attention_layernorm.weight", "layers.1.self_attn.k_norm.weight", "layers.1.self_attn.k_proj.weight", "layers.1.self_attn.o_proj.weight", "layers.1.self_attn.q_norm.weight", "layers.1.self_attn.q_proj.weight", "layers.1.self_attn.v_proj.weight",
        "layers.2.input_layernorm.weight", "layers.2.mlp.down_proj.weight", "layers.2.mlp.gate_proj.weight", "layers.2.mlp.up_proj.weight", "layers.2.post_attention_layernorm.weight", "layers.2.self_attn.k_norm.weight", "layers.2.self_attn.k_proj.weight", "layers.2.self_attn.o_proj.weight", "layers.2.self_attn.q_norm.weight", "layers.2.self_attn.q_proj.weight", "layers.2.self_attn.v_proj.weight",
        "markov_head.markov_w1.weight", "markov_head.markov_w2.weight", "norm.weight",
    };
    try std.testing.expectEqual(@as(usize, 40), keys.len);
}

test "glm dsa dflash rejects Flash drafter geometry" {
    const flash = "{\"architectures\":[\"DFlash2DraftModel\"],\"model_type\":\"qwen3\",\"hidden_size\":4096}";
    try std.testing.expectError(error.NotDFlash2, parse(std.testing.allocator, flash));
}

test "glm dsa draft conf temp parses floats and refuses bad values" {
    try std.testing.expectEqual(@as(f64, 1.0), try parseConfTemp(null));
    try std.testing.expectEqual(@as(f64, 1.0), try parseConfTemp("1"));
    try std.testing.expectEqual(@as(f64, 2.5), try parseConfTemp("2.5"));
    try std.testing.expectEqual(@as(f64, 0.05), try parseConfTemp("0.05"));
    try std.testing.expectEqual(@as(f64, 20.0), try parseConfTemp("20"));
    const bad_values = [_][]const u8{ "0", "-1", "0.04", "21", "abc", "nan", "inf", "" };
    for (bad_values) |bad| try std.testing.expectError(error.BadDraftConfTemp, parseConfTemp(bad));
}

/// Host reference for the target engine's vocab-sharded embed gather followed by owner selection.
pub fn embedShardsRef(allocator: std.mem.Allocator, shards: []const []const u16, ids: []const usize, shard_rows: usize, width: usize) ![]u16 {
    if (shards.len != 2) return error.BadEmbedMock;
    for (shards) |shard| if (shard.len != shard_rows * width) return error.BadEmbedMock;
    const out = try allocator.alloc(u16, ids.len * width);
    errdefer allocator.free(out);
    for (ids, 0..) |id, row| {
        const rank = id / shard_rows;
        const local = id % shard_rows;
        if (rank >= shards.len) return error.BadEmbedMock;
        @memcpy(out[row * width ..][0..width], shards[rank][local * width ..][0..width]);
    }
    return out;
}

test "two-rank vocab-sharded embed selects rows from both halves" {
    const shard_rows: usize = 77_440;
    const width: usize = 2;
    const r0 = try std.testing.allocator.alloc(u16, shard_rows * width);
    defer std.testing.allocator.free(r0);
    const r1 = try std.testing.allocator.alloc(u16, shard_rows * width);
    defer std.testing.allocator.free(r1);
    for (0..shard_rows) |row| {
        r0[row * width] = 10;
        r0[row * width + 1] = @intCast(row % 65_536);
        r1[row * width] = 30;
        r1[row * width + 1] = @intCast(row % 65_536);
    }
    const ids = [_]usize{ 0, 154856, 154879 };
    const rows = try embedShardsRef(std.testing.allocator, &.{ r0, r1 }, &ids, shard_rows, width);
    defer std.testing.allocator.free(rows);
    try std.testing.expectEqualSlices(u16, &.{ 10, 0, 30, 11_880, 30, 11_903 }, rows);
}

test "glm dsa drafter runtime geometry requests plain hidden taps" {
    try std.testing.expectEqual(@as(usize, 1), runtimeGeometry(.{}).tap_streams);
}

test "draft module declarations analyze" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Drafter);
    std.testing.refAllDecls(Config);
    std.testing.refAllDecls(MemoryPlan);
}

test "glm dsa dflash memory plan includes device scratch, rings, target head and host books" {
    const g: Geometry = .{};
    const one = try memoryPlan(g, 4, 2, 1);
    const sixteen = try memoryPlan(g, 4, 2, 16);
    try std.testing.expectEqual(try weightBytes(g, 4, 2), one.weights);
    try std.testing.expectEqual(try draftHeadBytes(g, 2), one.draft_head);
    try std.testing.expectEqual(ringBytes(g), one.rings);
    try std.testing.expectEqual(@as(u64, 158_597_120), one.host_books);
    try std.testing.expectEqual(one.scratch, sixteen.scratch);
    try std.testing.expectEqual(one.rings * 16, sixteen.rings);
    try std.testing.expect(one.scratch > 0 and one.host_tables > 0);
    try std.testing.expectEqual(one.weights + one.draft_head + one.scratch + one.rings, one.deviceBytes());
    try std.testing.expectEqual(one.host_books + one.host_tables, one.hostBytes());
}

test "glm dsa dflash config arrays fail closed when not arrays" {
    const malformed = "{\"architectures\":\"DFlash2DraftModel\"}";
    try std.testing.expectError(error.DrafterConfigUnexpected, parse(std.testing.allocator, malformed));
}
