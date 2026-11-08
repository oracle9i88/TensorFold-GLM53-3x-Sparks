// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 CUDA forward for one TP2 + DCP2 rank (WP9): decode windows (`Forward.whole`) and prefill chunks
//! (`Forward.pieced`) through the real launch wrappers of WP2 (qmmf `.fp8b`), WP4 (EXL3 experts), WP6 (MLA),
//! WP7 (indexer), WP8 (MoE/MLP/low-rank/glue) and WP10 (fp32 head).
//! Exchange and glue patterns adapted from ashhart/TensorFold@d1018b31:zig/src/families/glm/cuda_forward.zig and
//! cuda_glue.zig (Apache-2.0). Layer order and DCP2 flow follow bertholomus/bertholomus/TensorFold@ddfad35:
//! src/tensorfold/families/glm_moe_dsa/cuda/forward.py and dcp.py (Apache-2.0); the rope table follows rope.py `_table`.
//! Changes: Zig orchestration for this engine, gather-on-prefill (prefill doc §3.4), explicit exchange plans.
const std = @import("std");
const cuda = @import("cuda");
const mla = @import("mla.zig");
pub const index = @import("index.zig");
pub const moe = @import("moe.zig");
const mlp = @import("mlp.zig");
const experts = @import("experts.zig");
const sampler = @import("sampler.zig");
const weights = @import("weights.zig");
const state = @import("state.zig");
const prefill_gemm = @import("prefill_gemm");
const types = @import("types");
const ablate_mod = @import("ablate");
pub const taps_mod = @import("taps.zig");

pub const hidden = state.hidden;
/// Widest decode/verify window: 16 for copy rounds under TF_GLM53_VERIFY_MAX=16 (the served width is Geometry.decode_rows,
/// 8 by default, so buffers and the memory plan do not grow). The decode MLA kernels stay at mla.max_decode_rows (8):
/// wider rounds take the generic absorb/expand_v kernels.
pub const max_decode_rows = 16;
/// Rows the fast exact decode kernels (decode MLA, A3 attendVerify) take; TF_GLM53_COPYD copy windows stay within it.
pub const decode_mla_rows = mla.max_decode_rows;
/// Largest verify round that may use the decode side stream (see the side_on comment in the attention piece).
pub const side_max_rows: usize = 8;
comptime {
    std.debug.assert(mla.max_decode_rows <= max_decode_rows and mla.verify_max_rows <= max_decode_rows and side_max_rows <= max_decode_rows);
}
pub const prefill_chunk_rows = 4096;
pub const max_prefill_chunk_rows = 8192;
pub const world = 2;
pub const HeadMask = struct { bits: u64 = 0, words: u32 = 0 };

pub fn collFusePartialPlan(rank: usize, block_bytes: usize) struct { send: usize, pair: usize, peer: usize, need: usize } {
    const peer_rank = 1 - rank;
    const pair = (world + 1) * block_bytes;
    return .{ .send = peer_rank * block_bytes, .pair = pair, .peer = pair + peer_rank * block_bytes, .need = (world + 3) * block_bytes };
}
pub const rms_eps: f32 = 1e-5;
pub const index_eps: f32 = 1e-6;
pub const rope_theta: f64 = 8e6;
pub const rope_half = mla.rope_dim / 2;

/// Flash's glue image (glm_glue.cu, tf_glm_*): norms, residual add, embed rows, router. Argument layouts are
/// glm/cuda_glue.zig's (rmsnorm, residualAdd, embedB16, and the router kernels moe.zig's `route` launches).
pub const Glue = struct {
    rmsnorm_6144: cuda.Function,
    rmsnorm_2048: cuda.Function,
    residual_add: cuda.Function,
    embed_b16: cuda.Function,
    router_part: cuda.Function,
    router_tile: cuda.Function,
    topk: cuda.Function,
    slice_sum: cuda.Function,

    pub fn load(m: cuda.Module) !Glue {
        var g: Glue = undefined;
        const names = @typeInfo(Glue).@"struct".field_names;
        inline for (names) |name| @field(g, name) = try m.function("tf_glm_" ++ name);
        return g;
    }

    /// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))), eps from config.json (1e-5), widths 6144 and 2048.
    pub fn rmsnorm(g: *const Glue, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, rows: usize, d: usize, eps: f32) !void {
        if (rows == 0) return;
        const f, const warps: u32 = switch (d) {
            6144 => .{ g.rmsnorm_6144, 8 },
            2048 => .{ g.rmsnorm_2048, 4 },
            else => return error.NoRmsnormWidth,
        };
        var a: cuda.Args = .{};
        inline for (.{ x, @as(i64, @intCast(x_stride)), w, out, @as(i64, @intCast(o_stride)), @as(u64, 0), eps, @as(c_int, 0) }) |v| a.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
    }

    /// glue.residual_add: xout = bf16(x + bf16(sum of `world` fp32 partials [world][rows][d], rank 0 first)).
    pub fn residualAdd(g: *const Glue, stream: cuda.Stream, x: u64, xout: u64, parts: u64, rows: usize, d: usize, tp_world: usize) !void {
        if (rows == 0) return;
        if (d % 8 != 0 or x % 16 != 0 or xout % 16 != 0 or parts % 16 != 0) return error.RowsUnaligned;
        var a: cuda.Args = .{};
        inline for (.{ x, xout, parts, @as(i64, @intCast(rows * d)), @as(i64, @intCast(d)), @as(c_int, @intCast(tp_world)) }) |v| a.add(v);
        try cuda.launch.launch(g.residual_add, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d + 1023) / 1024) }, .block = .{ .x = 128 } }, stream, &a);
    }

    /// glue.embed of a bf16 table: row r = table[ids[r]] (ids are rank-local rows; non-owned rows read row 0).
    pub fn embed(g: *const Glue, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        inline for (.{ ids, table, out, @as(i64, @intCast(d)), @as(c_int, 1) }) |v| a.add(v);
        try cuda.launch.launch(g.embed_b16, .{ .grid = .{ .x = @intCast(rows), .y = @intCast((d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
};

/// Every kernel family the forward launches. The lane and the EXL3 experts are loaded once per engine.
pub const Kernels = struct {
    glue: Glue,
    glue53: moe.Kernels,
    residual_rmsnorm_6144: cuda.Function,
    ablate: ?cuda.Function = null,
    swiglu_plain: cuda.Function,
    mla: mla.Mla,
    index: index.Index,
    head: sampler.Head,
    lane: *const cuda.qmmf.Lane,
    experts: *const cuda.exl3_experts.Experts,
    /// TF_GLM53_PREFILL_W8A8=1: the W8A8 o_proj kernels (cuda.w8a8, glm53_w8a8 image), set at setup; null otherwise.
    w8a8: ?cuda.w8a8.Fns = null,

    pub const Modules = struct { glm_glue: cuda.Module, glm53_glue: cuda.Module, glm53_mla: cuda.Module, glm53_index: cuda.Module, glm53_head: cuda.Module };

    pub fn load(mods: Modules, lane: *const cuda.qmmf.Lane, ex: *const cuda.exl3_experts.Experts, index_mma_rows: usize) !Kernels {
        return .{
            .glue = try Glue.load(mods.glm_glue),
            .glue53 = try moe.Kernels.load(mods.glm53_glue),
            .residual_rmsnorm_6144 = try mods.glm53_glue.function("tf_glm53_residual_rmsnorm_6144"),
            .swiglu_plain = try mods.glm53_glue.function("tf_glm53_swiglu_plain"),
            .mla = try mla.Mla.load(mods.glm53_mla),
            .index = try index.Index.load(mods.glm53_index, index_mma_rows),
            .head = try sampler.Head.resolve(mods.glm53_head),
            .lane = lane,
            .experts = ex,
        };
    }

    pub fn router(k: *const Kernels) moe.RouterKernels {
        return .{ .router_part = k.glue.router_part, .router_tile = k.glue.router_tile, .topk = k.glue.topk, .slice_sum = k.glue.slice_sum };
    }

    pub fn mlpKernels(k: *const Kernels) mlp.Kernels {
        return .{ .linear = k.lane, .swiglu_plain = k.swiglu_plain };
    }
};

/// The exchanges of one forward step, by what they carry.
pub const Phase = enum { embed, stage_fetch, index_candidates, absorbed_query, shard_qkv_a, attention_partial, attention_output, moe_partial, head };

pub const QueryPairLayout = struct {
    qa: usize,
    qpe: usize,
    rank_bytes: usize,

    pub fn init(qa_bytes: usize, qpe_bytes: usize) QueryPairLayout {
        return .{ .qa = 0, .qpe = qa_bytes, .rank_bytes = qa_bytes + qpe_bytes };
    }

    pub fn rankBase(layout: QueryPairLayout, rank: usize) usize {
        return rank * layout.rank_bytes;
    }
};

/// Host reference for the byte layout used by the decode staging gather.
pub fn packQueryPairBytes(dst: []u8, qa: []const u8, qpe: []const u8) !void {
    const layout = QueryPairLayout.init(qa.len, qpe.len);
    if (dst.len != layout.rank_bytes) return error.BadQueryPairBuffer;
    @memcpy(dst[layout.qa..][0..qa.len], qa);
    @memcpy(dst[layout.qpe..][0..qpe.len], qpe);
}

/// The TP group's transport as the forward uses it (glm/cuda_forward.zig `Exchange` over cuda.tp_group.Group):
/// rank-ordered all-gathers, and paired send/recv with the peer. `phase` is for plans and recording only.
pub const Collective = struct {
    ctx: *anyopaque,
    all_gather_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream, mine: u64, all: u64, bytes: usize, phase: Phase) anyerror!void,
    one_shot_gather_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream, mine: u64, all: u64, bytes: usize) anyerror!void = noOneShotGather,
    send_recv_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: usize, phase: Phase) anyerror!void,
    gate_result_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream, token: u64, status: u64) anyerror!void = noGate,

    fn noGate(_: *anyopaque, _: cuda.Stream, _: u64, _: u64) anyerror!void {}
    fn noOneShotGather(_: *anyopaque, _: cuda.Stream, _: u64, _: u64, _: usize) anyerror!void { return error.RoceUnavailable; }

    pub fn allGather(c: Collective, stream: cuda.Stream, mine: u64, all: u64, bytes: usize, phase: Phase) !void {
        if (bytes == 0) return;
        // MK8 flush point: the transport enqueues on stream.handle, behind any deferred segment.
        try stream.flushDeferred();
        try c.all_gather_fn(c.ctx, stream, mine, all, bytes, phase);
    }

    pub fn oneShotGather(c: Collective, stream: cuda.Stream, mine: u64, all: u64, bytes: usize) !void {
        if (bytes == 0) return;
        try stream.flushDeferred();
        try c.one_shot_gather_fn(c.ctx, stream, mine, all, bytes);
    }

    /// `send_bytes` to the peer and `recv_bytes` from it, completing in `stream` order (NCCL's grouped send/recv):
    /// an event recorded on `stream` afterwards is the cross-rank completion signal for the received bytes.
    pub fn sendRecv(c: Collective, stream: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: usize, phase: Phase) !void {
        if (send_bytes == 0 and recv_bytes == 0) return;
        try stream.flushDeferred(); // MK8 flush point
        try c.send_recv_fn(c.ctx, stream, send, send_bytes, recv, recv_bytes, peer, phase);
    }

    pub fn gateResult(c: Collective, stream: cuda.Stream, token: u64, status: u64) !void {
        try stream.flushDeferred(); // MK8 flush point
        try c.gate_result_fn(c.ctx, stream, token, status);
    }

    /// Production transport: cuda.tp_group.Group (RoCE for small all-gathers on two ranks, NCCL otherwise).
    pub fn group(g: *const cuda.tp_group.Group) Collective {
        return .{ .ctx = @ptrCast(@constCast(g)), .all_gather_fn = groupGather, .one_shot_gather_fn = groupOneShotGather, .send_recv_fn = groupSendRecv, .gate_result_fn = groupGateResult };
    }

    fn groupGather(ctx: *anyopaque, stream: cuda.Stream, mine: u64, all: u64, bytes: usize, _: Phase) anyerror!void {
        const g: *const cuda.tp_group.Group = @ptrCast(@alignCast(ctx));
        try g.allGather(stream, mine, all, bytes);
    }

    fn copyDevice(stream: cuda.Stream, dst: u64, src: u64, bytes: usize, what: [:0]const u8) !void {
        if (bytes == 0) return;
        try stream.flushDeferred();
        try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), what);
    }

    fn groupOneShotGather(ctx: *anyopaque, stream: cuda.Stream, mine: u64, all: u64, bytes: usize) anyerror!void {
        const g: *const cuda.tp_group.Group = @ptrCast(@alignCast(ctx));
        try g.oneShotGather(stream, mine, all, bytes);
    }

    fn groupGateResult(ctx: *anyopaque, stream: cuda.Stream, token: u64, status: u64) anyerror!void {
        const g: *const cuda.tp_group.Group = @ptrCast(@alignCast(ctx));
        try g.gateResult(stream, token, status);
    }

    /// One send and one receive in an NCCL group, as tp_group.Group.connect pairs them.
    fn groupSendRecv(ctx: *anyopaque, stream: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: usize, _: Phase) anyerror!void {
        const g: *const cuda.tp_group.Group = @ptrCast(@alignCast(ctx));
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            if (send_bytes != 0) g.send(stream, send, send_bytes, peer) catch |err| break :post err;
            if (recv_bytes != 0) g.recv(stream, recv, recv_bytes, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try posted;
    }
};

/// The side stream's own NCCL communicator (review M3): the stage fetch never shares a communicator with the
/// compute stream's collectives, so the two streams' NCCL kernels cannot interleave on one communicator. Opened
/// once at init over the same two ranks: rank 0 makes a unique id, the main group gathers it, both ranks init.
pub const SideGroup = struct {
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    rank: usize,
    world: usize,

    /// Both communicators come from NCCL unique ids (review N7), never from the NCCL_COMM_ID bootstrap address: the
    /// main group is opened from a unique id rank 0 made and handed to rank 1 out of band (tf-glm53-bringup
    /// --idfile), and this side group from a second unique id rank 0 makes here and the main group gathers. With
    /// NCCL_COMM_ID set, ncclGetUniqueId would encode that same fixed address and the second bootstrap root would
    /// bind it again, so it is refused. The native server removes NCCL_COMM_ID from its process environment before any NCCL call when TF_GLM53_ATTN_PIPE=1 or TF_GLM53_PREFETCH=1 (it keeps the address only for its rank sockets).
    pub fn open(main: *const cuda.tp_group.Group, d: *const cuda.Driver, s: cuda.Stream) !SideGroup {
        if (std.c.getenv("NCCL_COMM_ID") != null) return error.CommIdSetForSideComm;
        var mine: cuda.nccl.UniqueId = std.mem.zeroes(cuda.nccl.UniqueId);
        if (main.rank == 0) try main.lib.check(main.lib.api.ncclGetUniqueId(&mine), "ncclGetUniqueId (stage side)");
        var all: [3]cuda.nccl.UniqueId = undefined;
        if (main.world != 2 and main.world != 3) return error.BadWorld;
        try main.gather(cuda.nccl.UniqueId, d, s, mine, all[0..main.world]);
        var comm: cuda.nccl.Comm = null;
        try main.lib.check(main.lib.api.ncclCommInitRank(&comm, @intCast(main.world), all[0], @intCast(main.rank)), "ncclCommInitRank (stage side)");
        return .{ .lib = &main.lib, .comm = comm, .rank = main.rank, .world = main.world };
    }

    pub fn close(g: *SideGroup) void {
        _ = g.lib.api.ncclCommDestroy(g.comm);
        g.* = undefined;
    }

    pub fn collective(g: *const SideGroup) Collective {
        return .{ .ctx = @ptrCast(@constCast(g)), .all_gather_fn = gather, .send_recv_fn = sendRecv };
    }

    fn gather(ctx: *anyopaque, stream: cuda.Stream, mine: u64, all: u64, bytes: usize, _: Phase) anyerror!void {
        const g: *const SideGroup = @ptrCast(@alignCast(ctx));
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, stream.handle), "ncclAllGather (stage side)");
    }

    fn sendRecv(ctx: *anyopaque, stream: cuda.Stream, send: u64, send_bytes: usize, recv: u64, recv_bytes: usize, peer: usize, _: Phase) anyerror!void {
        const g: *const SideGroup = @ptrCast(@alignCast(ctx));
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            if (send_bytes != 0) g.lib.check(g.lib.api.ncclSend(send, send_bytes, .u8, @intCast(peer), g.comm, stream.handle), "ncclSend (stage side)") catch |err| break :post err;
            if (recv_bytes != 0) g.lib.check(g.lib.api.ncclRecv(recv, recv_bytes, .u8, @intCast(peer), g.comm, stream.handle), "ncclRecv (stage side)") catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        try posted;
    }
};

/// Host-to-device row inputs (positions, rope rows, ids). Production enqueues through cuda.copy_source.
pub const Upload = struct {
    ctx: ?*anyopaque = null,
    upload_fn: *const fn (ctx: ?*anyopaque, stream: cuda.Stream, dst: u64, bytes: []const u8) anyerror!void = copySource,

    pub fn put(u: Upload, stream: cuda.Stream, dst: u64, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        try stream.flushDeferred(); // MK8 flush point
        try u.upload_fn(u.ctx, stream, dst, bytes);
    }

    fn copySource(_: ?*anyopaque, stream: cuda.Stream, dst: u64, bytes: []const u8) anyerror!void {
        try cuda.copy_source.enqueue(stream.d, dst, bytes, stream);
    }
};

pub const Mode = enum { decode, prefill };

pub fn tboSwitch() bool {
    const value = std.c.getenv("TF_GLM53_TBO") orelse return false;
    return tboSelected(std.mem.span(value));
}

pub fn shardQkvASwitch() bool {
    const value = std.c.getenv("TF_GLM53_SHARD_QKV_A") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn tboSelected(value: ?[]const u8) bool {
    const raw = value orelse return false;
    return std.mem.eql(u8, raw, "1");
}

pub fn profilePrefillSwitch() bool {
    const value = std.c.getenv("TF_GLM53_PROFILE_PREFILL") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

/// TF_GLM53_PROFILE_DECODE_SUB=1 (bring-up profiling only, with TF_GLM53_PROFILE_PREFILL=1): also time the attn_pre
/// sub-stages in decode steps.
pub fn profileDecodeSubSwitch() bool {
    const value = std.c.getenv("TF_GLM53_PROFILE_DECODE_SUB") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn validateProfilePrefillTbo(profile_prefill: bool, tbo_enabled: bool) !void {
    if (profile_prefill and tbo_enabled) {
        if (!@import("builtin").is_test) std.log.err("TF_GLM53_PROFILE_PREFILL=1 does not support TF_GLM53_TBO=1", .{});
        return error.ProfilePrefillWithTbo;
    }
}

pub fn piecePipeSwitch() bool {
    const value = std.c.getenv("TF_GLM53_PIECE_PIPE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn moePipeSwitch() bool {
    const value = std.c.getenv("TF_GLM53_MOE_PIPE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn wsTrimSwitch() bool {
    const value = std.c.getenv("TF_GLM53_WS_TRIM") orelse return false;
    return value[0] == '1' and value[1] == 0;
}

pub fn moePostSwitch() bool {
    const value = std.c.getenv("TF_GLM53_MOE_POST") orelse return false;
    return value[0] == '1' and value[1] == 0;
}

pub fn commPrioSwitch() bool {
    const value = std.c.getenv("TF_GLM53_COMM_PRIO") orelse return false;
    return value[0] == '1' and value[1] == 0;
}

pub fn indexMergeSwitch() u8 {
    const value = std.c.getenv("TF_GLM53_INDEX_MERGE") orelse return 0;
    return if (std.mem.eql(u8, std.mem.span(value), "2")) 2 else 0;
}

pub fn indexPipeSwitch() bool {
    const value = std.c.getenv("TF_GLM53_INDEX_PIPE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn attnPipeSwitch() bool {
    const value = std.c.getenv("TF_GLM53_ATTN_PIPE") orelse return false;
    return attnPipeSelected(std.mem.span(value));
}

pub fn attnPipeSelected(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}

pub const PipeOp = enum { attention, reduce_issue, reduce_finish, ffn_pre, routed_group };

pub const IndexPipeOp = struct { kind: enum { score, ready, gather, merge_wait, merge }, block: u16 = 0 };

/// Host model for one piece: each gather is queued in identical rank order and the merge follows every gather.
pub fn indexPipeSchedule(blocks: usize, out: []IndexPipeOp) !usize {
    if (blocks == 0 or out.len < blocks * 3 + 2) return error.ScheduleBufferTooSmall;
    var at: usize = 0;
    for (0..blocks) |block| {
        out[at] = .{ .kind = .score, .block = @intCast(block) };
        out[at + 1] = .{ .kind = .ready, .block = @intCast(block) };
        out[at + 2] = .{ .kind = .gather, .block = @intCast(block) };
        at += 3;
    }
    out[at] = .{ .kind = .merge_wait };
    out[at + 1] = .{ .kind = .merge };
    return at + 2;
}

/// Host-side order model for the grouped prefill schedule; used to guard the required overlap and flush point.
pub fn piecePipeSchedule(enabled: bool, pieces: usize, out: []PipeOp) !usize {
    var at: usize = 0;
    const needed = if (enabled) pieces * 4 + 1 else pieces * 2 + 1;
    if (out.len < needed) return error.ScheduleBufferTooSmall;
    for (0..pieces) |_| {
        out[at] = .attention;
        at += 1;
        if (enabled) {
            out[at] = .reduce_issue;
            at += 1;
            if (at > 2) {
                out[at] = .reduce_finish;
                out[at + 1] = .ffn_pre;
                at += 2;
            }
        } else {
            out[at] = .ffn_pre;
            at += 1;
        }
    }
    if (enabled and pieces != 0) {
        out[at] = .reduce_finish;
        out[at + 1] = .ffn_pre;
        at += 2;
    }
    out[at] = .routed_group;
    return at + 1;
}

test "piece pipe issues next attention before finishing prior piece and flushes before routed group" {
    var ops: [13]PipeOp = undefined;
    const n = try piecePipeSchedule(true, 3, &ops);
    const attention_next = 1 + std.mem.indexOfScalar(PipeOp, ops[1..n], .attention).?;
    const first_finish = std.mem.indexOfScalar(PipeOp, ops[0..n], .reduce_finish).?;
    const first_pre = std.mem.indexOfScalar(PipeOp, ops[0..n], .ffn_pre).?;
    const routed = std.mem.indexOfScalar(PipeOp, ops[0..n], .routed_group).?;
    try std.testing.expect(attention_next < first_finish and first_finish < first_pre);
    try std.testing.expect(ops[routed - 2] == .reduce_finish and ops[routed - 1] == .ffn_pre);
    var off: [7]PipeOp = undefined;
    const off_n = try piecePipeSchedule(false, 3, &off);
    try std.testing.expectEqualSlices(PipeOp, &.{ .attention, .ffn_pre, .attention, .ffn_pre, .attention, .ffn_pre, .routed_group }, off[0..off_n]);
}

test "index pipe queues gathers in rank order and merges only after every gather" {
    var ops: [26]IndexPipeOp = undefined;
    const n = try indexPipeSchedule(8, &ops);
    try std.testing.expectEqual(@as(usize, 26), n);
    for (0..8) |block| {
        try std.testing.expectEqual(IndexPipeOp{ .kind = .score, .block = @intCast(block) }, ops[block * 3]);
        try std.testing.expectEqual(IndexPipeOp{ .kind = .ready, .block = @intCast(block) }, ops[block * 3 + 1]);
        try std.testing.expectEqual(IndexPipeOp{ .kind = .gather, .block = @intCast(block) }, ops[block * 3 + 2]);
    }
    try std.testing.expectEqual(IndexPipeOp{ .kind = .merge_wait }, ops[24]);
    try std.testing.expectEqual(IndexPipeOp{ .kind = .merge }, ops[25]);
    // Both ranks run the same deterministic host schedule before enqueueing collectives.
    var peer: [26]IndexPipeOp = undefined;
    const peer_n = try indexPipeSchedule(8, &peer);
    try std.testing.expectEqual(n, peer_n);
    try std.testing.expectEqualSlices(IndexPipeOp, &ops, &peer);
}

pub const MoePipeOp = enum { slot_wait, block_post, ready_record, comm_wait, reduce_issue, residual_add, done_record, reduce_finish, piece_wait, next_attention, drain, routed_group, taps, final_norm };

/// Host model of the issue order with the switch on: per piece, main waits a reused slot's done (two slots), posts
/// the partial and records ready; the comm stream waits ready, gathers, adds the residual and records done. The next
/// layer's attention of each piece waits that piece's done; the group drains before routedGroup, taps and the head.
/// Off: blockPost then the synchronous reduce per piece.
pub fn moePipeSchedule(enabled: bool, pieces: usize, out: []MoePipeOp) !usize {
    const needed = if (enabled) pieces * 9 + 6 else pieces * 2 + 3;
    if (out.len < needed) return error.ScheduleBufferTooSmall;
    var at: usize = 0;
    const put = struct {
        fn f(o: []MoePipeOp, i: *usize, op: MoePipeOp) void {
            o[i.*] = op;
            i.* += 1;
        }
    }.f;
    for (0..pieces) |piece| {
        if (enabled and piece >= 2) put(out, &at, .slot_wait);
        put(out, &at, .block_post);
        if (enabled) {
            for ([_]MoePipeOp{ .ready_record, .comm_wait, .reduce_issue, .residual_add, .done_record }) |op| put(out, &at, op);
        } else put(out, &at, .reduce_finish);
    }
    if (enabled) for (0..pieces) |_| {
        put(out, &at, .piece_wait);
        put(out, &at, .next_attention);
    };
    if (enabled) put(out, &at, .drain);
    put(out, &at, .routed_group);
    if (enabled) put(out, &at, .drain);
    put(out, &at, .taps);
    if (enabled) put(out, &at, .drain);
    put(out, &at, .final_norm);
    return at;
}

test "MoE pipe waits on slot reuse and piece completion before next attention and drains boundaries" {
    var ops: [33]MoePipeOp = undefined;
    const n = try moePipeSchedule(true, 3, &ops);
    try std.testing.expectEqualSlices(MoePipeOp, &.{ .block_post, .ready_record, .comm_wait, .reduce_issue, .residual_add, .done_record, .block_post, .ready_record, .comm_wait, .reduce_issue, .residual_add, .done_record, .slot_wait, .block_post, .ready_record, .comm_wait, .reduce_issue, .residual_add, .done_record, .piece_wait, .next_attention, .piece_wait, .next_attention, .piece_wait, .next_attention, .drain, .routed_group, .drain, .taps, .drain, .final_norm }, ops[0..n]);
    const residual = std.mem.indexOfScalar(MoePipeOp, ops[0..n], .residual_add).?;
    const first_wait = std.mem.indexOfScalar(MoePipeOp, ops[0..n], .piece_wait).?;
    try std.testing.expect(residual < first_wait and ops[first_wait + 1] == .next_attention);
    const reuse = std.mem.indexOfScalar(MoePipeOp, ops[0..n], .slot_wait).?;
    try std.testing.expect(ops[reuse + 1] == .block_post);
    try std.testing.expect(ops[n - 1] == .final_norm);
    var off: [9]MoePipeOp = undefined;
    const off_n = try moePipeSchedule(false, 3, &off);
    try std.testing.expectEqualSlices(MoePipeOp, &.{ .block_post, .reduce_finish, .block_post, .reduce_finish, .block_post, .reduce_finish, .routed_group, .taps, .final_norm }, off[0..off_n]);
}

pub const TboOp = enum { compute_piece, collective_piece, consume_piece, compute_a, ready_a, collective_a, compute_b, ready_b, collective_b, wait_a, consume_a, wait_b, consume_b };

pub fn tboSchedule(enabled: bool, out: []TboOp) !usize {
    if (enabled) {
        const planned = [_]TboOp{ .compute_a, .ready_a, .collective_a, .compute_b, .ready_b, .collective_b, .wait_a, .consume_a, .wait_b, .consume_b };
        if (out.len < planned.len) return error.ScheduleBufferTooSmall;
        @memcpy(out[0..planned.len], &planned);
        return planned.len;
    }
    const planned = [_]TboOp{ .compute_piece, .collective_piece, .consume_piece };
    if (out.len < planned.len) return error.ScheduleBufferTooSmall;
    @memcpy(out[0..planned.len], &planned);
    return planned.len;
}

test "TBO launch schedule preserves off order and waits before both half consumers" {
    const off_expected = [_]TboOp{ .compute_piece, .collective_piece, .consume_piece };
    var off: [off_expected.len]TboOp = undefined;
    const off_len = try tboSchedule(false, &off);
    try std.testing.expectEqualSlices(TboOp, &off_expected, off[0..off_len]);
    const on_expected = [_]TboOp{ .compute_a, .ready_a, .collective_a, .compute_b, .ready_b, .collective_b, .wait_a, .consume_a, .wait_b, .consume_b };
    var rank_zero: [on_expected.len]TboOp = undefined;
    var rank_one: [on_expected.len]TboOp = undefined;
    const zero_len = try tboSchedule(true, &rank_zero);
    const one_len = try tboSchedule(true, &rank_one);
    try std.testing.expectEqualSlices(TboOp, &on_expected, rank_zero[0..zero_len]);
    try std.testing.expectEqualSlices(TboOp, rank_zero[0..zero_len], rank_one[0..one_len]);
    const wait_a = std.mem.indexOfScalar(TboOp, rank_zero[0..zero_len], .wait_a).?;
    const consume_a = std.mem.indexOfScalar(TboOp, rank_zero[0..zero_len], .consume_a).?;
    const wait_b = std.mem.indexOfScalar(TboOp, rank_zero[0..zero_len], .wait_b).?;
    const consume_b = std.mem.indexOfScalar(TboOp, rank_zero[0..zero_len], .consume_b).?;
    try std.testing.expect(wait_a < consume_a and wait_b < consume_b);
}

test "TBO switch is opt-in" {
    try std.testing.expect(!tboSelected(null));
    try std.testing.expect(!tboSelected("0"));
    try std.testing.expect(tboSelected("1"));
}

test "pre-fuse and prep-fuse switches are opt-in and mode-specific" {
    try std.testing.expectEqual(PreFuse.off, preFuseSelected(null));
    try std.testing.expectEqual(PreFuse.off, preFuseSelected("0"));
    try std.testing.expectEqual(PreFuse.on, preFuseSelected("1"));
    try std.testing.expectEqual(PreFuse.prefetch, preFuseSelected("2"));
    try std.testing.expectEqual(PreFuse.off, preFuseSelected("3"));
    try std.testing.expectEqual(PreFuse.off, preFuseSelected("on"));
    try std.testing.expectEqual(PrepFuse.off, prepFuseSelected(null));
    try std.testing.expectEqual(PrepFuse.off, prepFuseSelected("0"));
    try std.testing.expectEqual(PrepFuse.on, prepFuseSelected("1"));
    try std.testing.expectEqual(PrepFuse.split, prepFuseSelected("2"));
    try std.testing.expectEqual(PrepFuse.off, prepFuseSelected("3"));
    try std.testing.expectEqual(PrepFuse.off, prepFuseSelected("on"));
    try std.testing.expect(!groupedPreFor(.decode, .off, .on));
    try std.testing.expect(groupedPreFor(.decode, .on, .off));
    try std.testing.expect(!groupedPreFor(.prefill, .prefetch, .off));
    try std.testing.expect(groupedPreFor(.prefill, .off, .split));
}
pub const LayerKind = enum { dense, moe };
pub const Stage = enum {
    embed,
    attn_pre,
    attn_pre_input_norm,
    attn_pre_q_a_proj,
    attn_pre_kv_a_proj, // empty when groupedPre runs q_a, kv_a and wk together in attn_pre_q_a_proj
    attn_pre_q_a_norm,
    attn_pre_q_b_proj,
    attn_pre_kv_a_norm_rope,
    attn_pre_qkv_gather,
    attn_pre_indexer_weights_proj,
    attn_pre_indexer_key_proj, // empty when groupedPre runs q_a, kv_a and wk together in attn_pre_q_a_proj
    attn_pre_indexer_keys_write,
    attn_pre_indexer_query_proj,
    attn_pre_indexer_scoring,
    attn_pre_indexer_topk,
    attn_pre_indexer_gather,
    attn_pre_indexer_merge,
    attn_pre_q_pe_pack,
    attn_pre_absorb,
    attn_pre_smla_p_scores,
    attn_comm,
    attn_post,
    expand_v,
    route,
    routed,
    shared,
    lowrank,
    moe_post, // decode side-stream path: the routed + shared combine after the side stream joins
    moe_comm,
    dense_mlp,
    dense_comm,
    head,
};
pub const stage_count = std.enums.values(Stage).len;
const profile_extra_pairs: usize = blk: {
    var total: usize = 0;
    for (std.enums.values(Stage)) |stage| total += stagePairs(stage) - 1;
    break :blk total;
};

pub fn stagePairs(stage: Stage) usize {
    return switch (stage) {
        .attn_pre => 10,
        .attn_pre_indexer_scoring, .attn_pre_indexer_topk, .attn_pre_indexer_gather, .attn_pre_indexer_merge => 16,
        .attn_comm, .moe_comm => 16,
        .attn_post => 3,
        else => 1,
    };
}

pub fn isSubStage(stage: Stage) bool {
    return switch (stage) {
        .attn_pre_input_norm, .attn_pre_q_a_proj, .attn_pre_kv_a_proj, .attn_pre_q_a_norm,
        .attn_pre_q_b_proj, .attn_pre_kv_a_norm_rope, .attn_pre_qkv_gather,
        .attn_pre_indexer_weights_proj, .attn_pre_indexer_key_proj, .attn_pre_indexer_keys_write,
        .attn_pre_indexer_query_proj, .attn_pre_indexer_scoring, .attn_pre_indexer_topk,
        .attn_pre_indexer_gather, .attn_pre_indexer_merge, .attn_pre_q_pe_pack,
        .attn_pre_absorb, .attn_pre_smla_p_scores => true,
        else => false,
    };
}

pub fn stagePairBase(layer: usize, stage: Stage) usize {
    var extra: usize = 0;
    for (std.enums.values(Stage)) |prior| {
        if (@intFromEnum(prior) >= @intFromEnum(stage)) break;
        extra += stagePairs(prior) - 1;
    }
    return layer * (stage_count + profile_extra_pairs) + @intFromEnum(stage) + extra;
}

/// Caller-owned CUDA event pool and accumulated decode timings. Split stages keep multiple reusable pairs per layer.
pub const Profile = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    max_layers: usize,
    events: []cuda.Event,
    completed: []bool,
    open: []bool,
    started: []u64,
    sequence: u64 = 0,
    totals_ms: [stage_count]f64 = @splat(0),
    counts: [stage_count]u64 = @splat(0),

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, max_layers: usize) !Profile {
        const pairs = try std.math.mul(usize, max_layers, stage_count + profile_extra_pairs);
        const events = try gpa.alloc(cuda.Event, try std.math.mul(usize, pairs, 2));
        errdefer gpa.free(events);
        var made: usize = 0;
        errdefer for (events[0..made]) |*ev| ev.deinit();
        for (events) |*ev| {
            ev.* = try cuda.Event.init(d, true);
            made += 1;
        }
        const completed = try gpa.alloc(bool, pairs);
        @memset(completed, false);
        errdefer gpa.free(completed);
        const open = try gpa.alloc(bool, pairs);
        @memset(open, false);
        errdefer gpa.free(open);
        const started = try gpa.alloc(u64, pairs);
        errdefer gpa.free(started);
        return .{ .gpa = gpa, .d = d, .max_layers = max_layers, .events = events, .completed = completed, .open = open, .started = started };
    }

    pub fn deinit(p: *Profile) void {
        for (p.events) |*ev| ev.deinit();
        p.gpa.free(p.events);
        p.gpa.free(p.completed);
        p.gpa.free(p.open);
        p.gpa.free(p.started);
        p.* = undefined;
    }

    fn pair(p: *Profile, layer: usize, stage: Stage) !usize {
        if (layer >= p.max_layers) return error.ProfileLayerOutOfRange;
        return stagePairBase(layer, stage);
    }

    fn freeSlot(p: *const Profile, layer: usize, stage: Stage) ?usize {
        const base = stagePairBase(layer, stage);
        for (0..stagePairs(stage)) |slot| {
            const i = base + slot;
            if (!p.open[i] and !p.completed[i]) return slot;
        }
        return null;
    }

    /// Whether begin(layer, stage) runs out of reusable pairs and must collect completed events.
    pub fn wouldCollect(p: *const Profile, layer: usize, stage: Stage) bool {
        if (layer >= p.max_layers) return false;
        return p.freeSlot(layer, stage) == null;
    }

    pub fn begin(p: *Profile, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        const base = try p.pair(layer, stage);
        if (p.freeSlot(layer, stage) == null) {
            try stream.synchronize();
            try p.collect();
        }
        const slot = p.freeSlot(layer, stage) orelse return error.ProfilePairCapacity;
        const i = base + slot;
        try p.events[i * 2].record(stream);
        p.sequence +%= 1;
        p.started[i] = p.sequence;
        p.open[i] = true;
    }

    pub fn end(p: *Profile, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        const base = try p.pair(layer, stage);
        var found: ?usize = null;
        var oldest: u64 = std.math.maxInt(u64);
        for (0..stagePairs(stage)) |slot| {
            const i = base + slot;
            if (p.open[i] and p.started[i] < oldest) {
                found = slot;
                oldest = p.started[i];
            }
        }
        const slot = found orelse return error.ProfilePairCapacity;
        const i = base + slot;
        try p.events[i * 2 + 1].record(stream);
        p.completed[i] = true;
        p.open[i] = false;
    }

    pub fn collect(p: *Profile) !void {
        for (0..p.max_layers) |layer| {
            for (std.enums.values(Stage)) |stage| {
                const base = stagePairBase(layer, stage);
                for (0..stagePairs(stage)) |slot| {
                    const i = base + slot;
                    if (!p.completed[i]) continue;
                    const ms = try cuda.Event.elapsedMs(p.events[i * 2], p.events[i * 2 + 1]);
                    const stage_index = @intFromEnum(stage);
                    p.totals_ms[stage_index] += ms;
                    p.counts[stage_index] += 1;
                    p.completed[i] = false;
                }
            }
        }
    }

    /// One log line per stage that ran: its milliseconds per verify window over `windows` windows of `rows` rows in
    /// all, behind position `position`. Sub-stages (attn_pre.*) also count inside their parent stage.
    pub fn report(p: *const Profile, windows: u64, rows: u64, position: usize) void {
        if (windows == 0) return;
        const w: f64 = @floatFromInt(windows);
        std.log.info("glm53 profile: {d} windows, {d:.2} rows each, position {d}", .{ windows, @as(f64, @floatFromInt(rows)) / w, position });
        for (std.enums.values(Stage)) |stage| {
            const i = @intFromEnum(stage);
            if (p.counts[i] == 0) continue;
            std.log.info("glm53 profile: {s:<34} {d:>8.3} ms/window", .{ name(stage), p.totals_ms[i] / w });
        }
    }

    pub fn reset(p: *Profile) void {
        p.totals_ms = @splat(0);
        p.counts = @splat(0);
        @memset(p.open, false);
        @memset(p.started, 0);
        p.sequence = 0;
        @memset(p.completed, false);
    }

    pub fn name(stage: Stage) []const u8 {
        return switch (stage) {
            .attn_pre_input_norm => "attn_pre.input_norm",
            .attn_pre_q_a_proj => "attn_pre.q_a_proj",
            .attn_pre_kv_a_proj => "attn_pre.kv_a_proj",
            .attn_pre_q_a_norm => "attn_pre.q_a_norm",
            .attn_pre_q_b_proj => "attn_pre.q_b_proj",
            .attn_pre_kv_a_norm_rope => "attn_pre.kv_a_norm_rope",
            .attn_pre_qkv_gather => "attn_pre.qkv_gather",
            .attn_pre_indexer_weights_proj => "attn_pre.indexer.weights_proj",
            .attn_pre_indexer_key_proj => "attn_pre.indexer.key_proj",
            .attn_pre_indexer_keys_write => "attn_pre.indexer.keys_write",
            .attn_pre_indexer_query_proj => "attn_pre.indexer.query_proj",
            .attn_pre_indexer_scoring => "attn_pre.indexer.scoring",
            .attn_pre_indexer_topk => "attn_pre.indexer.topk",
            .attn_pre_indexer_gather => "attn_comm.indexer.gather",
            .attn_pre_indexer_merge => "attn_pre.indexer.merge",
            .attn_pre_q_pe_pack => "attn_pre.q_pe_pack",
            .attn_pre_absorb => "attn_pre.absorb",
            .attn_pre_smla_p_scores => "attn_pre.smla_p_scores",
            else => @tagName(stage),
        };
    }
};
/// A layer as plans see it (Forward derives it from weights.DeviceLayer).
pub const Layer = struct { index: usize, kind: LayerKind, index_full: bool = false, has_lowrank: bool = false };

pub fn layerOf(l: weights.DeviceLayer) Layer {
    return .{ .index = l.index, .kind = if (l.ffn == .dense) .dense else .moe, .index_full = l.attn.indexer != null, .has_lowrank = l.ffn == .moe and l.ffn.moe.lowrank != null };
}

/// Head output per row: none (prefill rows before the last), greedy top-1 + local LSE, or k fp32 candidates.
pub const HeadMode = union(enum) {
    none,
    greedy,
    candidates: usize,

    pub fn rowBytes(h: HeadMode) usize {
        return switch (h) {
            .none => 0,
            .greedy => 16,
            .candidates => |k| k * 8 + 8,
        };
    }
};

/// One exchange phase of a forward step: bytes this rank sends in total, and the number of collective calls.
pub const Exchange = struct { phase: Phase, bytes: usize = 0, calls: usize = 0, all_gather: bool = true, when: []const u8 = "" };

pub const ExchangePlan = struct {
    entries: [@typeInfo(Phase).@"enum".field_names.len]Exchange,
    two_batch_overlap: bool = false,
    gather_on_prefill: bool = false,

    pub fn of(p: *const ExchangePlan, phase: Phase) Exchange {
        return p.entries[@backingInt(phase)];
    }

    fn add(p: *ExchangePlan, phase: Phase, bytes: usize, calls: usize) void {
        p.entries[@backingInt(phase)].bytes += bytes;
        p.entries[@backingInt(phase)].calls += calls;
    }
};

pub const Bytes = struct {
    pub const embed_row = hidden * 2;
    pub const absorbed_query_row = state.local_heads * (mla.latent_dim + mla.rope_dim) * 2;
    pub const attention_partial_row = state.decode_partial_row_bytes;
    pub const reduce_row = hidden * 4;
    pub const moe_partial_row = hidden * 2;
    pub const index_candidates_row = index.top_k * 8;
    pub const stage_row = state.kv_row_bytes;
};

/// What one forward step exchanges, from its shape: rows and absolute positions, the layers, the head, the pieces.
pub const StepShape = struct {
    mode: Mode,
    positions: []const usize,
    layers: []const Layer,
    head: HeadMode = .greedy,
    head_rows: usize = 0,
    head_block: usize = max_decode_rows,
    score_rows: usize = 128,
    /// Attention/FFN piece rows inside a prefill chunk (cuts at absolute multiples); decode is one piece.
    piece_rows: usize = 1024,
    /// This rank (the stage fetch sends this rank's own rows below chunk_start).
    rank: usize = 0,
    /// Models the opt-in decode plan for host collective-count assertions.
    coll_fuse: bool = false,
};

/// Rows that select from the indexer (position >= 2048).
fn sparseIn(positions: []const usize) bool {
    for (positions) |p| if (p >= index.dense_limit) return true;
    return false;
}

/// The piece of a window starting at row `at`: decode is one piece; prefill cuts at absolute multiples of
/// `piece_rows`, so a resumed prompt cuts where a fresh one does.
pub fn pieceAt(mode: Mode, positions: []const usize, piece_rows: usize, at: usize) Chunk {
    if (mode == .decode or piece_rows == 0) return .{ .start = at, .end = positions.len };
    const bound = (positions[at] / piece_rows + 1) * piece_rows;
    var end = at;
    while (end < positions.len and positions[end] < bound) end += 1;
    return .{ .start = at, .end = end };
}

/// Assign local ordinals to absolute prefill pieces in one group.
pub fn pieceOrdinals(positions: []const usize, piece_rows: usize, group_at: usize, group_end: usize, out: []usize) usize {
    var at = group_at;
    var count: usize = 0;
    while (at < group_end) {
        std.debug.assert(count < out.len);
        const piece = pieceAt(.prefill, positions, piece_rows, at);
        out[count] = count;
        count += 1;
        at = piece.end;
    }
    return count;
}

/// Decode windows are keyed by the power-of-two bucket of their last position (at least the 2048 dense limit):
/// whether any row is sparse, and how many local keys the indexer scores, are functions of the bucket.
pub fn decodeKeyBucket(last_position: usize) usize {
    return @max(index.dense_limit, std.math.ceilPowerOfTwoAssert(usize, last_position + 1));
}

/// Rank `owner`'s cache rows below `chunk_start`: positions p < chunk_start with p % 2 == owner, slots 0.. of its
/// plane. Counted per owner (an odd chunk_start gives rank 0 one row more than rank 1).
pub fn stageFetchSlots(chunk_start: usize, owner: usize) usize {
    return (chunk_start + world - 1 - owner) / world;
}

pub fn plan(s: StepShape) !ExchangePlan {
    const rows = s.positions.len;
    if (rows == 0 or s.piece_rows == 0 or s.score_rows == 0) return error.BadExchangePlan;
    var p: ExchangePlan = .{ .entries = undefined, .gather_on_prefill = s.mode == .prefill };
    for (&p.entries, 0..) |*e, i| e.* = .{ .phase = @fromBackingInt(@intCast(i)) };
    p.entries[@backingInt(Phase.embed)].when = "per piece, after local vocab-row embedding: bf16 [world][piece][6144], owner pick";
    p.entries[@backingInt(Phase.stage_fetch)] = .{ .phase = .stage_fetch, .all_gather = false, .when = "prefill, before each layer's attention: peer latent+rope rows below chunk_start into stage[p/2]" };
    p.entries[@backingInt(Phase.index_candidates)].when = "full layers with sparse rows: packed local top-2048 {score, position} per score block";
    p.entries[@backingInt(Phase.absorbed_query)].when = "decode: qa [world][R][32][512] and q_pe [world][R][32][64]";
    p.entries[@backingInt(Phase.shard_qkv_a)].when = "prefill and decode: packed per-rank q_a and kv_a rows";
    p.entries[@backingInt(Phase.attention_partial)] = .{ .phase = .attention_partial, .all_gather = s.mode == .decode and s.coll_fuse, .when = if (s.mode == .decode and s.coll_fuse) "decode: one-shot rank gather of bf16 partials [R][32][512] + fp32 LSE [R][32]" else "decode: peer-only bf16 partials [R][32] + fp32 LSE [R][32]" };
    p.entries[@backingInt(Phase.attention_output)].when = "per piece, after o_proj: fp32 rank partials, rank-order sum, residual";
    p.entries[@backingInt(Phase.moe_partial)].when = "per piece: dense MLP fp32 or MoE BF16 rank partials, rank-order sum, residual";
    p.entries[@backingInt(Phase.head)].when = "after the fp32 head: top-1 + LSE (16 B/row) or k candidates (8k B/row)";
    var npieces: usize = 0;
    var at: usize = 0;
    while (at < rows) : (npieces += 1) {
        const pc = pieceAt(s.mode, s.positions, s.piece_rows, at);
        p.add(.embed, (pc.end - pc.start) * Bytes.embed_row, 1);
        at = pc.end;
    }
    p.two_batch_overlap = npieces > 1;
    const chunk_start = s.positions[0];
    for (s.layers) |l| {
        if (s.mode == .prefill and chunk_start > 0) p.add(.stage_fetch, stageFetchSlots(chunk_start, s.rank) * Bytes.stage_row, 2);
        at = 0;
        while (at < rows) {
            const pc = pieceAt(s.mode, s.positions, s.piece_rows, at);
            const n = pc.end - pc.start;
            const pos = s.positions[pc.start..pc.end];
            if (l.index_full and sparseIn(pos)) {
                var b: usize = 0;
                while (b < n) : (b += s.score_rows) {
                    const m = @min(s.score_rows, n - b);
                    if (sparseIn(pos[b..][0..m])) p.add(.index_candidates, m * Bytes.index_candidates_row, 1);
                }
            }
            if (s.mode == .decode) {
                p.add(.absorbed_query, n * Bytes.absorbed_query_row, if (s.coll_fuse) 1 else 2);
                p.add(.attention_partial, n * Bytes.attention_partial_row, 1);
            }
            p.add(.attention_output, n * Bytes.reduce_row, 1);
            const partial_row: usize = if (l.kind == .moe) Bytes.moe_partial_row else Bytes.reduce_row;
            p.add(.moe_partial, n * partial_row, 1);
            at = pc.end;
        }
    }
    if (s.head != .none and s.head_rows > 0) p.add(.head, s.head_rows * s.head.rowBytes(), (s.head_rows + s.head_block - 1) / s.head_block);
    return p;
}

pub const Chunk = struct { start: usize, end: usize };

/// TF_GLM53_MK=1 turns on the MK8 deferred launch in decode steps (default off; prefill never defers).
pub fn megakernelSwitch() bool {
    if (std.c.getenv("TF_GLM_WORLD")) |w| {
        if (std.mem.eql(u8, std.mem.span(w), "3")) return false;
    }
    const v = std.c.getenv("TF_GLM53_MK") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The decode side stream is opt-in until Spark parity and timing are signed off.
pub fn decodeSideSwitch() bool {
    const value = std.c.getenv("TF_GLM53_SIDE") orelse return false;
    return decodeSideSelected(std.mem.span(value));
}

pub fn decodeSideSelected(value: ?[]const u8) bool {
    const raw = value orelse return false;
    return !std.mem.eql(u8, raw, "0");
}


/// TF_GLM53_PROMPT_EXPERTS is opt-in: 1 selects 64-member kernels, 2 the wide kernels, 3 PMX, and 10 is =3 with p5a kernels for its <= 64-member items.
pub const PromptExperts = enum { off, tiles, wide, pmx, p5a };

/// The Forward/Geometry flags of a TF_GLM53_PROMPT_EXPERTS mode. Every setup site uses this; the switch is exhaustive.
pub const PromptExpertsFlags = struct { on: bool, wide: bool, pmx: bool, p5a: bool };
pub fn promptExpertsFlags(mode: PromptExperts) PromptExpertsFlags {
    return switch (mode) {
        .off => .{ .on = false, .wide = false, .pmx = false, .p5a = false },
        .tiles => .{ .on = true, .wide = false, .pmx = false, .p5a = false },
        .wide => .{ .on = true, .wide = true, .pmx = false, .p5a = false },
        .pmx => .{ .on = true, .wide = true, .pmx = true, .p5a = false },
        .p5a => .{ .on = true, .wide = true, .pmx = true, .p5a = true },
    };
}

pub fn promptExpertsMode() PromptExperts {
    const value = std.c.getenv("TF_GLM53_PROMPT_EXPERTS") orelse return .off;
    return promptExpertsModeSelected(std.mem.span(value));
}

pub fn promptExpertsModeSelected(value: ?[]const u8) PromptExperts {
    const raw = value orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .tiles;
    if (std.mem.eql(u8, raw, "2")) return .wide;
    if (std.mem.eql(u8, raw, "3")) return .pmx;
    if (std.mem.eql(u8, raw, "10")) return .p5a;
    return .off;
}

test "promptExpertsFlags: =10 is =3's flags plus p5a" {
    try std.testing.expectEqual(PromptExpertsFlags{ .on = false, .wide = false, .pmx = false, .p5a = false }, promptExpertsFlags(.off));
    try std.testing.expectEqual(PromptExpertsFlags{ .on = true, .wide = false, .pmx = false, .p5a = false }, promptExpertsFlags(.tiles));
    try std.testing.expectEqual(PromptExpertsFlags{ .on = true, .wide = true, .pmx = false, .p5a = false }, promptExpertsFlags(.wide));
    try std.testing.expectEqual(PromptExpertsFlags{ .on = true, .wide = true, .pmx = true, .p5a = false }, promptExpertsFlags(.pmx));
    try std.testing.expectEqual(PromptExpertsFlags{ .on = true, .wide = true, .pmx = true, .p5a = true }, promptExpertsFlags(.p5a));
}

/// Whether TF_GLM53_PROMPT_EXPERTS selects a prompt path: the workspace carve and the prefill route.
pub fn promptExpertsSwitch() bool {
    return promptExpertsMode() != .off;
}

pub fn promptExpertsSelected(value: ?[]const u8) bool {
    return promptExpertsModeSelected(value) != .off;
}

/// TF_GLM53_PROMPT_MLA selects the fixed 16x8 kernel with "1" and grouped 64x64 kernel with "2".
pub const PromptMlaMode = mla.PromptMlaMode;

pub fn promptMlaSwitch() PromptMlaMode {
    const value = std.c.getenv("TF_GLM53_PROMPT_MLA") orelse return .off;
    return promptMlaSelected(std.mem.span(value));
}

pub fn promptMlaSelected(value: ?[]const u8) PromptMlaMode {
    const raw = value orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .tile16;
    if (std.mem.eql(u8, raw, "2")) return .tile64;
    return .off;
}

/// SMLA-P (the query-blocked prefill attend, mla.cu tf_glm53_smla_p) is opt-in; only the exact value "1" enables it.
/// Decode keeps tf_glm53_attend_mma / tf_glm53_attend.
pub fn smlaPSwitch() bool {
    const value = std.c.getenv("TF_GLM53_SMLA_P") orelse return false;
    return smlaPSelected(std.mem.span(value));
}

pub fn smlaPSelected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

/// SMLA-FP8 (the e4m3 score GEMM inside SMLA-P, mla.cu tf_glm53_smla_p_fp8) is opt-in and lossy (KLD gate); only the
/// exact value "1" enables it, and setup refuses it without TF_GLM53_SMLA_P=1. The KV cache stays bf16.
pub fn smlaFp8Switch() bool {
    const value = std.c.getenv("TF_GLM53_SMLA_FP8") orelse return false;
    return smlaFp8Selected(std.mem.span(value));
}

pub fn smlaFp8Selected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

/// SMLA-P3 (SMLA-P with the step unions precomputed by tf_glm53_smla_union and 1 full + 2 named barriers a step, mla.cu
/// tf_glm53_smla_p3; bit-identical to SMLA-P) is opt-in; only the exact value "1" enables it. Setup reads it once,
/// carries it in Geometry (the union table carve) and Forward, and refuses it without TF_GLM53_SMLA_P=1, with
/// TF_GLM53_SMLA_FP8=1 or without its kernels.
pub fn smlaP3Switch() bool {
    const value = std.c.getenv("TF_GLM53_SMLA_P3") orelse return false;
    return smlaP3Selected(std.mem.span(value));
}

pub fn smlaP3Selected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

/// A3 decode verify attend is opt-in; only the exact value "1" enables it.
pub fn decodeUnionSwitch() bool {
    const value = std.c.getenv("TF_GLM53_DECODE_UNION") orelse return false;
    return decodeUnionSelected(std.mem.span(value));
}

pub fn decodeUnionSelected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

/// The wider verify attend path is opt-in; only the exact value "1" enables it.
pub fn decodeUnion16Switch() bool {
    const value = std.c.getenv("TF_GLM53_DECODE_UNION16") orelse return false;
    return decodeUnion16Selected(std.mem.span(value));
}

pub fn decodeUnion16Selected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

pub fn decodeUnionRows(decode_union: bool, union16: bool, verify_max: usize) !usize {
    if (!union16) return mla.max_decode_rows;
    if (!decode_union) return error.DecodeUnion16WithoutDecodeUnion;
    if (verify_max != 16) return error.DecodeUnion16NeedsVerifyMax16;
    return mla.verify_max_rows;
}
/// W8A8 prefill o_proj (cuda.w8a8: e4m3 rows with a fp32 scale per row and 128-K block x the FP8B weight) is
/// opt-in and lossy (behind a KLD gate); only the exact value "1" enables it. Decode keeps qmmf.
/// Setup (native.zig, the bring-up tool) reads it once and carries it in Geometry and Forward.
pub fn prefillW8a8Switch() bool {
    const value = std.c.getenv("TF_GLM53_PREFILL_W8A8") orelse return false;
    return prefillW8a8Selected(std.mem.span(value));
}

pub fn prefillW8a8Selected(value: ?[]const u8) bool {
    return if (value) |raw| std.mem.eql(u8, raw, "1") else false;
}

/// Native prefill stage-plane fetch overlap is opt-in until both ranks have the same communicator plan.
pub fn prefetchSwitch() bool {
    const value = std.c.getenv("TF_GLM53_PREFETCH") orelse return false;
    return prefetchSelected(std.mem.span(value));
}

pub fn prefetchSelected(value: ?[]const u8) bool {
    const raw = value orelse return false;
    return std.mem.eql(u8, raw, "1");
}

/// TF_GLM53_PRE_FUSE (decode): "1" groups q_a | kv_a | indexer wk into one qmmf launch (bit-identical per element);
/// "2" also prefetches the layer's q_b codes into L2 from that launch. Anything else or unset: off.
pub const PreFuse = state.PreFuse;
pub fn preFuseSwitch() PreFuse {
    const value = std.c.getenv("TF_GLM53_PRE_FUSE") orelse return .off;
    return preFuseSelected(std.mem.span(value));
}
pub fn preFuseSelected(value: ?[]const u8) PreFuse {
    const raw = value orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .on;
    if (std.mem.eql(u8, raw, "2")) return .prefetch;
    return .off;
}
/// TF_GLM53_PREP_FUSE (prefill): "1" the grouped projection with qmmf's own plans; "2" with the forced split-K cluster
/// plan (also for indexer weights_proj). Same bits either way. Plan P2 adds the row-local prep kernel under it.
pub const PrepFuse = state.PrepFuse;
pub fn prepFuseSwitch() PrepFuse {
    const value = std.c.getenv("TF_GLM53_PREP_FUSE") orelse return .off;
    return prepFuseSelected(std.mem.span(value));
}
pub fn prepFuseSelected(value: ?[]const u8) PrepFuse {
    const raw = value orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .on;
    if (std.mem.eql(u8, raw, "2")) return .split;
    return .off;
}
pub fn groupedPreFor(mode: Mode, pre: PreFuse, prep: PrepFuse) bool {
    return switch (mode) { .decode => pre != .off, .prefill => prep != .off };
}

/// Decode MLA modes: 1 uses v1; 2 uses v1 absorb and v2 expand_v; 3 uses v3 absorb and v2 expand_v.
pub const DecodeMlaMode = enum { off, v1, v2, v3 };

pub fn decodeMlaModeSelected(value: ?[]const u8) DecodeMlaMode {
    const raw = value orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .v1;
    if (std.mem.eql(u8, raw, "2")) return .v2;
    if (std.mem.eql(u8, raw, "3")) return .v3;
    return if (std.mem.eql(u8, raw, "0")) .off else .v1;
}

pub fn decodeMlaModeSwitch() DecodeMlaMode {
    const value = std.c.getenv("TF_GLM53_DECODE_MLA") orelse return .off;
    return decodeMlaModeSelected(std.mem.span(value));

}

/// Chunks are cut at absolute multiples of `chunk_rows`, so a resumed prompt cuts where a fresh one does.
pub fn chunkAt(position: usize, chunk_rows: usize) !Chunk {
    if (chunk_rows == 0 or chunk_rows > max_prefill_chunk_rows) return error.BadChunk;
    const start = position / chunk_rows * chunk_rows;
    return .{ .start = start, .end = start + chunk_rows };
}

/// Chunk segmentation is defined from absolute positions, so cache writes and RoPE inputs are invariant to call grouping.
pub fn chunkPlan(positions: []const usize, chunk_rows: usize, out: []Chunk) !usize {
    if (out.len < positions.len) return error.BadChunk;
    var n: usize = 0;
    var previous: ?usize = null;
    for (positions) |position| {
        const c = try chunkAt(position, chunk_rows);
        if (previous == null or c.start != previous.?) {
            out[n] = c;
            n += 1;
            previous = c.start;
        }
    }
    return n;
}

/// The row range of `positions` (consecutive, ascending) that falls in each absolute chunk.
pub fn chunkRows(positions: []const usize, chunk_rows: usize, at: usize) !Chunk {
    const c = try chunkAt(positions[at], chunk_rows);
    var end = at;
    while (end < positions.len and positions[end] < c.end) : (end += 1) {
        if (end > at and positions[end] != positions[end - 1] + 1) return error.NonConsecutivePositions;
    }
    return .{ .start = at, .end = end };
}

pub fn adaptSelected(limit_position: usize, positions: []const usize) bool {
    return positions.len != 0 and positions[positions.len - 1] < limit_position;
}

test "adaptive expert mode selects whole windows by their last position" {
    try std.testing.expect(!adaptSelected(10, &.{}));
    try std.testing.expect(adaptSelected(10, &.{ 2, 9 }));
    try std.testing.expect(!adaptSelected(10, &.{ 2, 10 }));
    try std.testing.expect(!adaptSelected(10, &.{ 9, 10, 11 }));
}

pub const LocalToken = struct { global: i32, local_slot: i32 };

/// Host check of WP7's DCP coordinates: a global position p maps to rank p % 2's local slot p / 2 (filter_remap's rule).
pub fn rankLocalToMla(rank: usize, positions: []const i32, slots: []const i32, out: []LocalToken) !void {
    if (rank > 1 or positions.len != slots.len or out.len < positions.len) return error.TokenMapShape;
    for (positions, slots, 0..) |position, slot, i| {
        if (position < 0) {
            out[i] = .{ .global = -1, .local_slot = -1 };
            continue;
        }
        if (@as(usize, @intCast(position)) % 2 != rank or slot != @as(i32, @intCast(@as(usize, @intCast(position)) / 2))) return error.TokenOwnershipMismatch;
        out[i] = .{ .global = position, .local_slot = slot };
    }
}

/// rope.py `_table` row for an absolute position, in fp32: inv = 2^(-(2i/64) log2(theta)), phase = P * inv.
/// Built from positions[r] per row (review L-WP6-r4-3), so verify and tree rows need not be consecutive.
/// Host libm cos/sin/exp2 stand in for Triton's; the table bits are a Spark parity item.
pub fn ropeRow(position: usize, cos: []f32, sin: []f32) void {
    // rope.py _table in fp32, op for op: P = float(pos); inv = exp2(-(2.0 * i / (2 * HALF)) * log2(THETA));
    // phase = P * inv; cos(phase), sin(phase). Triton lowers exp2/log2 (and maybe cos/sin) to PTX approximations;
    // these host ops follow the same order in fp32. PTX-confirm on Spark (dump _table's PTX, compare tables).
    const p: f32 = @floatFromInt(position);
    const theta: f32 = @floatCast(rope_theta);
    const log2_theta: f32 = @log2(theta);
    for (0..rope_half) |i| {
        const fi: f32 = @floatFromInt(i);
        const e: f32 = -((2.0 * fi) / @as(f32, @floatFromInt(2 * rope_half))) * log2_theta;
        const inv: f32 = @exp2(e);
        const phase: f32 = p * inv;
        cos[i] = @cos(phase);
        sin[i] = @sin(phase);
    }
}

/// Embedding owner and rank-local row for a token (WP8 embed_pick's owners; the vocab is split in equal halves).
/// A token another rank owns reads local row 0; embed_pick discards it.
pub const EmbedRow = struct { owner: i32, local: i32 };
pub fn embedRowFor(token: u32, rank: usize, vocab_local: usize) EmbedRow {
    if (vocab_local == 51712 or vocab_local == 51584) {
        const o: usize = if (token < 51712) 0 else if (token < 103296) 1 else 2;
        const offset: u32 = if (o == 0) 0 else if (o == 1) 51712 else 103296;
        return .{ .owner = @intCast(o), .local = if (o == rank) @intCast(token - offset) else 0 };
    }
    const o = token / vocab_local;
    return .{ .owner = @intCast(o), .local = if (o == rank) @intCast(token - o * vocab_local) else 0 };
}

/// Per-row device inputs of one launch window, built on the host from absolute positions.
pub const RowInputs = struct {
    positions: []i64,
    dense_len: []i32,
    cos: []f32,
    sin: []f32,
    ids: []i32,
    owners: []i32,

    /// Decode and verify are at most eight rows. Keeping their small upload staging on the stack removes the
    /// per-window allocator traffic; prefill still uses allocator-backed slices sized to its chunk.
    pub const DecodeStorage = struct {
        positions: [max_decode_rows]i64 = undefined,
        dense_len: [max_decode_rows]i32 = undefined,
        cos: [max_decode_rows * rope_half]f32 = undefined,
        sin: [max_decode_rows * rope_half]f32 = undefined,
        ids: [max_decode_rows]i32 = undefined,
        owners: [max_decode_rows]i32 = undefined,

        pub fn inputs(s: *DecodeStorage, rows: usize) !RowInputs {
            if (rows > max_decode_rows) return error.RowInputsTooSmall;
            return .{
                .positions = s.positions[0..rows],
                .dense_len = s.dense_len[0..rows],
                .cos = s.cos[0 .. rows * rope_half],
                .sin = s.sin[0 .. rows * rope_half],
                .ids = s.ids[0..rows],
                .owners = s.owners[0..rows],
            };
        }
    };

    pub fn fill(r: *RowInputs, positions: []const usize, tokens: []const u32, rank: usize, vocab_local: usize) !void {
        const n = positions.len;
        if (tokens.len != 0 and tokens.len != n) return error.BadForwardShape;
        if (r.positions.len < n or r.dense_len.len < n or r.cos.len < n * rope_half or r.sin.len < n * rope_half or r.ids.len < n or r.owners.len < n) return error.RowInputsTooSmall;
        for (positions, 0..) |p, i| {
            r.positions[i] = @intCast(p);
            // TokenList.dense_len: position + 1 below the 2048 dense threshold, 0 for sparse rows.
            r.dense_len[i] = if (p < index.dense_limit) @intCast(p + 1) else 0;
            ropeRow(p, r.cos[i * rope_half ..][0..rope_half], r.sin[i * rope_half ..][0..rope_half]);
            const e = if (tokens.len == 0) EmbedRow{ .owner = -1, .local = 0 } else embedRowFor(tokens[i], rank, vocab_local);
            r.ids[i] = e.local;
            r.owners[i] = e.owner;
        }
    }

    pub fn init(gpa: std.mem.Allocator, positions: []const usize, tokens: []const u32, rank: usize, vocab_local: usize) !RowInputs {
        const n = positions.len;
        const positions_out = try gpa.alloc(i64, n);
        errdefer gpa.free(positions_out);
        const dense_len = try gpa.alloc(i32, n);
        errdefer gpa.free(dense_len);
        const cos = try gpa.alloc(f32, n * rope_half);
        errdefer gpa.free(cos);
        const sin = try gpa.alloc(f32, n * rope_half);
        errdefer gpa.free(sin);
        const ids = try gpa.alloc(i32, n);
        errdefer gpa.free(ids);
        const owners = try gpa.alloc(i32, n);
        errdefer gpa.free(owners);
        var r: RowInputs = .{ .positions = positions_out, .dense_len = dense_len, .cos = cos, .sin = sin, .ids = ids, .owners = owners };
        try r.fill(positions, tokens, rank, vocab_local);
        return r;
    }

    pub fn deinit(r: *RowInputs, gpa: std.mem.Allocator) void {
        gpa.free(r.positions);
        gpa.free(r.dense_len);
        gpa.free(r.cos);
        gpa.free(r.sin);
        gpa.free(r.ids);
        gpa.free(r.owners);
        r.* = undefined;
    }
};

/// Called after each head block's exchange, before the next block reuses the buffers: rows [first, first+rows)
/// of the window, `gathered` holding every rank's head output ([world][per_rank] bytes).
pub const HeadSink = struct {
    ctx: *anyopaque,
    sink_fn: *const fn (ctx: *anyopaque, stream: cuda.Stream, first: usize, rows: usize, gathered: u64, per_rank: usize) anyerror!void,
};

/// The prefill stage fetch on a second stream (prefill doc §3.4 step 3), with the round-7 ordering rules:
/// `fetched[b]` is recorded after fetch into buffer b, `attended[b]` after the last attend that read it, `ready`
/// orders the side stream after the compute stream's earlier work. `coll` is the side stream's own communicator
/// (SideGroup.collective; review M3) and must not be the forward's main collective.
pub const Prefetch = struct { side: cuda.Stream, coll: Collective, ready: cuda.Event, fetched: [2]cuda.Event, attended: [2]cuda.Event };

/// Host-visible launch contract used to keep both ranks' stage-fetch order explicit.
pub const PrefetchOp = enum { compute_fetch, side_fetch, fetched_wait, attend, attended_record };

pub fn prefetchSchedule(enabled: bool, layers: usize, out: []PrefetchOp) !usize {
    var at: usize = 0;
    const emit = struct {
        fn op(buf: []PrefetchOp, pos: *usize, value: PrefetchOp) !void {
            if (pos.* == buf.len) return error.ScheduleBufferTooSmall;
            buf[pos.*] = value;
            pos.* += 1;
        }
    }.op;
    if (enabled and layers != 0) try emit(out, &at, .side_fetch);
    for (0..layers) |layer| {
        if (enabled) {
            try emit(out, &at, .fetched_wait);
            try emit(out, &at, .attend);
            try emit(out, &at, .attended_record);
            if (layer + 1 < layers) try emit(out, &at, .side_fetch);
        } else {
            try emit(out, &at, .compute_fetch);
            try emit(out, &at, .attend);
        }
    }
    return at;
}

test "prefill host launch plan keeps the old order off and rank order equal with prefetch" {
    const off_expected = [_]PrefetchOp{ .compute_fetch, .attend, .compute_fetch, .attend };
    var off: [off_expected.len]PrefetchOp = undefined;
    const off_len = try prefetchSchedule(false, 2, &off);
    try std.testing.expectEqual(off_expected.len, off_len);
    try std.testing.expectEqualSlices(PrefetchOp, &off_expected, off[0..off_len]);

    const enabled_expected = [_]PrefetchOp{ .side_fetch, .fetched_wait, .attend, .attended_record, .side_fetch, .fetched_wait, .attend, .attended_record };
    var rank_zero: [enabled_expected.len]PrefetchOp = undefined;
    var rank_one: [enabled_expected.len]PrefetchOp = undefined;
    const zero_len = try prefetchSchedule(true, 2, &rank_zero);
    const one_len = try prefetchSchedule(true, 2, &rank_one);
    try std.testing.expectEqual(enabled_expected.len, zero_len);
    try std.testing.expectEqualSlices(PrefetchOp, &enabled_expected, rank_zero[0..zero_len]);
    try std.testing.expectEqualSlices(PrefetchOp, rank_zero[0..zero_len], rank_one[0..one_len]);
}

test "prefetch and TBO environment selections are opt-in" {
    try std.testing.expect(!prefetchSelected(null));
    try std.testing.expect(!prefetchSelected("0"));
    try std.testing.expect(prefetchSelected("1"));
    try std.testing.expect(!tboSelected(null));
    try std.testing.expect(!tboSelected("0"));
    try std.testing.expect(tboSelected("1"));
}

test "attention pipe switch accepts only the exact opt-in value" {
    try std.testing.expect(!attnPipeSelected(null));
    try std.testing.expect(!attnPipeSelected("0"));
    try std.testing.expect(attnPipeSelected("1"));
    try std.testing.expect(!attnPipeSelected("2"));
}

/// TBO design for each 512-row prefill piece (not active yet): retain the compute stream C and create communication
/// stream M on both ranks. For halves A=[start,start+256) and B=[start+256,end), enqueue the unchanged row kernels
/// for A on C, record readyA on C, make M wait readyA, then enqueue A's all-gather on M. While M runs that
/// collective, enqueue B's unchanged row kernels on C and record readyB. Make M wait readyB and enqueue B's
/// all-gather after A's (same collective order on both ranks). Record reducedA/reducedB after each gather on M;
/// C waits reducedA before consuming gathered A and reducedB before consuming gathered B. Each event is reused only
/// after its prior wait has been enqueued. The refactor must split every piece consumer at the row boundary and
/// keep one ordered collective stream per rank; without that, asynchronous gathers can race the shared workspace.

/// WP8's routed-expert callback state: this layer's EXL3 weights and the forward's scratch.
const Routed = struct { f: *const Forward, layer: experts.Loaded, mode: Mode = .decode };

pub const Forward = struct {
    gpa: std.mem.Allocator,
    k: *const Kernels,
    w: *const weights.ModelWeights,
    st: *const state.State,
    stream: cuda.Stream,
    coll: Collective,
    upload: Upload = .{},
    rank: usize,
    head: HeadMode = .greedy,
    mask: HeadMask = .{},
    /// Gate capture: every prefill row through the head (sampler.headWindow), in head_rows blocks.
    gate_mode: bool = false,
    chunk_rows: usize = prefill_chunk_rows,
    /// Internal half selector while launching a TBO piece.
    tbo_half: ?usize = null,
    /// Row offset within the piece's shared normalized-input workspace.
    tbo_xn_off: usize = 0,
    sink: ?HeadSink = null,
    /// null: the stage fetch runs in order on the compute stream before each layer's attention.
    prefetch: ?*const Prefetch = null,
    /// Null selects the production qmmf MMA path. The BF16 path is a fixed N/K cuBLASLt plan set at startup.
    prefill_gemms: ?*prefill_gemm.Gemms = null,
    /// MK8 deferred launch (megakernel/defer.zig): with TF_GLM53_MK=1 a decode step hangs this sink on its stream,
    /// so dispatchable launches run as persistent megakernel segments. null, or the switch off: per-kernel launches.
    mk: ?*const cuda.Stream.Deferred = null,
    ablate: ?*const ablate_mod.Ablation = null,
    taps: ?taps_mod.Taps = null,
    /// null reads TF_GLM53_SIDE like production; tests can select the path without mutating process environment.
    decode_side: ?bool = null,

    /// null reads TF_GLM53_PROMPT_EXPERTS; prompt experts run only in prefill mode.
    prompt_experts: ?bool = null,
    /// With prompt experts on: the 128-member wide kernels (TF_GLM53_PROMPT_EXPERTS=2). null reads the switch.
    prompt_experts_wide: ?bool = null,
    prompt_experts_pmx: ?bool = null,
    prompt_experts_p5a: ?bool = null,
    /// null reads TF_GLM53_PROMPT_MLA; tests can select the path without mutating process environment.
    prompt_mla: ?PromptMlaMode = null,
    /// null reads TF_GLM53_SMLA_P; setup (native.zig, bring-up) reads it once and passes it here. Prefill attend only.
    smla_p: ?bool = null,
    /// null reads TF_GLM53_SHARD_QKV_A; setup reads it once and passes it here.
    shard_qkv_a: ?bool = null,
    /// null reads TF_GLM53_SMLA_FP8; setup (native.zig, bring-up) reads it once and passes it here. Requires SMLA-P
    /// (validateSetup refuses it otherwise). Prefill attend only.
    smla_fp8: ?bool = null,
    /// TF_GLM53_SMLA_P3=1, captured once at setup (Geometry.smla_p3, which carves ws.smla_union): the SMLA-P prefill
    /// attend runs as tf_glm53_smla_union + tf_glm53_smla_p3. Off (the default) launches exactly what it did before.
    smla_p3: bool = false,
    /// TF_GLM53_DECODE_UNION=1 (A3), captured once at setup: decode windows of 1..decode_union_rows attend with
    /// tf_glm53_attend_verify + tf_glm53_attend_split_merge (byte-identical to attend_mma + split_merge per row).
    /// TF_GLM53_DECODE_UNION16=1 widens this to 16 rows with TF_GLM53_VERIFY_MAX=16.
    /// Off (the default) launches exactly what it did before.
    decode_union: bool = false,
    decode_union_rows: usize = mla.max_decode_rows,
    /// TF_GLM53_PREFILL_W8A8=1, captured once at setup (Geometry.prefill_w8a8): prefill o_proj through k.w8a8.
    /// Off (the default) launches exactly what it did before the switch existed.
    prefill_w8a8: bool = false,
    /// TF_GLM53_PRE_FUSE, captured once at setup (Geometry.pre_fuse). Off (the default) launches exactly what it did before.
    pre_fuse: PreFuse = .off,
    /// TF_GLM53_PREP_FUSE, captured once at setup (Geometry.prep_fuse). Off (the default) launches exactly what it did before.
    prep_fuse: PrepFuse = .off,
    /// null reads TF_GLM53_PREFETCH like production; a Forward override keeps host launch-order tests independent
    /// of process environment.
    prefetch_enabled: ?bool = null,
    /// null reads TF_GLM53_TBO like production; a Forward override is available to host-side planners/tests.
    tbo_enabled: ?bool = null,
    piece_pipe: ?bool = null,
    /// Captured once at setup. This moves only grouped-prefill MoE reductions to the ordered comm stream.
    moe_pipe: bool = false,
    ws_trim: bool = false,
    moe_post: bool = false,
    comm_prio: bool = false,
    /// Captured once at setup; 2 selects exact radix-threshold/linear index merge.
    index_merge: u8 = 0,
    /// Captured once at setup; overlaps index candidate gathers on the ordered communication stream.
    index_pipe: bool = false,
    /// TF_GLM53_ATTN_PIPE=1: the comm stream's own communicator (a second SideGroup), captured at setup; null: comm-stream gathers share `coll` (today).
    pipe_coll: ?Collective = null,
    /// Decode low-rank selector captured at setup; host tests can override it directly.
    lowrank_decode: bool = true,
    /// Captured once during setup and carried in Geometry for the index MMA CTA row tile.
    index_mma_rows: usize = 4,
    /// Captured once during setup and carried in Geometry; decode-only collective and query fusion.
    coll_fuse: bool = false,
    /// Set from TF_GLM53_ROW_PRUNE during native setup; tests can select the path directly.
    row_prune: ?bool = null,
/// null uses the setup value in Geometry; tests can select the MLA decode kernels directly.
    decode_mla: ?bool = null,
    decode_mla_v2: ?bool = null,
    decode_mla_v3: ?bool = null,

    prof: ?*Profile = null,
    /// Bench-only prefill stage profiling; uses the decode profiler's event pairs and collect path.
    prof_prefill: bool = false,
    prof_decode_sub: bool = false,

    pub fn dcpRank(f: *const Forward) usize {
        return if (f.st.world == 3) 0 else f.rank;
    }

    pub fn dcpWorld(f: *const Forward) usize {
        return if (f.st.world == 3) 1 else 2;
    }

    fn decodeSideEnabled(f: *const Forward) bool {
        return f.decode_side orelse decodeSideSwitch();
    }


    fn promptExpertsEnabled(f: *const Forward) bool {
        return f.prompt_experts orelse promptExpertsSwitch();
    }

    fn promptExpertsWide(f: *const Forward) bool {
        const mode = promptExpertsMode();
        return f.prompt_experts_wide orelse (mode == .wide or mode == .pmx or mode == .p5a);
    }

    fn promptExpertsPmx(f: *const Forward) bool {
        return f.prompt_experts_pmx orelse (promptExpertsMode() == .pmx or promptExpertsMode() == .p5a);
    }
    fn promptExpertsP5a(f: *const Forward) bool {
        return f.prompt_experts_p5a orelse (promptExpertsMode() == .p5a);
    }

    pub fn validateSetup(f: *const Forward) !void {
        if (f.promptExpertsP5a() and (!f.promptExpertsPmx() or !f.promptExpertsWide() or !f.promptExpertsEnabled())) return error.PromptExpertsP5aNeedsPmx;
        if (f.smlaPEnabled() and f.k.mla.attend_kernel == .scalar) return error.SmlaPWithScalarAttend;
        if (f.smlaFp8Enabled() and !f.smlaPEnabled()) return error.SmlaFp8WithoutSmlaP;
        if (f.smla_p3) {
            if (!f.smlaPEnabled()) return error.SmlaP3WithoutSmlaP;
            if (f.smlaFp8Enabled()) return error.SmlaP3WithSmlaFp8;
            if (f.k.mla.smla_union_fn == null or f.k.mla.smla_p3_fn == null) return error.MissingSmlaP3Kernels;
            if (f.st.ws.sizes.smla_union < mla.smlaP3TableBytes(f.st.ws.piece_rows)) return error.SmlaP3Table;
        }
        if (f.decode_union) {
            if (f.k.mla.attend_kernel != .mma) return error.DecodeUnionWithScalarAttend;
            if (f.k.mla.attend_verify_fn == null) return error.MissingDecodeUnionKernel;
            if (f.st.ws.sizes.attend_split == 0) return error.DecodeUnionWorkspace;
            if (f.decode_union_rows > mla.max_decode_rows and (f.decode_union_rows > f.st.ws.decode_rows or f.st.ws.sizes.attend_split < mla.splitWorkspaceBytes(f.decode_union_rows, 2 * state.local_heads))) return error.DecodeUnionWorkspace;
        }
        if (!f.decode_union and f.decode_union_rows != mla.max_decode_rows) return error.DecodeUnion16WithoutDecodeUnion;
        if (f.prefill_w8a8) {
            if (f.k.w8a8 == null) return error.MissingW8a8Kernels;
            for (f.w.layers) |l| if (!cuda.w8a8.supports(l.attn.o_proj)) return error.W8a8Weight;
            const max_heads: usize = if (f.st.world == 3) 22 else state.local_heads;
            const vo_width = max_heads * mla.value_dim;
            if (f.st.ws.sizes.part < cuda.w8a8.scratchBytes(f.st.ws.piece_rows, vo_width)) return error.W8a8Scratch;
        }
        if (f.moe_pipe and f.tboEnabled()) return error.MoePipeWithTbo;
        if (f.index_pipe and f.tboEnabled()) return error.IndexPipeWithTbo;
        if (f.pipe_coll) |pipe_coll| {
            if (f.st.tbo_stream == null) return error.AttnPipeWithoutCommStream;
            if (pipe_coll.ctx == f.coll.ctx) return error.SharedCommunicator;
            if (f.prefetch) |prefetch| if (pipe_coll.ctx == prefetch.coll.ctx) return error.SharedCommunicator;
            if (f.tboEnabled()) return error.AttnPipeWithTbo;
        }
        if (f.moe_pipe and f.rowPruneEnabled()) return error.MoePipeWithRowPrune;
        if (f.ws_trim) {
            if (!f.promptExpertsEnabled()) return error.WsTrimNeedsPromptExperts;
            for (f.w.layers) |layer| switch (layer.ffn) {
                .dense => {},
                .moe => |m| {
                    const routed = m.routed.layer();
                    if (!cuda.exl3_experts.prompt.eligibleTiles(routed)) return error.WsTrimNeedsPromptExperts;
                    if (f.promptExpertsPmx()) {
                        if (!cuda.exl3_experts.pmx.eligible(routed)) return error.WsTrimNeedsPromptExperts;
                    } else if (!cuda.exl3_experts.prompt.eligible(routed)) return error.WsTrimNeedsPromptExperts;
                },
            };
        }
        if (!f.ws_trim and f.st.ws.generic_rows < f.st.ws.expert_rows) {
            if (!f.promptExpertsEnabled()) return error.MemslimNeedsPromptExperts;
            for (f.w.layers) |layer| switch (layer.ffn) {
                .dense => {},
                .moe => |m| {
                    const routed = m.routed.layer();
                    if (!cuda.exl3_experts.prompt.eligibleTiles(routed)) return error.MemslimNeedsPromptExperts;
                },
            };
        }
        if (f.moe_post) {
            if (!f.moe_pipe or !f.promptExpertsEnabled() or f.tboEnabled()) return error.MoePostNeedsPromptMoePipe;
            for (f.w.layers) |layer| switch (layer.ffn) {
                .dense => {},
                .moe => |m| {
                    const routed = m.routed.layer();
                    if (!cuda.exl3_experts.prompt.eligibleTiles(routed)) return error.MoePostNeedsPromptExperts;
                    if (f.promptExpertsPmx()) {
                        if (!cuda.exl3_experts.pmx.eligible(routed)) return error.MoePostNeedsPromptExperts;
                    } else if (!cuda.exl3_experts.prompt.eligible(routed)) return error.MoePostNeedsPromptExperts;
                },
            };
        }
        if (f.st.adapt) |ad| {
            if (!f.promptExpertsEnabled() or f.tboEnabled()) return error.ExpertRowsAdaptNeedsPromptExperts;
            for (f.w.layers) |dl| {
                const m = switch (dl.ffn) {
                    .moe => |moe_w| moe_w,
                    .dense => continue,
                };
                const l = m.routed.layer();
                if (!cuda.exl3_experts.prompt.eligibleTiles(l)) return error.ExpertRowsAdaptNeedsPromptExperts;
                if (f.promptExpertsPmx()) {
                    if (!cuda.exl3_experts.pmx.eligible(l)) return error.ExpertRowsAdaptNeedsPromptExperts;
                } else if (f.promptExpertsWide() and !cuda.exl3_experts.prompt.eligible(l)) return error.ExpertRowsAdaptNeedsPromptExperts;
            }
            try state.validateMoePipeGeometry(.{ .expert_rows = ad.rows, .chunk_rows = f.st.ws.rows, .piece_rows = f.st.ws.piece_rows, .decode_rows = f.st.ws.decode_rows, .moe_pipe = f.moe_pipe });
        } else {
            try state.validateMoePipeGeometry(.{ .expert_rows = f.st.ws.expert_rows, .chunk_rows = f.st.ws.rows, .piece_rows = f.st.ws.piece_rows, .decode_rows = f.st.ws.decode_rows, .moe_pipe = f.moe_pipe });
        }
        try state.validatePromptExpertsCarve(f.promptExpertsEnabled(), &f.st.ws);
        if (f.pre_fuse != .off or f.prep_fuse != .off) try f.validatePreGroup();
    }

    fn promptMlaMode(f: *const Forward) PromptMlaMode {
        return f.prompt_mla orelse promptMlaSwitch();
    }

    fn smlaPEnabled(f: *const Forward) bool {
        return f.smla_p orelse smlaPSwitch();
    }

    fn shardQkvAEnabled(f: *const Forward) bool {
        return f.shard_qkv_a orelse shardQkvASwitch();
    }

    fn groupedPre(f: *const Forward, mode: Mode) bool { return groupedPreFor(mode, f.pre_fuse, f.prep_fuse); }
    fn preGroupOuts(f: *const Forward, a: weights.Attention, buf: *[3]cuda.qmmf.GroupOut) []const cuda.qmmf.GroupOut {
        const ws = &f.st.ws;
        buf[0] = .{ .w = a.q_a, .y = ws.q_a };
        buf[1] = .{ .w = a.kv_a, .y = ws.kv };
        if (a.indexer) |ix| { buf[2] = .{ .w = ix.wk, .y = ws.ik }; return buf[0..3]; }
        return buf[0..2];
    }
    fn projectPreGroup(f: *const Forward, mode: Mode, s: cuda.Stream, xn: u64, rows: usize, a: weights.Attention) !void {
        var buf: [3]cuda.qmmf.GroupOut = undefined;
        const prefetch = mode == .decode and f.pre_fuse == .prefetch;
        try f.k.lane.matmulGroup(s, xn, hidden, rows, f.preGroupOuts(a, &buf), .{ .split = mode == .prefill and f.prep_fuse == .split, .prefetch = if (prefetch) a.q_b.codes else 0, .prefetch_bytes = if (prefetch) @as(usize, a.q_b.npad) * a.q_b.k / @as(usize, if (a.q_b.mode == .fp4) 2 else 1) else 0 });
    }
    fn validatePreGroup(f: *const Forward) !void {
        if (f.shardQkvAEnabled()) return error.PreFuseWithShardQkvA;
        if (f.pre_fuse != .off and megakernelSwitch()) return error.PreFuseWithMegakernel;
        if (f.prep_fuse != .off and f.prefill_gemms != null) return error.PrepFuseWithPrefillBf16;
        if (!f.k.lane.clusters) return error.PreFuseNeedsClusters;
        for (f.k.lane.group_fns) |g| if (g == null) return error.MissingGroupKernels;
        const decode_rows = [_]usize{ 1, f.st.ws.decode_rows };
        const prefill_rows = [_]usize{ 1, 17, 33, 256, f.st.ws.piece_rows };
        for (f.w.layers) |l| {
            var buf: [3]cuda.qmmf.GroupOut = undefined;
            const outs = f.preGroupOuts(l.attn, &buf);
            if (f.pre_fuse != .off) for (decode_rows) |m| { _ = try cuda.qmmf.groupPlan(m, hidden, outs, true, false); };
            if (f.prep_fuse != .off) for (prefill_rows) |m| {
                _ = try cuda.qmmf.groupPlan(m, hidden, outs, true, f.prep_fuse == .split);
                if (f.prep_fuse == .split) if (l.attn.indexer) |ix| { const w = ix.weights_proj; if (!cuda.qmmf.splitPlan(m, w.n, w.k, true, w.mode).cluster) return error.PrepFuseSplitPlan; };
            };
        }
    }

    fn smlaFp8Enabled(f: *const Forward) bool {
        return f.smla_fp8 orelse smlaFp8Switch();
    }

    fn prefetchEnabled(f: *const Forward) bool {
        if (f.dcpWorld() <= 1) return false;
        return f.prefetch_enabled orelse (f.prefetch != null or prefetchSwitch());
    }

    fn tboEnabled(f: *const Forward) bool {
        return f.tbo_enabled orelse tboSwitch();
    }

    fn piecePipeEnabled(f: *const Forward) bool {
        return f.piece_pipe orelse piecePipeSwitch();
    }

    fn moePipeEnabled(f: *const Forward) bool {
        return f.moe_pipe;
    }

    fn indexPipeEnabled(f: *const Forward) bool {
        return f.index_pipe;
    }

    /// Comm-stream gathers can use a dedicated communicator. If FETCH_COMM merges later, its comm-stream fetch must use this too.
    fn commColl(f: *const Forward) Collective {
        return f.pipe_coll orelse f.coll;
    }

    fn activePrefetch(f: *const Forward) ?*const Prefetch {
        if (!f.prefetchEnabled()) return null;
        return f.prefetch;
    }

    fn gatherBuffer(f: *const Forward) u64 {
        return if (f.tbo_half) |half| f.st.ws.tbo_gathered[half] else f.st.ws.gathered;
    }

    fn rowPruneEnabled(f: *const Forward) bool {
        return f.row_prune orelse false;
    }

    fn decodeMlaEnabled(f: *const Forward) bool {
        return f.decode_mla orelse f.st.decode_mla;
    }

    fn decodeMlaV2Enabled(f: *const Forward) bool {
        return f.decode_mla_v2 orelse f.st.decode_mla_v2;
    }

    fn decodeMlaV3Enabled(f: *const Forward) bool {
        return f.decode_mla_v3 orelse f.st.decode_mla_v3;
    }

    fn profileBegin(f: *const Forward, mode: Mode, layer: usize, stage: Stage) !void {
        try f.profileBeginStream(mode, layer, stage, f.stream);
    }

    fn subProfiled(f: *const Forward, mode: Mode) bool {
        return (mode == .prefill and f.prof_prefill) or (mode == .decode and f.prof_decode_sub);
    }

    fn profileSubBegin(f: *const Forward, mode: Mode, layer: usize, stage: Stage) !void {
        if (f.subProfiled(mode)) try f.profileBegin(mode, layer, stage);
    }

    fn profileSubEnd(f: *const Forward, mode: Mode, layer: usize, stage: Stage) !void {
        if (f.subProfiled(mode)) try f.profileEnd(mode, layer, stage);
    }

    fn profileSubBeginStream(f: *const Forward, mode: Mode, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        if (f.subProfiled(mode)) try f.profileBeginStream(mode, layer, stage, stream);
    }

    fn profileSubEndStream(f: *const Forward, mode: Mode, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        if (f.subProfiled(mode)) try f.profileEndStream(mode, layer, stage, stream);
    }

    /// Before a prefill profile collect: with TF_GLM53_MOE_PIPE=1 the comm stream holds .moe_comm events too.
    fn profileSync(f: *const Forward) !void {
        try f.stream.synchronize();
        if (f.moePipeEnabled() or f.indexPipeEnabled()) if (f.st.tbo_stream) |comm| try comm.synchronize();
    }

    fn profileEnd(f: *const Forward, mode: Mode, layer: usize, stage: Stage) !void {
        if (mode == .decode or (mode == .prefill and f.prof_prefill)) if (f.prof) |p| try p.end(layer, stage, f.stream);
    }

    const MoeProfile = struct { f: *const Forward, layer: usize, mode: Mode = .decode };
    fn moeProfileBegin(ctx: *anyopaque, stage: u8) anyerror!void {
        const m: *MoeProfile = @ptrCast(@alignCast(ctx));
        const s: Stage = switch (stage) { 0 => .route, 1 => .routed, 2 => .shared, 3 => .lowrank, else => unreachable };
        try m.f.profileBegin(m.mode, m.layer, s);
    }
    fn moeProfileEnd(ctx: *anyopaque, stage: u8) anyerror!void {
        const m: *MoeProfile = @ptrCast(@alignCast(ctx));
        const s: Stage = switch (stage) { 0 => .route, 1 => .routed, 2 => .shared, 3 => .lowrank, else => unreachable };
        try m.f.profileEnd(m.mode, m.layer, s);
    }

    pub const Args = struct {
        positions: []const usize,
        tokens: []const u32,
        observer_ctx: ?*anyopaque = null,
        observer_fn: ?*const fn (ctx: *anyopaque, start: usize, rows: usize) anyerror!void = null,
    };

    /// Decode or verify window: R <= verify max (8 or 16) rows at absolute positions, DCP2 attention, every row through the head.
    pub fn whole(f: *const Forward, args: Args) !void {
        const rows = args.positions.len;
        if (rows == 0 or rows > max_decode_rows or rows > f.st.ws.decode_rows) return error.BadForwardShape;
        try f.step(.decode, args.positions, args.tokens, .{ .start = 0, .end = rows });
    }

    /// Prefill: chunks cut at absolute multiples of chunk_rows, gather-on-prefill attention, the head on the last
    /// row (or every row in gate mode).
    pub fn pieced(f: *const Forward, args: Args) !void {
        if (args.positions.len == 0 or args.tokens.len != args.positions.len) return error.BadForwardShape;
        // The whole prompt must fit before any chunk runs (a later chunk failing would leave earlier rows written).
        if (args.positions[args.positions.len - 1] >= f.st.capacity) return error.ContextFull;
        var at: usize = 0;
        while (at < args.positions.len) {
            const c = try chunkRows(args.positions, f.chunk_rows, at);
            const rows = c.end - c.start;
            const last = c.end == args.positions.len;
            const window = sampler.headWindow(rows, f.gate_mode);
            const head: Chunk = if (f.gate_mode or last) .{ .start = window.start, .end = window.start + window.count } else .{ .start = 0, .end = 0 };
            try f.step(.prefill, args.positions[c.start..c.end], args.tokens[c.start..c.end], head);
            if (args.observer_fn) |observe| try observe(args.observer_ctx.?, args.positions[c.start], rows);
            at = c.end;
        }
    }

    /// The exchange plan this forward follows for one window (what `launch` sends, by phase).
    pub fn stepPlan(f: *const Forward, mode: Mode, positions: []const usize, head_rows: usize, layers: []Layer) !ExchangePlan {
        for (f.w.layers, layers) |l, *out| out.* = layerOf(l);
        return plan(.{ .mode = mode, .positions = positions, .layers = layers, .head = f.head, .head_rows = head_rows, .head_block = f.st.ws.head_rows, .score_rows = f.st.ws.score_rows, .piece_rows = f.st.ws.piece_rows, .rank = f.rank });
    }

    fn step(f: *const Forward, mode: Mode, positions: []const usize, tokens: []const u32, head: Chunk) !void {
        try f.prepare(mode, positions, tokens);
        try f.launch(mode, positions, head);
    }

    fn project(f: *const Forward, mode: Mode, s: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, out: cuda.qmmf.Out, part: u64) !void {
        if (mode != .prefill or f.prefill_gemms == null or w.mode != .fp8b) return f.k.lane.matmulOut(s, x, ldx, rows, w, y, out, part);
        const dequant_stream = f.st.dequant_stream orelse return error.MissingDequantStream;
        const ready = f.st.dequant_ready orelse return error.MissingDequantEvent;
        const consumed = f.st.dequant_consumed orelse return error.MissingDequantConsumeEvent;
        try dequant_stream.wait(consumed);
        try f.k.lane.dequantizeFp8bBf16RowMajor(dequant_stream, w, f.st.ws.prefill_weight);
        try ready.record(dequant_stream);
        try s.wait(ready);
        const dtype: cuda.cublaslt.DataType = if (out == .f32) .f32 else .bf16;
        try f.prefill_gemms.?.run(s, x, f.st.ws.prefill_weight, y, rows, w.n, w.k, dtype);
        try consumed.record(s);
    }

    /// QMMF projection planned as if it had `pin_n` output columns, while writing its actual width.
    fn projectPinned(f: *const Forward, s: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, out: cuda.qmmf.Out, part: u64, pin_n: usize) !void {
        return f.k.lane.matmulOutPinned(s, x, ldx, rows, w, y, out, part, pin_n);
    }
    /// TF_GLM53_PREFILL_W8A8=1 prefill o_proj: the same fp32 rank partial rows as project(.f32), from the e4m3
    /// rows of `x` and the FP8B weight. Scratch: ws.part, the attend partials expand_v has already consumed on this
    /// stream (the next piece's attend rewrites it after this call, also in stream order); xq | xs fit in it.
    fn oProjW8a8(f: *const Forward, s: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64) !void {
        const fns = f.k.w8a8 orelse return error.MissingW8a8Kernels;
        const ws = &f.st.ws;
        try fns.run(s, x, ldx, rows, w, y, .f32, ws.part, ws.sizes.part);
    }

    fn mlpProject(ctx: ?*anyopaque, s: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, out: cuda.qmmf.Out, part: u64) anyerror!void {
        const f: *const Forward = @ptrCast(@alignCast(ctx.?));
        return f.project(.prefill, s, x, ldx, rows, w, y, out, part);
    }

    /// Host side of a window: shape checks and the row inputs uploaded (outside any graph capture).
    pub fn prepare(f: *const Forward, mode: Mode, positions: []const usize, tokens: []const u32) !void {
        const ws = &f.st.ws;
        const rows = positions.len;
        if (rows == 0 or tokens.len != rows or rows > ws.rows) return error.BadForwardShape;
        for (positions[1..], 0..) |p, i| if (p <= positions[i]) return error.NonAscendingPositions;
        if (positions[rows - 1] >= f.st.capacity) return error.ContextFull;
        const vocab_local = @max(f.w.vocab.len(), 1);
        var storage: RowInputs.DecodeStorage = .{};
        var in: RowInputs = if (mode == .decode) blk: {
            if (rows > max_decode_rows) return error.BadForwardShape;
            break :blk try storage.inputs(rows);
        } else try RowInputs.init(f.gpa, positions, tokens, f.rank, vocab_local);
        defer if (mode == .prefill) in.deinit(f.gpa);
        if (mode == .decode) try in.fill(positions, tokens, f.rank, vocab_local);
        try f.upload.put(f.stream, ws.positions, std.mem.sliceAsBytes(in.positions));
        try f.upload.put(f.stream, ws.dense_len, std.mem.sliceAsBytes(in.dense_len));
        try f.upload.put(f.stream, ws.cos, std.mem.sliceAsBytes(in.cos));
        try f.upload.put(f.stream, ws.sin, std.mem.sliceAsBytes(in.sin));
        try f.upload.put(f.stream, ws.ids, std.mem.sliceAsBytes(in.ids));
        try f.upload.put(f.stream, ws.owners, std.mem.sliceAsBytes(in.owners));
    }

    /// Device side of a window: every launch and exchange, no host reads (graph-capturable for decode keys: its
    /// host decisions are functions of the row count and decodeKeyBucket only). Layer-major; inside a layer the
    /// rows run in pieces (absolute multiples of piece_rows), attention then FFN per piece: a piece attends only
    /// to keys of rows at or before it, which earlier pieces already wrote.
    pub fn launch(f: *const Forward, mode: Mode, positions: []const usize, head: Chunk) !void {
        if (mode == .prefill) if (f.st.adapt) |ad| if (adaptSelected(ad.limit_position, positions)) {
            var st2 = f.st.*;
            st2.ws = ad.ws;
            var g = f.*;
            g.st = &st2;
            return g.launchSteps(mode, positions, head);
        };
        // MK8: decode only (prefill untouched); the end of the step flushes the last segment.
        if (mode == .decode) if (f.mk) |sink| if (megakernelSwitch()) {
            var g = f.*;
            g.mk = null;
            g.stream = f.stream.withDeferred(sink);
            g.launchSteps(mode, positions, head) catch |err| {
                // What was recorded before the failure runs, as it would have without the sink.
                g.stream.flushDeferred() catch {};
                return err;
            };
            return g.stream.flushDeferred();
        };
        return f.launchSteps(mode, positions, head);
    }

    fn launchSteps(f: *const Forward, mode: Mode, positions: []const usize, head: Chunk) !void {
        // This build has no MTP layer: only the config's layers run.
        if (f.w.config.mtp != 0 or f.w.layers.len > f.w.config.layers) return error.MtpLayerNotInBuild;
        for (f.w.layers) |l| if (l.index >= f.w.config.layers) return error.MtpLayerNotInBuild;
        // Review N2: planes are indexed by a layer's position among the loaded layers (plane 0..n-1), and every one
        // of them, and every full-indexer ordinal, must exist in State before the first launch.
        var full: usize = 0;
        for (f.w.layers) |l| full += @intFromBool(l.attn.indexer != null);
        try f.st.checkLayers(f.w.layers.len, full);
        // A workspace carved for cluster lanes holds no split-K partials; a lane without clusters would need them.
        if (f.st.ws.lane_clusters and !f.k.lane.clusters) return error.LanePartNeedsClusters;
        const prefill = mode == .prefill;
        if (prefill and f.activePrefetch() != null) {
            const pf = f.activePrefetch().?;
            try state.validateStage(&f.st.stage, true);
            if (pf.coll.ctx == f.coll.ctx) return error.SharedCommunicator;
            // The side stream starts after the compute stream's earlier work (previous chunks' kv writes).
            try pf.ready.record(f.stream);
            try pf.side.wait(pf.ready);
            try f.prefetchLayer(0, positions[0]);
        }
        if (prefill and f.prefetchEnabled() and f.prefetch == null) return error.MissingPrefetch;
        var at: usize = 0;
        while (at < positions.len) {
            const p = pieceAt(mode, positions, f.st.ws.piece_rows, at);
            try f.profileBegin(mode, 0, .embed);
            try f.embed(p);
            try f.profileEnd(mode, 0, .embed);
            if (prefill and f.prof_prefill) if (f.prof) |profile| {
                try f.profileSync();
                try profile.collect();
            };
            at = p.end;
        }
        var ordinal: usize = 0;
        var moe_pending: MoePipePendingSet = .{};
        for (f.w.layers, 0..) |l, i| {
            // `i` is the plane (and stage-pair) index of this layer; `next` the next layer's.
            const next: ?usize = if (i + 1 < f.w.layers.len) i + 1 else null;
            // Gather-on-prefill (§3.4): the peer's cached keys below chunk_start into stage[p / 2], once a layer. Without
            // a prefetch stream it runs here, in order; kv_write adds this chunk's peer rows (disjoint slots).
            if (prefill and f.dcpWorld() > 1 and f.activePrefetch() == null) try f.fetch(f.coll, f.stream, i, positions[0]);
            at = 0;
            // Last-layer pruning keeps cache writes for every row and limits post-attention work to rows needed by the head/taps.
            const prune_layer = prefill and f.w.config.layers != 0 and l.index == @as(usize, f.w.config.layers - 1) and f.rowPruneEnabled();
            var keep: sampler.RowWindow = if (head.end > head.start) sampler.layer77Window(positions.len, f.gate_mode, true) else .{ .start = positions.len, .count = 0 };
            if (prune_layer) if (f.taps) |t| if (taps_mod.slotOf(t.layers, l.index) != null) {
                const tap_start = @max(positions[0], t.from);
                const tap_row = @min(positions.len, tap_start - positions[0]);
                keep.start = @min(keep.start, tap_row);
                keep.count = positions.len - keep.start;
            };
            if (prune_layer) {
                // Keep every row's KV/indexer-key writes, then run post-attention work for only the
                // suffix required by the head or a drafter tap on the model's final layer.
                while (at < positions.len) {
                    const p = pieceAt(mode, positions, f.st.ws.piece_rows, at);
                    if (f.moePipeEnabled()) try f.finishMoePendingFor(&moe_pending, p);
                    try f.attention(l, i, mode, positions, p, ordinal, next, 0, true, false, null);
                    const first = @max(p.start, keep.start);
                    if (first < p.end) {
                        const kept = Chunk{ .start = first, .end = p.end };
                        if (f.moePipeEnabled()) try f.finishMoePendingFor(&moe_pending, kept);
                        try f.attention(l, i, mode, positions, kept, ordinal, next, 0, false, true, null);
                        try f.ffn(l, kept, mode);
                    }
                    at = p.end;
                }
            } else if (prefill and !f.tboEnabled() and state.groupedExperts(f.st.ws.expert_rows, f.st.ws.piece_rows)) switch (l.ffn) {
                .moe => try f.groupedMoeLayer(l, i, positions, ordinal, next, &moe_pending),
                .dense => while (at < positions.len) {
                    const p = pieceAt(mode, positions, f.st.ws.piece_rows, at);
                    if (f.moePipeEnabled()) try f.finishMoePendingFor(&moe_pending, p);
                    try f.attention(l, i, mode, positions, p, ordinal, next, 0, false, false, null);
                    try f.ffn(l, p, mode);
                    at = p.end;
                },
            } else while (at < positions.len) {
                const p = pieceAt(mode, positions, f.st.ws.piece_rows, at);
                if (f.moePipeEnabled()) try f.finishMoePendingFor(&moe_pending, p);
                if (prefill and f.tboEnabled() and f.ablate == null and p.end - p.start > 1) {
                    try f.tboPiece(l, i, positions, p, ordinal, next);
                } else {
                    try f.attention(l, i, mode, positions, p, ordinal, next, 0, false, false, null);
                    try f.ffn(l, p, mode);
                }
                if (prefill and f.prof_prefill) if (f.prof) |profile| {
                    try f.profileSync();
                    try profile.collect();
                };
                at = p.end;
            }
            // Taps observe the complete layer output, including every deferred MoE residual.
            if (f.moePipeEnabled() and f.taps != null) try f.finishAllMoePending(&moe_pending);
            if (f.taps) |t| try taps_mod.at(t, f.stream, l.index, f.st.ws.x, positions[0], positions.len);
            if (l.attn.indexer != null) ordinal += 1;
        }
        if (f.moePipeEnabled()) try f.finishAllMoePending(&moe_pending);
        if (head.end > head.start and f.head != .none) {
            try f.profileBegin(mode, f.w.layers.len, .head);
            try f.headRows(head.start, head.end - head.start);
            try f.profileEnd(mode, f.w.layers.len, .head);
            if (prefill and f.prof_prefill) if (f.prof) |profile| {
                try f.profileSync();
                try profile.collect();
            };
        }
    }

    fn tboPiece(f: *const Forward, l: weights.DeviceLayer, plane: usize, positions: []const usize, p: Chunk, ordinal: usize, next: ?usize) !void {
        const mid = p.start + (p.end - p.start) / 2;
        const halves = [2]Chunk{ .{ .start = p.start, .end = mid }, .{ .start = mid, .end = p.end } };
        var work_a = f.*;
        work_a.tbo_half = 0;
        work_a.tbo_xn_off = 0;
        try work_a.attention(l, plane, .prefill, positions, halves[0], ordinal, next, 0, false, false, null);
        var work_b = f.*;
        work_b.tbo_half = 1;
        work_b.tbo_xn_off = halves[0].end - halves[0].start;
        try work_b.attention(l, plane, .prefill, positions, halves[1], ordinal, next, work_b.tbo_xn_off, false, false, null);
        try f.finishTboReduce(halves[0], false, l.post_norm, work_a.tbo_xn_off, 0);
        try f.finishTboReduce(halves[1], false, l.post_norm, work_b.tbo_xn_off, 1);

        work_a = f.*;
        work_a.tbo_half = 0;
        work_a.tbo_xn_off = 0;
        try work_a.ffn(l, halves[0], .prefill);
        work_b = f.*;
        work_b.tbo_half = 1;
        work_b.tbo_xn_off = halves[0].end - halves[0].start;
        try work_b.ffn(l, halves[1], .prefill);
        const moe_partial = switch (l.ffn) { .moe => true, .dense => false };
        try f.finishTboReduce(halves[0], moe_partial, null, 0, 0);
        try f.finishTboReduce(halves[1], moe_partial, null, 0, 1);
    }

    /// Stage fill for `layer`: this rank's own cache rows below chunk_start to the peer, the peer's into this
    /// rank's stage pair (layer % 2), counted per owner (rule 5), on `coll`'s communicator.
    fn fetch(f: *const Forward, coll: Collective, stream: cuda.Stream, layer: usize, chunk_start: usize) !void {
        if (f.dcpWorld() <= 1) return;
        const peer = 1 - f.rank;
        const mine = stageFetchSlots(chunk_start, f.rank);
        const theirs = stageFetchSlots(chunk_start, peer);
        try coll.sendRecv(stream, f.st.latentAt(layer), mine * mla.latent_dim * 2, state.stageLatent(&f.st.stage, layer), theirs * mla.latent_dim * 2, peer, .stage_fetch);
        try coll.sendRecv(stream, f.st.ropeAt(layer), mine * mla.rope_dim * 2, state.stageRope(&f.st.stage, layer), theirs * mla.rope_dim * 2, peer, .stage_fetch);
    }

    /// fetch(L) on the side stream and its own communicator, at most one layer ahead (rule 3). Pair L % 2 was last
    /// read by attend(L-2) (fetch(L+1) waits on attend(L-1), rule 1); the event after it gates attend(L) (rule 2).
    fn prefetchLayer(f: *const Forward, layer: usize, chunk_start: usize) !void {
        if (f.dcpWorld() <= 1) return;
        const pf = f.prefetch.?;
        const b = layer % state.stage_pairs;
        try pf.side.wait(pf.attended[b]);
        try f.fetch(pf.coll, pf.side, layer, chunk_start);
        try pf.fetched[b].record(pf.side);
    }

    /// Vocab-split embed of piece `p`: each rank writes its owned rows, the ranks all-gather bf16 rows, then the
    /// owner pick into the residual rows.
    fn embed(f: *const Forward, p: Chunk) !void {
        const ws = &f.st.ws;
        const n = p.end - p.start;
        const bytes = n * hidden * 2;
        const mine = ws.gathered + f.rank * bytes;
        try f.k.glue.embed(f.stream, ws.ids + p.start * 4, f.w.embed, mine, n, hidden);
        try f.coll.allGather(f.stream, mine, ws.gathered, bytes, .embed);
        try moe.embedPick(f.k.glue53.embed_pick, f.stream, ws.gathered, ws.owners + p.start * 4, ws.x + p.start * hidden * 2, n, hidden, @intCast(f.st.world));
    }

    /// This rank's fp32 partial slot for a piece of `n` rows in the per-piece reduce buffer [world][n][6144].
    fn slot(f: *const Forward, n: usize) u64 {
        return f.gatherBuffer() + f.rank * n * hidden * 4;
    }

    /// The rank-order TP sum of piece `p`: attention/dense partials are fp32, MoE partials are BF16;
    /// attention also fuses residual add and the post-attention RMSNorm.
    const CollectiveProfile = struct { mode: Mode, layer: usize, stage: Stage, after: ?Stage = null };

    const PipePending = struct { p: Chunk, off: usize, slot: usize };
    const MoePipePending = struct { p: Chunk, piece_index: usize, layer: usize };
    const MoePipePendingSet = struct {
        items: [state.Workspace.max_moe_pipe_pieces]?MoePipePending = @splat(null),
        count: usize = 0,
    };

    fn reduceIssue(f: *const Forward, p: Chunk, ring_slot: usize) !void {
        const ws = &f.st.ws;
        const comm = f.st.tbo_stream orelse return error.MissingTboStream;
        try f.st.pipe_ready[ring_slot].?.record(f.stream);
        try comm.wait(f.st.pipe_ready[ring_slot].?);
        try f.commColl().allGather(comm, ws.pipe_src[ring_slot], ws.pipe_gather[ring_slot], (p.end - p.start) * hidden * 4, .attention_output);
        try f.st.pipe_done[ring_slot].?.record(comm);
    }

    fn moeReduceIssue(f: *const Forward, p: Chunk, ring_slot: usize, piece_index: usize, layer: usize) !void {
        const ws = &f.st.ws;
        const comm = f.st.tbo_stream orelse return error.MissingTboStream;
        try f.st.moe_pipe_ready[ring_slot].?.record(f.stream);
        try comm.wait(f.st.moe_pipe_ready[ring_slot].?);
        const n = p.end - p.start;
        const mine = if (f.moe_post) ws.moe_pipe_gather[ring_slot] + f.rank * n * hidden * 2 else ws.moe_pipe_src[ring_slot];
        try f.commColl().allGather(comm, mine, ws.moe_pipe_gather[ring_slot], n * hidden * 2, .moe_partial);
        try moe.residualAddBf16(f.k.glue53.residual_add_bf16, comm, ws.x + p.start * hidden * 2, ws.moe_pipe_gather[ring_slot], p.end - p.start, f.st.world);
        if (f.ablate) |ab| if (ab.vec(layer, .ffn)) |v| try ab.launch(f.k.ablate.?, comm, ws.x + p.start * hidden * 2, ws.moe_pipe_gather[ring_slot], .bf16_parts, v, p.end - p.start, 0, 0, f.w.config.eps);
        try f.st.moe_pipe_done[ring_slot].?.record(comm);
        // One event per piece lets next-layer attention wait for precisely the rows it will read.
        try f.st.moe_pipe_pdone[piece_index].?.record(comm);
    }

    fn finishMoePending(f: *const Forward, pending: *MoePipePendingSet, at: usize) !void {
        const item = pending.items[at].?;
        try f.profileBegin(.prefill, item.layer, .moe_comm);
        try f.stream.wait(f.st.moe_pipe_pdone[item.piece_index].?);
        try f.profileEnd(.prefill, item.layer, .moe_comm);
        var i = at;
        while (i + 1 < pending.count) : (i += 1) pending.items[i] = pending.items[i + 1];
        pending.count -= 1;
        pending.items[pending.count] = null;
    }

    fn finishMoePendingFor(f: *const Forward, pending: *MoePipePendingSet, p: Chunk) !void {
        var i: usize = 0;
        while (i < pending.count) {
            const item = pending.items[i].?;
            if (item.p.start == p.start and item.p.end == p.end) {
                try f.finishMoePending(pending, i);
            } else i += 1;
        }
    }

    fn finishAllMoePending(f: *const Forward, pending: *MoePipePendingSet) !void {
        while (pending.count != 0) try f.finishMoePending(pending, pending.count - 1);
    }

    fn addMoePending(pending: *MoePipePendingSet, item: MoePipePending) !void {
        if (pending.count == pending.items.len) return error.MoePipeRingOverflow;
        pending.items[pending.count] = item;
        pending.count += 1;
    }

    fn profileBeginStream(f: *const Forward, mode: Mode, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        if (mode == .decode or (mode == .prefill and f.prof_prefill)) if (f.prof) |p| {
            if (f.moePipeEnabled() and p.wouldCollect(layer, stage)) try f.profileSync();
            try p.begin(layer, stage, stream);
        };
    }

    fn profileEndStream(f: *const Forward, mode: Mode, layer: usize, stage: Stage, stream: cuda.Stream) !void {
        if (mode == .decode or (mode == .prefill and f.prof_prefill)) if (f.prof) |p| try p.end(layer, stage, stream);
    }

    fn reduceFinish(f: *const Forward, p: Chunk, off: usize, ring_slot: usize, norm: u64, layer: usize) !void {
        const ws = &f.st.ws;
        try f.profileBegin(.prefill, layer, .attn_comm);
        try f.stream.wait(f.st.pipe_done[ring_slot].?);
        try f.profileEnd(.prefill, layer, .attn_comm);
        try f.profileBegin(.prefill, layer, .attn_post);
        const n = p.end - p.start;
        var args: cuda.Args = .{};
        const xn = ws.xn + off * hidden * 2;
        inline for (.{ ws.x + p.start * hidden * 2, ws.pipe_gather[ring_slot], norm, xn, @as(c_int, @intCast(n | (f.st.world << 24))), f.w.config.eps }) |v| args.add(v);
        try cuda.launch.launch(f.k.residual_rmsnorm_6144, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, f.stream, &args);
        try f.profileEnd(.prefill, layer, .attn_post);
    }

    fn ffnPre(f: *const Forward, l: weights.DeviceLayer, m: anytype, mlp_k: anytype, p: Chunk, off: usize) !void {
        const ws = &f.st.ws;
        const n = p.end - p.start;
        const scratch: mlp.Scratch = .{ .gate_up = ws.mlp_gate_up, .activation = ws.mlp_act, .mm_part = ws.lane_part };
        const lr: ?moe.LowRankWeights = if (m.lowrank) |r| .{ .a = r.a, .b = r.b, .mid = ws.lowrank_mid + off * 8 * (r.rank_local * 4), .rank_local = r.rank_local } else null;
        if (m.lowrank) |r| if (r.rank_local > state.lowrank_max) return error.LowRankTooWide;
        var lowrank_k = f.k.glue53;
        // Grouped prefill retains the established kernels regardless of the decode selector.
        lowrank_k.decode_fast = false;
        try f.profileBegin(.prefill, l.index, .route);
        try moe.route(f.k.router(), f.stream, ws.xn + off * hidden * 2, hidden, m.router, m.bias, ws.router_logits, ws.router_part, ws.picks, ws.wts, n);
        try f.profileEnd(.prefill, l.index, .route);
        try f.profileBegin(.prefill, l.index, .shared);
        try mlp.block(mlp_k, f.stream, m.shared, scratch, ws.xn + off * hidden * 2, n, m.shared.down.k, ws.shared_out + off * hidden * 4);
        try f.profileEnd(.prefill, l.index, .shared);
        if (lr) |lr_weights| {
            try f.profileBegin(.prefill, l.index, .lowrank);
            try moe.lowrankPre(lowrank_k, f.stream, ws.xn + off * hidden * 2, hidden, n, lr_weights);
            try f.profileEnd(.prefill, l.index, .lowrank);
        }
        try copyRows(f.stream, ws.picks8 + off * state.routed_slots * 4, state.routed_slots * 4, ws.picks, state.route_slots * 4, state.routed_slots * 4, n, "GLM-5.3 compact route picks");
        try copyRows(f.stream, ws.wts8 + off * state.routed_slots * 4, state.routed_slots * 4, ws.wts, state.route_slots * 4, state.routed_slots * 4, n, "GLM-5.3 compact route weights");
    }

    fn reduce(f: *const Forward, p: Chunk, phase: Phase, source: u64, bf16_partial: bool, next_norm: ?u64, xn_off: usize, prof: ?CollectiveProfile) !void {
        const ws = &f.st.ws;
        const n = p.end - p.start;
        const bytes = n * hidden * (if (bf16_partial) @as(usize, 2) else 4);
        if (f.tbo_half) |half| {
            const comm = f.st.tbo_stream orelse return error.MissingTboStream;
            try f.st.tbo_ready[half].?.record(f.stream);
            try comm.wait(f.st.tbo_ready[half].?);
            try f.commColl().allGather(comm, source, f.gatherBuffer(), bytes, phase);
            try f.st.tbo_done[half].?.record(comm);
            return;
        }
        if (prof) |timing| try f.profileBegin(timing.mode, timing.layer, timing.stage);
        try f.coll.allGather(f.stream, source, ws.gathered, bytes, phase);
        if (prof) |timing| try f.profileEnd(timing.mode, timing.layer, timing.stage);
        if (prof) |timing| if (timing.after) |stage| try f.profileBegin(timing.mode, timing.layer, stage);
        if (next_norm) |weight| {
            if (bf16_partial) return error.BadFusedNormPartial;
            var args: cuda.Args = .{};
            const xn = ws.xn + xn_off * hidden * 2;
            const rows_arg: c_int = @intCast(n | (f.st.world << 24));
            inline for (.{ ws.x + p.start * hidden * 2, ws.gathered, weight, xn, rows_arg, f.w.config.eps }) |v| args.add(v);
            try cuda.launch.launch(f.k.residual_rmsnorm_6144, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, f.stream, &args);
        } else if (bf16_partial) {
            try moe.residualAddBf16(f.k.glue53.residual_add_bf16, f.stream, ws.x + p.start * hidden * 2, ws.gathered, n, f.st.world);
        } else {
            try f.k.glue.residualAdd(f.stream, ws.x + p.start * hidden * 2, ws.x + p.start * hidden * 2, ws.gathered, n, hidden, f.st.world);
        }
    }

    fn finishTboReduce(f: *const Forward, p: Chunk, bf16_partial: bool, next_norm: ?u64, xn_off: usize, half: usize) !void {
        const ws = &f.st.ws;
        try f.stream.wait(f.st.tbo_done[half].?);
        const n = p.end - p.start;
        const gathered = ws.tbo_gathered[half];
        if (next_norm) |weight| {
            var args: cuda.Args = .{};
            const xn = ws.xn + xn_off * hidden * 2;
            const rows_arg: c_int = @intCast(n | (f.st.world << 24));
            inline for (.{ ws.x + p.start * hidden * 2, gathered, weight, xn, rows_arg, f.w.config.eps }) |v| args.add(v);
            try cuda.launch.launch(f.k.residual_rmsnorm_6144, .{ .grid = .{ .x = @intCast(n) }, .block = .{ .x = 256 } }, f.stream, &args);
        } else if (bf16_partial) {
            try moe.residualAddBf16(f.k.glue53.residual_add_bf16, f.stream, ws.x + p.start * hidden * 2, gathered, n, f.st.world);
        } else {
            try f.k.glue.residualAdd(f.stream, ws.x + p.start * hidden * 2, ws.x + p.start * hidden * 2, gathered, n, hidden, f.st.world);
        }
    }

    /// One piece's attention: attn_norm -> q_a, q_a_norm, q_b, kv_a -> kv write -> indexer -> q_pe, absorb ->
    /// DCP2 attention (decode: gather 64 heads, attend own keys, exchange partials, LSE merge; prefill: own 32
    /// heads over own + stage keys, no merge) -> expand_v -> o_proj -> TP sum -> residual + post-attention RMSNorm.
    fn attention(f: *const Forward, l: weights.DeviceLayer, plane: usize, mode: Mode, positions: []const usize, p: Chunk, ordinal: usize, next: ?usize, xn_off: usize, cache_only: bool, skip_cache: bool, pipe_slot: ?usize) !void {
        // Decode side stream for the key path and shared expert after bertholomus/TensorFold glm-dsa-tp4 commit f14e7f7 (Apache-2.0, Albert Lee); independent Zig.
        const ws = &f.st.ws;
        const k = f.k;
        const a = l.attn;
        const lo = p.start;
        const rows = p.end - p.start;
        const xn_rows = if (state.groupedExperts(ws.expert_rows, ws.piece_rows)) ws.expert_rows else ws.piece_rows;
        if (xn_off + rows > xn_rows) return error.ExpertBatchBounds;
        const xn = ws.xn + xn_off * hidden * 2;
        const pos = positions[p.start..p.end];
        const s = f.stream;
        const prefill = mode == .prefill;
        // The side stream is used for rounds of at most 8 rows (DFlash2 blocks). 12- and 16-row copy rounds
        // (TF_GLM53_VERIFY_MAX=16) run on the main stream: side parity at 12 rows failed on 2026-10-30 (planes
        // differed, tokens and logits equal) and its cause is still open.
        const side_on = mode == .decode and rows <= side_max_rows and f.decodeSideEnabled() and f.ablate == null and !megakernelSwitch();
        const eps = f.w.config.eps;
        const expected_heads: usize = if (f.st.world == 3) (if (f.rank == 2) 20 else 22) else state.local_heads;
        if (a.local_heads != expected_heads) return error.BadHeadShare;
        // Row inputs and the window-wide token list at this piece's rows.
        const positions_at = ws.positions + lo * 8;
        const cos = ws.cos + lo * rope_half * 4;
        const sin = ws.sin + lo * rope_half * 4;
        const dense_len = ws.dense_len + lo * 4;
        const tokens = ws.global_tokens + lo * index.top_k * 4;
        const counts = ws.global_counts + lo * 4;
        try f.profileBegin(mode, plane, .attn_pre);
        try f.profileSubBegin(mode, plane, .attn_pre_input_norm);
        try k.glue.rmsnorm(s, ws.x + lo * hidden * 2, hidden, l.in_norm, xn, hidden, rows, hidden, eps);
        try f.profileSubEnd(mode, plane, .attn_pre_input_norm);
        if (cache_only) {
            try f.profileSubBegin(mode, plane, .attn_pre_kv_a_proj);
            try f.project(mode, s, xn, hidden, rows, a.kv_a, ws.kv, .bf16, ws.lane_part);
            try f.profileSubEnd(mode, plane, .attn_pre_kv_a_proj);
            const cache_latent = f.st.latentAt(plane);
            const cache_rope = f.st.ropeAt(plane);
            const cache_stage: ?types.StagePlanePair = if (f.dcpWorld() == 1) null else f.st.stage.forLayer(plane).*;
            try f.profileSubBegin(mode, plane, .attn_pre_kv_a_norm_rope);
            try k.mla.prep(s, ws.kv, a.kv_a_norm, cos, sin, positions_at, cache_latent, cache_rope, cache_stage, rows, state.kv_a_width, f.dcpRank(), f.dcpWorld(), true);
            try f.profileSubEnd(mode, plane, .attn_pre_kv_a_norm_rope);
            if (a.indexer) |ix| {
                const cache_index_plane = f.st.indexAt(ordinal);
                try f.profileSubBegin(mode, plane, .attn_pre_indexer_key_proj);
                try f.project(mode, s, xn, hidden, rows, ix.wk, ws.ik, .bf16, ws.lane_part);
                try f.profileSubEnd(mode, plane, .attn_pre_indexer_key_proj);
                try f.profileSubBegin(mode, plane, .attn_pre_indexer_keys_write);
                try k.index.keys(s, ws.ik, ix.k_norm, ix.k_bias, cos, sin, positions_at, cache_index_plane, rows, index.dim, 0, f.dcpRank(), f.dcpWorld(), f.w.config.indexer_eps);
                try f.profileSubEnd(mode, plane, .attn_pre_indexer_keys_write);
            }
            if (f.activePrefetch()) |pf| {
                const cache_pair = plane % state.stage_pairs;
                if (p.start == 0) try s.wait(pf.fetched[cache_pair]);
                if (p.end == positions.len) try pf.attended[cache_pair].record(s);
            }
            try f.profileSubEnd(mode, plane, .attn_pre);
            return;
        }
        // TF_GLM53_PRE_FUSE / TF_GLM53_PREP_FUSE: q_a, kv_a and indexer wk in one grouped launch on the main stream,
        // before the decode side stream forks (the side stream then runs only the kv write and the key write).
        const grouped = !skip_cache and f.groupedPre(mode);
        if (grouped) {
            try f.profileSubBegin(mode, plane, .attn_pre_q_a_proj);
            try f.projectPreGroup(mode, s, xn, rows, a);
            try f.profileSubEnd(mode, plane, .attn_pre_q_a_proj);
        }
        if (side_on) {
            const side = f.st.side orelse return error.MissingDecodeSideStream;
            try (f.st.fork orelse return error.MissingDecodeSideEvents).record(s);
            try side.wait(f.st.fork.?);
        }
        if (f.shardQkvAEnabled()) {
            const kvw: usize = if (f.rank == 0) 256 else 320;
            try f.profileSubBegin(mode, plane, .attn_pre_q_a_proj);
            try f.projectPinned(s, xn, hidden, rows, a.q_a, ws.sh_q, .bf16, ws.lane_part, 2048);
            try f.profileSubEnd(mode, plane, .attn_pre_q_a_proj);
            try f.profileSubBegin(mode, plane, .attn_pre_kv_a_proj);
            try f.projectPinned(s, xn, hidden, rows, a.kv_a, ws.sh_kv, .bf16, ws.lane_part, 576);
            try f.profileSubEnd(mode, plane, .attn_pre_kv_a_proj);
            try copyRows(s, ws.sh_mine, 2688, ws.sh_q, 2048, 2048, rows, "shard q_a");
            try copyRows(s, ws.sh_mine + 2048, 2688, ws.sh_kv, kvw * 2, kvw * 2, rows, "shard kv_a");
            try f.profileSubBegin(mode, plane, .attn_pre_qkv_gather);
            try f.coll.allGather(s, ws.sh_mine, ws.sh_all, rows * 2688, .shard_qkv_a);
            try f.profileSubEnd(mode, plane, .attn_pre_qkv_gather);
            try copyRows(s, ws.q_a, state.q_lora * 2, ws.sh_all, 2688, 2048, rows, "assemble q_a rank 0");
            try copyRows(s, ws.q_a + 2048, state.q_lora * 2, ws.sh_all + 2688, 2688, 2048, rows, "assemble q_a rank 1");
            try copyRows(s, ws.kv, state.kv_a_width * 2, ws.sh_all + 2048, 2688, 512, rows, "assemble kv rank 0");
            try copyRows(s, ws.kv + 512, state.kv_a_width * 2, ws.sh_all + 2688 + 2048, 2688, 640, rows, "assemble kv rank 1");
            if (side_on) {
                try (f.st.fork orelse return error.MissingDecodeSideEvents).record(s);
                try (f.st.side orelse return error.MissingDecodeSideStream).wait(f.st.fork.?);
            }
        } else if (!grouped) {
            try f.profileSubBegin(mode, plane, .attn_pre_q_a_proj);
            try f.project(mode, s, xn, hidden, rows, a.q_a, ws.q_a, .bf16, ws.lane_part);
            try f.profileSubEnd(mode, plane, .attn_pre_q_a_proj);
        }
        try f.profileSubBegin(mode, plane, .attn_pre_q_a_norm);
        try k.glue.rmsnorm(s, ws.q_a, state.q_lora, a.q_a_norm, ws.q_resid, state.q_lora, rows, state.q_lora, eps);
        try f.profileSubEnd(mode, plane, .attn_pre_q_a_norm);
        try f.profileSubBegin(mode, plane, .attn_pre_q_b_proj);
        try f.project(mode, s, ws.q_resid, state.q_lora, rows, a.q_b, ws.q, .bf16, ws.lane_part);
        try f.profileSubEnd(mode, plane, .attn_pre_q_b_proj);
        const side = if (side_on) f.st.side.? else s;
        const side_part = if (side_on) ws.lane_part_side else ws.lane_part;
        // with the q_a / kv_a split, ws.kv was already assembled from both ranks' gathered rows above
        if (!skip_cache and !f.shardQkvAEnabled() and !grouped) {
            if (side_on) {
                try f.profileSubBeginStream(mode, plane, .attn_pre_kv_a_proj, side);
                try f.project(mode, side, xn, hidden, rows, a.kv_a, ws.kv, .bf16, side_part);
                try f.profileSubEndStream(mode, plane, .attn_pre_kv_a_proj, side);
            } else {
                try f.profileSubBegin(mode, plane, .attn_pre_kv_a_proj);
                try f.project(mode, s, xn, hidden, rows, a.kv_a, ws.kv, .bf16, ws.lane_part);
                try f.profileSubEnd(mode, plane, .attn_pre_kv_a_proj);
            }
        }
        const latent = f.st.latentAt(plane);
        const rope = f.st.ropeAt(plane);
        const b = plane % state.stage_pairs;
        // The shared StagePlanes pair of this layer (types.zig contract); decode passes none.
        const stage: ?types.StagePlanePair = if (prefill and f.dcpWorld() > 1) f.st.stage.forLayer(plane).* else null;
        if (!skip_cache) {
            try f.profileSubBeginStream(mode, plane, .attn_pre_kv_a_norm_rope, if (side_on) side else s);
            try k.mla.prep(if (side_on) side else s, ws.kv, a.kv_a_norm, cos, sin, positions_at, latent, rope, stage, rows, state.kv_a_width, f.dcpRank(), f.dcpWorld(), prefill);
            try f.profileSubEndStream(mode, plane, .attn_pre_kv_a_norm_rope, if (side_on) side else s);
        }
        if (a.indexer) |ix| {
            const iplane = f.st.indexAt(ordinal);
            // Keys for every row (later rows score them); the query side only when a row selects (review L9).
            if (!skip_cache) {
                if (side_on) {
                    if (!grouped) {
                        try f.profileSubBeginStream(mode, plane, .attn_pre_indexer_key_proj, side);
                        try f.project(mode, side, xn, hidden, rows, ix.wk, ws.ik, .bf16, side_part);
                        try f.profileSubEndStream(mode, plane, .attn_pre_indexer_key_proj, side);
                    }
                    try f.profileSubBeginStream(mode, plane, .attn_pre_indexer_keys_write, side);
                    try k.index.keys(side, ws.ik, ix.k_norm, ix.k_bias, cos, sin, positions_at, iplane, rows, index.dim, 0, f.dcpRank(), f.dcpWorld(), f.w.config.indexer_eps);
                    try f.profileSubEndStream(mode, plane, .attn_pre_indexer_keys_write, side);
                    try (f.st.kv orelse return error.MissingDecodeSideEvents).record(side);
                    try s.wait(f.st.kv.?);
                } else {
                    if (!grouped) {
                        try f.profileSubBegin(mode, plane, .attn_pre_indexer_key_proj);
                        try f.project(mode, s, xn, hidden, rows, ix.wk, ws.ik, .bf16, ws.lane_part);
                        try f.profileSubEnd(mode, plane, .attn_pre_indexer_key_proj);
                    }
                    try f.profileSubBegin(mode, plane, .attn_pre_indexer_keys_write);
                    try k.index.keys(s, ws.ik, ix.k_norm, ix.k_bias, cos, sin, positions_at, iplane, rows, index.dim, 0, f.dcpRank(), f.dcpWorld(), f.w.config.indexer_eps);
                    try f.profileSubEnd(mode, plane, .attn_pre_indexer_keys_write);
                }
            }
            if (sparseIn(pos)) {
                try f.profileSubBegin(mode, plane, .attn_pre_indexer_weights_proj);
                if (mode == .prefill and f.prep_fuse == .split) {
                    try f.k.lane.matmulOutSplit(s, xn, hidden, rows, ix.weights_proj, ws.iw, .bf16, ws.lane_part);
                } else try f.project(mode, s, xn, hidden, rows, ix.weights_proj, ws.iw, .bf16, ws.lane_part);
                try f.profileSubEnd(mode, plane, .attn_pre_indexer_weights_proj);
                try f.profileSubBegin(mode, plane, .attn_pre_indexer_query_proj);
                try f.project(mode, s, ws.q_resid, state.q_lora, rows, ix.wq_b, ws.iq, .bf16, ws.lane_part);
                try k.index.q(s, ws.iq, cos, sin, ws.qi, rows);
                try f.profileSubEnd(mode, plane, .attn_pre_indexer_query_proj);
                try f.select(mode, plane, pos, lo, iplane);
            }
        } else if (side_on) {
            try (f.st.kv orelse return error.MissingDecodeSideEvents).record(side);
            try s.wait(f.st.kv.?);
        }
        // kv_b holds this rank's heads only, so the MLA kernels index it with local heads: head0 = 0 (kvbLocalHead0).
        const kvb_head0 = 0;
        const prompt_mla = if (prefill) f.promptMlaMode() else .off;
        const qa_bytes = rows * a.local_heads * mla.latent_dim * 2;
        const qpe_bytes = rows * a.local_heads * mla.rope_dim * 2;
        const qa = if (prefill) ws.qa else ws.qa + f.rank * qa_bytes;
        const qpe = if (prefill) ws.qpe else ws.qpe + f.rank * qpe_bytes;
        try f.profileSubBegin(mode, plane, .attn_pre_q_pe_pack);
        try k.mla.packQpe(s, ws.q, cos, sin, qpe, rows, a.local_heads);
        try f.profileSubEnd(mode, plane, .attn_pre_q_pe_pack);

        // 12/16-row copy rounds (TF_GLM53_VERIFY_MAX=16) exceed the decode MLA kernels' 8 rows: they use the generic kernels.
        const use_decode_mla = mode == .decode and rows <= mla.max_decode_rows and f.decodeMlaEnabled();
        try f.profileSubBegin(mode, plane, .attn_pre_absorb);
        if (prefill and prompt_mla != .off) {
            try k.mla.absorbPrompt(s, ws.q, a.kv_b.codes, a.kv_b.scales, qa, rows, a.local_heads, kvb_head0, prompt_mla);
        } else if (use_decode_mla and f.decodeMlaV3Enabled()) {
            try k.mla.absorbV3(s, ws.q, a.kv_b.codes, a.kv_b.scales, qa, rows, a.local_heads, kvb_head0);
        } else {
            try k.mla.absorb(s, ws.q, a.kv_b.codes, a.kv_b.scales, qa, rows, a.local_heads, kvb_head0, use_decode_mla);
        }
        try f.profileSubEnd(mode, plane, .attn_pre_absorb);

        if (prefill) {
            // Rule 2 (round-7 review): the first piece's attend waits on the event recorded after fetch(L) on the
            // side stream; the grouped send/recv completes in that stream's order, so it is the cross-rank signal.
            if (f.activePrefetch()) |pf| if (p.start == 0) try s.wait(pf.fetched[b]);
            if (f.smlaPEnabled()) {
                try f.profileSubBegin(mode, plane, .attn_pre_smla_p_scores);
                if (f.smla_p3) {
                    try k.mla.attendSmlaP3(s, ws.qa, ws.qpe, latent, rope, stage, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, ws.part, rows, a.local_heads, f.dcpRank(), f.dcpWorld(), ws.smla_union, ws.sizes.smla_union);
                } else if (f.smlaFp8Enabled()) {
                    try k.mla.attendSmlaPFp8(s, ws.qa, ws.qpe, latent, rope, stage, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, ws.part, rows, a.local_heads, f.dcpRank(), f.dcpWorld());
                } else try k.mla.attendSmlaP(s, ws.qa, ws.qpe, latent, rope, stage, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, ws.part, rows, a.local_heads, f.dcpRank(), f.dcpWorld());
                try f.profileSubEnd(mode, plane, .attn_pre_smla_p_scores);
            } else try k.mla.attend(s, ws.qa, ws.qpe, latent, rope, stage, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, ws.part, rows, a.local_heads, f.dcpRank(), f.dcpWorld(), true, 0);
            if (f.activePrefetch()) |pf| if (p.end == positions.len) {
                try pf.attended[b].record(s);
                if (next) |n| try f.prefetchLayer(n, positions[0]);
            };
            try f.profileEnd(mode, plane, .attn_pre);
            try f.profileBegin(mode, plane, .expand_v);

            if (prompt_mla != .off) {
                try k.mla.expandVPrompt(s, ws.part, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, a.local_heads, kvb_head0, prompt_mla);
            } else {
                try k.mla.expandV(s, ws.part, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, a.local_heads, kvb_head0, false);
            }

            try f.profileEnd(mode, plane, .expand_v);
        } else if (f.st.world == 3) {
            // TP=3 Replicated KV (dcp=1): No query gathering, no peer exchange, no LSE merge.
            // Each rank attends its own local heads directly against local KV cache into ws.ol.
            try f.profileEnd(mode, plane, .attn_pre);
            try f.profileBegin(mode, plane, .attn_pre);
            try k.mla.attend(s, qa, qpe, latent, rope, null, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, ws.ol, rows, a.local_heads, 0, 1, false, ws.attend_split);
            try f.profileEnd(mode, plane, .attn_pre);
            try f.profileBegin(mode, plane, .expand_v);
            if (use_decode_mla and f.decodeMlaV2Enabled()) {
                try k.mla.expandVV2(s, ws.ol, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, a.local_heads, kvb_head0);
            } else {
                try k.mla.expandV(s, ws.ol, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, a.local_heads, kvb_head0, use_decode_mla);
            }
            try f.profileEnd(mode, plane, .expand_v);
        } else {
            try f.profileEnd(mode, plane, .attn_pre);
            try f.profileBegin(mode, plane, .attn_comm);
            if (f.coll_fuse) {
                const layout = QueryPairLayout.init(qa_bytes, qpe_bytes);
                const mine = ws.fused_queries + layout.rankBase(f.rank);
                const all = ws.fused_queries;
                try Collective.copyDevice(s, mine + layout.qa, qa, qa_bytes, "GLM-5.3 pack absorbed query qa");
                try Collective.copyDevice(s, mine + layout.qpe, qpe, qpe_bytes, "GLM-5.3 pack absorbed query qpe");
                try f.coll.oneShotGather(s, mine, all, layout.rank_bytes);
                for (0..world) |rank| {
                    const base = ws.fused_queries + layout.rankBase(rank);
                    try Collective.copyDevice(s, ws.qa + rank * qa_bytes, base + layout.qa, qa_bytes, "GLM-5.3 unpack absorbed query qa");
                    try Collective.copyDevice(s, ws.qpe + rank * qpe_bytes, base + layout.qpe, qpe_bytes, "GLM-5.3 unpack absorbed query qpe");
                }
            } else {
                try f.coll.allGather(s, qa, ws.qa, qa_bytes, .absorbed_query);
                try f.profileEnd(mode, plane, .attn_comm);
                try f.profileBegin(mode, plane, .attn_comm);
                try f.coll.allGather(s, qpe, ws.qpe, qpe_bytes, .absorbed_query);
            }
            try f.profileEnd(mode, plane, .attn_comm);
            // Each rank keeps its contiguous [R][32][512] bf16 + [R][32] fp32 LSE block; only the peer block moves.
            const block_bytes = rows * state.decode_partial_row_bytes;
            const mine = ws.part;
            const peer = 1 - f.rank;
            const local_partial = mine + f.rank * block_bytes;
            const send_partial = mine + peer * block_bytes;
            const peer_partial = mine + world * block_bytes;
            try f.profileBegin(mode, plane, .attn_pre);
            if (f.decode_union and rows >= 1 and rows <= f.decode_union_rows) {
                try k.mla.attendVerify(s, ws.qa, ws.qpe, latent, rope, positions_at, ws.local_slots, ws.local_counts, mine, rows, 2 * state.local_heads, f.rank, world, ws.attend_split);
            } else try k.mla.attend(s, ws.qa, ws.qpe, latent, rope, null, positions_at, tokens, ws.local_slots, counts, ws.local_counts, dense_len, mine, rows, 2 * state.local_heads, f.rank, world, false, ws.attend_split);
            try f.profileEnd(mode, plane, .attn_pre);
            try f.profileBegin(mode, plane, .attn_comm);
            if (f.coll_fuse) {
                const fuse_plan = collFusePartialPlan(f.rank, block_bytes);
                if (fuse_plan.need > ws.sizes.part) return error.CollFusePartialScratch;
                try f.coll.oneShotGather(s, mine + fuse_plan.send, mine + fuse_plan.pair, block_bytes);
            } else try f.coll.sendRecv(s, send_partial, block_bytes, peer_partial, block_bytes, peer, .attention_partial);
            try f.profileEnd(mode, plane, .attn_comm);
            try f.profileBegin(mode, plane, .attn_post);
            const merge_peer = if (f.coll_fuse) mine + collFusePartialPlan(f.rank, block_bytes).peer else peer_partial;
            try k.mla.merge(s, local_partial, merge_peer, ws.ol, ws.lse_out, rows, state.local_heads, f.rank);
            try f.profileEnd(mode, plane, .attn_post);
            try f.profileBegin(mode, plane, .expand_v);
            if (use_decode_mla and f.decodeMlaV2Enabled()) {
                try k.mla.expandVV2(s, ws.ol, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, state.local_heads, kvb_head0);
            } else {
                try k.mla.expandV(s, ws.ol, a.kv_b.codes, a.kv_b.scales, ws.vo, rows, state.local_heads, kvb_head0, use_decode_mla);
            }
            try f.profileEnd(mode, plane, .expand_v);
        }
        try f.profileBegin(mode, plane, .attn_post);
        const vo_width = a.local_heads * mla.value_dim;
        const output = if (pipe_slot) |ring_slot| ws.pipe_src[ring_slot] else f.slot(rows);
        if (prefill and f.prefill_w8a8) {
            try f.oProjW8a8(s, ws.vo, vo_width, rows, a.o_proj, output);
        } else try f.project(mode, s, ws.vo, vo_width, rows, a.o_proj, output, .f32, ws.lane_part);
        try f.profileEnd(mode, plane, .attn_post);
        if (pipe_slot) |ring_slot| {
            try f.reduceIssue(p, ring_slot);
        } else try f.reduce(p, .attention_output, output, false, l.post_norm, xn_off, .{ .mode = mode, .layer = plane, .stage = .attn_comm, .after = .attn_post });
        if (pipe_slot == null) try f.profileEnd(mode, plane, .attn_post);
        if (pipe_slot == null and f.tbo_half == null) if (f.ablate) |ab| if (ab.vec(l.index, .attn)) |v| try ab.launch(f.k.ablate.?, f.stream, ws.x + p.start * hidden * 2, ws.gathered, .f32_parts, v, p.end - p.start, l.post_norm, xn, f.w.config.eps);
    }

    /// DSA selection on a full layer for one piece (rows `lo`.. of the window): local scores over this rank's
    /// keys, local exact top-2048 packed {score, position} records, one all-gather, then fused global merge/filter/
    /// remap into prefill tokens and DCP local slots. Scored in score_rows blocks.
    fn select(f: *const Forward, mode: Mode, layer: usize, positions: []const usize, lo: usize, plane: u64) !void {
        if (mode == .prefill and f.indexPipeEnabled()) return f.selectIndexPipe(mode, layer, positions, lo, plane);
        const ws = &f.st.ws;
        const k = f.k;
        var index_k = k.index;
        index_k.score_mma_rows = f.index_mma_rows;
        const s = f.stream;
        const rows = positions.len;
        const top = positions[rows - 1];
        // Decode scores a whole key bucket (masked past each row's position), so a decode graph replays across
        // positions inside its bucket; prefill scores exactly the keys below the piece's last row.
        const nkeys = if (mode == .decode) @min(decodeKeyBucket(top) / f.dcpWorld(), ws.index_slots) else if (top >= f.dcpRank()) (top - f.dcpRank()) / f.dcpWorld() + 1 else 0;
        if (nkeys == 0 or nkeys > ws.index_slots or nkeys > index.max_local_keys) return error.IndexCapacity;
        var at: usize = 0;
        while (at < rows) : (at += ws.score_rows) {
            const n = @min(ws.score_rows, rows - at);
            if (!sparseIn(positions[at..][0..n])) continue;
            const pos = ws.positions + (lo + at) * 8;
            try f.profileSubBegin(mode, layer, .attn_pre_indexer_scoring);
            try index_k.scores(s, ws.qi + at * index.heads * index.dim * 2, ws.iw + at * index.heads * 2, index.heads, 0, plane, pos, ws.scores, n, nkeys, nkeys, f.dcpRank(), f.dcpWorld());
            try f.profileSubEnd(mode, layer, .attn_pre_indexer_scoring);
            try f.profileSubBegin(mode, layer, .attn_pre_indexer_topk);
            try k.index.topk(s, ws.scores, nkeys, ws.topk_tokens, ws.topk_counts, ws.candidate_records, pos, n, nkeys, index.top_k, f.dcpRank(), f.dcpWorld());
            try f.profileSubEnd(mode, layer, .attn_pre_indexer_topk);
            if (f.dcpWorld() == 1) {
                try f.profileSubBegin(mode, layer, .attn_pre_indexer_merge);
                if (f.index_merge == 2) {
                    try k.index.merge2(s, ws.candidate_records, ws.global_tokens + (lo + at) * index.top_k * 4, ws.global_counts + (lo + at) * 4, ws.local_slots + (lo + at) * index.top_k * 4, ws.local_counts + (lo + at) * 4, pos, 0, 1, n, index.top_k, n);
                } else {
                    try k.index.merge(s, ws.candidate_records, ws.global_tokens + (lo + at) * index.top_k * 4, ws.global_counts + (lo + at) * 4, ws.local_slots + (lo + at) * index.top_k * 4, ws.local_counts + (lo + at) * 4, pos, 0, 1, n, index.top_k);
                }
                try f.profileSubEnd(mode, layer, .attn_pre_indexer_merge);
            } else {
                const bytes = n * index.top_k * 8;
                try f.profileEnd(mode, layer, .attn_pre);
                try f.profileBegin(mode, layer, .attn_comm);
                try f.profileSubBegin(mode, layer, .attn_pre_indexer_gather);
                try f.coll.allGather(s, ws.candidate_records, ws.gathered_candidates, bytes, .index_candidates);
                try f.profileSubEnd(mode, layer, .attn_pre_indexer_gather);
                try f.profileEnd(mode, layer, .attn_comm);
                try f.profileBegin(mode, layer, .attn_pre);
                try f.profileSubBegin(mode, layer, .attn_pre_indexer_merge);
                if (f.index_merge == 2) {
                    try k.index.merge2(s, ws.gathered_candidates, ws.global_tokens + (lo + at) * index.top_k * 4, ws.global_counts + (lo + at) * 4, ws.local_slots + (lo + at) * index.top_k * 4, ws.local_counts + (lo + at) * 4, pos, f.dcpRank(), f.dcpWorld(), n, index.top_k, n);
                } else {
                    try k.index.merge(s, ws.gathered_candidates, ws.global_tokens + (lo + at) * index.top_k * 4, ws.global_counts + (lo + at) * 4, ws.local_slots + (lo + at) * index.top_k * 4, ws.local_counts + (lo + at) * 4, pos, f.dcpRank(), f.dcpWorld(), n, index.top_k);
                }
                try f.profileSubEnd(mode, layer, .attn_pre_indexer_merge);
            }
        }
    }

    fn selectIndexPipe(f: *const Forward, mode: Mode, layer: usize, positions: []const usize, lo: usize, plane: u64) !void {
        const ws = &f.st.ws;
        const k = f.k;
        var index_k = k.index;
        index_k.score_mma_rows = f.index_mma_rows;
        const main = f.stream;
        const comm = f.st.tbo_stream orelse return error.MissingTboStream;
        const rows = positions.len;
        if (rows > ws.piece_rows) return error.IndexPipeRows;
        const top = positions[rows - 1];
        const nkeys = if (top >= f.dcpRank()) (top - f.dcpRank()) / f.dcpWorld() + 1 else 0;
        if (nkeys == 0 or nkeys > ws.index_slots or nkeys > index.max_local_keys) return error.IndexCapacity;
        if (!sparseIn(positions)) return;
        const blocks = (rows + ws.score_rows - 1) / ws.score_rows;
        if (blocks > state.Workspace.max_index_blocks) return error.IndexPipeTooManyBlocks;
        var at: usize = 0;
        var block: usize = 0;
        var b0: usize = 0;
        try f.profileEnd(mode, layer, .attn_pre);
        try f.profileBegin(mode, layer, .attn_comm);
        while (at < rows) : (block += 1) {
            const n = @min(ws.score_rows, rows - at);
            if (!sparseIn(positions[at..][0..n])) {
                b0 += 1;
                at += n;
                continue;
            }
            const pos = ws.positions + (lo + at) * 8;
            const mine = ws.candidate_records + block * ws.score_rows * index.top_k * 8;
            const gathered = ws.gathered_candidates + block * f.dcpWorld() * ws.score_rows * index.top_k * 8;
            try f.profileSubBegin(mode, layer, .attn_pre_indexer_scoring);
            try index_k.scores(main, ws.qi + at * index.heads * index.dim * 2, ws.iw + at * index.heads * 2, index.heads, 0, plane, pos, ws.scores, n, nkeys, nkeys, f.dcpRank(), f.dcpWorld());
            try f.profileSubEnd(mode, layer, .attn_pre_indexer_scoring);
            try f.profileSubBegin(mode, layer, .attn_pre_indexer_topk);
            try k.index.topk(main, ws.scores, nkeys, ws.topk_tokens, ws.topk_counts, mine, pos, n, nkeys, index.top_k, f.dcpRank(), f.dcpWorld());
            try f.profileSubEnd(mode, layer, .attn_pre_indexer_topk);
            const ready = try wsEvent(f.st.index_pipe_ready[block]);
            try ready.record(main);
            try comm.wait(ready);
            try f.profileSubBeginStream(mode, layer, .attn_pre_indexer_gather, comm);
            try f.commColl().allGather(comm, mine, gathered, n * index.top_k * 8, .index_candidates);
            try f.profileSubEndStream(mode, layer, .attn_pre_indexer_gather, comm);
            at += n;
        }
        const done = f.st.index_pipe_done orelse return error.MissingIndexPipeEvents;
        try done.record(comm);
        try main.wait(done);
        try f.profileEnd(mode, layer, .attn_comm);
        try f.profileBegin(mode, layer, .attn_pre);
        try f.profileSubBegin(mode, layer, .attn_pre_indexer_merge);
        const skipped_rows = b0 * ws.score_rows;
        try k.index.merge2(main, ws.gathered_candidates + b0 * f.dcpWorld() * ws.score_rows * index.top_k * 8, ws.global_tokens + (lo + skipped_rows) * index.top_k * 4, ws.global_counts + (lo + skipped_rows) * 4, ws.local_slots + (lo + skipped_rows) * index.top_k * 4, ws.local_counts + (lo + skipped_rows) * 4, ws.positions + (lo + skipped_rows) * 8, f.dcpRank(), f.dcpWorld(), rows - skipped_rows, index.top_k, ws.score_rows);
        try f.profileSubEnd(mode, layer, .attn_pre_indexer_merge);
    }

    fn wsEvent(event: ?cuda.Event) !cuda.Event {
        return event orelse error.MissingIndexPipeEvents;
    }

    fn groupedMoeLayer(f: *const Forward, l: weights.DeviceLayer, plane: usize, positions: []const usize, ordinal: usize, next: ?usize, pending_moe: *MoePipePendingSet) !void {
        const ws = &f.st.ws;
        const m = switch (l.ffn) { .moe => |moe_weights| moe_weights, .dense => return error.ExpectedMoeLayer };
        var mlp_k = f.k.mlpKernels();
        if (f.prefill_gemms != null) {
            mlp_k.project_ctx = @constCast(f);
            mlp_k.project_fn = mlpProject;
        }
        var group_at: usize = 0;
        while (group_at < positions.len) {
            // Expert groups cut at absolute multiples of expert_rows (validExpertRows: a multiple of piece_rows), like
            // pieces and chunks, so a resumed or restored prompt batches its experts the way a fresh one does.
            const group_end = pieceAt(.prefill, positions, ws.expert_rows, group_at).end;
            std.debug.assert(group_end - group_at <= ws.expert_rows);
            var row_at = group_at;
            var piece_ordinal: usize = 0;
            var pending: ?PipePending = null;
            var moe_slot_used = [_]bool{ false, false };
            const pipe_on = f.piecePipeEnabled();
            while (row_at < group_end) {
                const p = pieceAt(.prefill, positions, ws.piece_rows, row_at);
                const n = p.end - p.start;
                const off = row_at - group_at;
                std.debug.assert(off + n <= ws.expert_rows);
                if (f.moePipeEnabled()) try f.finishMoePendingFor(pending_moe, p);
                if (pipe_on) {
                    const ring_slot = piece_ordinal % 2;
                    try f.attention(l, plane, .prefill, positions, p, ordinal, next, off, false, false, ring_slot);
                    if (pending) |previous| {
                        try f.reduceFinish(previous.p, previous.off, previous.slot, l.post_norm, l.index);
                        try f.ffnPre(l, m, mlp_k, previous.p, previous.off);
                    }
                    // ffnPre(q) reads only q's xn/shared_out/picks8/wts8/lowrank_mid rows. It is ordered on
                    // the main stream after attention(p); attention(p) does not read q's x/xn rows. Gather(p)
                    // reads pipe_src[i%2], writes pipe_gather[i%2], both outside the phase-union scratch.
                    // Reusing a ring ring_slot at p+2 is safe because p's reduceFinish waited for pipe_done[k].
                    pending = .{ .p = p, .off = off, .slot = ring_slot };
                } else {
                    try f.attention(l, plane, .prefill, positions, p, ordinal, next, off, false, false, null);
                    try f.ffnPre(l, m, mlp_k, p, off);
                }
                row_at = p.end;
                piece_ordinal += 1;
            }
            if (pending) |last| {
                try f.reduceFinish(last.p, last.off, last.slot, l.post_norm, l.index);
                try f.ffnPre(l, m, mlp_k, last.p, last.off);
                pending = null;
            }
            // Before routedGroup reuses ws.experts, retire prior-layer slots whose pieces occur in later groups.
            if (f.moePipeEnabled()) try f.finishAllMoePending(pending_moe);
            const group_rows = row_at - group_at;
            var routed: Routed = .{ .f = f, .layer = m.routed, .mode = .prefill };
            try f.profileBegin(.prefill, l.index, .routed);
            const post_y = try routedGroup(&routed, f.stream, ws.xn, ws.picks8, ws.wts8, group_rows, ws.routed_out);
            try f.profileEnd(.prefill, l.index, .routed);
            row_at = group_at;
            piece_ordinal = 0;
            while (row_at < group_end) {
                const p = pieceAt(.prefill, positions, ws.piece_rows, row_at);
                const n = p.end - p.start;
                const off = row_at - group_at;
                const lr: ?moe.LowRankWeights = if (m.lowrank) |r| .{ .a = r.a, .b = r.b, .mid = ws.lowrank_mid + off * 8 * (r.rank_local * 4), .rank_local = r.rank_local } else null;
                const ring_slot = if (f.moePipeEnabled()) piece_ordinal % ws.moe_pipe_slots else 0;
                if (f.moePipeEnabled() and moe_slot_used[ring_slot]) try f.stream.wait(f.st.moe_pipe_done[ring_slot].?);
                if (f.moePipeEnabled()) moe_slot_used[ring_slot] = true;
                const gathered_bytes = n * hidden * f.st.world * 2;
                const moe_out = if (f.moe_post) ws.moe_pipe_gather[ring_slot] + f.rank * n * hidden * 2 else if (f.moePipeEnabled()) ws.moe_pipe_src[ring_slot] else f.gatherBuffer() + gathered_bytes;
                var legacy_lowrank_k = f.k.glue53;
                legacy_lowrank_k.decode_fast = false;
                // Between this blockPost and its pdone event, main-stream work only produces later pieces'
                // routed/shared outputs and distinct ring slots; it does not touch this piece's ws.x rows.
                // The comm stream owns the residual add to those rows. Before main can reuse this slot,
                // done[slot] proves both its gather read and residual add have completed.
                if (f.moe_post) {
                    comptime std.debug.assert(state.routed_slots == moe.top_k);
                    const y = post_y orelse return error.MoePostPromptOutputMissing;
                    try moe.postPrompt(legacy_lowrank_k, f.stream, y + off * state.routed_slots * hidden * 4, ws.wts8 + off * state.routed_slots * 4, ws.shared_out + off * hidden * 4, lr, n, moe_out);
                } else {
                    try moe.blockPost(legacy_lowrank_k, f.stream, ws.routed_out + off * hidden * 4, ws.shared_out + off * hidden * 4, lr, n, moe_out);
                }
                if (f.moePipeEnabled()) {
                    const piece_index = piece_ordinal;
                    try f.moeReduceIssue(p, ring_slot, piece_index, l.index);
                    try addMoePending(pending_moe, .{ .p = p, .piece_index = piece_index, .layer = l.index });
                } else {
                    try f.profileBegin(.prefill, l.index, .moe_comm);
                    try f.reduce(p, .moe_partial, moe_out, true, null, 0, null);
                    if (f.ablate) |ab| if (ab.vec(l.index, .ffn)) |v| try ab.launch(f.k.ablate.?, f.stream, ws.x + p.start * hidden * 2, ws.gathered, .bf16_parts, v, n, 0, 0, f.w.config.eps);
                    try f.profileEnd(.prefill, l.index, .moe_comm);
                }
                row_at = p.end;
                piece_ordinal += 1;
            }
            if (f.prof_prefill) if (f.prof) |profile| {
                try f.profileSync();
                try profile.collect();
            };
            group_at = group_end;
        }
    }

    /// One piece's FFN consumes the normalized attention output, then runs MoE or dense MLP -> TP sum -> residual.
    fn ffn(f: *const Forward, l: weights.DeviceLayer, p: Chunk, mode: Mode) !void {
        // Decode side stream for the key path and shared expert after bertholomus/TensorFold glm-dsa-tp4 commit f14e7f7 (Apache-2.0, Albert Lee); independent Zig.
        const ws = &f.st.ws;
        const k = f.k;
        const s = f.stream;
        const n = p.end - p.start;
        const xn = ws.xn + f.tbo_xn_off * hidden * 2;
        const scratch: mlp.Scratch = .{ .gate_up = ws.mlp_gate_up, .activation = ws.mlp_act, .mm_part = ws.lane_part };
        const out = f.slot(n);
        const dense = switch (l.ffn) { .dense => true, .moe => false };
        var mlp_k = k.mlpKernels();
        if (mode == .prefill and f.prefill_gemms != null) {
            mlp_k.project_ctx = @constCast(f);
            mlp_k.project_fn = mlpProject;
        }
        switch (l.ffn) {
            .dense => |d| {
                try f.profileBegin(mode, l.index, .dense_mlp);
                try mlp.block(mlp_k, s, d, scratch, xn, n, d.down.k, out);
            },
            .moe => |m| {
                var routed: Routed = .{ .f = f, .layer = m.routed, .mode = mode };
                const lr: ?moe.LowRankWeights = if (m.lowrank) |r| .{ .a = r.a, .b = r.b, .mid = ws.lowrank_mid, .rank_local = r.rank_local } else null;
                if (m.lowrank) |r| if (r.rank_local > state.lowrank_max) return error.LowRankTooWide;
                // The BF16 partial lives after this piece's rank-major all-gather span; router_logits is too small.
                const gathered_bytes = n * hidden * f.st.world * 2;
                const moe_out = f.gatherBuffer() + gathered_bytes;
                // The routed fp32 output is never all-gathered, so it sits at the start of ws.gathered on EVERY rank:
                // f.slot(n) put rank 1's at gathered + 4*n*hidden, the same address as moe_out, and the combine kernel
                // (reads fp32 routed[i], writes bf16 out[i]) then overwrote its own input on rank 1: nondeterministic text.
                const routed_out = f.gatherBuffer();
                const side_on = mode == .decode and p.end - p.start <= side_max_rows and f.decodeSideEnabled() and f.ablate == null and !megakernelSwitch();
                if (side_on) {
                    const side = f.st.side orelse return error.MissingDecodeSideStream;
                    try (f.st.fork2 orelse return error.MissingDecodeSideEvents).record(s);
                    try side.wait(f.st.fork2.?);
                    // Side MLP uses the ordinary MLP scratch: the main route and EXL3 expert path touch router,
                    // pick, expert and routed-output regions, which are disjoint from gate_up/activation/lane_part.
                    // TF_GLM53_PROFILE times route and routed on the main stream and shared on the side stream (they
                    // overlap: the window pays the longer of routed and shared, then moe_post).
                    try f.profileBegin(mode, l.index, .route);
                    try moe.route(k.router(), s, xn, hidden, m.router, m.bias, ws.router_logits, ws.router_part, ws.picks, ws.wts, n);
                    try f.profileEnd(mode, l.index, .route);
                    try f.profileBegin(mode, l.index, .routed);
                    try runRouted(&routed, s, xn, hidden, ws.picks, ws.wts, n, routed_out);
                    try f.profileEnd(mode, l.index, .routed);
                    try f.profileBeginStream(mode, l.index, .shared, side);
                    try mlp.block(mlp_k, side, m.shared, scratch, xn, n, m.shared.down.k, ws.shared_out);
                    try f.profileEndStream(mode, l.index, .shared, side);
                    // The low-rank partial/reduce that moe.block runs (review of 2b9a090e: without it blockPost combined
                    // stale lowrank_mid). lowrank_mid is disjoint from the main route/expert scratch.
                    var lowrank_k = k.glue53;
                    lowrank_k.decode_fast = f.lowrank_decode;
                    if (lr) |w| try moe.lowrankPre(lowrank_k, side, xn, hidden, n, w);
                    try (f.st.shared orelse return error.MissingDecodeSideEvents).record(side);
                    try s.wait(f.st.shared.?);
                    try f.profileBegin(mode, l.index, .moe_post);
                    try moe.blockPost(lowrank_k, s, routed_out, ws.shared_out, lr, n, moe_out);
                    try f.profileEnd(mode, l.index, .moe_post);
                } else {
                    var timing: MoeProfile = .{ .f = f, .layer = l.index, .mode = mode };
                    const hooks: ?moe.ProfileHooks = if ((mode == .decode or (mode == .prefill and f.prof_prefill)) and f.prof != null) .{ .ctx = @ptrCast(&timing), .begin_fn = moeProfileBegin, .end_fn = moeProfileEnd } else null;
                    var lowrank_k = k.glue53;
                    lowrank_k.decode_fast = f.lowrank_decode and mode == .decode;
                    try moe.blockWithHooks(lowrank_k, k.router(), mlp_k, s, m.shared, scratch, &routed, runRouted, xn, hidden, m.router, m.bias, ws.router_logits, ws.router_part, ws.picks, ws.wts, ws.shared_out, n, routed_out, moe_out, lr, hooks);
                }
                try f.reduce(p, .moe_partial, moe_out, true, null, 0, .{ .mode = mode, .layer = l.index, .stage = .moe_comm });
                if (f.tbo_half == null) if (f.ablate) |ab| if (ab.vec(l.index, .ffn)) |v| try ab.launch(f.k.ablate.?, f.stream, ws.x + p.start * hidden * 2, ws.gathered, .bf16_parts, v, n, 0, 0, f.w.config.eps);
                return;
            },
        }
        if (dense) try f.profileEnd(mode, l.index, .dense_mlp);
        try f.reduce(p, .moe_partial, out, false, null, 0, if (dense) .{ .mode = mode, .layer = l.index, .stage = .dense_comm } else null);
        if (f.tbo_half == null) if (f.ablate) |ab| if (ab.vec(l.index, .ffn)) |v| try ab.launch(f.k.ablate.?, f.stream, ws.x + p.start * hidden * 2, ws.gathered, .f32_parts, v, n, 0, 0, f.w.config.eps);
    }

    /// moe.RoutedRunner: route picks/weights [R][9] (slot 8 = shared) compacted to the 8 routed slots, then WP4's
    /// grouped EXL3 experts with the fp32 slot-order weighted sum (down_combine) into `out`, in expert_rows chunks.
    /// With the prompt switch off, decode and prefill rows take the same kernels and have equal bits.
    fn runRouted(ctx: *anyopaque, stream: cuda.Stream, x: u64, x_stride: usize, picks: u64, wts: u64, rows: usize, out: u64) anyerror!void {
        const r: *const Routed = @ptrCast(@alignCast(ctx));
        const f = r.f;
        const ws = &f.st.ws;
        try copyRows(stream, ws.picks8, state.routed_slots * 4, picks, state.route_slots * 4, state.routed_slots * 4, rows, "GLM-5.3 compact route picks");
        try copyRows(stream, ws.wts8, state.routed_slots * 4, wts, state.route_slots * 4, state.routed_slots * 4, rows, "GLM-5.3 compact route weights");
        const l = r.layer.layer();
        const prompt_tiles_eligible = cuda.exl3_experts.prompt.eligibleTiles(l);
        const prompt_wide_eligible = cuda.exl3_experts.prompt.eligible(l);
        if (r.mode == .prefill and f.promptExpertsEnabled() and prompt_tiles_eligible) {
            try state.validatePromptExpertsCarveForLayer(true, ws, l);
            const router = f.k.experts.router;
            const prompt_plan = promptPlan(ws.prompt_plan, ws.expert_rows, state.routed_slots, state.routed_experts);
            const prompt_scratch = try cuda.exl3_experts.prompt.scratch(ws.prompt_experts, l, ws.expert_rows, state.routed_slots, ws.scratch_overlay);
            const wide = f.promptExpertsWide() and prompt_wide_eligible;
            const pmx = f.promptExpertsPmx() and cuda.exl3_experts.pmx.eligible(l);
            var at_prompt: usize = 0;
            while (at_prompt < rows) : (at_prompt += ws.expert_rows) {
                const n = @min(ws.expert_rows, rows - at_prompt);
                try stream.flushDeferred();
                const piece_x = x + at_prompt * x_stride * 2;
                const piece_picks = ws.picks8 + at_prompt * state.routed_slots * 4;
                if (pmx and f.promptExpertsP5a()) {
                    const pmx_kernels = f.k.experts.pmx orelse return error.PromptExpertsPmxKernelsMissing;
                    const p5a_kernels = f.k.experts.p5a orelse return error.PromptExpertsP5aKernelsMissing;
                    try f.k.experts.prompt.runP5a(&pmx_kernels, &p5a_kernels, f.k.experts, router, stream, l, piece_x, x_stride, piece_picks, n, prompt_scratch, prompt_plan);
                } else if (pmx) {
                    const pmx_kernels = f.k.experts.pmx orelse return error.PromptExpertsPmxKernelsMissing;
                    try f.k.experts.prompt.runPmx(&pmx_kernels, f.k.experts, router, stream, l, piece_x, x_stride, piece_picks, n, prompt_scratch, prompt_plan);
                } else if (wide) {
                    try f.k.experts.prompt.runWide(f.k.experts, router, stream, l, piece_x, x_stride, piece_picks, n, prompt_scratch, prompt_plan);
                } else {
                    try f.k.experts.prompt.runTile(f.k.experts, router, stream, l, piece_x, x_stride, piece_picks, n, prompt_scratch, prompt_plan, cuda.exl3_experts.prompt.tile);
                }
                try f.k.experts.combinePrompt(stream, l, prompt_scratch.y, ws.wts8 + at_prompt * state.routed_slots * 4, n, state.routed_slots, out + at_prompt * hidden * 4, prompt_scratch.ids);
            }
            return;
        }
        if (ws.adapt_big) return error.ExpertRowsAdaptGenericPath;
        const generic_rows = ws.generic_rows;
        const sc = try cuda.exl3_experts.scratch(ws.experts, l, generic_rows, state.routed_slots);
        var at: usize = 0;
        while (at < rows) : (at += generic_rows) {
            const n = @min(generic_rows, rows - at);
            // MK8 flush point: the EXL3 runner may clear its counters with a stream memset.
            try stream.flushDeferred();
            try f.k.experts.run(stream, l, x + at * x_stride * 2, x_stride, .bf16, ws.picks8 + at * state.routed_slots * 4, ws.wts8 + at * state.routed_slots * 4, n, out + at * hidden * 4, sc);
        }
    }

    fn routedGroup(r: *const Routed, stream: cuda.Stream, x: u64, picks: u64, wts: u64, rows: usize, out: u64) !?u64 {
        const f = r.f;
        const ws = &f.st.ws;
        const layer = r.layer.layer();
        if (rows > ws.expert_rows) return error.ExpertBatchBounds;
        const prompt_tiles_eligible = cuda.exl3_experts.prompt.eligibleTiles(layer);
        const prompt_wide_eligible = cuda.exl3_experts.prompt.eligible(layer);
        if (f.promptExpertsEnabled() and prompt_tiles_eligible) {
            try state.validatePromptExpertsCarveForLayer(true, ws, layer);
            const router = f.k.experts.router;
            const prompt_plan = promptPlan(ws.prompt_plan, ws.expert_rows, state.routed_slots, state.routed_experts);
            const prompt_scratch = try cuda.exl3_experts.prompt.scratch(ws.prompt_experts, layer, ws.expert_rows, state.routed_slots, ws.scratch_overlay);
            try stream.flushDeferred();
            if (f.promptExpertsP5a() and f.promptExpertsPmx() and cuda.exl3_experts.pmx.eligible(layer)) {
                const pmx_kernels = f.k.experts.pmx orelse return error.PromptExpertsPmxKernelsMissing;
                const p5a_kernels = f.k.experts.p5a orelse return error.PromptExpertsP5aKernelsMissing;
                try f.k.experts.prompt.runP5a(&pmx_kernels, &p5a_kernels, f.k.experts, router, stream, layer, x, hidden, picks, rows, prompt_scratch, prompt_plan);
            } else if (f.promptExpertsPmx() and cuda.exl3_experts.pmx.eligible(layer)) {
                const pmx_kernels = f.k.experts.pmx orelse return error.PromptExpertsPmxKernelsMissing;
                try f.k.experts.prompt.runPmx(&pmx_kernels, f.k.experts, router, stream, layer, x, hidden, picks, rows, prompt_scratch, prompt_plan);
            } else if (f.promptExpertsWide() and prompt_wide_eligible) {
                try f.k.experts.prompt.runWide(f.k.experts, router, stream, layer, x, hidden, picks, rows, prompt_scratch, prompt_plan);
            } else {
                try f.k.experts.prompt.runTile(f.k.experts, router, stream, layer, x, hidden, picks, rows, prompt_scratch, prompt_plan, cuda.exl3_experts.prompt.tile);
            }
            if (f.moe_post) return prompt_scratch.y;
            try f.k.experts.combinePrompt(stream, layer, prompt_scratch.y, wts, rows, state.routed_slots, out, prompt_scratch.ids);
            return null;
        }
        if (ws.adapt_big) return error.ExpertRowsAdaptGenericPath;
        if (rows > ws.generic_rows) return error.WsTrimGenericBounds;
        const scratch = try cuda.exl3_experts.scratch(ws.experts, layer, ws.generic_rows, state.routed_slots);
        try stream.flushDeferred();
        try f.k.experts.run(stream, layer, x, hidden, .bf16, picks, wts, rows, out, scratch);
        return null;
    }

    /// GPU parity harness entry point: exercise the same grouped prefill runner with caller-carved workspaces.
    pub fn routedGroupParity(k: *const Kernels, ws: state.Workspace, stream: cuda.Stream, layer: experts.Loaded, x: u64, picks: u64, wts: u64, rows: usize, out: u64) !void {
        var st: state.State = undefined;
        st.ws = ws;
        var f: Forward = undefined;
        f.k = k;
        f.st = &st;
        f.prompt_experts = true;
        f.prompt_experts_wide = true;
        f.prompt_experts_pmx = false;
        f.prompt_experts_p5a = false;
        f.moe_post = false; // the harness checks routed output directly, never the deferred MOE_POST combine
        f.ws_trim = false;
        const routed: Routed = .{ .f = &f, .layer = layer, .mode = .prefill };
        _ = try routedGroup(&routed, stream, x, picks, wts, rows, out);
    }

    fn promptPlan(base: u64, rows: usize, slots: usize, experts_n: usize) cuda.grouped.Plan {
        const layout = state.promptPlanLayout(rows, slots, experts_n);
        var at = base;
        const members = at;
        at += state.Workspace.align256(@intCast(layout.pairs * @sizeOf(i32)));
        const items = at;
        at += state.Workspace.align256(@intCast(layout.items * @sizeOf(i32)));
        const counts = at;
        at += state.Workspace.align256(@intCast(layout.counts * @sizeOf(i32)));
        const rank = if (layout.rank == 0) 0 else at;
        at += state.Workspace.align256(@intCast(layout.rank * @sizeOf(i32)));
        const hist = if (layout.hist == 0) 0 else at;
        return .{ .members = members, .items = items, .counts = counts, .rank = rank, .hist = hist };
    }

    /// Final norm, this rank's fp32 lm_head rows, then top-1 + LSE or k candidates, gathered rank-major, in blocks
    /// of head_rows (WP10 launches at most 64 rows). The sink reads each block before the next one.
    fn headRows(f: *const Forward, first: usize, count: usize) !void {
        const ws = &f.st.ws;
        const k = f.k;
        const s = f.stream;
        const head = f.w.head orelse return error.NoHead;
        const vocab = f.w.vocab.len();
        if (ws.head_rows == 0 or ws.head_rows > state.max_head_rows) return error.BadHeadRows;
        var at: usize = 0;
        while (at < count) : (at += ws.head_rows) {
            const n = @min(ws.head_rows, count - at);
            try k.glue.rmsnorm(s, ws.x + (first + at) * hidden * 2, hidden, f.w.norm, ws.xn, hidden, n, hidden, f.w.config.eps);
            try k.lane.matmulF32(s, ws.xn, hidden, n, head, ws.logits, ws.lane_part);
            if (f.mask.bits != 0) try k.head.mask(s, ws.logits, vocab, f.w.vocab.lo, f.mask.bits + @as(u64, at) * @as(u64, f.mask.words) * 4, f.mask.words, n);
            const per = n * f.head.rowBytes();
            const mine = ws.head_gather + f.rank * per;
            switch (f.head) {
                .none => unreachable,
                .greedy => {
                    try k.head.top1(s, ws.logits, vocab, f.w.vocab.lo, mine, mine + n * 4, n);
                    try k.head.lse(s, ws.logits, vocab, mine + n * 8, n);
                },
                .candidates => |kc| {
                    if (kc == 0 or kc > ws.candidate_k) return error.InvalidCandidateCount;
                    try k.head.candidates(s, ws.logits, vocab, f.w.vocab.lo, kc, mine, mine + n * kc * 4, ws.cand_scratch, ws.sizes.cand_scratch, n);
                    try k.head.lse(s, ws.logits, vocab, mine + n * kc * 8, n);
                },
            }
            try f.coll.allGather(s, mine, ws.head_gather, per, .head);
            try f.coll.gateResult(s, ws.roce_token, ws.roce_status);
            if (f.sink) |sink| {
                try s.flushDeferred(); // MK8 flush point: the sink reads the head results on the host
                try sink.sink_fn(sink.ctx, s, first + at, n, ws.head_gather, per);
            }
        }
    }
};

/// `height` rows of `width` bytes, `spitch` apart at `src`, to `dpitch` apart at `dst` (glm/cuda_exl3.zig copyRows).
fn copyRows(stream: cuda.Stream, dst: u64, dpitch: usize, src: u64, spitch: usize, width: usize, height: usize, what: []const u8) !void {
    if (height == 0) return;
    try stream.flushDeferred(); // MK8 flush point
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = spitch, .dst = dst, .dst_pitch = dpitch, .width = width, .height = height };
    try stream.d.check(stream.d.api.cuMemcpy2DAsync_v2(&m, stream.handle), what);
}

test "decode and prefill exchange plans: per-row bytes, gather-on-prefill and the overlap flag" {
    const layers = [_]Layer{ .{ .index = 0, .kind = .dense, .index_full = true }, .{ .index = 1, .kind = .moe, .has_lowrank = true } };
    const decode = try plan(.{ .mode = .decode, .positions = &.{ 100, 3000 }, .layers = &layers, .head_rows = 2 });
    try std.testing.expectEqual(@as(usize, 2 * 6144 * 2), decode.of(.embed).bytes);
    try std.testing.expectEqual(@as(usize, 36864), Bytes.absorbed_query_row);
    try std.testing.expectEqual(@as(usize, 32896), Bytes.attention_partial_row);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 36864), decode.of(.absorbed_query).bytes);
    try std.testing.expectEqual(@as(usize, 2 * 2 * 32896), decode.of(.attention_partial).bytes);
    try std.testing.expectEqual(@as(usize, 4), decode.of(.absorbed_query).calls);
    try std.testing.expect(!decode.of(.attention_partial).all_gather);
    try std.testing.expectEqual(@as(usize, 2 * 16384), decode.of(.index_candidates).bytes); // one full layer, one block
    try std.testing.expectEqual(@as(usize, 2 * 2 * 24576), decode.of(.attention_output).bytes);
    try std.testing.expectEqual(@as(usize, 32), decode.of(.head).bytes);
    try std.testing.expectEqual(@as(usize, 0), decode.of(.stage_fetch).bytes);
    try std.testing.expect(!decode.two_batch_overlap and !decode.gather_on_prefill);
    const fused = try plan(.{ .mode = .decode, .positions = &.{ 100, 3000 }, .layers = &layers, .head_rows = 2, .coll_fuse = true });
    try std.testing.expectEqual(@as(usize, 2), fused.of(.absorbed_query).calls);
    try std.testing.expectEqual(decode.of(.absorbed_query).bytes, fused.of(.absorbed_query).bytes);
    try std.testing.expect(fused.of(.attention_partial).all_gather);
    try std.testing.expectEqual(decode.of(.attention_partial).calls, fused.of(.attention_partial).calls);
    try std.testing.expectEqual(decode.of(.index_candidates).calls, fused.of(.index_candidates).calls);
    var rows: [4096]usize = undefined;
    for (&rows, 0..) |*p, i| p.* = 4096 + i;
    const prefill = try plan(.{ .mode = .prefill, .positions = &rows, .layers = &layers, .head_rows = 1 });
    try std.testing.expect(prefill.two_batch_overlap and prefill.gather_on_prefill);
    try std.testing.expectEqual(@as(usize, 0), prefill.of(.absorbed_query).bytes);
    try std.testing.expectEqual(@as(usize, 0), prefill.of(.attention_partial).bytes);
    try std.testing.expectEqual(@as(usize, 2 * 2048 * 1152), prefill.of(.stage_fetch).bytes);
    // 4 pieces of 1024 rows, each 8 score blocks of 128 rows, one full layer: 32 packed exchanges.
    try std.testing.expectEqual(@as(usize, 32), prefill.of(.index_candidates).calls);
    try std.testing.expectEqual(@as(usize, 2 * 4), prefill.of(.moe_partial).calls);
    try std.testing.expectEqual(@as(usize, 4), prefill.of(.embed).calls);
    // An unaligned resume start cuts its pieces at absolute multiples: [1000, 1024), [1024, 2048), [2048, 2100).
    var tail: [1100]usize = undefined;
    for (&tail, 0..) |*q, i| q.* = 1000 + i;
    try std.testing.expectEqual(Chunk{ .start = 0, .end = 24 }, pieceAt(.prefill, &tail, 1024, 0));
    try std.testing.expectEqual(Chunk{ .start = 24, .end = 1048 }, pieceAt(.prefill, &tail, 1024, 24));
    try std.testing.expectEqual(Chunk{ .start = 0, .end = 1100 }, pieceAt(.decode, &tail, 1024, 0));
}

test "piece ordinals follow pieces for aligned and unaligned groups" {
    var aligned: [2048]usize = undefined;
    for (&aligned, 0..) |*position, i| position.* = 4096 + i;
    var aligned_ordinals: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 4), pieceOrdinals(&aligned, 512, 0, aligned.len, &aligned_ordinals));
    for (aligned_ordinals, 0..) |piece_ordinal, i| {
        const off: usize = i * 512;
        try std.testing.expectEqual(off / 512, piece_ordinal);
    }

    var unaligned: [2044]usize = undefined;
    for (&unaligned, 0..) |*position, i| position.* = 4100 + i;
    var unaligned_ordinals: [4]usize = undefined;
    try std.testing.expectEqual(@as(usize, 4), pieceOrdinals(&unaligned, 512, 0, unaligned.len, &unaligned_ordinals));
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 2, 3 }, &unaligned_ordinals);
    for (0..unaligned_ordinals.len - 1) |i| try std.testing.expect((unaligned_ordinals[i] % 2) != (unaligned_ordinals[i + 1] % 2));

    var short: [100]usize = undefined;
    for (&short, 0..) |*position, i| position.* = 4100 + i;
    var short_ordinal: [1]usize = undefined;
    try std.testing.expectEqual(@as(usize, 1), pieceOrdinals(&short, 512, 0, short.len, &short_ordinal));
    try std.testing.expectEqual(@as(usize, 0), short_ordinal[0]);
}

test "packed query pair layout preserves byte-identical QA and QPE views" {
    const qa = [_]u8{ 0, 1, 2, 3, 4, 5 };
    const qpe = [_]u8{ 8, 9, 10 };
    var gathered: [18]u8 = undefined;
    try packQueryPairBytes(gathered[0..9], &qa, &qpe);
    try packQueryPairBytes(gathered[9..18], &qa, &qpe);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3, 4, 5, 8, 9, 10 }, gathered[0..9]);
    const layout = QueryPairLayout.init(qa.len, qpe.len);
    const rank_one = layout.rankBase(1);
    try std.testing.expectEqualSlices(u8, &qa, gathered[rank_one + layout.qa ..][0..qa.len]);
    try std.testing.expectEqualSlices(u8, &qpe, gathered[rank_one + layout.qpe ..][0..qpe.len]);
}

test "rope rows follow each row's absolute position and dense_len marks rows below 2048" {
    var c: [rope_half]f32 = undefined;
    var s: [rope_half]f32 = undefined;
    ropeRow(0, &c, &s);
    try std.testing.expectEqual(@as(f32, 1), c[0]);
    try std.testing.expectEqual(@as(f32, 0), s[5]);
    ropeRow(7, &c, &s);
    try std.testing.expectEqual(@as(f32, @cos(@as(f32, 7))), c[0]);
    // fp32 op order of rope.py _table: inv_i = exp2(-(2i/64) * log2f(8e6)), phase = 7 * inv_i.
    const l2: f32 = @log2(@as(f32, 8e6));
    try std.testing.expectEqual(@as(f32, @cos(@as(f32, 7) * @exp2(-((2.0 * @as(f32, 5)) / 64.0) * l2))), c[5]);
    var in = try RowInputs.init(std.testing.allocator, &.{ 2047, 2048, 9 }, &.{ 5, 77440 + 3, 77439 }, 1, 77440);
    defer in.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(i32, &.{ 2048, 0, 10 }, in.dense_len);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 0 }, in.owners);
    try std.testing.expectEqualSlices(i32, &.{ 0, 3, 0 }, in.ids);
    var expect_c: [rope_half]f32 = undefined;
    ropeRow(9, &expect_c, &s);
    try std.testing.expectEqualSlices(f32, &expect_c, in.cos[2 * rope_half ..][0..rope_half]);
}

test "decode row staging is fixed capacity and matches allocator-backed row inputs" {
    const positions = [_]usize{ 17, 2047, 2048, 4095, 4096, 8191, 8192, 16383 };
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var storage: RowInputs.DecodeStorage = .{};
    var staged = try storage.inputs(positions.len);
    try staged.fill(&positions, &tokens, 1, 77440);
    var allocated = try RowInputs.init(std.testing.allocator, &positions, &tokens, 1, 77440);
    defer allocated.deinit(std.testing.allocator);
    try std.testing.expectEqualSlices(i64, allocated.positions, staged.positions);
    try std.testing.expectEqualSlices(i32, allocated.dense_len, staged.dense_len);
    try std.testing.expectEqualSlices(f32, allocated.cos, staged.cos);
    try std.testing.expectEqualSlices(f32, allocated.sin, staged.sin);
    try std.testing.expectEqualSlices(i32, allocated.ids, staged.ids);
    try std.testing.expectEqualSlices(i32, allocated.owners, staged.owners);
    var small = try storage.inputs(1);
    try std.testing.expectError(error.RowInputsTooSmall, small.fill(&positions, &tokens, 1, 77440));
}

test "chunk plan depends on absolute positions and is invariant to input batching" {
    const positions = [_]usize{ 0, 2, 4095, 4096, 4097, 8191, 8192 };
    var a: [7]Chunk = undefined;
    var b: [7]Chunk = undefined;
    const an = try chunkPlan(&positions, 4096, &a);
    const first = try chunkPlan(positions[0..3], 4096, &b);
    const second = try chunkPlan(positions[3..], 4096, b[first..]);
    try std.testing.expectEqual(@as(usize, 3), an);
    try std.testing.expectEqual(@as(usize, 3), first + second);
    try std.testing.expectEqualSlices(Chunk, a[0..an], &.{ b[0], b[first], b[first + 1] });
    var prompt: [9000]usize = undefined;
    for (&prompt, 0..) |*p, i| p.* = i;
    try std.testing.expectEqual(Chunk{ .start = 0, .end = 4096 }, try chunkRows(&prompt, 4096, 0));
    try std.testing.expectEqual(Chunk{ .start = 8192, .end = 9000 }, try chunkRows(&prompt, 4096, 8192));
    try std.testing.expectEqual(Chunk{ .start = 100, .end = 4096 }, try chunkRows(&prompt, 4096, 100));
}

test "prefill accepts fixed chunks through 8192 rows and rejects larger tiles" {
    try std.testing.expectEqual(Chunk{ .start = 8192, .end = 16384 }, try chunkAt(16383, max_prefill_chunk_rows));
    try std.testing.expectError(error.BadChunk, chunkAt(0, max_prefill_chunk_rows + 1));
}

test "rank-local slots convert explicitly to global positions at the WP9 seam" {
    const pos = [_]i32{ 4, -1 };
    const slots = [_]i32{ 2, -1 };
    var out: [2]LocalToken = undefined;
    try rankLocalToMla(0, &pos, &slots, &out);
    try std.testing.expectEqual(@as(i32, 4), out[0].global);
    try std.testing.expectError(error.TokenOwnershipMismatch, rankLocalToMla(1, &pos, &slots, &out));
}

test "COLL_FUSE partial plan sends the peer head block and preserves sendRecv merge inputs" {
    const rows = 16;
    const block = rows * state.decode_partial_row_bytes;
    const part_scratch: usize = @intCast(state.workspaceLayout(.{ .decode_rows = rows, .piece_rows = 512 }, 8192).sizes.part);
    try std.testing.expect(5 * block <= part_scratch);
    for (0..world) |rank| {
        const fuse_plan = collFusePartialPlan(rank, block);
        try std.testing.expectEqual((1 - rank) * block, fuse_plan.send);
        try std.testing.expectEqual(3 * block, fuse_plan.pair);
        try std.testing.expectEqual((3 + (1 - rank)) * block, fuse_plan.peer);
        try std.testing.expectEqual(5 * block, fuse_plan.need);

        var mine: [5]u8 = undefined;
        var peer_mine: [5]u8 = undefined;
        mine[0] = @intCast(rank * 2);
        mine[1] = @intCast(rank * 2 + 1);
        peer_mine[0] = @intCast((1 - rank) * 2);
        peer_mine[1] = @intCast((1 - rank) * 2 + 1);
        var gathered: [2]u8 = undefined;
        gathered[rank] = mine[(fuse_plan.send / block)];
        gathered[1 - rank] = peer_mine[rank];
        try std.testing.expectEqual(@as(u8, @intCast(rank * 2 + rank)), mine[rank]);
        try std.testing.expectEqual(@as(u8, @intCast((1 - rank) * 2 + rank)), gathered[1 - rank]);
        try std.testing.expectEqual(gathered[1 - rank], peer_mine[rank]);
    }
}

test "profile stage capacities bracket every decode collective fixture entry" {
    const Fixture = struct {
        active: ?Stage = null,
        seen: [stage_count]usize = @splat(0),
        fn begin(self: *@This(), stage: Stage) !void {
            if (self.active != null) return error.NestedCollectiveStage;
            self.active = stage;
        }
        fn collective(self: *@This(), stage: Stage) !void {
            if (self.active != stage) return error.UnprofiledCollective;
            self.seen[@intFromEnum(stage)] += 1;
        }
        fn end(self: *@This(), stage: Stage) !void {
            if (self.active != stage) return error.MismatchedCollectiveStage;
            self.active = null;
        }
    };
    var fixture: Fixture = .{};
    const collectives = [_]Stage{ .attn_comm, .attn_comm, .attn_comm, .attn_comm, .attn_comm, .moe_comm, .dense_comm };
    for (collectives) |stage| {
        try fixture.begin(stage);
        try fixture.collective(stage);
        try fixture.end(stage);
    }
    try std.testing.expectEqual(@as(usize, 5), fixture.seen[@intFromEnum(Stage.attn_comm)]);
    try std.testing.expectEqual(@as(usize, 1), fixture.seen[@intFromEnum(Stage.moe_comm)]);
    try std.testing.expectEqual(@as(usize, 1), fixture.seen[@intFromEnum(Stage.dense_comm)]);
    try std.testing.expectEqual(@as(?Stage, null), fixture.active);
    try std.testing.expectEqual(@as(usize, 10), stagePairs(.attn_pre));
    try std.testing.expectEqual(@as(usize, 16), stagePairs(.attn_comm));
    try std.testing.expectEqual(@as(usize, 16), stagePairs(.moe_comm));
    try std.testing.expectEqual(@as(usize, 3), stagePairs(.attn_post));
}

test "force semantic analysis of every forward declaration" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Glue);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(Collective);
    std.testing.refAllDecls(Upload);
    std.testing.refAllDecls(HeadMode);
    std.testing.refAllDecls(ExchangePlan);
    std.testing.refAllDecls(RowInputs);
    std.testing.refAllDecls(RowInputs.DecodeStorage);
    std.testing.refAllDecls(Forward);
}

test "embedRowFor TP3 boundary tokens" {
    // Rank 0 (vocab_local = 51712): tokens 0..51711
    const r0_t0 = embedRowFor(0, 0, 51712);
    try std.testing.expectEqual(@as(i32, 0), r0_t0.owner);
    try std.testing.expectEqual(@as(i32, 0), r0_t0.local);

    const r0_last = embedRowFor(51711, 0, 51712);
    try std.testing.expectEqual(@as(i32, 0), r0_last.owner);
    try std.testing.expectEqual(@as(i32, 51711), r0_last.local);

    // Rank 1 (vocab_local = 51584): tokens 51712..103295
    const r1_first = embedRowFor(51712, 1, 51584);
    try std.testing.expectEqual(@as(i32, 1), r1_first.owner);
    try std.testing.expectEqual(@as(i32, 0), r1_first.local);

    const r1_last = embedRowFor(103295, 1, 51584);
    try std.testing.expectEqual(@as(i32, 1), r1_last.owner);
    try std.testing.expectEqual(@as(i32, 51583), r1_last.local);

    // Rank 2 (vocab_local = 51584): tokens 103296..154879
    const r2_first = embedRowFor(103296, 2, 51584);
    try std.testing.expectEqual(@as(i32, 2), r2_first.owner);
    try std.testing.expectEqual(@as(i32, 0), r2_first.local);

    const r2_last = embedRowFor(154879, 2, 51584);
    try std.testing.expectEqual(@as(i32, 2), r2_last.owner);
    try std.testing.expectEqual(@as(i32, 51583), r2_last.local);

    // Token owned by another rank gives local 0
    const r0_on_r1 = embedRowFor(51712, 0, 51712);
    try std.testing.expectEqual(@as(i32, 1), r0_on_r1.owner);
    try std.testing.expectEqual(@as(i32, 0), r0_on_r1.local);
}
