// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 CUDA per-rank cache planes, prefill stage plane and forward workspace (WP9/WP13).
//! The workspace is carved like cuda/exl3_experts.zig scratch() (256-byte aligned regions, one allocation).
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const mla = @import("mla.zig");
const index = @import("index.zig");
const sampler = @import("sampler.zig");
/// The low-rank decode split count of moe.zig (`decode_splits`); moe.zig is a separate build module, so it is mirrored here.
const lowrank_decode_splits: u64 = 128;
const types = @import("types");

pub const CacheFormat = enum { bf16, fp8 };
pub const PrefillGemm = enum { mma, bf16 };
/// TF_GLM53_PRE_FUSE (decode) and TF_GLM53_PREP_FUSE (prefill): grouped pre-attention projections. Defined here so
/// Geometry can carry them; forward.zig re-exports them.
pub const PreFuse = enum { off, on, prefetch };
pub const PrepFuse = enum { off, on, split };

pub fn prefillGemmSelected(raw: ?[]const u8) error{UnknownPrefillGemm}!PrefillGemm {
    const value = raw orelse return .mma;
    if (std.mem.eql(u8, value, "mma")) return .mma;
    if (std.mem.eql(u8, value, "bf16")) return .bf16;
    return error.UnknownPrefillGemm;
}

pub fn prefillGemmFromEnv() error{UnknownPrefillGemm}!PrefillGemm {
    const raw = std.c.getenv("TF_GLM53_PREFILL_GEMM") orelse return .mma;
    return prefillGemmSelected(std.mem.span(raw));
}

test "prefill projection backend defaults to MMA and accepts the BF16 override" {
    try std.testing.expectEqual(PrefillGemm.mma, try prefillGemmSelected(null));
    try std.testing.expectEqual(PrefillGemm.mma, try prefillGemmSelected("mma"));
    try std.testing.expectEqual(PrefillGemm.bf16, try prefillGemmSelected("bf16"));
    try std.testing.expectError(error.UnknownPrefillGemm, prefillGemmSelected("cutlass"));
}

test "sample candidate K defaults to 256 and rejects invalid or oversized widths" {
    try std.testing.expectEqual(@as(usize, 256), try parseSampleK(null));
    try std.testing.expectEqual(@as(usize, 128), try parseSampleK("128"));
    try std.testing.expectError(error.BadSampleK, parseSampleK("0"));
    try std.testing.expectError(error.BadSampleK, parseSampleK("2049"));
}
pub const dcp_world = 2;
pub fn dcpWorld(world: usize) usize {
    return if (world == 3) 1 else 2;
}
pub const default_chunk_rows = 4096;
pub const dense_limit = index.dense_limit;
pub const kv_row_bytes = (mla.latent_dim + mla.rope_dim) * 2;
pub const latent_row_bytes = mla.latent_dim * 2;
pub const rope_row_bytes = mla.rope_dim * 2;
pub const index_row_bytes = index.dim * 2;

pub fn parkBytes(layers: usize, full_index_layers: usize, rows: usize) u64 {
    const latent: u64 = @intCast(mla.latent_dim * 2 + mla.rope_dim * 2);
    const indexer: u64 = @intCast(index.dim * 2);
    return @as(u64, @intCast(rows / 2)) * (@as(u64, @intCast(layers)) * latent + @as(u64, @intCast(full_index_layers)) * indexer);
}

pub const ParkSegment = struct { plane: u64, park: u64, len: u64 };

pub fn parkPlaneCount(layers: usize, full: usize) usize {
    return 2 * layers + full;
}

pub fn parkSegment(i: usize, layers: usize, full: usize, slots: usize, cap: usize, bases: [3]u64, park_base: u64, first: usize, count: usize, tail_slot: usize) ParkSegment {
    const latent_row: usize = mla.latent_dim * 2;
    const rope_row: usize = mla.rope_dim * 2;
    const index_row: usize = index.dim * 2;
    var base = bases[0];
    var ord = i;
    var row = latent_row;
    var block: u64 = 0;
    if (i < layers) {
        block = 0;
    } else if (i < 2 * layers) {
        base = bases[1];
        ord = i - layers;
        row = rope_row;
        block = @as(u64, @intCast(layers * cap * latent_row));
    } else {
        base = bases[2];
        ord = i - 2 * layers;
        row = index_row;
        block = @as(u64, @intCast(layers * cap * latent_row + layers * cap * rope_row));
    }
    _ = full;
    const plane = base + @as(u64, @intCast(ord * slots * row)) + @as(u64, @intCast(first * row));
    const park = if (tail_slot != 0) base + @as(u64, @intCast(ord * slots * row)) + @as(u64, @intCast(tail_slot * row)) else park_base + block + @as(u64, @intCast(ord * cap * row));
    return .{ .plane = plane, .park = park, .len = @as(u64, @intCast(count * row)) };
}

/// Historical digest value retained as the default for settings identity.
pub const graphs_runtime_placeholder: u64 = 1 << 30;
pub const default_graph_mib: u64 = 64;
pub const default_runtime_reserve_mib: u64 = 2560;
pub const default_serve_margin_mib: u64 = 4096;
pub const default_serve_min_avail_gib: f64 = 8.0;
pub const default_serve_min_free_gib: f64 = 4.0;

/// Number of decode graph shape keys needed for a context. Graph buckets double from 2048 rows.
pub fn bucketCount(context: usize) usize {
    var rounded: usize = 1;
    while (rounded < context) : (rounded <<= 1) {}
    var count: usize = 1;
    while (rounded > 2048) : (rounded >>= 1) count += 1;
    return count;
}

pub const GraphShape = struct { rows: u32, bucket: u32 };
/// Decode widths by expected use: the full 8-row verify (drafter block / 8-row copy), the plain 1-row step, the shorter verifies, then the 12/16-row copy windows. The lanes only verify widths 1..8, 12 and 16.
pub const default_graph_rows = [_]u32{ 8, 1, 2, 3, 4, 5, 6, 7, 12, 16 };

pub fn longTierFrom(raw: ?[]const u8, context: usize) !usize {
    const value = raw orelse return if (context > 200000) 200000 else 0;
    const parsed = std.fmt.parseInt(usize, value, 10) catch return error.BadLongCtxRows;
    if (parsed == 0 or parsed >= context) return 0;
    if (parsed < 2048 or parsed % 2 != 0) return error.BadLongCtxRows;
    return parsed;
}

pub fn parkTailFits(tail_slot: usize, park_slots: usize, slots: usize, adapt_limit_slot: ?usize) bool {
    if (tail_slot + park_slots > slots) return false;
    if (adapt_limit_slot) |limit| if (tail_slot + park_slots > limit) return false;
    return true;
}

pub fn parseGraphRows(raw: ?[]const u8, verify_max: usize, out: []u32) !usize {
    if (raw == null) {
        var count: usize = 0;
        for (default_graph_rows) |rows| {
            if (rows <= verify_max) {
                if (count == out.len) return error.BadGraphRows;
                out[count] = rows;
                count += 1;
            }
        }
        return count;
    }
    const value = raw.?;
    if (value.len == 0) return error.BadGraphRows;
    var count: usize = 0;
    var iter = std.mem.splitScalar(u8, value, ',');
    while (iter.next()) |item| {
        if (item.len == 0 or count == out.len) return error.BadGraphRows;
        const parsed = std.fmt.parseInt(u32, item, 10) catch return error.BadGraphRows;
        if (parsed == 0 or parsed > verify_max) return error.BadGraphRows;
        for (out[0..count]) |seen| if (seen == parsed) return error.BadGraphRows;
        out[count] = parsed;
        count += 1;
    }
    return count;
}

pub fn graphBudget(raw: ?[]const u8) !?u64 {
    const value = raw orelse return null;
    const parsed = std.fmt.parseInt(u64, value, 10) catch return error.BadGraphBudget;
    if (parsed > 1048576) return error.BadGraphBudget;
    return parsed;
}

pub fn graphPlan(verify_max: usize, context: usize, budget_mib: u64, mib_per_graph: u64, rows: []const u32, out: []GraphShape) !usize {
    if (mib_per_graph == 0) return error.BadGraphMiB;
    const possible = rows.len * bucketCount(context);
    const budget_count: usize = @intCast(@min(budget_mib / mib_per_graph, @as(u64, @intCast(possible))));
    if (out.len < budget_count) return error.GraphPlanTooSmall;
    var count: usize = 0;
    var rounded: usize = 1;
    while (rounded < context) : (rounded <<= 1) {}
    for (rows) |row| {
        var bucket: usize = 2048;
        while (true) {
            if (count == budget_count) return count;
            out[count] = .{ .rows = row, .bucket = @intCast(bucket) };
            count += 1;
            if (bucket == rounded) break;
            bucket <<= 1;
        }
    }
    _ = verify_max;
    return count;
}

pub fn graphLiveRows(rows_n: usize, budget_mib: ?u64, mib_per_graph: u64) !usize {
    if (mib_per_graph == 0) return error.BadGraphMiB;
    if (budget_mib) |budget| {
        const budget_rows: usize = @intCast(budget / mib_per_graph);
        const live_rows: usize = @min(rows_n, budget_rows);
        return live_rows;
    }
    return rows_n;
}

pub fn graphLivePlan(context: usize, live_rows: usize, rows: []const u32, out: []GraphShape) !usize {
    if (live_rows > rows.len) return error.BadGraphRows;
    var rounded: usize = 2048;
    while (rounded < context) : (rounded <<= 1) {}
    var count: usize = 0;
    for (rows[0..live_rows]) |row| {
        var bucket: usize = 2048;
        while (true) {
            if (count == out.len) return error.GraphPlanTooSmall;
            out[count] = .{ .rows = row, .bucket = @intCast(bucket) };
            count += 1;
            if (bucket == rounded) break;
            bucket <<= 1;
        }
    }
    return count;
}

pub fn graphBytes(verify_max: usize, context: usize, mib_per_graph: u64) !u64 {
    const count = std.math.mul(u64, @intCast(verify_max), @intCast(bucketCount(context))) catch return error.BadGraphReserve;
    const mib = std.math.mul(u64, count, mib_per_graph) catch return error.BadGraphReserve;
    return std.math.mul(u64, mib, 1 << 20) catch return error.BadGraphReserve;
}

pub fn graphMiB(raw: []const u8) !u64 {
    return std.fmt.parseInt(u64, raw, 10) catch return error.BadGraphMiB;
}

pub fn graphMiBFromEnv() !u64 {
    const raw = std.c.getenv("TF_GLM53_GRAPH_MIB") orelse return default_graph_mib;
    return graphMiB(std.mem.span(raw));
}

pub fn memslimFrom(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadMemslim;
}

pub fn memslimFromEnv() !bool {
    const raw = std.c.getenv("TF_GLM53_MEMSLIM");
    return memslimFrom(if (raw) |value| std.mem.span(value) else null);
}

/// TF_GLM53_GRAPH_RESERVE_MIB overrides computed graph bytes; otherwise graph count × TF_GLM53_GRAPH_MIB is used.
pub fn graphsReserveFromEnv(verify_max: usize, context: usize) !u64 {
    const mib_per_graph = try graphMiBFromEnv();
    if (try memslimFromEnv()) {
        if (std.c.getenv("TF_GLM53_GRAPH_RESERVE_MIB") != null) return error.GraphLiveAndReserve;
        const budget = if (std.c.getenv("TF_GLM53_GRAPH_BUDGET_MIB")) |raw| try graphBudget(std.mem.span(raw)) else null;
        var rows_buf: [16]u32 = undefined;
        const rows_n = try parseGraphRows(if (std.c.getenv("TF_GLM53_GRAPH_ROWS")) |v| std.mem.span(v) else null, verify_max, &rows_buf);
        const live = try graphLiveRows(rows_n, budget, mib_per_graph);
        return @as(u64, @intCast(live)) * mib_per_graph * (1 << 20);
    }
    if (std.c.getenv("TF_GLM53_GRAPH_BUDGET_MIB")) |raw| {
        if (std.c.getenv("TF_GLM53_GRAPH_RESERVE_MIB") != null) return error.GraphBudgetAndReserve;
        const budget = try graphBudget(std.mem.span(raw)) orelse 0;
        var rows_buf: [16]u32 = undefined;
        const rows_n = try parseGraphRows(if (std.c.getenv("TF_GLM53_GRAPH_ROWS")) |v| std.mem.span(v) else null, verify_max, &rows_buf);
        var shapes: [128]GraphShape = undefined;
        const n = try graphPlan(verify_max, context, budget, mib_per_graph, rows_buf[0..rows_n], &shapes);
        return @as(u64, @intCast(n)) * mib_per_graph * (1 << 20);
    }
    if (std.c.getenv("TF_GLM53_GRAPH_RESERVE_MIB")) |raw| return graphsReserveMiB(std.mem.span(raw));
    return graphBytes(verify_max, context, mib_per_graph);
}

pub fn graphsReserveMiB(raw: []const u8) !u64 {
    const mib = std.fmt.parseInt(u64, raw, 10) catch return error.BadGraphReserve;
    return std.math.mul(u64, mib, 1 << 20) catch return error.BadGraphReserve;
}

pub fn runtimeReserveMiB(raw: []const u8) !u64 {
    const mib = std.fmt.parseInt(u64, raw, 10) catch return error.BadRuntimeReserve;
    return std.math.mul(u64, mib, 1 << 20) catch return error.BadRuntimeReserve;
}

pub fn runtimeReserveFromEnv() !u64 {
    const raw = std.c.getenv("TF_GLM53_RUNTIME_RESERVE_MIB");
    return runtimeReserveFrom(if (raw) |v| std.mem.span(v) else null);
}

pub fn runtimeReserveFrom(raw: ?[]const u8) !u64 {
    return if (raw) |value| runtimeReserveMiB(value) else default_runtime_reserve_mib * (1 << 20);
}

pub fn serveMarginBytes(raw: ?[]const u8) !u64 {
    const mib = if (raw) |v| (std.fmt.parseInt(u64, v, 10) catch return error.BadServeMargin) else default_serve_margin_mib;
    if (mib > 1048576) return error.BadServeMargin;
    return mib * (1 << 20);
}

pub fn serveMarginFromEnv() !u64 {
    if (std.c.getenv("TF_GLM53_SERVE_MARGIN_GIB") != null) {
        std.log.err("TF_GLM53_SERVE_MARGIN_GIB is now TF_GLM53_SERVE_MARGIN_MIB (MiB, integer)", .{});
        return error.ServeMarginGibRenamed;
    }
    const raw = std.c.getenv("TF_GLM53_SERVE_MARGIN_MIB");
    return serveMarginBytes(if (raw) |v| std.mem.span(v) else null);
}

fn serveThresholdBytes(raw: ?[]const u8, default_gib: f64, comptime err: anyerror) !u64 {
    const value = if (raw) |v| (std.fmt.parseFloat(f64, v) catch return err) else default_gib;
    if (!std.math.isFinite(value) or value < 0 or value > 1024) return err;
    return @intFromFloat(value * 1024.0 * 1024.0 * 1024.0);
}

pub fn serveMinAvailableBytes(raw: ?[]const u8) !u64 {
    return serveThresholdBytes(raw, default_serve_min_avail_gib, error.BadServeMinAvail);
}

pub fn serveMinFreeBytes(raw: ?[]const u8) !u64 {
    return serveThresholdBytes(raw, default_serve_min_free_gib, error.BadServeMinFree);
}

pub fn serveMinAvailableFromEnv(total: u64) !u64 {
    _ = total;
    const raw = std.c.getenv("TF_GLM53_SERVE_MIN_AVAIL_GIB");
    return serveMinAvailableBytes(if (raw) |v| std.mem.span(v) else null);
}

pub fn serveMinFreeFromEnv(total: u64) !u64 {
    _ = total;
    const raw = std.c.getenv("TF_GLM53_SERVE_MIN_FREE_GIB");
    return serveMinFreeBytes(if (raw) |v| std.mem.span(v) else null);
}

pub const GateVerdict = enum { ok, low_available, low_free };

/// Serving gate gives the hard free-memory stop precedence over the softer available-memory target.
pub fn serveGate(avail: u64, free: u64, min_avail: u64, min_free: u64) GateVerdict {
    if (free < min_free) return .low_free;
    if (avail < min_avail) return .low_available;
    return .ok;
}

/// Allocation geometry. Admission is separate (admitRank), done once per rank before weights and State.
pub const Geometry = struct {
    world: usize = 2,
    layers: usize = 78,
    full_index_layers: usize = 21,
    cache_format: CacheFormat = .bf16,
    chunk_rows: usize = default_chunk_rows,
    /// Attention and FFN run in pieces of this many rows inside a chunk, cut at absolute multiples (row-local
    /// kernels, so the bits do not depend on it; only the workspace does).
    piece_rows: usize = 512,
    /// DtoD side-call park capacity in positions; zero disables the allocation.
    park_rows: usize = 0,
    park_tail_slot: usize = 0,
    index_slots: usize = 131073,
    decode_rows: usize = 8,
    /// Decode graph pool reservation, computed from verify rows and context buckets by native setup.
    graphs_bytes: u64 = 0,
    /// Runtime modules, CUDA context and communication reserve; not part of State's device arenas.
    runtime_bytes: u64 = 0,
    /// Maximum drafter work buffer overlaid on the forward phase union when TF_GLM53_DRAFTER_SHARE=1.
    drafter_share_bytes: u64 = 0,
    /// Indexer scoring rows per sub-block (index scores are [score_rows][index_slots] fp32).
    score_rows: usize = 64,
    /// Index MMA query rows grouped into one CTA (4, 8, or 16), captured once at setup.
    index_mma_rows: usize = 4,
    /// Routed-expert rows per WP4 grouped call (EXL3 scratch scales with it).
    expert_rows: usize = 128,
    /// Borrowed KV tails temporarily host larger prompt expert batches; zero disables it.
    expert_rows_adapt: usize = 0,
    expert_rows_adapt_maxpos: usize = 0,
    /// Head rows per call (decode window, or gate-mode prefill blocks; WP10 launches at most 64).
    head_rows: usize = 8,
    /// Largest per-rank candidate list (WP10 cand kernel, k <= 2048).
    candidate_k: usize = 64,
    /// Rank-local vocabulary rows (154,880 / 2).
    vocab_local: usize = 77440,
    /// The stage fetch on a side stream (Forward.prefetch) needs two stage pairs; in order on one stream, one pair
    /// serves every layer (the fetch for layer L+1 runs after layer L's attend).
    prefetch: bool = false,
    /// Reserve two half-piece collective destinations and the ordered communication stream for opt-in TBO.
    tbo: bool = false,
    shard_qkv_a: bool = false,
    /// Overlap grouped-prefill attention gathers with the next piece's attention.
    piece_pipe: bool = false,
    /// Overlap grouped-prefill MoE all-gathers on the communication stream.
    moe_pipe: bool = false,
    /// Trim generic expert scratch to decode rows and carve dedicated prompt scratch.
    ws_trim: bool = false,
    /// Fuse grouped prompt combine into the MoE communication pack.
    moe_post: bool = false,
    /// Use highest CUDA priority for the communication stream.
    comm_prio: bool = false,
    /// TF_GLM53_INDEX_MERGE=2 selects the exact linear candidate merge.
    index_merge: u8 = 0,
    /// TF_GLM53_INDEX_PIPE=1 expands index scratch to a piece and allocates gather ordering events.
    index_pipe: bool = false,
    /// Prompt MLA tile choice captured once during setup and carried into Forward.
    prompt_mla: mla.PromptMlaMode = .off,
    /// SMLA-P prefill attend choice captured once during setup and carried into Forward.
    smla_p: bool = false,
    /// TF_GLM53_SMLA_FP8: lossy e4m3 score GEMM inside SMLA-P (tf_glm53_smla_p_fp8), captured once during setup and
    /// carried into Forward; setup refuses it without smla_p. No workspace change (the KV cache stays bf16).
    smla_fp8: bool = false,
    /// TF_GLM53_SMLA_P3: SMLA-P with precomputed step unions (tf_glm53_smla_union + tf_glm53_smla_p3, bit-identical),
    /// captured once during setup and carried into Forward; setup refuses it without smla_p or with smla_fp8. Carves the
    /// union table (mla.smlaP3TableBytes(piece rows)) at the end of the attention phase; off carves nothing.
    smla_p3: bool = false,
    /// TF_GLM53_PREFILL_W8A8=1 captured once during setup and carried into Forward (prefill o_proj in W8A8).
    /// It does not change allocation geometry: the e4m3 rows and scales reuse the attention partials (`part`).
    prefill_w8a8: bool = false,
    /// no workspace effect; carried to Forward
    pre_fuse: PreFuse = .off,
    /// no workspace effect; carried to Forward
    prep_fuse: PrepFuse = .off,
    /// Allocate a dedicated dequant stream/event when `TF_GLM53_PREFILL_GEMM=bf16` is selected at engine startup.
    prefill_bf16: bool = false,
    /// Reuse expert decode scratch for prompt scratch when lifetimes permit.
    scratch_overlay: bool = false,
    /// TF_GLM53_MEMSLIM=1 with the overlay and prompt experts: generic expert scratch and lowrank_mid sized for decode rows (as WS_TRIM), prompt scratch stays overlaid. Forward refuses it unless every MoE layer takes the prompt kernels.
    slim_generic: bool = false,
    /// Opt in to coalesced decode/verify MLA kernels; it does not change allocation geometry.
    decode_mla: bool = false,
    decode_mla_v2: bool = false,
    decode_mla_v3: bool = false,
    /// Allocate the decode side stream and its four ordering events when enabled at engine startup.
    decode_side: bool = false,
    /// Reserve prompt tensor-core routed-expert scratch when TF_GLM53_PROMPT_EXPERTS=1 or 2 at startup.
    prompt_experts: bool = false,
    /// Reserve a decode-only packed query staging region for TF_GLM53_COLL_FUSE=1.
    coll_fuse: bool = false,
    /// TF_GLM53_PROMPT_EXPERTS=2 at startup: the prompt path runs the 128-member wide kernels (same carve and bits).
    prompt_experts_wide: bool = false,
    prompt_experts_pmx: bool = false,
    /// TF_GLM53_PROMPT_EXPERTS=10: =3 plus the glm53_p5a small-item kernels (same carve and bits).
    prompt_experts_p5a: bool = false,
    /// qmmf slices meet in a cluster (sm_90+, GB10 is sm_121), so no split-K partial buffer is needed. The forward
    /// refuses a lane without clusters for a workspace carved this way.
    lane_clusters: bool = true,
};

pub fn validateMoePipeGeometry(g: Geometry) !void {
    if (!g.moe_pipe) return;
    const pr: u64 = @intCast(@max(@min(g.piece_rows, @max(g.chunk_rows, g.decode_rows)), g.decode_rows));
    if (!groupedExperts(g.expert_rows, @intCast(pr))) return error.MoePipeRequiresGroupedExperts;
    const pieces = (g.expert_rows + @as(usize, @intCast(pr)) - 1) / @as(usize, @intCast(pr));
    if (pieces > Workspace.max_moe_pipe_pieces) return error.MoePipeTooManyPieces;
    const stride = std.math.mul(u64, pr, hidden * 2) catch return error.MoePipeRingTooLarge;
    const slot_bytes = std.math.mul(u64, stride, if (g.moe_post) g.world else 1 + g.world) catch return error.MoePipeRingTooLarge;
    _ = std.math.mul(u64, slot_bytes, Workspace.moe_pipe_ring_slots) catch return error.MoePipeRingTooLarge;
}

/// Device allocations are made in a few arenas (weights by class, KV planes, stage, workspace), each rounded once
/// to this granularity, so admitted bytes are the bytes CUDA allocates (assumed cuMemAlloc page; Spark-measure).
pub const arena_granularity: u64 = 2 << 20;

pub fn arenaBytes(n: u64) u64 {
    return std.mem.alignForward(u64, n, arena_granularity);
}

/// Placement alignment of each tensor inside an arena (at least what every kernel's vector loads need).
pub const tensor_alignment: u64 = 256;

pub const PlaneBytes = struct { latent: u64, rope: u64, index: u64, slots: usize };

/// Position p lives at local slot p / 2 on rank p % 2. One extra slot permits the design's sentinel row.
pub fn planeBytes(capacity: usize, layers: usize, full_index_layers: usize, format: CacheFormat) PlaneBytes {
    return planeBytesDcp(capacity, layers, full_index_layers, format, dcp_world);
}

/// The planes of `g`: TP=3 runs DCP=1, so every rank holds every position (slot p) and needs the full window.
pub fn planeBytesFor(g: Geometry, capacity: usize) PlaneBytes {
    return planeBytesDcp(capacity, g.layers, g.full_index_layers, g.cache_format, dcpWorld(g.world));
}

/// Rank-local cache rows for a window: DCP=2 splits positions by parity, DCP=1 keeps all of them.
pub fn localSlots(capacity: usize, dcp_w: usize) usize {
    return if (dcp_w == 1) capacity + 2 else (capacity + 1) / 2 + 1;
}

/// The largest window DCP=1 can serve: one rank's indexer scores every key, bounded by its slot table and top-k.
pub fn dcp1MaxContext(g: Geometry) usize {
    return @min(g.index_slots, index.max_local_keys);
}

pub fn planeBytesDcp(capacity: usize, layers: usize, full_index_layers: usize, format: CacheFormat, dcp_w: usize) PlaneBytes {
    const slots = localSlots(capacity, dcp_w);
    const kv_element_bytes: u64 = if (format == .bf16) 2 else 1;
    const index_element_bytes: u64 = if (format == .bf16) 2 else 1;
    return .{
        .latent = @as(u64, @intCast(layers)) * slots * mla.latent_dim * kv_element_bytes,
        .rope = @as(u64, @intCast(layers)) * slots * mla.rope_dim * kv_element_bytes,
        .index = @as(u64, @intCast(full_index_layers)) * slots * index.dim * index_element_bytes,
        .slots = slots,
    };
}

pub fn owner(position: usize) usize {
    return position % dcp_world;
}
pub fn localSlot(position: usize) usize {
    return position / dcp_world;
}

/// The prefill stage plane is types.StagePlanes (the shared contract, round-6/7 review decision): pairs of
/// latent 512 + rope 64 bf16 rows shared by every layer (layer L uses pair L % 2), each pair
/// StagePlanes.rowsForCapacity(context, 2) rows, mirroring the peer's keys for every position a chunk attends to.
/// Two pairs with a prefetch stream (0.28 GiB at 262k); one pair otherwise (both buffer slots name it, 0.14 GiB).
pub const stage_pairs = 2;

pub fn stagePairsFor(geometry: Geometry) usize {
    return if (geometry.prefetch) 2 else 1;
}

pub fn stageRows(context: usize) usize {
    return types.StagePlanes.rowsForCapacity(context, dcp_world) catch unreachable;
}

/// The single sizing rule for the stage plane allocation (both pairs).
pub fn stagePlaneBytes(geometry: Geometry, context: usize) u64 {
    return @as(u64, @intCast(stagePairsFor(geometry) * stageRows(context))) * kv_row_bytes;
}

/// The shared StagePlanes carved from one allocation at `base`: pair b = latent [rows][512], then rope [rows][64].
/// The planes borrow the allocation (State owns it).
pub fn stagePlanes(driver: *const cuda.Driver, base: u64, context: usize, pairs: usize) types.StagePlanes {
    const rows = stageRows(context);
    var out: types.StagePlanes = .{ .buffers = undefined, .rows = rows, .capacity = context, .world = dcp_world };
    for (&out.buffers, 0..) |*pair, b| {
        const latent = base + @as(u64, @intCast((b % pairs) * rows * kv_row_bytes));
        const rope = latent + @as(u64, @intCast(rows * mla.latent_dim * 2));
        pair.* = .{
            .latent = .{ .data = .{ .d = driver, .ptr = latent, .len = rows * mla.latent_dim * 2, .borrowed = true }, .dtype = .bf16, .rows = rows, .cols = mla.latent_dim, .stride = mla.latent_dim },
            .rope = .{ .data = .{ .d = driver, .ptr = rope, .len = rows * mla.rope_dim * 2, .borrowed = true }, .dtype = .bf16, .rows = rows, .cols = mla.rope_dim, .stride = mla.rope_dim },
        };
    }
    return out;
}

/// types.StagePlanes.validate (extent, dtype, layout) plus round-7 rule 6 when a prefetch stream runs: no two of
/// the four planes may overlap, so fetch(L+1) into one pair can never write rows attend(L) reads from the other.
/// Without a prefetch the two slots may name one pair (fetch and attend are in order on one stream).
pub fn validateStage(p: *const types.StagePlanes, prefetch: bool) !void {
    try p.validate();
    if (!prefetch) {
        for (p.buffers) |pair| if (pair.latent.data.ptr == 0 or pair.rope.data.ptr == 0) return error.MissingStagePlane;
        return;
    }
    var spans: [4][2]u64 = undefined;
    for (p.buffers, 0..) |pair, b| {
        spans[2 * b] = .{ pair.latent.data.ptr, pair.latent.data.ptr + pair.latent.data.len };
        spans[2 * b + 1] = .{ pair.rope.data.ptr, pair.rope.data.ptr + pair.rope.data.len };
    }
    for (spans, 0..) |x, i| {
        if (x[0] == 0 or x[1] <= x[0]) return error.MissingStagePlane;
        for (spans[i + 1 ..]) |y| if (x[0] < y[1] and y[0] < x[1]) return error.StageBuffersAliased;
    }
}

/// Device addresses of a stage pair (the raw pointers the stage fetch sends into).
pub fn stageLatent(p: *const types.StagePlanes, layer: usize) u64 {
    return p.forLayer(layer).latent.data.ptr;
}
pub fn stageRope(p: *const types.StagePlanes, layer: usize) u64 {
    return p.forLayer(layer).rope.data.ptr;
}

/// Per-rank weight bytes by class: weights.planRank computes them exactly from the shard headers and the share
/// rules (weights.ModelWeights.classes gives the loaded bytes, which must equal them). The drafter counts only
/// when one is loaded.
pub const WeightClasses = struct {
    routed: u64 = 0,
    backbone_fp8: u64 = 0,
    embed_head: u64 = 0,
    router_norms: u64 = 0,
    lowrank: u64 = 0,
    drafter: ?u64 = null,

    pub fn total(w: WeightClasses) u64 {
        return w.routed + w.backbone_fp8 + w.embed_head + w.router_norms + w.lowrank + (w.drafter orelse 0);
    }

    /// Design §5's per-rank figures (estimates), for planning tables before a checkpoint is at hand.
    pub fn designEstimate() WeightClasses {
        return .{ .routed = gibBytes(86.00), .backbone_fp8 = gibBytes(8.39), .embed_head = gibBytes(1.77), .router_norms = gibBytes(0.23), .lowrank = gibBytes(0.06) };
    }
};

pub fn gibBytes(v: f64) u64 {
    return @intFromFloat(v * 1024.0 * 1024.0 * 1024.0);
}

pub fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

/// Every row of a rank's memory, for the decision table printed at admission.
pub const Breakdown = struct {
    weights: WeightClasses,
    latent_rope: u64,
    index_keys: u64,
    stage: u64,
    workspace: u64,
    expert_batch: u64,
    prefill_weight: u64,
    park: u64 = 0,
    graphs: u64,
    runtime: u64 = 0,
    /// State's arenas as allocated (each rounded once to arena_granularity).
    state_arenas: u64,

    /// What a rank allocates: weights (arena-rounded when they come from planRank), State's arenas, the runtime.
    pub fn total(b: Breakdown) u64 {
        return b.weights.total() + b.state_arenas + b.graphs + b.runtime;
    }

    pub fn logicalState(b: Breakdown) u64 {
        return b.latent_rope + b.index_keys + b.stage + b.workspace + b.prefill_weight + b.park;
    }

    pub fn report(b: Breakdown, context: usize) void {
        const w = b.weights;
        std.debug.print("GLM-5.3 CUDA per-rank memory at context {d} (GiB)\n", .{context});
        const rows = .{
            .{ "weights: routed EXL3 experts (halves)", w.routed },
            .{ "weights: FP8 backbone (.fp8b codes + scales)", w.backbone_fp8 },
            .{ "weights: embed half + lm_head half (bf16)", w.embed_head },
            .{ "weights: router, norms, bias, weights_proj", w.router_norms },
            .{ "weights: low-rank (fp32, R/2)", w.lowrank },
        };
        inline for (rows) |r| std.debug.print("  {s:<48} {d:>8.2}\n", .{ r[0], gib(r[1]) });
        if (w.drafter) |d| std.debug.print("  {s:<48} {d:>8.2}\n", .{ "drafter (DFlash2 + draft head + ring + host allocations)", gib(d) }) else std.debug.print("  {s:<48} {s:>8}\n", .{ "drafter", "not loaded" });
        const more = .{
            .{ "KV planes: latent + rope (DCP2, bf16)", b.latent_rope },
            .{ "indexer keys (DCP2, bf16, full layers)", b.index_keys },
            .{ "prefill stage pairs (ceil(ctx/2) rows each)", b.stage },
            .{ "forward workspace (carved, union of phases)", b.workspace },
            .{ "expert batch workspace increment", b.expert_batch },
            .{ "prefill BF16 weight scratch (largest projection)", b.prefill_weight },
            .{ "decode graph reservation", b.graphs },
            .{ "runtime: CUDA context, module images, load gap (measured)", b.runtime },
        };
        inline for (more) |r| std.debug.print("  {s:<48} {d:>8.2}\n", .{ r[0], gib(r[1]) });
        if (b.park != 0) std.debug.print("  {s:<48} {d:>8.2}\n", .{ "side park (TF_GLM53_SIDE_ROWS)", gib(b.park) });
        std.debug.print("  {s:<48} {d:>8.2} (logical {d:.4}, rounded {d:.4})\n", .{ "State arenas as allocated (2 MiB rounding)", gib(b.state_arenas), gib(b.logicalState()), gib(b.state_arenas) });
        std.debug.print("  {s:<48} {d:>8.2}\n", .{ "total (allocated)", gib(b.total()) });
    }
};

/// Logical bytes of every row (weights as given; State rows unrounded).
pub fn breakdown(geometry: Geometry, context: usize, w: WeightClasses) Breakdown {
    const planes = planeBytesFor(geometry, context);
    const lr = planes.latent + planes.rope;
    const stage = stagePlaneBytes(geometry, context);
    const prefill_weight = if (geometry.prefill_bf16) prefill_weight_scratch_bytes else 0;
    const ws = workspaceLayout(geometry, context).bytes - prefill_weight;
    var baseline = geometry;
    baseline.expert_rows = @min(@as(usize, 128), geometry.piece_rows);
    const expert_batch = workspaceLayout(geometry, context).bytes -| workspaceLayout(baseline, context).bytes;
    // KV arena: latent | rope | index keys, each placed at tensor_alignment.
    const kv = std.mem.alignForward(u64, planes.latent, tensor_alignment) + std.mem.alignForward(u64, planes.rope, tensor_alignment) + planes.index;
    const park: u64 = if (geometry.park_rows == 0 or geometry.park_tail_slot != 0) 0 else parkBytes(geometry.layers, geometry.full_index_layers, geometry.park_rows);
    const park_arena: u64 = if (park == 0) 0 else arenaBytes(park);
    return .{ .weights = w, .latent_rope = lr, .index_keys = planes.index, .stage = stage, .workspace = ws, .expert_batch = expert_batch, .prefill_weight = prefill_weight, .park = park, .graphs = geometry.graphs_bytes, .runtime = geometry.runtime_bytes, .state_arenas = arenaBytes(kv) + arenaBytes(stage) + arenaBytes(ws + prefill_weight) + park_arena };
}

/// State arenas: KV planes (latent, rope, index keys), stage pairs, workspace and optional park, each rounded once.
pub const StateArenas = struct { kv: u64, stage: u64, workspace: u64, park: u64 = 0 };

pub fn stateArenas(geometry: Geometry, context: usize) StateArenas {
    const planes = planeBytesFor(geometry, context);
    const kv = std.mem.alignForward(u64, planes.latent, tensor_alignment) + std.mem.alignForward(u64, planes.rope, tensor_alignment) + planes.index;
    const park_arena: u64 = if (geometry.park_rows == 0 or geometry.park_tail_slot != 0) 0 else arenaBytes(parkBytes(geometry.layers, geometry.full_index_layers, geometry.park_rows));
    return .{ .kv = arenaBytes(kv), .stage = arenaBytes(stagePlaneBytes(geometry, context)), .workspace = arenaBytes(workspaceLayout(geometry, context).bytes), .park = park_arena };
}

pub const AdaptTail = struct { kv_offset: u64, len: u64 };
pub const AdaptPlan = struct { limit_slot: usize, limit_position: usize, need: u64, pool: u64, tails: []AdaptTail, kv_bytes: u64 };

pub fn adaptNeedBytes(g: Geometry, context: usize) u64 {
    var gb = g;
    gb.expert_rows = g.expert_rows_adapt;
    gb.scratch_overlay = false;
    gb.prompt_experts = true;
    const ref = workspaceLayout(gb, context);
    const lowrank_mid_big = @as(u64, @intCast(8 * g.expert_rows_adapt * lowrank_max * 4));
    const sizes = [_]u64{ ref.sizes.xn, ref.sizes.picks8, ref.sizes.wts8, ref.sizes.shared_out, ref.sizes.routed_out, lowrank_mid_big, promptPlanBytes(g.expert_rows_adapt, routed_slots, routed_experts), cuda.exl3_experts.prompt.bytes(expertShape(), g.expert_rows_adapt, routed_slots, false) catch unreachable };
    var total: u64 = 0;
    for (sizes) |n| total += Workspace.align256(n);
    return total;
}

pub fn planAdapt(gpa: std.mem.Allocator, g: Geometry, context: usize, granularity: u64) !AdaptPlan {
    if (granularity == 0) return error.AdaptBadGranularity;
    const p = planeBytesFor(g, context);
    const elem: u64 = if (g.cache_format == .bf16) 2 else 1;
    const row_bytes = [_]u64{ @as(u64, mla.latent_dim) * elem, @as(u64, mla.rope_dim) * elem, @as(u64, index.dim) * elem };
    const plane_lens = [_]u64{ p.latent, p.rope, p.index };
    const bases = [_]u64{ 0, std.mem.alignForward(u64, p.latent, tensor_alignment), std.mem.alignForward(u64, std.mem.alignForward(u64, p.latent, tensor_alignment) + p.rope, tensor_alignment) };
    // Planes are layer-major (State.latentAt: base + layer * slots * row): each layer of a plane has its own tail past
    // the limit slot, and each layer's last row stays out of the lease.
    const counts = [_]u64{ @intCast(g.layers), @intCast(g.layers), @intCast(g.full_index_layers) };
    const need = adaptNeedBytes(g, context);
    if (adaptPool(1, bases, counts, row_bytes, p.slots, granularity) < need) return error.AdaptPoolTooSmall;
    var lo: usize = 1;
    var hi: usize = p.slots - 1;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        if (adaptPool(mid, bases, counts, row_bytes, p.slots, granularity) >= need) lo = mid else hi = mid - 1;
    }
    var tails: std.ArrayList(AdaptTail) = .empty;
    errdefer tails.deinit(gpa);
    for (0..3) |i| {
        if (plane_lens[i] == 0) continue;
        const stride = @as(u64, @intCast(p.slots)) * row_bytes[i];
        for (0..@intCast(counts[i])) |l| {
            const lb = bases[i] + @as(u64, @intCast(l)) * stride;
            const start = std.mem.alignForward(u64, lb + @as(u64, @intCast(lo)) * row_bytes[i], granularity);
            const end = std.mem.alignBackward(u64, lb + stride - row_bytes[i], granularity);
            if (end > start) try tails.append(gpa, .{ .kv_offset = start, .len = end - start });
        }
    }
    // A slot holds dcpWorld positions (p / 2 under DCP=2, p itself under TP=3's DCP=1).
    const per_slot = dcpWorld(g.world);
    const maxpos = if (g.expert_rows_adapt_maxpos == 0) per_slot * lo else @min(per_slot * lo, g.expert_rows_adapt_maxpos);
    return .{ .limit_slot = lo, .limit_position = maxpos, .need = need, .pool = adaptPool(lo, bases, counts, row_bytes, p.slots, granularity), .tails = try tails.toOwnedSlice(gpa), .kv_bytes = stateArenas(g, context).kv };
}

/// Bytes leasable past `slots` across every layer of the three KV planes (granularity-aligned per layer tail).
pub fn adaptPool(slots: usize, bases: [3]u64, counts: [3]u64, rows: [3]u64, plane_slots: usize, gran: u64) u64 {
    var sum: u64 = 0;
    for (0..3) |i| {
        if (counts[i] == 0) continue;
        const stride = @as(u64, @intCast(plane_slots)) * rows[i];
        for (0..@intCast(counts[i])) |l| {
            const lb = bases[i] + @as(u64, @intCast(l)) * stride;
            const start = std.mem.alignForward(u64, lb + @as(u64, @intCast(slots)) * rows[i], gran);
            const end = std.mem.alignBackward(u64, lb + stride - rows[i], gran);
            if (end > start) sum += end - start;
        }
    }
    return sum;
}

/// The bytes State.init allocates: its three arenas as rounded.
pub fn stateBytes(geometry: Geometry, context: usize) u64 {
    const a = stateArenas(geometry, context);
    return a.kv + a.stage + a.workspace + a.park;
}

pub fn parseSideRows(raw: ?[]const u8, piece_rows: usize, context: usize) !usize {
    const value = raw orelse return 0;
    if (value.len == 0 or std.mem.eql(u8, value, "0")) return 0;
    const rows = std.fmt.parseInt(usize, value, 10) catch return error.BadSideRows;
    if (piece_rows == 0 or rows < piece_rows or rows > context or rows % piece_rows != 0) return error.BadSideRows;
    return rows;
}

pub fn sideRowsFromEnv(piece_rows: usize, context: usize) !usize {
    const raw = std.c.getenv("TF_GLM53_SIDE_ROWS") orelse return 0;
    return parseSideRows(std.mem.span(raw), piece_rows, context);
}

/// What is kept free below MemAvailable: the TF_GLM53_MEM_FLOOR_GIB floor (default 4 GiB), or, only with
/// TF_GLM53_UMA_HEADROOM=1, the GB10 headroom of 2 GiB + 5% of MemTotal. Never both.
pub const Reserve = struct {
    kind: enum { floor, uma_headroom },
    bytes: u64,
    guard_floor: u64,
    serve_margin: u64,
};

pub fn umaHeadroomBytes(mem_total: u64) u64 {
    return 2 * (1 << 30) + mem_total / 20;
}

pub fn reserveFor(mem_total: u64, uma_headroom: bool, floor: u64) Reserve {
    return reserveForWithMargin(mem_total, uma_headroom, floor, 0);
}

/// Startup's additive serving margin is separate from the legacy guard reserve.
pub fn reserveForWithMargin(mem_total: u64, uma_headroom: bool, floor: u64, margin: u64) Reserve {
    const guard = if (uma_headroom) umaHeadroomBytes(mem_total) else floor;
    return .{ .kind = if (uma_headroom) .uma_headroom else .floor, .bytes = guard + margin, .guard_floor = guard, .serve_margin = margin };
}

/// The process environment's reserve: TF_GLM53_UMA_HEADROOM=1 selects the headroom, else the floor.
pub fn reserveFromEnv(mem_total: u64) !Reserve {
    const on = if (std.c.getenv("TF_GLM53_UMA_HEADROOM")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const margin = try serveMarginFromEnv();
    return reserveForWithMargin(mem_total, on, if (on) 0 else try floorFromEnv(), margin);
}

/// One rank's admission, made once before the weights load and State: exact weights + State + graphs against
/// MemAvailable minus the reserve. ModelWeights.loadConfig and State.init each verify their bytes against it.
pub const Admission = struct {
    context: usize,
    breakdown: Breakdown,
    state_bytes: u64,
    /// The unrounded State bytes, so a geometry that rounds to the same arenas is still told apart.
    state_logical: u64,
    mem_available: u64,
    reserve: Reserve,

    pub fn report(a: Admission) void {
        a.breakdown.report(a.context);
        const label = if (a.reserve.kind == .floor) "guard floor" else "UMA headroom";
        std.debug.print("  {s:<48} {d:>8.2}\n  reserve {d:.2} ({s}) + {d:.2} (serve margin)\n", .{ "MemAvailable", gib(a.mem_available), gib(a.reserve.guard_floor), label, gib(a.reserve.serve_margin) });
    }
};

pub fn admitRank(geometry: Geometry, context: usize, weights: WeightClasses, mem_available: u64, r: Reserve) !Admission {
    if (context == 0) return error.DcpShape;
    if (geometry.expert_rows > geometry.chunk_rows) {
        if (!builtin.is_test) std.log.err("expert_rows {d} exceeds chunk_rows {d}; set TF_GLM53_CHUNK_ROWS >= expert_rows on both ranks", .{ geometry.expert_rows, geometry.chunk_rows });
        return error.ExpertRowsExceedChunkRows;
    }
    if (!validExpertRows(geometry)) return error.BadExpertRows;
    const a: Admission = .{ .context = context, .breakdown = breakdown(geometry, context, weights), .state_bytes = stateBytes(geometry, context), .state_logical = breakdown(geometry, context, .{}).logicalState(), .mem_available = mem_available, .reserve = r };
    if (!builtin.is_test) a.report();
    if (a.breakdown.total() > mem_available -| r.bytes) {
        if (!builtin.is_test) std.log.err("GLM-5.3 CUDA rank needs {d:.2} GiB; MemAvailable {d:.2} GiB less the {s} {d:.2} GiB leaves {d:.2}", .{ gib(a.breakdown.total()), gib(mem_available), @tagName(r.kind), gib(r.bytes), gib(mem_available -| r.bytes) });
        return error.MemoryBudget;
    }
    return a;
}

/// Phase 1 (routed experts deferred over groups of pieces) runs only when it is asked for explicitly: more than 512
/// rows and more than one piece. Below that (and for every default) each piece runs its routed experts itself.
pub fn groupedExperts(expert_rows: usize, piece_rows: usize) bool {
    return expert_rows > 512 and expert_rows > piece_rows;
}

pub fn validExpertRows(g: Geometry) bool {
    const er = g.expert_rows;
    if (er == 0 or er > 8192) return false;
    if (er > g.chunk_rows) return false;
    if (er > g.piece_rows) return g.piece_rows != 0 and er % g.piece_rows == 0;
    return er % 16 == 0;
}

/// Parse TF_GLM53_MEM_FLOOR_GIB; unset uses the design's 4 GiB safety floor.
pub fn floorBytes(raw: ?[]const u8) !u64 {
    const gib_value = if (raw) |value| (std.fmt.parseFloat(f64, value) catch return error.BadMemoryFloor) else 4.0;
    if (!std.math.isFinite(gib_value) or gib_value < 0 or gib_value > 1024) return error.BadMemoryFloor;
    return @intFromFloat(gib_value * 1024.0 * 1024.0 * 1024.0);
}

/// The floor from the process environment (TF_GLM53_MEM_FLOOR_GIB), default 4 GiB; a malformed value is refused.
pub fn floorFromEnv() !u64 {
    const raw = std.c.getenv("TF_GLM53_MEM_FLOOR_GIB");
    return floorBytes(if (raw) |v| std.mem.span(v) else null);
}

/// GLM-5.3 shapes the workspace is carved for (the kernels' constants; config.zig refuses other shapes).
pub const hidden = 6144;
pub const q_lora = 2048;
pub const local_heads = 32;
pub const q_head = mla.nope_dim + mla.rope_dim;
pub const kv_a_width = mla.latent_dim + mla.rope_dim;
pub const index_heads = index.heads;
pub const index_dim = index.dim;
pub const top_k = index.top_k;
pub const routed_experts = 256;
pub const routed_slots = 8;
pub const route_slots = 9;
pub const dense_width = 6144;
/// Largest single FP8 backbone projection on one TP rank: dense gate/up [12288,6144], BF16 scratch.
pub const prefill_weight_scratch_bytes: u64 = 12288 * 6144 * 2;
pub const shared_width = 1024;
pub const expert_width = 1024;
pub const lowrank_max = 64;
/// One peer-only DCP attention block per decode row: bf16 partials [32][512] then fp32 LSE [32].
pub const decode_partial_row_bytes = local_heads * (mla.latent_dim * 2 + 4);

/// Largest qmmf split-K partial for any GLM-5.3 linear at `rows` rows (0 when slices meet in a cluster).
pub fn lanePartBytes(rows: usize, clusters: bool) usize {
    const shapes = [_][2]usize{ .{ q_lora, hidden }, .{ local_heads * q_head, q_lora }, .{ kv_a_width, hidden }, .{ index_dim, hidden }, .{ index_heads, hidden }, .{ index_heads * index_dim, q_lora }, .{ hidden, local_heads * mla.value_dim }, .{ 2 * dense_width, hidden }, .{ hidden, dense_width }, .{ 2 * shared_width, hidden }, .{ hidden, shared_width }, .{ 77440, hidden } };
    var most: usize = 0;
    const m = @min(rows, cuda.qmmf.fused_rows - 1);
    for (shapes) |s| {
        const p = cuda.qmmf.planForMode(m, s[0], s[1], clusters, .fp8b);
        if (p.reduce) most = @max(most, @as(usize, p.sk) * m * s[0] * 4);
    }
    return most;
}

/// WP10 candidate scratch for `rows` rows at `k` candidates (round 7: sized by k, reused per 64-row tile).
pub fn candidateScratchBytes(rows: usize, k: usize) !usize {
    return sampler.Head.candidateScratchBytes(@min(rows, max_head_rows), @max(k, 1));
}

/// WP10 launches at most this many rows of candidates; gate-mode heads loop in blocks of it.
pub const max_head_rows = 64;

/// Candidate head width used by sampled requests; each rank returns its local top K.
pub fn sampleKFromEnv() !usize {
    const raw = std.c.getenv("TF_GLM53_SAMPLE_K") orelse return 256;
    return parseSampleK(std.mem.span(raw));
}

pub fn parseSampleK(raw: ?[]const u8) !usize {
    const k = if (raw) |value| std.fmt.parseInt(usize, value, 10) catch return error.BadSampleK else 256;
    if (k == 0 or k > 2048) return error.BadSampleK;
    return k;
}

/// Routed-expert shape for WP4 scratch sizing (this rank's 1024-wide halves of 256 experts).
pub fn expertShape() cuda.exl3_experts.Layer {
    return .{ .gate = 0, .up = 0, .down = 0, .gate_k2 = 0, .up_k2 = 0, .down_k2 = 0, .suh_g = 0, .suh_u = 0, .svh_g = 0, .svh_u = 0, .suh_d = 0, .svh_d = 0, .experts = routed_experts, .dims = hidden, .width = expert_width, .codebook = .mcg_e4m3, .widths_ok = true };
}

/// Every forward buffer of one rank, carved from one allocation. Persistent regions live across a forward step;
/// the attention, FFN and head regions are never live together, so they share one union region (§3.9 trims).
pub const Workspace = struct {
    pub const max_moe_pipe_pieces = 16;
    pub const moe_pipe_ring_slots = 2;
    pub const max_index_blocks = 64;
    rows: usize,
    piece_rows: usize,
    lane_clusters: bool,
    decode_rows: usize,
    score_rows: usize,
    expert_rows: usize,
    scratch_overlay: bool = false,
    generic_rows: usize = 0,
    adapt_big: bool = false,
    head_rows: usize,
    candidate_k: usize,
    index_slots: usize,
    // persistent
    x: u64,
    xn: u64,
    gathered: u64,
    tbo_gathered: [2]u64,
    pipe_src: [2]u64,
    pipe_gather: [2]u64,
    moe_pipe_src: [moe_pipe_ring_slots]u64,
    moe_pipe_gather: [moe_pipe_ring_slots]u64,
    moe_pipe_slots: usize,
    moe_pipe: u64,
    global_tokens: u64,
    global_counts: u64,
    local_slots: u64,
    local_counts: u64,
    roce_token: u64,
    roce_status: u64,
    positions: u64,
    dense_len: u64,
    cos: u64,
    sin: u64,
    ids: u64,
    owners: u64,
    lane_part: u64,
    lane_part_side: u64,
    // attention phase
    q_a: u64,
    sh_q: u64,
    sh_kv: u64,
    sh_mine: u64,
    sh_all: u64,
    q_resid: u64,
    q: u64,
    kv: u64,
    ik: u64,
    iw: u64,
    iq: u64,
    qi: u64,
    qa: u64,
    qpe: u64,
    fused_queries: u64,
    part: u64,
    ol: u64,
    lse_out: u64,
    /// Decode attend's key-split partials (mla.splitWorkspaceBytes for both 32-head groups).
    attend_split: u64,
    vo: u64,
    scores: u64,
    topk_tokens: u64,
    topk_counts: u64,
    candidate_records: u64,
    gathered_candidates: u64,
    /// SMLA-P3 step union table of one piece (Geometry.smla_p3); zero-length at the attention-phase end when off.
    smla_union: u64,
    // FFN phase
    router_logits: u64,
    router_part: u64,
    picks: u64,
    wts: u64,
    picks8: u64,
    wts8: u64,
    shared_out: u64,
    routed_out: u64,
    lowrank_mid: u64,
    mlp_gate_up: u64,
    mlp_act: u64,
    experts: u64,
    prompt_experts: u64,
    prompt_plan: u64,
    // head phase
    logits: u64,
    cand_values: u64,
    cand_ids: u64,
    cand_scratch: u64,
    head_gather: u64,
    /// Reused BF16 dequant destination for one backbone projection (kept live through its cuBLASLt GEMM).
    prefill_weight: u64,
    /// Bytes from the base to the end of the last region.
    bytes: u64,
    /// Start and extent of the phase union, also the drafter's safe overlay area.
    phase: u64,
    phase_bytes: u64,
    /// Region sizes, for launch-contract checks.
    sizes: Sizes,

    /// Bytes of every region, by field name.
    pub const Sizes = struct {
        x: u64 = 0,
        xn: u64 = 0,
        gathered: u64 = 0,
        tbo_gathered: [2]u64 = .{ 0, 0 },
        pipe_src: [2]u64 = .{ 0, 0 },
        pipe_gather: [2]u64 = .{ 0, 0 },
        moe_pipe: u64 = 0,
        global_tokens: u64 = 0,
        global_counts: u64 = 0,
        local_slots: u64 = 0,
        local_counts: u64 = 0,
        roce_token: u64 = 0,
        roce_status: u64 = 0,
        positions: u64 = 0,
        dense_len: u64 = 0,
        cos: u64 = 0,
        sin: u64 = 0,
        ids: u64 = 0,
        owners: u64 = 0,
        lane_part: u64 = 0,
        lane_part_side: u64 = 0,
        q_a: u64 = 0,
        sh_q: u64 = 0,
        sh_kv: u64 = 0,
        sh_mine: u64 = 0,
        sh_all: u64 = 0,
        q_resid: u64 = 0,
        q: u64 = 0,
        kv: u64 = 0,
        ik: u64 = 0,
        iw: u64 = 0,
        iq: u64 = 0,
        qi: u64 = 0,
        qa: u64 = 0,
        qpe: u64 = 0,
        fused_queries: u64 = 0,
        part: u64 = 0,
        ol: u64 = 0,
        lse_out: u64 = 0,
        attend_split: u64 = 0,
        vo: u64 = 0,
        scores: u64 = 0,
        topk_tokens: u64 = 0,
        topk_counts: u64 = 0,
        candidate_records: u64 = 0,
        gathered_candidates: u64 = 0,
        smla_union: u64 = 0,
        router_logits: u64 = 0,
        router_part: u64 = 0,
        picks: u64 = 0,
        wts: u64 = 0,
        picks8: u64 = 0,
        wts8: u64 = 0,
        shared_out: u64 = 0,
        routed_out: u64 = 0,
        lowrank_mid: u64 = 0,
        mlp_gate_up: u64 = 0,
        mlp_act: u64 = 0,
        experts: u64 = 0,
        prompt_experts: u64 = 0,
        prompt_plan: u64 = 0,
        logits: u64 = 0,
        cand_values: u64 = 0,
        cand_ids: u64 = 0,
        cand_scratch: u64 = 0,
        head_gather: u64 = 0,
        prefill_weight: u64 = 0,
    };

    pub fn align256(n: u64) u64 {
        return (n + 255) / 256 * 256;
    }

    fn take(at: *u64, bytes: u64) u64 {
        const r = at.*;
        at.* = r + align256(bytes);
        return r;
    }

    /// Head exchange per row: greedy top-1 (value, id) plus local LSE = 16 B, or k candidates plus local LSE.
    pub fn headRowBytes(k: usize) u64 {
        return @max(16, @as(u64, @intCast(k)) * 8 + 8);
    }

    pub fn carve(base: u64, g: Geometry, context: usize) Workspace {
        // r: rows a window may hold (chunk or decode); pr: rows one attention/FFN piece holds.
        const r: u64 = @intCast(@max(g.chunk_rows, g.decode_rows));
        const pr: u64 = @intCast(@max(@min(g.piece_rows, @max(g.chunk_rows, g.decode_rows)), g.decode_rows));
        const dr: u64 = @intCast(g.decode_rows);
        const sr: u64 = @intCast(@min(g.score_rows, pr));
        const grouped = groupedExperts(g.expert_rows, @intCast(pr));
        const trim_generic = g.ws_trim or g.slim_generic;
        if (grouped) std.debug.assert(g.expert_rows <= @as(usize, @intCast(r)));
        const er: usize = if (grouped) @min(g.expert_rows, @as(usize, @intCast(r))) else @min(g.expert_rows, @as(usize, @intCast(pr)));
        const batch_rows: u64 = if (grouped) @intCast(@min(g.expert_rows, @as(usize, @intCast(r)))) else pr;
        const hr: u64 = @intCast(g.head_rows);
        const k: u64 = @intCast(g.candidate_k);
        const slots: u64 = @intCast(@min(g.index_slots, localSlots(context, dcpWorld(g.world))));
        var w: Workspace = undefined;
        w.adapt_big = false;
        w.rows = @intCast(r);
        w.piece_rows = @intCast(pr);
        w.decode_rows = g.decode_rows;
        w.score_rows = @intCast(sr);
        w.expert_rows = er;
        w.scratch_overlay = g.scratch_overlay;
        w.generic_rows = if (trim_generic) @max(g.decode_rows, 16) else er;
        w.head_rows = g.head_rows;
        w.candidate_k = g.candidate_k;
        w.index_slots = @intCast(slots);
        var at = base;
        // Persistent across a window: the residual rows, the token list shared layers reuse, the row inputs.
        w.sizes.x = r * hidden * 2;
        w.x = take(&at, w.sizes.x);
        w.sizes.xn = batch_rows * hidden * 2;
        w.xn = take(&at, w.sizes.xn);
        // The TP reduce (and the bf16 embed gather) of one piece: [tp_world][piece][6144] fp32.
        const tp_world = g.world;
        w.sizes.gathered = tp_world * pr * hidden * 4;
        w.gathered = take(&at, w.sizes.gathered);
        if (g.tbo) {
            const half_rows = (pr + 1) / 2;
            const half_bytes = tp_world * half_rows * hidden * 4;
            for (0..2) |half| {
                w.sizes.tbo_gathered[half] = half_bytes;
                w.tbo_gathered[half] = take(&at, half_bytes);
            }
        } else {
            // Keep the disabled layout byte-identical: zero-length halves sit at the cursor.
            for (0..2) |half| {
                w.sizes.tbo_gathered[half] = 0;
                w.tbo_gathered[half] = at;
            }
        }
        if (g.piece_pipe) {
            for (0..2) |ring_slot| {
                w.sizes.pipe_src[ring_slot] = pr * hidden * 4;
                w.pipe_src[ring_slot] = take(&at, w.sizes.pipe_src[ring_slot]);
                w.sizes.pipe_gather[ring_slot] = tp_world * pr * hidden * 4;
                w.pipe_gather[ring_slot] = take(&at, w.sizes.pipe_gather[ring_slot]);
            }
        } else {
            for (0..2) |ring_slot| {
                w.sizes.pipe_src[ring_slot] = 0;
                w.sizes.pipe_gather[ring_slot] = 0;
                w.pipe_src[ring_slot] = at;
                w.pipe_gather[ring_slot] = at;
            }
        }
        w.moe_pipe_slots = 0;
        if (g.moe_pipe) {
            const moe_slots: usize = Workspace.moe_pipe_ring_slots;
            const stride = pr * hidden * 2;
            const slot_bytes = stride * @as(u64, if (g.moe_post) tp_world else 1 + tp_world);
            const ring_bytes = @as(u64, @intCast(moe_slots)) * slot_bytes;
            w.sizes.moe_pipe = ring_bytes;
            w.moe_pipe_slots = moe_slots;
            w.moe_pipe = at;
            for (0..moe_slots) |slot| {
                const slot_start = take(&at, slot_bytes);
                w.moe_pipe_src[slot] = slot_start;
                w.moe_pipe_gather[slot] = if (g.moe_post) slot_start else slot_start + stride;
            }
        } else {
            w.sizes.moe_pipe = 0;
            w.moe_pipe = at;
            @memset(&w.moe_pipe_src, at);
            @memset(&w.moe_pipe_gather, at);
        }
        w.sizes.global_tokens = r * top_k * 4;
        w.global_tokens = take(&at, w.sizes.global_tokens);
        w.sizes.global_counts = r * 4;
        w.global_counts = take(&at, w.sizes.global_counts);
        // Fused DCP merge writes local slots/counts for every sparse selection row; prefill uses global tokens.
        w.sizes.local_slots = r * top_k * 4;
        w.local_slots = take(&at, w.sizes.local_slots);
        w.sizes.local_counts = r * 4;
        w.local_counts = take(&at, w.sizes.local_counts);
        w.sizes.roce_token = 4;
        w.roce_token = take(&at, w.sizes.roce_token);
        w.sizes.roce_status = 4;
        w.roce_status = take(&at, w.sizes.roce_status);
        w.sizes.positions = r * 8;
        w.positions = take(&at, w.sizes.positions);
        w.sizes.dense_len = r * 4;
        w.dense_len = take(&at, w.sizes.dense_len);
        w.sizes.cos = r * (mla.rope_dim / 2) * 4;
        w.cos = take(&at, w.sizes.cos);
        w.sizes.sin = r * (mla.rope_dim / 2) * 4;
        w.sin = take(&at, w.sizes.sin);
        w.sizes.ids = r * 4;
        w.ids = take(&at, w.sizes.ids);
        w.sizes.owners = r * 4;
        w.owners = take(&at, w.sizes.owners);
        w.lane_clusters = g.lane_clusters;
        w.sizes.lane_part = @max(lanePartBytes(@intCast(pr), g.lane_clusters), 4);
        w.lane_part = take(&at, w.sizes.lane_part);
        w.sizes.lane_part_side = @max(lanePartBytes(@intCast(dr), g.lane_clusters), 4);
        w.lane_part_side = take(&at, w.sizes.lane_part_side);
        // Grouped MoE keeps these results live while the next piece's attention runs.
        // Other shapes retain the original FFN-phase placement byte for byte.
        if (grouped) {
            w.sizes.picks8 = batch_rows * routed_slots * 4;
            w.picks8 = take(&at, w.sizes.picks8);
            w.sizes.wts8 = batch_rows * routed_slots * 4;
            w.wts8 = take(&at, w.sizes.wts8);
            w.sizes.shared_out = batch_rows * hidden * 4;
            w.shared_out = take(&at, w.sizes.shared_out);
            w.sizes.routed_out = if (g.moe_post) 0 else if (g.expert_rows > @as(usize, @intCast(pr))) batch_rows * hidden * 4 else 0;
            w.routed_out = take(&at, w.sizes.routed_out);
            w.sizes.lowrank_mid = if (trim_generic) @max(8 * batch_rows, @as(u64, lowrank_decode_splits) * dr) * lowrank_max * 4 else 128 * batch_rows * lowrank_max * 4;
            w.lowrank_mid = take(&at, w.sizes.lowrank_mid);
        }
        const phase = at;
        // attention phase
        w.sizes.q_a = pr * q_lora * 2;
        w.q_a = take(&at, w.sizes.q_a);
        w.sizes.sh_q = if (g.shard_qkv_a) pr * 1024 * 2 else 0;
        w.sh_q = if (g.shard_qkv_a) take(&at, w.sizes.sh_q) else at;
        w.sizes.sh_kv = if (g.shard_qkv_a) pr * 320 * 2 else 0;
        w.sh_kv = if (g.shard_qkv_a) take(&at, w.sizes.sh_kv) else at;
        w.sizes.sh_mine = if (g.shard_qkv_a) pr * 2688 else 0;
        w.sh_mine = if (g.shard_qkv_a) take(&at, w.sizes.sh_mine) else at;
        w.sizes.sh_all = if (g.shard_qkv_a) 2 * pr * 2688 else 0;
        w.sh_all = if (g.shard_qkv_a) take(&at, w.sizes.sh_all) else at;
        w.sizes.q_resid = pr * q_lora * 2;
        w.q_resid = take(&at, w.sizes.q_resid);
        w.sizes.q = pr * local_heads * q_head * 2;
        w.q = take(&at, w.sizes.q);
        w.sizes.kv = pr * kv_a_width * 2;
        w.kv = take(&at, w.sizes.kv);
        w.sizes.ik = pr * index_dim * 2;
        w.ik = take(&at, w.sizes.ik);
        w.sizes.iw = pr * index_heads * 2;
        w.iw = take(&at, w.sizes.iw);
        w.sizes.iq = pr * index_heads * index_dim * 2;
        w.iq = take(&at, w.sizes.iq);
        w.sizes.qi = pr * index_heads * index_dim * 2;
        w.qi = take(&at, w.sizes.qi);
        const dcp_w = dcpWorld(g.world);
        w.sizes.qa = @max(dcp_w * dr * local_heads * mla.latent_dim * 2, pr * local_heads * mla.latent_dim * 2);
        w.qa = take(&at, w.sizes.qa);
        w.sizes.qpe = @max(dcp_w * dr * local_heads * mla.rope_dim * 2, pr * local_heads * mla.rope_dim * 2);
        w.qpe = take(&at, w.sizes.qpe);
        w.sizes.fused_queries = if (g.coll_fuse) dcp_w * dr * local_heads * (mla.latent_dim + mla.rope_dim) * 2 else 0;
        w.fused_queries = take(&at, w.sizes.fused_queries);
        // Decode computes both 32-head groups locally (2 blocks) and receives only the peer's matching block.
        w.sizes.part = @max((dcp_w + 1) * dr * decode_partial_row_bytes, pr * local_heads * (mla.latent_dim * 2 + 4));
        w.part = take(&at, w.sizes.part);
        w.sizes.ol = dr * local_heads * mla.latent_dim * 2;
        w.ol = take(&at, w.sizes.ol);
        w.sizes.lse_out = dr * local_heads * 4;
        w.lse_out = take(&at, w.sizes.lse_out);
        w.sizes.attend_split = mla.splitWorkspaceBytes(@intCast(dr), 2 * @as(usize, @intCast(local_heads)));
        w.attend_split = take(&at, w.sizes.attend_split);
        w.sizes.vo = pr * local_heads * mla.value_dim * 2;
        w.vo = take(&at, w.sizes.vo);
        w.sizes.scores = sr * slots * 4;
        w.scores = take(&at, w.sizes.scores);
        w.sizes.topk_tokens = sr * top_k * 4;
        w.topk_tokens = take(&at, w.sizes.topk_tokens);
        w.sizes.topk_counts = sr * 4;
        w.topk_counts = take(&at, w.sizes.topk_counts);
        const index_rows = if (g.index_pipe) pr else sr;
        w.sizes.candidate_records = index_rows * top_k * 8;
        w.candidate_records = take(&at, w.sizes.candidate_records);
        w.sizes.gathered_candidates = dcp_w * index_rows * top_k * 8;
        w.gathered_candidates = take(&at, w.sizes.gathered_candidates);
        if (g.smla_p3) {
            // Written by tf_glm53_smla_union and read by tf_glm53_smla_p3 inside one piece's attend: attention-phase only.
            w.sizes.smla_union = mla.smlaP3TableBytes(@intCast(pr));
            w.smla_union = take(&at, w.sizes.smla_union);
        } else {
            // Off: a zero-length span at the cursor (nothing moves, the layout is byte-identical).
            w.sizes.smla_union = 0;
            w.smla_union = at;
        }
        const attention_end = at;
        // FFN phase
        at = phase;
        w.sizes.router_logits = pr * routed_experts * 4;
        w.router_logits = take(&at, w.sizes.router_logits);
        w.sizes.router_part = 8 * pr * routed_experts * 4;
        w.router_part = take(&at, w.sizes.router_part);
        w.sizes.picks = pr * route_slots * 4;
        w.picks = take(&at, w.sizes.picks);
        w.sizes.wts = pr * route_slots * 4;
        w.wts = take(&at, w.sizes.wts);
        if (!grouped) {
            w.sizes.picks8 = batch_rows * routed_slots * 4;
            w.picks8 = take(&at, w.sizes.picks8);
            w.sizes.wts8 = batch_rows * routed_slots * 4;
            w.wts8 = take(&at, w.sizes.wts8);
            w.sizes.shared_out = batch_rows * hidden * 4;
            w.shared_out = take(&at, w.sizes.shared_out);
            w.sizes.routed_out = if (g.moe_post) 0 else if (g.expert_rows > @as(usize, @intCast(pr))) batch_rows * hidden * 4 else 0;
            w.routed_out = take(&at, w.sizes.routed_out);
            w.sizes.lowrank_mid = if (trim_generic) @max(8 * batch_rows, @as(u64, lowrank_decode_splits) * dr) * lowrank_max * 4 else 128 * batch_rows * lowrank_max * 4;
            w.lowrank_mid = take(&at, w.sizes.lowrank_mid);
        }
        w.sizes.mlp_gate_up = pr * 2 * dense_width * 2;
        w.mlp_gate_up = take(&at, w.sizes.mlp_gate_up);
        w.sizes.mlp_act = pr * dense_width * 2;
        w.mlp_act = take(&at, w.sizes.mlp_act);
        const generic_rows: usize = if (trim_generic) @max(g.decode_rows, 16) else er;
        w.sizes.experts = cuda.exl3_experts.scratchBytes(expertShape(), generic_rows, routed_slots) catch unreachable;
        w.experts = take(&at, w.sizes.experts);
        // Lifetime check for the ws.experts/prompt_experts overlay, in per-layer stream order:
        // 1. The prompt plan is written at the head of ws.experts by router.route; prompt rotate writes xg/xu.
        // 2. Prompt =1 reads xg/xu and writes xd/y; =2 runs its small gate/up, wide up (reads xu, writes y as uu),
        //    wide gate (reads xg/y, writes xd), then down (reads xd and writes y). combinePrompt reads y.
        // 3. Ineligible prefill shapes take routedGroup/runRouted's grouped expert path on ws.experts instead;
        //    those calls do not touch prompt scratch. No prompt kernel remains live in that branch.
        // 4. The ordinary per-piece runRouted path has the same eligible-prompt vs grouped-expert exclusivity.
        //    Decode, including the optional side-stream route, never takes the prefill prompt branch.
        // 5. All prompt/expert work is enqueued on the layer's main stream, which orders it before the next
        //    layer reuses these buffers. No plan or scratch contents are consumed after that layer.
        // Thus there is no same-layer grouped/decode expert use while prompt scratch is live and no cross-layer
        // retention; the overlay is safe for both ordinary and grouped prefill configurations.
        if (g.ws_trim and g.prompt_experts) {
            w.sizes.prompt_plan = promptPlanBytes(er, routed_slots, routed_experts);
            w.prompt_plan = take(&at, w.sizes.prompt_plan);
            w.sizes.prompt_experts = cuda.exl3_experts.prompt.bytes(expertShape(), er, routed_slots, false) catch unreachable;
            w.prompt_experts = take(&at, w.sizes.prompt_experts);
        } else if (g.prompt_experts) {
            w.sizes.prompt_experts = cuda.exl3_experts.prompt.bytes(expertShape(), er, routed_slots, g.scratch_overlay) catch unreachable;
            if (g.scratch_overlay) {
                const offset = align256(promptPlanBytes(er, routed_slots, routed_experts));
                w.prompt_experts = w.experts + offset;
                const needed = offset + w.sizes.prompt_experts;
                if (w.sizes.experts < needed) at += needed - w.sizes.experts;
            } else {
                w.prompt_experts = take(&at, w.sizes.prompt_experts);
            }
            // The plan is temporary while the prompt path runs and uses the head of ws.experts.
            w.sizes.prompt_plan = 0;
            w.prompt_plan = w.experts;
        } else {
            // Off: zero-length spans at the current cursor (in range, nothing else moves).
            w.sizes.prompt_experts = 0;
            w.prompt_experts = at;
            w.sizes.prompt_plan = 0;
            w.prompt_plan = at;
        }
        const ffn_end = at;
        // head phase
        at = phase;
        const vocab: u64 = @intCast(g.vocab_local);
        w.sizes.logits = hr * vocab * 4;
        w.logits = take(&at, w.sizes.logits);
        w.sizes.cand_values = hr * @max(k, 1) * 4;
        w.cand_values = take(&at, w.sizes.cand_values);
        w.sizes.cand_ids = hr * @max(k, 1) * 4;
        w.cand_ids = take(&at, w.sizes.cand_ids);
        w.sizes.cand_scratch = candidateScratchBytes(@intCast(hr), g.candidate_k) catch unreachable;
        w.cand_scratch = take(&at, w.sizes.cand_scratch);
        w.sizes.head_gather = g.world * hr * headRowBytes(g.candidate_k);
        w.head_gather = take(&at, w.sizes.head_gather);
        const head_end = at;
        at = @max(attention_end, @max(ffn_end, head_end));
        const phase_bytes = at - phase;
        // `phase` is already an absolute cursor (like the other addresses in `Workspace`).
        // A zero-sized overlay therefore stays at the current union end and adds no allocation.
        if (g.drafter_share_bytes != 0) at = @max(at, phase + g.drafter_share_bytes);
        w.sizes.prefill_weight = if (g.prefill_bf16) prefill_weight_scratch_bytes else 0;
        w.prefill_weight = take(&at, w.sizes.prefill_weight);
        w.bytes = at - base;
        w.phase = phase;
        w.phase_bytes = @max(phase_bytes, g.drafter_share_bytes);
        return w;
    }
};

pub fn adaptWorkspace(base: Workspace, b: u64, g: Geometry, context: usize) Workspace {
    var gb = g;
    gb.expert_rows = g.expert_rows_adapt;
    gb.scratch_overlay = false;
    gb.prompt_experts = true;
    const ref = workspaceLayout(gb, context);
    var w = base;
    var at = b;
    w.xn = Workspace.take(&at, ref.sizes.xn); w.sizes.xn = ref.sizes.xn;
    w.picks8 = Workspace.take(&at, ref.sizes.picks8); w.sizes.picks8 = ref.sizes.picks8;
    w.wts8 = Workspace.take(&at, ref.sizes.wts8); w.sizes.wts8 = ref.sizes.wts8;
    w.shared_out = Workspace.take(&at, ref.sizes.shared_out); w.sizes.shared_out = ref.sizes.shared_out;
    w.routed_out = Workspace.take(&at, ref.sizes.routed_out); w.sizes.routed_out = ref.sizes.routed_out;
    w.sizes.lowrank_mid = Workspace.align256(@intCast(8 * g.expert_rows_adapt * lowrank_max * 4));
    w.lowrank_mid = Workspace.take(&at, w.sizes.lowrank_mid);
    w.sizes.prompt_plan = promptPlanBytes(g.expert_rows_adapt, routed_slots, routed_experts);
    w.prompt_plan = Workspace.take(&at, w.sizes.prompt_plan);
    w.sizes.prompt_experts = cuda.exl3_experts.prompt.bytes(expertShape(), g.expert_rows_adapt, routed_slots, false) catch unreachable;
    w.prompt_experts = Workspace.take(&at, w.sizes.prompt_experts);
    w.expert_rows = g.expert_rows_adapt;
    w.scratch_overlay = false;
    w.adapt_big = true;
    return w;
}

/// RoCE gathers read and write these as plain device operands; RoCE registers only its own pinned host region.
pub fn adaptRoceAligned(w: Workspace) bool {
    const pointers = [_]u64{ w.xn, w.picks8, w.wts8, w.shared_out, w.routed_out, w.lowrank_mid, w.prompt_plan, w.prompt_experts };
    for (pointers) |ptr| {
        if (ptr % 16 != 0) return false;
    }
    return true;
}

pub fn workspaceLayout(geometry: Geometry, context: usize) Workspace {
    return Workspace.carve(0, geometry, context);
}

pub const PromptPlanLayout = struct {
    pairs: usize,
    items: usize,
    counts: usize,
    rank: usize,
    hist: usize,
    bytes: u64,
};

/// Shared grouped-router plan layout for carving and sizing, using the prompt kernels' tile.
pub fn promptPlanLayout(rows: usize, slots: usize, experts: usize) PromptPlanLayout {
    const pairs = rows * slots;
    const items = cuda.grouped.maxItems(pairs, experts, cuda.exl3_experts.prompt.tile) * 3;
    const counts = experts;
    var rank: usize = 0;
    var hist: usize = 0;
    if (pairs > cuda.grouped.small) {
        rank = pairs;
        hist = ((pairs + 1023) / 1024) * experts;
    }
    const bytes = Workspace.align256(@intCast(pairs * @sizeOf(i32))) +
        Workspace.align256(@intCast(items * @sizeOf(i32))) +
        Workspace.align256(@intCast(counts * @sizeOf(i32))) +
        Workspace.align256(@intCast(rank * @sizeOf(i32))) +
        Workspace.align256(@intCast(hist * @sizeOf(i32)));
    return .{ .pairs = pairs, .items = items, .counts = counts, .rank = rank, .hist = hist, .bytes = bytes };
}

pub fn promptPlanBytes(rows: usize, slots: usize, experts: usize) u64 {
    return promptPlanLayout(rows, slots, experts).bytes;
}

pub fn validatePromptExpertsCarve(enabled: bool, ws: *const Workspace) !void {
    try validatePromptExpertsCarveForLayer(enabled, ws, expertShape());
}

pub fn validatePromptExpertsCarveForLayer(enabled: bool, ws: *const Workspace, layer: cuda.exl3_experts.Layer) !void {
    if (!enabled) return;
    const needed = cuda.exl3_experts.prompt.bytes(layer, ws.expert_rows, routed_slots, ws.scratch_overlay) catch return error.PromptExpertsNotCarved;
    if (ws.sizes.prompt_experts < @as(u64, @intCast(needed))) return error.PromptExpertsNotCarved;
}

/// Per-rank DCP2 cache planes, the prefill stage plane and the forward workspace. One owner keeps them alive.
pub const State = struct {
    pub const Adapt = struct { ws: Workspace, limit_position: usize, rows: usize, vmm: cuda.arena.Vmm, b: u64, b_len: u64, kv_va: u64, kv_len: u64 };
    /// The KV arena; latent, rope and index_keys are borrowed spans of it.
    kv_arena: cuda.DeviceBuffer,
    latent: cuda.DeviceBuffer,
    rope: cuda.DeviceBuffer,
    index_keys: cuda.DeviceBuffer,
    park: cuda.DeviceBuffer,
    park_slots: usize = 0,
    park_tail_slot: usize = 0,
    workspace: cuda.DeviceBuffer,
    /// Prefill mirror of peer DCP keys at absolute slots p / 2 (latent rows, then rope rows), stagePlaneBytes long.
    stage_plane: cuda.DeviceBuffer,
    /// Dedicated weight-dequant stream and event. Main compute waits on `ready` before the BF16 GEMM reads scratch.
    dequant_stream: ?cuda.Stream = null,
    dequant_ready: ?cuda.Event = null,
    dequant_consumed: ?cuda.Event = null,
    /// Decode side stream and layer-local forks. No collective is ever enqueued here.
    side: ?cuda.Stream = null,
    fork: ?cuda.Event = null,
    kv: ?cuda.Event = null,
    fork2: ?cuda.Event = null,
    shared: ?cuda.Event = null,
    /// TBO communication stream and per-half ready/done events. Collectives remain ordered on this one stream.
    tbo_stream: ?cuda.Stream = null,
    tbo_ready: [2]?cuda.Event = .{ null, null },
    tbo_done: [2]?cuda.Event = .{ null, null },
    pipe_ready: [2]?cuda.Event = .{ null, null },
    pipe_done: [2]?cuda.Event = .{ null, null },
    moe_pipe_ready: [Workspace.moe_pipe_ring_slots]?cuda.Event = @splat(null),
    moe_pipe_done: [Workspace.moe_pipe_ring_slots]?cuda.Event = @splat(null),
    moe_pipe_pdone: [Workspace.max_moe_pipe_pieces]?cuda.Event = @splat(null),
    index_pipe_ready: [Workspace.max_index_blocks]?cuda.Event = @splat(null),
    index_pipe_done: ?cuda.Event = null,
    /// Carved forward buffers inside `workspace`.
    ws: Workspace,
    capacity: usize,
    rank: usize,
    world: usize,
    slots: usize,
    stage: types.StagePlanes,
    layers: usize,
    full_index_layers: usize,
    decode_mla: bool,
    decode_mla_v2: bool,
    decode_mla_v3: bool,
    adapt: ?Adapt = null,

    /// Allocate what `admission` admitted (made once, by admitRank, before the weights load). Weights before State:
    /// the loader's transients are gone by the time these planes are allocated. `fresh_mem_available` is
    /// MemAvailable read after the load (and its page-cache drop): it must still hold State + graphs above the
    /// reserve, or this fails closed before allocating (review N4: a weight overrun never eats the floor).
    pub fn init(driver: *const cuda.Driver, geometry: Geometry, context: usize, rank: usize, world: usize, admission: *const Admission, fresh_mem_available: u64) !State {
        return initWith(driver, null, null, geometry, context, rank, world, admission, fresh_mem_available);
    }

    pub fn initWith(driver: *const cuda.Driver, device: ?cuda.abi.Device, maybe_vmm: ?cuda.arena.Vmm, geometry: Geometry, context: usize, rank: usize, world: usize, admission: *const Admission, fresh_mem_available: u64) !State {
        var incoming_vmm_owned = maybe_vmm != null;
        errdefer if (incoming_vmm_owned) {
            var v = maybe_vmm.?;
            v.unmapRange(v.va, v.va_len);
            v.close();
        };
        if ((world != dcp_world and world != 3) or rank >= world or context == 0) return error.DcpShape;
        if (geometry.expert_rows_adapt != 0 and maybe_vmm == null) return error.ExpertRowsAdaptNeedsVmm;
        if (maybe_vmm) |v| if (device) |d| if (v.device != d) {
            return error.ExpertRowsAdaptDeviceMismatch;
        };
        if (geometry.index_pipe and !geometry.piece_pipe and !geometry.moe_pipe) return error.IndexPipeRequiresCommStream;
        if (admission.context != context or admission.state_bytes != stateBytes(geometry, context) or admission.state_logical != breakdown(geometry, context, .{}).logicalState() or admission.breakdown.graphs != geometry.graphs_bytes or admission.breakdown.runtime != geometry.runtime_bytes) return error.AdmissionMismatch;
        try validateMoePipeGeometry(geometry);
        if (fresh_mem_available -| admission.reserve.bytes < admission.state_bytes + geometry.graphs_bytes) {
            if (!builtin.is_test) std.log.err("GLM-5.3 after load: MemAvailable {d:.2} GiB less reserve {d:.2} GiB cannot hold State {d:.2} + graphs {d:.2} GiB", .{ gib(fresh_mem_available), gib(admission.reserve.bytes), gib(admission.state_bytes), gib(geometry.graphs_bytes) });
            return error.MemoryBudgetAfterLoad;
        }
        const plane = planeBytesFor(geometry, context);
        if (geometry.park_tail_slot != 0 and !parkTailFits(geometry.park_tail_slot, geometry.park_rows / 2, plane.slots, null)) return error.LongTierParkTail;
        const layout = workspaceLayout(geometry, context);
        try validatePromptExpertsCarve(geometry.prompt_experts, &layout);
        const empty: cuda.DeviceBuffer = .{ .d = driver, .ptr = 0, .len = 0 };
        var out: State = .{
            .kv_arena = empty,
            .latent = empty,
            .rope = empty,
            .index_keys = empty,
            .park = empty,
            .workspace = empty,
            .stage_plane = empty,
            .ws = layout,
            .capacity = context,
            .rank = rank,
            .world = world,
            .slots = plane.slots,
            .stage = stagePlanes(driver, 0, context, stagePairsFor(geometry)),
            .layers = geometry.layers,
            .full_index_layers = geometry.full_index_layers,
            .decode_mla = geometry.decode_mla,
            .decode_mla_v2 = geometry.decode_mla_v2,
            .decode_mla_v3 = geometry.decode_mla_v3,
        };
        errdefer out.deinit();
        // KV planes (latent | rope | index keys), stage pairs, workspace and optional side park.
        const arenas = stateArenas(geometry, context);
        var pending_adapt: ?Adapt = null;
        if (geometry.expert_rows_adapt == 0) {
            out.kv_arena = try cuda.DeviceBuffer.alloc(driver, @intCast(arenas.kv));
        } else {
            var vmm = maybe_vmm.?;
            const plan = try planAdapt(std.heap.page_allocator, geometry, context, vmm.granularity);
            defer std.heap.page_allocator.free(plan.tails);
            if (geometry.park_tail_slot != 0 and !parkTailFits(geometry.park_tail_slot, geometry.park_rows / 2, plane.slots, plan.limit_slot)) return error.LongTierParkTail;
            if (vmm.va_len < arenas.kv + plan.pool) return error.AdaptVmmRangeTooSmall;
            const kv_va = vmm.va;
            const b = kv_va + arenas.kv;
            var off: u64 = 0;
            var tail_cursor: u64 = 0;
            for (plan.tails) |tail| {
                if (tail.kv_offset > off) try mapVmmSegment(&vmm, kv_va + off, tail.kv_offset - off, false, b, &tail_cursor);
                try mapVmmSegment(&vmm, kv_va + tail.kv_offset, tail.len, true, b, &tail_cursor);
                off = tail.kv_offset + tail.len;
            }
            if (off < arenas.kv) try mapVmmSegment(&vmm, kv_va + off, arenas.kv - off, false, b, &tail_cursor);
            try vmm.grantAccess(kv_va, @intCast(arenas.kv));
            if (plan.pool != 0) try vmm.grantAccess(b, @intCast(plan.pool));
            out.kv_arena = .{ .d = driver, .ptr = kv_va, .len = @intCast(arenas.kv), .borrowed = true };
            cuda.memory.accountExternalDeviceAllocation(@intCast(arenas.kv));
            pending_adapt = .{ .ws = out.ws, .limit_position = plan.limit_position, .rows = geometry.expert_rows_adapt, .vmm = vmm, .b = b, .b_len = plan.pool, .kv_va = kv_va, .kv_len = arenas.kv };
        }
        const latent_at = out.kv_arena.ptr;
        const rope_at = latent_at + std.mem.alignForward(u64, plane.latent, tensor_alignment);
        const index_at = rope_at + std.mem.alignForward(u64, plane.rope, tensor_alignment);
        if (index_at + plane.index > out.kv_arena.ptr + out.kv_arena.len) return error.ArenaOverflow;
        out.latent = .{ .d = driver, .ptr = latent_at, .len = @intCast(plane.latent), .borrowed = true };
        out.rope = .{ .d = driver, .ptr = rope_at, .len = @intCast(plane.rope), .borrowed = true };
        out.index_keys = .{ .d = driver, .ptr = index_at, .len = @intCast(plane.index), .borrowed = true };
        out.workspace = try cuda.DeviceBuffer.alloc(driver, @intCast(arenas.workspace));
        if (arenas.park != 0) {
            out.park = try cuda.DeviceBuffer.alloc(driver, @intCast(arenas.park));
        }
        out.park_slots = geometry.park_rows / 2;
        out.park_tail_slot = geometry.park_tail_slot;
        out.stage_plane = try cuda.DeviceBuffer.alloc(driver, @intCast(arenas.stage));
        out.ws = Workspace.carve(out.workspace.ptr, geometry, context);
        if (pending_adapt) |*ad| {
            ad.ws = adaptWorkspace(out.ws, ad.b, geometry, context);
            out.adapt = ad.*;
            incoming_vmm_owned = false;
        }
        out.stage = stagePlanes(driver, out.stage_plane.ptr, context, stagePairsFor(geometry));
        if (geometry.decode_side) {
            out.side = try cuda.Stream.init(driver, true);
            out.fork = try cuda.Event.init(driver, false);
            out.kv = try cuda.Event.init(driver, false);
            out.fork2 = try cuda.Event.init(driver, false);
            out.shared = try cuda.Event.init(driver, false);
        }
        if (geometry.tbo or geometry.piece_pipe or geometry.moe_pipe or geometry.index_pipe) {
            out.tbo_stream = if (geometry.comm_prio) try cuda.Stream.initPriority(driver, .highest) else try cuda.Stream.init(driver, true);
        }
        if (geometry.tbo) {
            for (0..2) |half| {
                out.tbo_ready[half] = try cuda.Event.init(driver, false);
                out.tbo_done[half] = try cuda.Event.init(driver, false);
            }
        }
        if (geometry.piece_pipe) {
            for (0..2) |slot| {
                out.pipe_ready[slot] = try cuda.Event.init(driver, false);
                out.pipe_done[slot] = try cuda.Event.init(driver, false);
            }
        }
        if (geometry.moe_pipe) {
            for (0..Workspace.moe_pipe_ring_slots) |slot| {
                out.moe_pipe_ready[slot] = try cuda.Event.init(driver, false);
                out.moe_pipe_done[slot] = try cuda.Event.init(driver, false);
            }
            for (0..Workspace.max_moe_pipe_pieces) |piece| out.moe_pipe_pdone[piece] = try cuda.Event.init(driver, false);
        }
        if (geometry.index_pipe) {
            const blocks = (@as(usize, @intCast(layout.piece_rows)) + @max(geometry.score_rows, 1) - 1) / @max(geometry.score_rows, 1);
            if (blocks > Workspace.max_index_blocks) return error.IndexPipeTooManyBlocks;
            for (0..blocks) |block| out.index_pipe_ready[block] = try cuda.Event.init(driver, false);
            out.index_pipe_done = try cuda.Event.init(driver, false);
        }
        if (geometry.prefill_bf16) {
            out.dequant_stream = try cuda.Stream.init(driver, true);
            out.dequant_ready = try cuda.Event.init(driver, false);
            out.dequant_consumed = try cuda.Event.init(driver, false);
            try out.dequant_consumed.?.record(out.dequant_stream.?);
        }
        return out;
    }

    fn mapVmmSegment(vmm: *cuda.arena.Vmm, at: u64, len: u64, tail: bool, b: u64, tail_cursor: *u64) !void {
        if (len == 0) return;
        const h = try vmm.createHandle(@intCast(len));
        errdefer vmm.releaseHandle(h);
        try vmm.mapHandle(at, @intCast(len), h);
        if (tail) {
            errdefer vmm.unmapRange(at, @intCast(len));
            try vmm.mapHandle(b + tail_cursor.*, @intCast(len), h);
            tail_cursor.* += len;
        }
        vmm.releaseHandle(h);
    }

    pub fn deinit(s: *State) void {
        if (s.tbo_stream) |stream| stream.synchronize() catch {};
        for (0..2) |slot| {
            if (s.pipe_ready[slot]) |*event| event.deinit();
            if (s.pipe_done[slot]) |*event| event.deinit();
        }
        for (0..Workspace.moe_pipe_ring_slots) |slot| {
            if (s.moe_pipe_ready[slot]) |*event| event.deinit();
            if (s.moe_pipe_done[slot]) |*event| event.deinit();
        }
        for (&s.moe_pipe_pdone) |*maybe_event| if (maybe_event.*) |*event| event.deinit();
        for (&s.index_pipe_ready) |*maybe_event| if (maybe_event.*) |*event| event.deinit();
        if (s.index_pipe_done) |*event| event.deinit();
        for (0..2) |half| {
            if (s.tbo_ready[half]) |*event| event.deinit();
            if (s.tbo_done[half]) |*event| event.deinit();
        }
        if (s.tbo_stream) |*stream| stream.deinit();
        if (s.side) |stream| stream.synchronize() catch {};
        if (s.fork) |*event| event.deinit();
        if (s.kv) |*event| event.deinit();
        if (s.fork2) |*event| event.deinit();
        if (s.shared) |*event| event.deinit();
        if (s.side) |*stream| stream.deinit();
        if (s.dequant_ready) |*event| event.deinit();
        if (s.dequant_consumed) |*event| event.deinit();
        if (s.dequant_stream) |*stream| stream.deinit();
        if (s.adapt) |*ad| {
            ad.vmm.unmapRange(ad.b, @intCast(ad.b_len));
            ad.vmm.unmapRange(ad.kv_va, @intCast(ad.kv_len));
            ad.vmm.close();
        }
        s.stage_plane.free();
        s.park.free();
        s.workspace.free();
        s.kv_arena.free(); // latent, rope and index_keys are spans of it
        s.* = undefined;
    }

    /// Review N2: a forward over `layers` layers (`full` of them with their own indexer) must fit these planes. The
    /// forward indexes planes by a layer's position among the loaded layers (0..n-1), not its checkpoint index, so a
    /// subset that does not start at 0 maps to planes 0..n-1.
    pub fn checkLayers(s: *const State, layers: usize, full: usize) !void {
        if (layers > s.layers) return error.LayerOutOfPlanes;
        if (full > s.full_index_layers) return error.IndexPlaneOutOfRange;
    }

    pub const ParkDir = enum { save, restore };

    pub fn parkCopy(s: *const State, stream: cuda.Stream, dir: ParkDir, first: usize, count: usize) !void {
        if (count == 0) return;
        if (count > s.park_slots or first + count > s.slots) return error.BadParkSlots;
        const bases = [_]u64{ s.latent.ptr, s.rope.ptr, s.index_keys.ptr };
        for (0..parkPlaneCount(s.layers, s.full_index_layers)) |i| {
            const seg = parkSegment(i, s.layers, s.full_index_layers, s.slots, s.park_slots, bases, s.park.ptr, first, count, s.park_tail_slot);
            const dst = if (dir == .save) seg.park else seg.plane;
            const src = if (dir == .save) seg.plane else seg.park;
            try s.latent.d.check(s.latent.d.api.cuMemcpyDtoDAsync_v2(dst, src, @intCast(seg.len), stream.handle), "glm53 side park copy");
        }
    }

    /// Plane `layer`'s latent rows [slots][512] bf16 (`layer` < layers; Forward checks with checkLayers first).
    pub fn latentAt(s: *const State, layer: usize) u64 {
        std.debug.assert(layer < s.layers);
        return s.latent.ptr + @as(u64, @intCast(layer * s.slots * mla.latent_dim * 2));
    }
    /// Layer `layer`'s rope plane [slots][64] bf16.
    pub fn ropeAt(s: *const State, layer: usize) u64 {
        std.debug.assert(layer < s.layers);
        return s.rope.ptr + @as(u64, @intCast(layer * s.slots * mla.rope_dim * 2));
    }
    /// Full-indexer layer ordinal `ordinal`'s key plane [slots][128] bf16.
    pub fn indexAt(s: *const State, ordinal: usize) u64 {
        std.debug.assert(ordinal <= s.full_index_layers);
        return s.index_keys.ptr + @as(u64, @intCast(ordinal * s.slots * index.dim * 2));
    }
};

test "DCP2 bf16 plane byte counts match design's 262k cache rows" {
    const p = planeBytes(262144, 78, 21, .bf16);
    try std.testing.expectEqual(@as(usize, 131073), p.slots);
    try std.testing.expectEqual(@as(u64, 10_469_062_656), p.latent);
    try std.testing.expectEqual(@as(u64, 1_308_632_832), p.rope);
    try std.testing.expectEqual(@as(u64, 704_648_448), p.index);
}

test "side park sizing is admitted as a rounded state arena" {
    try std.testing.expectEqual(@as(u64, 1_560_281_088), parkBytes(78, 21, 32768));
    const base: Geometry = .{};
    var off = base;
    off.park_rows = 0;
    var parked = base;
    parked.park_rows = 32768;
    try std.testing.expectEqual(stateBytes(base, 262144), stateBytes(off, 262144));
    try std.testing.expectEqual(breakdown(base, 262144, .{}).logicalState(), breakdown(off, 262144, .{}).logicalState());
    try std.testing.expectEqual(breakdown(base, 262144, .{}).logicalState() + parkBytes(78, 21, 32768), breakdown(parked, 262144, .{}).logicalState());
    try std.testing.expectEqual(arenaBytes(parkBytes(78, 21, 32768)), stateArenas(parked, 262144).park);
    try std.testing.expectEqual(stateBytes(base, 262144) + arenaBytes(parkBytes(78, 21, 32768)), stateBytes(parked, 262144));
}

test "park segments map every layer-major KV plane into disjoint park blocks" {
    const layers: usize = 3;
    const full: usize = 2;
    const slots: usize = 100;
    const cap: usize = 8;
    const first: usize = 5;
    const count: usize = 8;
    const bases = [_]u64{ 1000, 500000, 700000 };
    const plane_count = parkPlaneCount(layers, full);
    var len_sum: u64 = 0;
    var last_end: u64 = 0;
    for (0..plane_count) |i| {
            const seg = parkSegment(i, layers, full, slots, cap, bases, 900000, first, count, 0);
        try std.testing.expect(seg.park >= last_end);
        last_end = seg.park + seg.len;
        len_sum += seg.len;
        const row: usize = if (i < layers) mla.latent_dim * 2 else if (i < 2 * layers) mla.rope_dim * 2 else index.dim * 2;
        const ord = if (i < layers) i else if (i < 2 * layers) i - layers else i - 2 * layers;
        const base = if (i < layers) bases[0] else if (i < 2 * layers) bases[1] else bases[2];
        try std.testing.expectEqual(base + @as(u64, @intCast(ord * slots * row + first * row)), seg.plane);
    }
    const row_total: u64 = @as(u64, @intCast(layers * (mla.latent_dim * 2 + mla.rope_dim * 2) + full * index.dim * 2));
    try std.testing.expectEqual(@as(u64, @intCast(count)) * row_total, len_sum);
    try std.testing.expect(last_end <= 900000 + parkBytes(layers, full, cap * 2));
}

test "side rows environment parser accepts off and aligned in-context capacities" {
    try std.testing.expectEqual(@as(usize, 0), try parseSideRows(null, 512, 8192));
    try std.testing.expectEqual(@as(usize, 0), try parseSideRows("", 512, 8192));
    try std.testing.expectEqual(@as(usize, 4096), try parseSideRows("4096", 512, 8192));
    try std.testing.expectError(error.BadSideRows, parseSideRows("513", 512, 8192));
}

test "adaptive expert tail planner chooses the largest fitting KV slot" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 2048, 4096 }) |rows| {
        const g: Geometry = .{ .expert_rows = 1024, .chunk_rows = 4096, .piece_rows = 512, .expert_rows_adapt = rows };
        const p = try planAdapt(gpa, g, 262144, 2 << 20);
        defer gpa.free(p.tails);
        try std.testing.expect(p.pool >= p.need);
        try std.testing.expectEqual(stateArenas(g, 262144).kv, p.kv_bytes);
        const planes = planeBytes(262144, g.layers, g.full_index_layers, g.cache_format);
        const bases = [_]u64{ 0, std.mem.alignForward(u64, planes.latent, tensor_alignment), std.mem.alignForward(u64, std.mem.alignForward(u64, planes.latent, tensor_alignment) + planes.rope, tensor_alignment) };
        const lens = [_]u64{ planes.latent, planes.rope, planes.index };
        const elem: u64 = if (g.cache_format == .bf16) 2 else 1;
        const row = [_]u64{ mla.latent_dim * elem, mla.rope_dim * elem, index.dim * elem };
        var cursor: u64 = 0;
        for (p.tails) |tail| {
            try std.testing.expect(tail.kv_offset >= cursor);
            try std.testing.expectEqual(@as(u64, 0), tail.kv_offset % (2 << 20));
            try std.testing.expectEqual(@as(u64, 0), tail.len % (2 << 20));
            try std.testing.expect(tail.len > 0);
            var contained = false;
            for (0..3) |i| {
                if (lens[i] == 0) continue;
                const stride = @as(u64, @intCast(planes.slots)) * row[i];
                const layer_n = lens[i] / stride;
                var l: u64 = 0;
                while (l < layer_n) : (l += 1) {
                    const lb = bases[i] + l * stride;
                    if (tail.kv_offset >= lb + @as(u64, @intCast(p.limit_slot)) * row[i] and tail.kv_offset + tail.len <= lb + stride - row[i]) contained = true;
                }
            }
            try std.testing.expect(contained);
            cursor = tail.kv_offset + tail.len;
        }
        try std.testing.expect(poolForTest(g, 262144, p.limit_slot + 1, 2 << 20) < p.need);
        std.debug.print("adapt rows={d} limit_slot={d} limit_position={d}\n", .{ rows, p.limit_slot, p.limit_position });
    }
    const clipped = try planAdapt(gpa, .{ .expert_rows = 1024, .chunk_rows = 4096, .piece_rows = 512, .expert_rows_adapt = 2048, .expert_rows_adapt_maxpos = 1024 }, 262144, 2 << 20);
    defer gpa.free(clipped.tails);
    try std.testing.expectEqual(@as(usize, 1024), clipped.limit_position);
}

test "adaptive workspace only replaces prompt batch regions" {
    const g: Geometry = .{ .expert_rows = 1024, .chunk_rows = 4096, .piece_rows = 512, .expert_rows_adapt = 2048, .scratch_overlay = true };
    const base = workspaceLayout(g, 262144);
    const b: u64 = 0x100000000;
    const w = adaptWorkspace(base, b, g, 262144);
    const ref = workspaceLayout(.{ .expert_rows = 2048, .chunk_rows = 4096, .piece_rows = 512, .prompt_experts = true }, 262144);
    try std.testing.expect(w.adapt_big);
    try std.testing.expectEqual(@as(usize, 2048), w.expert_rows);
    try std.testing.expect(!w.scratch_overlay);
    try std.testing.expectEqual(ref.sizes.xn, w.sizes.xn);
    try std.testing.expectEqual(ref.sizes.picks8, w.sizes.picks8);
    try std.testing.expectEqual(ref.sizes.wts8, w.sizes.wts8);
    try std.testing.expectEqual(ref.sizes.shared_out, w.sizes.shared_out);
    try std.testing.expectEqual(ref.sizes.routed_out, w.sizes.routed_out);
    try std.testing.expect(w.sizes.prompt_experts >= cuda.exl3_experts.prompt.bytes(expertShape(), 2048, routed_slots, false) catch unreachable);
    try std.testing.expect(w.xn >= b and w.xn + w.sizes.xn <= b + adaptNeedBytes(g, 262144));
    try std.testing.expectEqual(base.x, w.x);
    try std.testing.expectEqual(base.gathered, w.gathered);
    try std.testing.expectEqual(base.positions, w.positions);
    try std.testing.expectEqual(base.phase, w.phase);
    try std.testing.expectEqual(base.experts, w.experts);
    try std.testing.expectEqual(base.smla_union, w.smla_union);
    const starts = [_]u64{ w.xn, w.picks8, w.wts8, w.shared_out, w.routed_out, w.lowrank_mid, w.prompt_plan, w.prompt_experts };
    const lens = [_]u64{ w.sizes.xn, w.sizes.picks8, w.sizes.wts8, w.sizes.shared_out, w.sizes.routed_out, w.sizes.lowrank_mid, w.sizes.prompt_plan, w.sizes.prompt_experts };
    for (0..starts.len) |i| {
        try std.testing.expect(starts[i] >= b and starts[i] + lens[i] <= b + adaptNeedBytes(g, 262144));
        for (i + 1..starts.len) |j| {
            try std.testing.expect(starts[i] + Workspace.align256(lens[i]) <= starts[j] or starts[j] + Workspace.align256(lens[j]) <= starts[i]);
        }
    }
}

test "adaptive workspace is RoCE-aligned for production geometries" {
    const contexts = [_]usize{ 65536, 131072, 262144 };
    const cases = [_]struct { expert_rows: usize, chunk_rows: usize, adapt: usize }{
        .{ .expert_rows = 1024, .chunk_rows = 2048, .adapt = 2048 },
        .{ .expert_rows = 1024, .chunk_rows = 4096, .adapt = 2048 },
        .{ .expert_rows = 1024, .chunk_rows = 4096, .adapt = 4096 },
    };
    for (contexts) |context| {
        for (cases) |case| {
            const g: Geometry = .{ .expert_rows = case.expert_rows, .chunk_rows = case.chunk_rows, .piece_rows = 512, .expert_rows_adapt = case.adapt, .scratch_overlay = true, .prompt_experts = true };
            if (!validExpertRows(g) or !groupedExperts(g.expert_rows, g.piece_rows)) continue;
            const arenas = stateArenas(g, context);
            try std.testing.expectEqual(@as(u64, 0), arenas.kv % 256);
            const b: u64 = 0x7f00_0000_0000 + arenas.kv;
            const base = Workspace.carve(0x7e00_0000_0000, g, context);
            try std.testing.expect(adaptRoceAligned(adaptWorkspace(base, b, g, context)));
        }
    }
}

test "a misaligned adapt base is caught" {
    const g: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .piece_rows = 512, .expert_rows_adapt = 2048, .scratch_overlay = true, .prompt_experts = true };
    try std.testing.expect(validExpertRows(g));
    try std.testing.expect(groupedExperts(g.expert_rows, g.piece_rows));
    const context: usize = 65536;
    const arenas = stateArenas(g, context);
    const b: u64 = 0x7f00_0000_0000 + arenas.kv;
    const base = Workspace.carve(0x7e00_0000_0000, g, context);
    try std.testing.expect(!adaptRoceAligned(adaptWorkspace(base, b + 8, g, context)));
}

test "adaptive switches explicitly off preserve workspace and state geometry" {
    for ([_]Geometry{ .{ .expert_rows = 1024, .chunk_rows = 2048 }, .{ .expert_rows = 4096, .chunk_rows = 4096 } }) |g| {
        var explicit = g;
        explicit.expert_rows_adapt = 0;
        explicit.expert_rows_adapt_maxpos = 0;
        try std.testing.expect(std.meta.eql(workspaceLayout(g, 262144), workspaceLayout(explicit, 262144)));
        try std.testing.expectEqual(stateBytes(g, 262144), stateBytes(explicit, 262144));
    }
}

fn poolForTest(g: Geometry, context: usize, slots: usize, granularity: u64) u64 {
    // Independent oracle: walk every layer's region of each plane (layer-major, as State.latentAt).
    const p = planeBytesFor(g, context);
    const bases = [_]u64{ 0, std.mem.alignForward(u64, p.latent, tensor_alignment), std.mem.alignForward(u64, std.mem.alignForward(u64, p.latent, tensor_alignment) + p.rope, tensor_alignment) };
    const elem: u64 = if (g.cache_format == .bf16) 2 else 1;
    const rows = [_]u64{ mla.latent_dim * elem, mla.rope_dim * elem, index.dim * elem };
    const layers = [_]usize{ g.layers, g.layers, g.full_index_layers };
    var sum: u64 = 0;
    for (0..3) |i| {
        const stride = @as(u64, @intCast(p.slots)) * rows[i];
        var l: usize = 0;
        while (l < layers[i]) : (l += 1) {
            const lb = bases[i] + @as(u64, @intCast(l)) * stride;
            const start = std.mem.alignForward(u64, lb + @as(u64, @intCast(slots)) * rows[i], granularity);
            const end = std.mem.alignBackward(u64, lb + stride - rows[i], granularity);
            if (end > start) sum += end - start;
        }
    }
    return sum;
}

test "stage plane sizing: one pair in order, two pairs with a prefetch stream, aliasing refused when prefetching" {
    const one: Geometry = .{};
    const two: Geometry = .{ .prefetch = true };
    try std.testing.expectEqual(@as(u64, 131072 * 1152), stagePlaneBytes(one, 262144));
    try std.testing.expectEqual(@as(u64, 2 * 131072 * 1152), stagePlaneBytes(two, 262144));
    try std.testing.expectEqual(@as(usize, 500), stageRows(999));
    const p = stagePlanes(undefined, 0x10000, 262144, 2);
    try validateStage(&p, true);
    try std.testing.expectEqual(p.buffers[1].latent.data.ptr, stageLatent(&p, 77));
    try std.testing.expectEqual(p.buffers[0].latent.data.ptr + 131072 * 1024, stageRope(&p, 2));
    const single = stagePlanes(undefined, 0x10000, 262144, 1);
    try std.testing.expectEqual(stageLatent(&single, 0), stageLatent(&single, 1));
    try validateStage(&single, false);
    try std.testing.expectError(error.StageBuffersAliased, validateStage(&single, true));
    var overlap = p;
    overlap.buffers[1].latent.data.ptr -= 4096;
    try std.testing.expectError(error.StageBuffersAliased, validateStage(&overlap, true));
}

test "workspace regions are aligned, inside the allocation, and phases share one union region" {
    const g: Geometry = .{};
    const w = Workspace.carve(0x10000, g, 262144);
    const info = @typeInfo(Workspace).@"struct";
    inline for (info.field_names, info.field_types) |name, T| {
        // prefill_weight is zero-sized (not a region) unless prefill_bf16 is on.
        if (comptime (T == u64 and !std.mem.eql(u8, name, "bytes") and !std.mem.eql(u8, name, "phase") and !std.mem.eql(u8, name, "phase_bytes"))) {
            const v = @field(w, name);
            if (@field(w.sizes, name) != 0) {
                try std.testing.expect(v % 256 == 0);
                try std.testing.expect(v >= 0x10000 and v < 0x10000 + w.bytes);
            }
        }
    }
    try std.testing.expectEqual(w.phase, w.q_a);
    try std.testing.expect(w.phase - 0x10000 + w.phase_bytes <= w.bytes);
    // FFN and head regions start where the attention region starts.
    try std.testing.expectEqual(w.q_a, w.router_logits);
    try std.testing.expectEqual(w.q_a, w.logits);
    try std.testing.expectEqual(@as(u64, 64 * 131073 * 4), w.sizes.scores);
    try std.testing.expect(w.sizes.part >= 3 * 8 * decode_partial_row_bytes and w.sizes.part >= 512 * 32 * 1028);
    // Clusters: no split-K partial buffer (4 B placeholder).
    try std.testing.expectEqual(@as(u64, 4), w.sizes.lane_part);
    try std.testing.expectEqual(@as(u64, 4), w.sizes.lane_part_side);
    const decode_scratch = Workspace.carve(0x20000, .{ .decode_rows = 8, .piece_rows = 512, .lane_clusters = false }, 8192);
    try std.testing.expectEqual(lanePartBytes(8, false), decode_scratch.sizes.lane_part_side);
    // Per-piece reduce, fused-selection slots/counts, chunk-wide token list and row inputs.
    try std.testing.expectEqual(@as(u64, 2 * 512 * 6144 * 4), w.sizes.gathered);
    try std.testing.expectEqual(@as(u64, 4096 * 2048 * 4), w.sizes.local_slots);
    try std.testing.expectEqual(@as(u64, 4096 * 2048 * 4), w.sizes.global_tokens);
    try std.testing.expectEqual(@as(u64, 4096 * 6144 * 2), w.sizes.x);
    try std.testing.expectEqual(@as(u64, 0), w.sizes.prefill_weight);
    const bf16 = Workspace.carve(0x10000, .{ .prefill_bf16 = true }, 262144);
    try std.testing.expectEqual(prefill_weight_scratch_bytes, bf16.sizes.prefill_weight);
    try std.testing.expectEqual(@as(u64, 0x10000) + bf16.bytes, bf16.prefill_weight + bf16.sizes.prefill_weight);
    try std.testing.expectEqual(@as(u64, 150_994_944), prefill_weight_scratch_bytes);
    try std.testing.expectEqual(@as(usize, 32_896), decode_partial_row_bytes);
}

test "drafter overlay extends only the phase union when larger than it" {
    const base = Workspace.carve(0, .{}, 262144);
    const extra: u64 = base.phase_bytes + 8192;
    const shared = Workspace.carve(0, .{ .drafter_share_bytes = extra }, 262144);
    try std.testing.expectEqual(base.bytes + 8192, shared.bytes);
    try std.testing.expectEqual(extra, shared.phase_bytes);
}

test "graph reserve environment value is MiB with checked conversion" {
    try std.testing.expectEqual(graphs_runtime_placeholder, try graphsReserveMiB("1024"));
    try std.testing.expectEqual(@as(u64, 0), try graphsReserveMiB("0"));
    try std.testing.expectError(error.BadGraphReserve, graphsReserveMiB("nope"));
    try std.testing.expectError(error.BadGraphReserve, graphsReserveMiB("18446744073709551615"));
}

test "memory admission includes graph and runtime reservations at the exact threshold" {
    const g: Geometry = .{ .layers = 1, .full_index_layers = 0, .chunk_rows = 16, .piece_rows = 16, .index_slots = 8, .decode_rows = 1, .graphs_bytes = 11, .runtime_bytes = 13, .expert_rows = 16, .head_rows = 1, .vocab_local = 16 };
    const reserve = reserveForWithMargin(0, false, 0, try serveMarginBytes(null));
    const b = breakdown(g, 8, .{});
    const exact = b.total() + reserve.bytes;
    _ = try admitRank(g, 8, .{}, exact, reserve);
    try std.testing.expectError(error.MemoryBudget, admitRank(g, 8, .{}, exact - 1, reserve));
}

test "graph buckets and per-graph byte accounting scale with context and verify rows" {
    try std.testing.expectEqual(@as(usize, 1), bucketCount(2048));
    try std.testing.expectEqual(@as(usize, 2), bucketCount(2049));
    try std.testing.expectEqual(@as(usize, 2), bucketCount(4096));
    try std.testing.expectEqual(@as(usize, 7), bucketCount(131072));
    try std.testing.expectEqual(@as(usize, 8), bucketCount(262144));
    try std.testing.expectEqual(@as(u64, 8 * 8 * 64 * (1 << 20)), try graphBytes(8, 262144, 64));
    try std.testing.expectEqual(@as(u64, 4 * 7 * 64 * (1 << 20)), try graphBytes(4, 131072, 64));
}

test "MEMSLIM graph rows and live graph plan follow the budget" {
    try std.testing.expectEqual(@as(usize, 4), try graphLiveRows(4, 256, 64));
    try std.testing.expectEqual(@as(usize, 8), try graphLiveRows(8, null, 64));
    try std.testing.expectEqual(@as(usize, 1), try graphLiveRows(8, 100, 64));
    try std.testing.expectEqual(@as(usize, 0), try graphLiveRows(8, 0, 64));
    try std.testing.expectError(error.BadGraphMiB, graphLiveRows(8, 100, 0));
    var shapes: [16]GraphShape = undefined;
    try std.testing.expectEqual(@as(usize, 16), try graphLivePlan(262144, 2, &.{ 8, 1, 2 }, &shapes));
    try std.testing.expectEqual(GraphShape{ .rows = 8, .bucket = 2048 }, shapes[0]);
    try std.testing.expectEqual(GraphShape{ .rows = 8, .bucket = 262144 }, shapes[7]);
    try std.testing.expectEqual(GraphShape{ .rows = 1, .bucket = 2048 }, shapes[8]);
    try std.testing.expectError(error.BadGraphRows, graphLivePlan(262144, 4, &.{ 8, 1 }, &shapes));
    try std.testing.expectEqual(bucketCount(1024), try graphLivePlan(1024, 1, &.{8}, &shapes));
    try std.testing.expectEqual(GraphShape{ .rows = 8, .bucket = 2048 }, shapes[0]);
}

test "MEMSLIM switch accepts only unset, zero and one" {
    try std.testing.expect(!(try memslimFrom(null)));
    try std.testing.expect(!(try memslimFrom("0")));
    try std.testing.expect(try memslimFrom("1"));
    try std.testing.expectError(error.BadMemslim, memslimFrom("2"));
}

test "runtime, graph and serving margin parsers reject malformed values" {
    try std.testing.expectEqual(default_runtime_reserve_mib * (1 << 20), try runtimeReserveFrom(null));
    try std.testing.expectError(error.BadRuntimeReserve, runtimeReserveMiB("bad"));
    try std.testing.expectError(error.BadRuntimeReserve, runtimeReserveMiB("18446744073709551615"));
    try std.testing.expectError(error.BadGraphMiB, graphMiB("bad"));
    try std.testing.expectError(error.BadServeMargin, serveMarginBytes("bad"));
    try std.testing.expectError(error.BadServeMargin, serveMarginBytes("-1"));
    try std.testing.expectEqual(@as(u64, 4096 << 20), try serveMarginBytes(null));
}

test "serving gate thresholds default to absolute GiB values and parse exact overrides" {
    try std.testing.expectEqual(gibBytes(default_serve_min_avail_gib), try serveMinAvailableBytes(null));
    try std.testing.expectEqual(gibBytes(default_serve_min_free_gib), try serveMinFreeBytes(null));
    try std.testing.expectEqual(gibBytes(12.5), try serveMinAvailableBytes("12.5"));
    try std.testing.expectEqual(gibBytes(3.25), try serveMinFreeBytes("3.25"));
}

test "serving gate threshold parsers reject malformed and out of range values" {
    try std.testing.expectError(error.BadServeMinAvail, serveMinAvailableBytes("bad"));
    try std.testing.expectError(error.BadServeMinAvail, serveMinAvailableBytes("-1"));
    try std.testing.expectError(error.BadServeMinAvail, serveMinAvailableBytes("NaN"));
    try std.testing.expectError(error.BadServeMinAvail, serveMinAvailableBytes("inf"));
    try std.testing.expectError(error.BadServeMinAvail, serveMinAvailableBytes("1025"));
    try std.testing.expectError(error.BadServeMinFree, serveMinFreeBytes("bad"));
    try std.testing.expectError(error.BadServeMinFree, serveMinFreeBytes("-1"));
    try std.testing.expectError(error.BadServeMinFree, serveMinFreeBytes("NaN"));
    try std.testing.expectError(error.BadServeMinFree, serveMinFreeBytes("inf"));
    try std.testing.expectError(error.BadServeMinFree, serveMinFreeBytes("1025"));
}

test "serving gate respects low free and available thresholds including exact boundaries" {
    try std.testing.expectEqual(GateVerdict.ok, serveGate(gibBytes(9), gibBytes(5), gibBytes(8), gibBytes(4)));
    try std.testing.expectEqual(GateVerdict.low_available, serveGate(gibBytes(7.9), gibBytes(5), gibBytes(8), gibBytes(4)));
    try std.testing.expectEqual(GateVerdict.low_free, serveGate(gibBytes(9), gibBytes(3.9), gibBytes(8), gibBytes(4)));
    try std.testing.expectEqual(GateVerdict.ok, serveGate(gibBytes(8), gibBytes(4), gibBytes(8), gibBytes(4)));
}

test "expert batch workspace scales monotonically with admitted rows" {
    var previous: u64 = 0;
    for ([_]struct { expert_rows: usize, chunk_rows: usize }{
        .{ .expert_rows = 128, .chunk_rows = 4096 },
        .{ .expert_rows = 512, .chunk_rows = 4096 },
        .{ .expert_rows = 2048, .chunk_rows = 4096 },
        .{ .expert_rows = 4096, .chunk_rows = 4096 },
        .{ .expert_rows = 8192, .chunk_rows = 8192 },
    }) |case| {
        const rows = case.expert_rows;
        const g: Geometry = .{ .expert_rows = rows, .chunk_rows = case.chunk_rows };
        const w = Workspace.carve(0x10000, g, 32768);
        try std.testing.expectEqual(rows, w.expert_rows);
        try std.testing.expectEqual(cuda.exl3_experts.scratchBytes(expertShape(), rows, routed_slots) catch unreachable, w.sizes.experts);
        // 672,768 B per row plus the scratch alignment padding (under 4 KiB in total)
        try std.testing.expect(w.sizes.experts >= @as(u64, 672_768) * rows and w.sizes.experts - @as(u64, 672_768) * rows < 4096);
        try std.testing.expect(w.bytes >= previous);
        previous = w.bytes;
        try std.testing.expectEqual(@as(u64, 0), w.xn % 256);
        try std.testing.expectEqual(@as(u64, 0), w.routed_out % 256);
    }
}

test "8192 chunk row workspace buffers cover the full batch" {
    const w = Workspace.carve(0x10000, .{ .chunk_rows = 8192 }, 32768);
    try std.testing.expectEqual(@as(usize, 8192), w.rows);
    try std.testing.expectEqual(@as(u64, 8192 * hidden * 2), w.sizes.x);
    try std.testing.expectEqual(@as(u64, 8192 * 8), w.sizes.positions);
    try std.testing.expectEqual(@as(u64, 8192 * top_k * 4), w.sizes.global_tokens);
    try std.testing.expectEqual(@as(u64, 8192 * top_k * 4), w.sizes.local_slots);
    try std.testing.expectEqual(@as(u64, 8192 * 4), w.sizes.ids);
    try std.testing.expectEqual(@as(u64, 8192 * 4), w.sizes.owners);
}

test "expert rows admit through 8192 and reject unsupported shapes" {
    for ([_]usize{ 4096, 8192 }) |rows| try std.testing.expect(validExpertRows(.{ .expert_rows = rows, .chunk_rows = rows }));
    try std.testing.expect(!validExpertRows(.{ .expert_rows = 8192, .chunk_rows = 4096 }));
    try std.testing.expectError(error.ExpertRowsExceedChunkRows, admitRank(.{ .expert_rows = 8192, .chunk_rows = 4096 }, 8192, .{}, 1 << 40, reserveFor(1 << 40, false, 0)));
    try std.testing.expect(!validExpertRows(.{ .expert_rows = 16384 }));
    try std.testing.expect(!validExpertRows(.{ .expert_rows = 4100 }));
}

test "expert batch carve totals at 2048 4096 and 8192 rows" {
    const context: usize = 32768;
    inline for (.{ false, true }) |overlay| {
        inline for (.{ false, true }) |prompt| {
            inline for (.{ 2048, 4096, 8192 }) |rows| {
                const w = Workspace.carve(0, .{ .expert_rows = rows, .piece_rows = 512, .chunk_rows = rows, .prompt_experts = prompt, .scratch_overlay = overlay }, context);
                std.debug.print("INFO expert-carve rows={d} overlay={s} prompt={s}: {d} bytes/rank\n", .{ rows, if (overlay) "on" else "off", if (prompt) "on" else "off", w.bytes });
            }
        }
    }
}

test "prompt experts admission adds prompt.bytes exactly and is zero when disabled" {
    const context: usize = 32768;
    const base_geometry: Geometry = .{ .expert_rows = 128 };
    const prompt_geometry: Geometry = .{ .expert_rows = 128, .prompt_experts = true };
    const off = workspaceLayout(base_geometry, context);
    const on = workspaceLayout(prompt_geometry, context);
    const expected = cuda.exl3_experts.prompt.bytes(expertShape(), off.expert_rows, routed_slots, false) catch unreachable;
    try std.testing.expectEqual(@as(u64, 0), off.sizes.prompt_experts);
    try std.testing.expectEqual(@as(u64, 0), off.sizes.prompt_plan);
    try std.testing.expectEqual(@as(u64, @intCast(expected)), on.sizes.prompt_experts);
    try std.testing.expectEqual(@as(u64, 0), on.sizes.prompt_plan);
    try std.testing.expect(promptPlanBytes(on.expert_rows, routed_slots, routed_experts) <= on.sizes.experts);
    try std.testing.expectEqual(@as(u64, @intCast(expected)), on.bytes - off.bytes);
    const b_off = breakdown(base_geometry, context, .{});
    const b_on = breakdown(prompt_geometry, context, .{});
    try std.testing.expectEqual(@as(u64, @intCast(expected)), b_on.logicalState() - b_off.logicalState());
}

test "scratch overlay relocates the off prompt layout after plan inside ws.experts" {
    const context: usize = 32768;
    const baseline = Workspace.carve(0x10000, .{ .expert_rows = 128 }, context);
    const off = Workspace.carve(0x10000, .{ .expert_rows = 128, .prompt_experts = true }, context);
    const over = Workspace.carve(0x10000, .{ .expert_rows = 128, .prompt_experts = true, .scratch_overlay = true }, context);
    const prompt_bytes = cuda.exl3_experts.prompt.bytes(expertShape(), 128, routed_slots, false) catch unreachable;
    const over_bytes = cuda.exl3_experts.prompt.bytes(expertShape(), 128, routed_slots, true) catch unreachable;
    const plan = std.mem.alignForward(u64, promptPlanBytes(128, routed_slots, routed_experts), 256);
    try std.testing.expectEqual(@as(u64, 0), baseline.sizes.prompt_experts);
    try std.testing.expectEqual(@as(u64, 0), baseline.sizes.prefill_weight);
    try std.testing.expectEqual(@as(u64, @intCast(prompt_bytes)), off.sizes.prompt_experts);
    try std.testing.expectEqual(baseline.bytes + @as(u64, @intCast(prompt_bytes)), off.bytes);
    try std.testing.expectEqual(over.experts + @as(u64, @intCast(plan)), over.prompt_experts);
    try std.testing.expect(over.prompt_plan + @as(u64, @intCast(plan)) <= over.prompt_experts);
    try std.testing.expectEqual(@as(u64, @intCast(over_bytes)), over.sizes.prompt_experts);
    try std.testing.expect(over.prompt_experts + over.sizes.prompt_experts <= over.experts + over.sizes.experts);
    try std.testing.expectEqual(@as(u64, @intCast(prompt_bytes)), @as(u64, @intCast(over_bytes)));
    try std.testing.expectEqual(off.sizes.prompt_experts, over.sizes.prompt_experts);
    const layer = expertShape();
    const sc = cuda.exl3_experts.prompt.scratch(0x100000, layer, 128, routed_slots, true) catch unreachable;
    const pairs = @as(u64, 128 * routed_slots);
    const off_sc = cuda.exl3_experts.prompt.scratch(0x200000, layer, 128, routed_slots, false) catch unreachable;
    try std.testing.expectEqual(off_sc.xg - 0x200000, sc.xg - 0x100000);
    try std.testing.expectEqual(off_sc.xu - 0x200000, sc.xu - 0x100000);
    try std.testing.expectEqual(off_sc.xd - 0x200000, sc.xd - 0x100000);
    try std.testing.expectEqual(off_sc.y - 0x200000, sc.y - 0x100000);
    try std.testing.expectEqual(off_sc.uu - 0x200000, sc.uu - 0x100000);
    try std.testing.expectEqual(off_sc.ids - 0x200000, sc.ids - 0x100000);
    try std.testing.expectEqual(@as(u64, @intCast(prompt_bytes)) - pairs * layer.dims * 4, sc.y - sc.xg);
    for ([_]usize{ 2048, 4096 }) |rows| {
        const old = Workspace.carve(0, .{ .expert_rows = rows, .prompt_experts = true }, context);
        const new = Workspace.carve(0, .{ .expert_rows = rows, .prompt_experts = true, .scratch_overlay = true }, context);
        try std.testing.expect(new.bytes <= old.bytes);
        std.debug.print("INFO scratch-overlay rows={d}: saved {d} bytes\n", .{ rows, old.bytes - new.bytes });
    }
}

test "prompt experts reject an enabled switch without carved scratch" {
    const off = workspaceLayout(.{ .expert_rows = 128 }, 32768);
    try std.testing.expectError(error.PromptExpertsNotCarved, validatePromptExpertsCarve(true, &off));
    try validatePromptExpertsCarve(false, &off);
}

test "prompt grouped plan sizing uses the shared 64 tile layout" {
    for ([_]usize{ 128, 512, 1024, 1536, 2048, 4096, 8192 }) |expert_rows| {
        const layout = promptPlanLayout(expert_rows, routed_slots, routed_experts);
        try std.testing.expectEqual(promptPlanBytes(expert_rows, routed_slots, routed_experts), layout.bytes);
        try std.testing.expect(layout.bytes <= workspaceLayout(.{ .expert_rows = expert_rows, .chunk_rows = expert_rows }, 32768).sizes.experts);
    }
}

test "expert batch carve aligns and separates every FFN region and preserves scratch row cost" {
    const base: u64 = 0x10000;
    const persistent = .{ "x", "xn", "gathered", "global_tokens", "global_counts", "local_slots", "local_counts", "roce_token", "roce_status", "positions", "dense_len", "cos", "sin", "ids", "owners", "lane_part", "lane_part_side" };
    const attention = .{ "q_a", "q_resid", "q", "kv", "ik", "iw", "iq", "qi", "qa", "qpe", "fused_queries", "part", "ol", "lse_out", "attend_split", "vo", "scores", "topk_tokens", "topk_counts", "candidate_records", "gathered_candidates" };
    const head = .{ "logits", "cand_values", "cand_ids", "cand_scratch", "head_gather" };
    const names = .{ "router_logits", "router_part", "picks", "wts", "picks8", "wts8", "shared_out", "routed_out", "lowrank_mid", "mlp_gate_up", "mlp_act", "experts" };
    for ([_]usize{ 128, 512, 2048 }) |er| {
        const w = Workspace.carve(base, .{ .expert_rows = er }, 32768);
        try std.testing.expectEqual(cuda.exl3_experts.scratchBytes(expertShape(), er, 8) catch unreachable, w.sizes.experts);
        try std.testing.expect(w.sizes.experts >= @as(u64, 672_768) * er and w.sizes.experts - @as(u64, 672_768) * er < 4096);
        inline for (@typeInfo(Workspace.Sizes).@"struct".field_names) |name| {
            const ptr = @field(w, name);
            const size = @field(w.sizes, name);
            if (comptime @TypeOf(ptr) == [2]u64) {
                inline for (0..2) |half| {
                    try std.testing.expectEqual(@as(u64, 0), ptr[half] % 256);
                    if (size[half] > 0) try std.testing.expect(ptr[half] >= base and ptr[half] + size[half] <= base + w.bytes);
                }
            } else {
                try std.testing.expectEqual(@as(u64, 0), ptr % 256);
                if (size > 0) try std.testing.expect(ptr >= base and ptr + size <= base + w.bytes);
            }
        }
        try expectWorkspacePhaseNonoverlap(&w, persistent);
        try expectWorkspacePhaseNonoverlap(&w, attention);
        try expectWorkspacePhaseNonoverlap(&w, names);
        try expectWorkspacePhaseNonoverlap(&w, head);
    }
    for ([_]usize{ 128, 2048 }) |er| {
        const w = Workspace.carve(base, .{ .expert_rows = er }, 32768);
        try std.testing.expect(w.sizes.experts >= @as(u64, 672_768) * er and w.sizes.experts - @as(u64, 672_768) * er < 4096);
    }
}

test "grouped expert buffers do not overlap attention phase regions" {
    const base: u64 = 0x10000;
    const attention = .{ "q_a", "q_resid", "q", "kv", "ik", "iw", "iq", "qi", "qa", "qpe", "fused_queries", "part", "ol", "lse_out", "attend_split", "vo", "scores", "topk_tokens", "topk_counts", "candidate_records", "gathered_candidates" };
    const routed = .{ "picks8", "wts8", "shared_out", "lowrank_mid", "routed_out" };
    for ([_]usize{ 1024, 2048 }) |er| {
        const w = Workspace.carve(base, .{ .expert_rows = er, .piece_rows = 512 }, 32768);
        inline for (attention) |left| inline for (routed) |right| {
            const a = @field(w, left);
            const an = @field(w.sizes, left);
            const b = @field(w, right);
            const bn = @field(w.sizes, right);
            if (an != 0 and bn != 0) try std.testing.expect(a + an <= b or b + bn <= a);
        };
    }
}

test "index merge keeps the default carve identical and index pipe reserves one piece" {
    const context = 32768;
    const base = workspaceLayout(.{ .expert_rows = 512 }, context);
    const off = workspaceLayout(.{ .expert_rows = 512, .index_merge = 2 }, context);
    try std.testing.expectEqual(base.bytes, off.bytes);
    try std.testing.expectEqual(base.candidate_records, off.candidate_records);
    try std.testing.expectEqual(base.gathered_candidates, off.gathered_candidates);
    const piped = workspaceLayout(.{ .expert_rows = 512, .piece_rows = 512, .index_pipe = true }, context);
    try std.testing.expectEqual(@as(u64, 512 * 2048 * 8), piped.sizes.candidate_records);
    try std.testing.expectEqual(@as(u64, 2 * 512 * 2048 * 8), piped.sizes.gathered_candidates);
    // The attention phase grows by the piece-sized candidate buffers; the workspace total is the largest phase, so it
    // may stay the same when the FFN phase is larger.
    try std.testing.expect(piped.bytes >= base.bytes);
}

fn expectWorkspacePhaseNonoverlap(w: *const Workspace, comptime names: anytype) !void {
    inline for (names, 0..) |left, i| inline for (names, 0..) |right, j| {
        if (comptime j <= i) continue;
        const a = @field(w, left);
        const an = @field(w.sizes, left);
        const b = @field(w, right);
        const bn = @field(w.sizes, right);
        if (an != 0 and bn != 0) try std.testing.expect(a + an <= b or b + bn <= a);
    };
}

/// Host "device": addresses handed out by a bump counter (no memory behind them), with a live-bytes count.
const Probe = struct {
    var next: usize = 0x1000;
    var live: usize = 0;
    fn alloc(_: *anyopaque, len: usize) ?cuda.abi.DevicePtr {
        const p = next;
        next += std.mem.alignForward(usize, len, 256);
        live += 1;
        return p;
    }
    fn free(_: *anyopaque, _: cuda.abi.DevicePtr) bool {
        live -= 1;
        return true;
    }
};

fn printTables() bool {
    const v = std.c.getenv("TF_GLM53_PRINT_TABLES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

test "262k table accounts graph, runtime and both serving reserves" {
    const g: Geometry = .{
        .graphs_bytes = graphBytes(8, 262144, default_graph_mib) catch unreachable,
        .runtime_bytes = runtimeReserveFrom(null) catch unreachable,
    };
    const target = WeightClasses.designEstimate();
    var drafted = target;
    drafted.drafter = gibBytes(1.3 + 0.24 + 0.1); // design §5 upper end: 4-bit DFlash2 + own draft head + ring
    var shared_drafter = target;
    shared_drafter.drafter = gibBytes(1.3 + 0.1 - 0.30); // WP11 option: drafter shares the target embed and lm_head rows
    // TF_GLM53_PRINT_TABLES=1 prints them (a passing test stays quiet, review N8).
    if (printTables()) for ([_]WeightClasses{ target, drafted, shared_drafter }) |w| breakdown(g, 262144, w).report(262144);
    const b = breakdown(g, 262144, target);
    try std.testing.expectEqual(@as(u64, 10_469_062_656 + 1_308_632_832), b.latent_rope);
    try std.testing.expectEqual(@as(u64, 704_648_448), b.index_keys);
    try std.testing.expectEqual(@as(u64, 0), b.prefill_weight);
    try std.testing.expectEqual(@as(u64, 4 << 30), b.graphs);
    try std.testing.expectEqual(@as(u64, 2560 << 20), b.runtime);
    try std.testing.expectEqual(@as(u64, 131072 * 1152), b.stage); // one pair (no prefetch stream)
    // The bit-neutral trims: pieces of 512, 64 score rows, 128 expert rows, clusters (no lane partials).
    try std.testing.expect(gib(b.workspace) < 0.3);
    try std.testing.expect(b.state_arenas >= b.logicalState() and b.state_arenas - b.logicalState() < 3 * arena_granularity);
    // With both 4 GiB serving reserves and graph/runtime accounting, 115.5 GiB is below the admission threshold.
    const avail = gibBytes(115.5);
    const floor = reserveForWithMargin(gibBytes(120), false, 4 << 30, try serveMarginBytes(null));
    try std.testing.expectEqual(gibBytes(8), floor.bytes);
    try std.testing.expectError(error.MemoryBudget, admitRank(g, 262144, target, avail, floor));
    try std.testing.expectError(error.MemoryBudget, admitRank(g, 262144, drafted, avail, floor));
    // UMA headroom replaces the guard floor: 2 GiB + 5% of 120 GiB, then the 4 GiB serve margin.
    const head = reserveForWithMargin(gibBytes(120), true, 4 << 30, try serveMarginBytes(null));
    try std.testing.expectEqual(gibBytes(12), head.bytes);
    try std.testing.expectError(error.MemoryBudget, admitRank(g, 262144, drafted, avail, head));
    try std.testing.expectEqual(@as(u64, 4 << 30), try floorBytes(null));
    try std.testing.expectError(error.BadMemoryFloor, floorBytes("-1"));
}

test "state allocates its three admitted arenas, refuses other bytes, and re-checks MemAvailable after the load" {
    var dummy: u8 = 0;
    cuda.memory.setHook(.{ .ctx = &dummy, .alloc = Probe.alloc, .free = Probe.free });
    defer cuda.memory.setHook(null);
    Probe.live = 0;
    const tiny = Geometry{ .layers = 2, .full_index_layers = 1, .chunk_rows = 64, .piece_rows = 32, .index_slots = 16, .decode_rows = 2, .graphs_bytes = 0, .expert_rows = 16, .head_rows = 2, .vocab_local = 256 };
    const adm = try admitRank(tiny, 16, .{}, 1 << 34, reserveFor(1 << 34, false, 1 << 20));
    const before = cuda.memory.usage(false).device;
    var s = try State.init(undefined, tiny, 16, 1, 2, &adm, 1 << 34);
    try std.testing.expectEqual(@as(usize, 3), Probe.live); // KV planes, stage, workspace
    try std.testing.expectEqual(s.latent.ptr + 9 * 512 * 2, s.latentAt(1));
    try std.testing.expect(s.index_keys.ptr + s.index_keys.len <= s.kv_arena.ptr + s.kv_arena.len);
    try std.testing.expectEqual(stageLatent(&s.stage, 0) + s.stage.rows * 1024, stageRope(&s.stage, 0));
    try std.testing.expectEqual(@as(usize, 8), s.stage.rows);
    try validateStage(&s.stage, false);
    try std.testing.expectEqual(@as(usize, 32), s.ws.piece_rows);
    try std.testing.expectEqual(before + adm.state_bytes, cuda.memory.usage(false).device);
    try s.checkLayers(2, 1);
    try std.testing.expectError(error.LayerOutOfPlanes, s.checkLayers(3, 1));
    try std.testing.expectError(error.IndexPlaneOutOfRange, s.checkLayers(2, 2));
    s.deinit();
    try std.testing.expectEqual(@as(usize, 0), Probe.live);
    var other = tiny;
    other.chunk_rows = 32;
    try std.testing.expectError(error.AdmissionMismatch, State.init(undefined, other, 16, 0, 2, &adm, 1 << 34));
    // A fresh MemAvailable after the load that no longer holds State above the reserve fails closed.
    try std.testing.expectError(error.MemoryBudgetAfterLoad, State.init(undefined, tiny, 16, 0, 2, &adm, adm.state_bytes));
    try std.testing.expectError(error.MemoryBudget, admitRank(tiny, 16, .{ .routed = 1 << 34 }, 1 << 34, reserveFor(1 << 34, false, 0)));
    try std.testing.expectEqual(@as(usize, 0), Probe.live);
}

test "state declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Geometry);
    std.testing.refAllDecls(PlaneBytes);
    std.testing.refAllDecls(CacheFormat);
    std.testing.refAllDecls(Workspace);
    std.testing.refAllDecls(WeightClasses);
    std.testing.refAllDecls(Breakdown);
    std.testing.refAllDecls(Admission);
    std.testing.refAllDecls(Reserve);
    std.testing.refAllDecls(StateArenas);
    std.testing.refAllDecls(Workspace.Sizes);
    std.testing.refAllDecls(State);
}

test "TBO workspace carves separate half reduction intervals" {
    const w = workspaceLayout(.{ .piece_rows = 512, .chunk_rows = 512, .tbo = true }, 4096);
    const a_start = w.tbo_gathered[0];
    const a_end = a_start + w.sizes.tbo_gathered[0];
    const b_start = w.tbo_gathered[1];
    const b_end = b_start + w.sizes.tbo_gathered[1];
    try std.testing.expect(a_end <= b_start or b_end <= a_start);
    try std.testing.expect(w.sizes.tbo_gathered[0] != 0 and w.sizes.tbo_gathered[1] != 0);
}

test "piece pipe workspace carves two source and gathered ring slots only when enabled" {
    const base: Geometry = .{ .piece_rows = 512, .chunk_rows = 2048, .expert_rows = 1024, .graphs_bytes = 0 };
    const off = workspaceLayout(base, 8192);
    var enabled = base;
    enabled.piece_pipe = true;
    const on = workspaceLayout(enabled, 8192);
    const src_bytes = @as(u64, 512) * hidden * 4;
    const gather_bytes = @as(u64, dcp_world) * 512 * hidden * 4;
    try std.testing.expectEqual(src_bytes, on.sizes.pipe_src[0]);
    try std.testing.expectEqual(src_bytes, on.sizes.pipe_src[1]);
    try std.testing.expectEqual(gather_bytes, on.sizes.pipe_gather[0]);
    try std.testing.expectEqual(gather_bytes, on.sizes.pipe_gather[1]);
    try std.testing.expectEqual(@as(u64, 0), off.sizes.pipe_src[0]);
    try std.testing.expectEqual(@as(u64, 0), off.sizes.pipe_gather[1]);
    try std.testing.expectEqual(2 * (src_bytes + gather_bytes), on.bytes - off.bytes);
}

test "MoE pipe ring stays outside the phase union and costs zero when disabled" {
    const context = 262144;
    for ([_]usize{ 2048, 4096, 8192 }) |rows| {
        for ([_]usize{ 512, 1024 }) |piece_rows| {
            const off = workspaceLayout(.{ .expert_rows = rows, .chunk_rows = rows, .piece_rows = piece_rows }, context);
            const on = workspaceLayout(.{ .expert_rows = rows, .chunk_rows = rows, .piece_rows = piece_rows, .moe_pipe = true }, context);
            const stride = @as(u64, @intCast(piece_rows)) * hidden * 2;
            const bytes = Workspace.moe_pipe_ring_slots * stride * (1 + dcp_world);
            const ring_end = on.moe_pipe + on.sizes.moe_pipe;
            const phase_end = on.phase + on.phase_bytes;
            try std.testing.expectEqual(@as(u64, 0), off.sizes.moe_pipe);
            try std.testing.expectEqual(bytes, on.sizes.moe_pipe);
            try std.testing.expect(ring_end <= on.phase or on.moe_pipe >= phase_end);
            try std.testing.expectEqual(Workspace.moe_pipe_ring_slots, on.moe_pipe_slots);
            try std.testing.expectEqual(on.moe_pipe, on.moe_pipe_src[0]);
            try std.testing.expectEqual(on.moe_pipe_src[0] + stride, on.moe_pipe_gather[0]);
            try std.testing.expect(on.moe_pipe_gather[1] + dcp_world * stride <= ring_end);
        }
    }
}

test "SMLA-P3 union table: off carves nothing; on adds one attention-phase region and moves nothing else" {
    const context = 262144;
    const attention = .{ "q_a", "q_resid", "q", "kv", "ik", "iw", "iq", "qi", "qa", "qpe", "fused_queries", "part", "ol", "lse_out", "attend_split", "vo", "scores", "topk_tokens", "topk_counts", "candidate_records", "gathered_candidates", "smla_union" };
    const routed = .{ "picks8", "wts8", "shared_out", "lowrank_mid", "routed_out" };
    for ([_]usize{ 512, 1024 }) |piece_rows| {
        for ([_]usize{ 1024, 4096 }) |er| {
            const base: Geometry = .{ .expert_rows = er, .chunk_rows = 4096, .piece_rows = piece_rows };
            const off = workspaceLayout(base, context);
            var g = base;
            g.smla_p3 = true;
            const on = workspaceLayout(g, context);
            // Off: a zero-length span at the attention-phase end (right after the last attention region).
            try std.testing.expectEqual(@as(u64, 0), off.sizes.smla_union);
            try std.testing.expectEqual(off.gathered_candidates + Workspace.align256(off.sizes.gathered_candidates), off.smla_union);
            // On: every region inside the phases keeps its address and size. The totals (bytes, phase_bytes) and the
            // regions placed after the phase may grow or shift when the attention phase becomes the larger one.
            const info = @typeInfo(Workspace).@"struct";
            const phase_end = off.phase + off.phase_bytes;
            inline for (info.field_names, info.field_types) |name, T| {
                if (comptime T == u64 and !std.mem.eql(u8, name, "smla_union") and !std.mem.eql(u8, name, "bytes") and !std.mem.eql(u8, name, "phase_bytes")) {
                    if (@field(off, name) < phase_end) try std.testing.expectEqual(@field(off, name), @field(on, name)) else try std.testing.expect(@field(on, name) >= @field(off, name));
                }
            }
            try std.testing.expect(on.bytes >= off.bytes and on.phase_bytes >= off.phase_bytes);
            const sizes = @typeInfo(Workspace.Sizes).@"struct";
            inline for (sizes.field_names, sizes.field_types) |name, T| {
                if (comptime T == u64 and !std.mem.eql(u8, name, "smla_union")) try std.testing.expectEqual(@field(off.sizes, name), @field(on.sizes, name));
            }
            try std.testing.expectEqual(mla.smlaP3TableBytes(piece_rows), on.sizes.smla_union);
            try std.testing.expect(on.smla_union % 256 == 0);
            try std.testing.expectEqual(on.gathered_candidates + Workspace.align256(on.sizes.gathered_candidates), on.smla_union);
            try std.testing.expect(on.smla_union >= on.phase and on.smla_union + on.sizes.smla_union <= on.phase + on.phase_bytes);
            try expectWorkspacePhaseNonoverlap(&on, attention);
            // Routed results sit outside the phase union only when grouped; otherwise they share the FFN phase by design.
            if (groupedExperts(er, piece_rows)) inline for (attention) |left| inline for (routed) |right| {
                const a = @field(on, left);
                const an = @field(on.sizes, left);
                const b = @field(on, right);
                const bn = @field(on.sizes, right);
                if (an != 0 and bn != 0) try std.testing.expect(a + an <= b or b + bn <= a);
            };
            // The totals grow by at most the table (zero while the FFN phase stays the larger one).
            try std.testing.expect(on.phase_bytes - off.phase_bytes <= Workspace.align256(on.sizes.smla_union));
            try std.testing.expect(on.bytes - off.bytes <= Workspace.align256(on.sizes.smla_union));
        }
    }
}

test "WS trim sizes generic scratch at decode rows and keeps prompt regions disjoint" {
    const context: usize = 262144;
    const base: Geometry = .{ .expert_rows = 2048, .chunk_rows = 2048, .prompt_experts = true };
    const off = workspaceLayout(base, context);
    var enabled = base;
    enabled.ws_trim = true;
    const on = workspaceLayout(enabled, context);
    const generic = cuda.exl3_experts.scratchBytes(expertShape(), @max(base.decode_rows, 16), routed_slots) catch unreachable;
    const prompt = cuda.exl3_experts.prompt.bytes(expertShape(), 2048, routed_slots, false) catch unreachable;
    try std.testing.expectEqual(@as(usize, 2048), off.generic_rows);
    try std.testing.expectEqual(@as(u64, @intCast(cuda.exl3_experts.scratchBytes(expertShape(), 2048, routed_slots) catch unreachable)), off.sizes.experts);
    try std.testing.expectEqual(@as(usize, 16), on.generic_rows);
    try std.testing.expectEqual(@as(u64, @intCast(generic)), on.sizes.experts);
    try std.testing.expectEqual(promptPlanBytes(2048, routed_slots, routed_experts), on.sizes.prompt_plan);
    try std.testing.expectEqual(@as(u64, @intCast(prompt)), on.sizes.prompt_experts);
    try std.testing.expect(on.experts + on.sizes.experts <= on.prompt_plan);
    try std.testing.expect(on.prompt_plan + on.sizes.prompt_plan <= on.prompt_experts);
    try std.testing.expect(on.prompt_experts + on.sizes.prompt_experts <= on.phase + on.phase_bytes);
    try std.testing.expect(on.sizes.lowrank_mid >= 8 * 2048 * lowrank_max * 4);
    const switches_off = workspaceLayout(.{ .expert_rows = 2048, .chunk_rows = 2048, .prompt_experts = true }, context);
    try std.testing.expectEqual(switches_off.bytes, off.bytes);
}

test "WS trim admission estimate at er1024 matches independent trim layout" {
    const context: usize = 262144;
    const base: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .prompt_experts = true, .moe_pipe = true };
    const base_layout = workspaceLayout(base, context);
    var trim = base;
    trim.ws_trim = true;
    trim.moe_post = true;
    const trim_layout = workspaceLayout(trim, context);
    var overlay = base;
    overlay.scratch_overlay = true;
    const overlay_layout = workspaceLayout(overlay, context);
    std.debug.print("WS_TRIM er1024/chunk2048/context262144 stateBytes={d} overlayBytes={d} deltaGiB={d:.3}\n", .{ stateBytes(trim, context), stateBytes(overlay, context), gib(stateBytes(trim, context)) - gib(stateBytes(overlay, context)) });
    try std.testing.expectEqual(base_layout.bytes, workspaceLayout(.{ .expert_rows = 1024, .chunk_rows = 2048, .prompt_experts = true, .moe_pipe = true }, context).bytes);
    try std.testing.expectEqual(@as(usize, 16), trim_layout.generic_rows);
    try std.testing.expectEqual(@as(u64, 0), trim_layout.sizes.routed_out);
    try std.testing.expect(overlay_layout.bytes > trim_layout.bytes);
}

test "MEMSLIM generic trim keeps the overlay prompt offset and frees the generic excess" {
    const context: usize = 262144;
    const base: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .piece_rows = 512, .prompt_experts = true, .scratch_overlay = true, .moe_pipe = true, .moe_post = true, .piece_pipe = true };
    var slim = base;
    slim.slim_generic = true;
    const off = workspaceLayout(base, context);
    const on = workspaceLayout(slim, context);
    const saving = off.bytes - on.bytes;
    const expected = (off.sizes.experts - (Workspace.align256(promptPlanBytes(1024, routed_slots, routed_experts)) + on.sizes.prompt_experts)) + (off.sizes.lowrank_mid - on.sizes.lowrank_mid);
    try std.testing.expectEqual(@as(usize, 1024), off.generic_rows);
    try std.testing.expectEqual(@as(usize, 16), on.generic_rows);
    try std.testing.expectEqual(@as(u64, @intCast(cuda.exl3_experts.scratchBytes(expertShape(), 16, routed_slots) catch unreachable)), on.sizes.experts);
    try std.testing.expectEqual(off.prompt_experts - off.experts, on.prompt_experts - on.experts);
    try std.testing.expectEqual(off.sizes.prompt_experts, on.sizes.prompt_experts);
    try std.testing.expectEqual(@as(u64, 2_097_152), on.sizes.lowrank_mid);
    try std.testing.expectEqual(@as(u64, 33_554_432), off.sizes.lowrank_mid);
    try std.testing.expectEqual(expected, saving);
    std.debug.print("MEMSLIM overlay saving={d} bytes ({d:.3} GiB)\n", .{ saving, gib(saving) });
}

test "MEMSLIM off is byte-identical" {
    const context: usize = 262144;
    const legacy: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .piece_rows = 512, .prompt_experts = true, .scratch_overlay = true, .moe_pipe = true, .moe_post = true, .piece_pipe = true };
    const disabled: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .piece_rows = 512, .prompt_experts = true, .scratch_overlay = true, .moe_pipe = true, .moe_post = true, .piece_pipe = true, .slim_generic = false };
    const a = workspaceLayout(legacy, context);
    const b = workspaceLayout(disabled, context);
    try std.testing.expectEqual(a.bytes, b.bytes);
    try std.testing.expectEqual(a.experts, b.experts);
    try std.testing.expectEqual(a.prompt_experts, b.prompt_experts);
    try std.testing.expectEqual(a.sizes.lowrank_mid, b.sizes.lowrank_mid);
    try std.testing.expectEqual(a.phase_bytes, b.phase_bytes);
}

test "MEMSLIM with WS_TRIM equals WS_TRIM" {
    const context: usize = 262144;
    const base: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .prompt_experts = true, .ws_trim = true };
    var slim = base;
    slim.slim_generic = true;
    try std.testing.expectEqual(workspaceLayout(base, context).bytes, workspaceLayout(slim, context).bytes);
}

test "MEMSLIM keeps the drafter share inside the phase union" {
    const context: usize = 262144;
    const geometry: Geometry = .{ .expert_rows = 1024, .chunk_rows = 2048, .piece_rows = 512, .prompt_experts = true, .scratch_overlay = true, .moe_pipe = true, .moe_post = true, .piece_pipe = true, .slim_generic = true, .drafter_share_bytes = 300 << 20 };
    const workspace = workspaceLayout(geometry, context);
    try std.testing.expect(workspace.phase_bytes >= (300 << 20));
    try std.testing.expect(workspace.bytes >= workspace.phase + workspace.phase_bytes);
}

test "MoE post removes routed output and stores in-place gather rows" {
    const context: usize = 262144;
    const base: Geometry = .{ .expert_rows = 2048, .chunk_rows = 2048, .piece_rows = 512, .moe_pipe = true };
    const off = workspaceLayout(base, context);
    var enabled = base;
    enabled.moe_post = true;
    const on = workspaceLayout(enabled, context);
    const stride: u64 = 512 * hidden * 2;
    try std.testing.expectEqual(@as(u64, 0), on.sizes.routed_out);
    try std.testing.expectEqual(@as(u64, 2) * stride * dcp_world, on.sizes.moe_pipe);
    try std.testing.expectEqual(@as(u64, 2048) * hidden * 4, off.sizes.routed_out);
    try std.testing.expectEqual(Workspace.moe_pipe_ring_slots, on.moe_pipe_slots);
    try std.testing.expectEqual(on.moe_pipe, on.moe_pipe_gather[0]);
    try std.testing.expectEqual(on.moe_pipe + stride * dcp_world, on.moe_pipe_gather[1]);
    try std.testing.expectEqual(off.bytes - 2048 * hidden * 4 - 2 * stride, on.bytes);
}
