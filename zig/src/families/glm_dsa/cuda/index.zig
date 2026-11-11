// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! DCP candidate merge follows ashhart/TensorFold PR #159 by drowzeys (Apache-2.0).
//! Host references and DCP2 layouts adapt bertholomus/bertholomus/TensorFold@ddfad35 `select.py` and `dcp.py` (Apache-2.0).
//! The radix kernel is ported from the zig blob f633f78e223d at ashhart/TensorFold@d1018b31.
const std = @import("std");
const cuda = @import("cuda");
const glm_config = @import("glm_config");
const contract = @import("contract.zig");

pub const heads = 32;
pub const dim = 128;
pub const rope_dim = 64;
pub const top_k = 2048;
pub const dense_limit = 2048;
pub const max_local_keys = contract.max_local_keys;
pub const max_global_keys = 262144;
pub const score_tile = 256;
pub const score_smem = 16 * dim * 4 + heads * 16 * 4 + heads * 4 + 16 * 16 * 4;
pub const key_smem = dim * @sizeOf(f32) * 2;
/// MMA uses 64 keys and a 4-row shared scratch tile; larger CTAs stream additional 4-row batches while retaining K.
pub const score_mma_keys = 64;
pub const score_mma_tile_rows = 4;
pub fn scoreMmaSmem(_: usize) u32 {
    return score_mma_tile_rows * heads * 136 * 2 + score_mma_keys * 136 * 2 + score_mma_tile_rows * heads * score_mma_keys * 4 + score_mma_tile_rows * heads * 4;
}
pub const score_mma_smem: u32 = scoreMmaSmem(4);
pub const ScoreKernel = enum { scalar, mma };

pub fn parseScoreRows(raw: ?[]const u8) !usize {
    const rows = if (raw) |value| std.fmt.parseInt(usize, value, 10) catch return error.BadIndexRows else 4;
    if (rows != 4 and rows != 8 and rows != 16) return error.BadIndexRows;
    return rows;
}

fn scoreKernelFromEnv() !ScoreKernel {
    const raw = std.c.getenv("TF_GLM53_INDEX") orelse return .mma;
    const v = std.mem.span(raw);
    if (std.mem.eql(u8, v, "mma")) return .mma;
    if (std.mem.eql(u8, v, "scalar")) return .scalar;
    return error.InvalidIndexKernel;
}

pub const Plan = struct {
    grid_x: u32,
    grid_y: u32,
    block: u32,
    dynamic_smem: u32,
};

pub const ScoreMmaPlan = struct { rows: usize, grid_x: u32, grid_y: u32, dynamic_smem: u32 };

pub fn scoreMmaPlan(rows: usize, keys: usize, mma_rows: usize) !ScoreMmaPlan {
    if (mma_rows != 4 and mma_rows != 8 and mma_rows != 16) return error.BadIndexRows;
    return .{ .rows = mma_rows, .grid_x = @intCast((keys + score_mma_keys - 1) / score_mma_keys), .grid_y = @intCast((rows + mma_rows - 1) / mma_rows), .dynamic_smem = scoreMmaSmem(mma_rows) };
}

/// Score plan: one CTA for each (row, 16-key tile); scratch comes from dynamic shared memory.
pub fn scoresPlan(rows: usize, keys: usize) Plan {
    return .{ .grid_x = @intCast((keys + 15) / 16), .grid_y = @intCast(rows), .block = score_tile, .dynamic_smem = score_smem };
}

/// Radix cluster sizing from DSV4.1 topk_cuda.cu: 512 threads, 32 entries/thread, <=8 CTAs.
pub fn topkPlan(keys: usize) !Plan {
    if (keys == 0 or keys > max_local_keys) return error.KeyCapacity;
    const entries_per_cta = contract.topk_threads * contract.topk_entries_per_thread;
    const clusters = @min(8, (keys + entries_per_cta - 1) / entries_per_cta);
    return .{ .grid_x = @intCast(clusters), .grid_y = 1, .block = contract.topk_threads, .dynamic_smem = contract.topk_dynamic_smem };
}

pub fn topkSmem(entries_per_thread: usize) !usize {
    if (entries_per_thread == 0 or entries_per_thread > 33) return error.EntryTile;
    return contract.topk_threads * entries_per_thread * @sizeOf(u32);
}

/// Map a rank-local DCP slot to its global sequence position, as in Python dcp.py.
pub fn slotToGlobal(slot: usize, rank: usize, world: usize) !usize {
    if (world == 0 or rank >= world) return error.DcpShape;
    if (slot > (std.math.maxInt(usize) - rank) / world) return error.DcpShape;
    return slot * world + rank;
}

pub const Pick = struct { score: f32, token: i32 };
pub const PackedCandidate = extern struct { score: f32, token: i32 };

/// Round an fp32 value to bf16 using round-to-nearest, ties-to-even, then return it as fp32.
pub fn bf16(value: f32) f32 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @bitCast(bits & 0xffff0000);
}

/// LayerNorm(1e-6) over projected keys, interleaved RoPE on columns 0..63, then bf16 output.
pub fn keysRef(projected: []const f32, rows: usize, stride: usize, col: usize, weight: []const f32, bias: []const f32, cos: []const f32, sin: []const f32, eps: f32, out: []f32) !void {
    if (stride < col + dim or weight.len < dim or bias.len < dim or
        cos.len < rows * rope_dim / 2 or sin.len < rows * rope_dim / 2 or projected.len < (rows -| 1) * stride + col + dim or out.len < rows * dim) return error.KeyShape;
    for (0..rows) |r| {
        var x: [dim]f32 = undefined;
        for (0..dim) |d| x[d] = bf16(projected[r * stride + col + d]);
        var mean: f32 = 0;
        for (x) |v| mean += v;
        mean /= @as(f32, @floatFromInt(dim));
        var variance: f32 = 0;
        for (x) |v| variance = @mulAdd(f32, v - mean, v - mean, variance);
        const denom = @sqrt(variance / @as(f32, @floatFromInt(dim)) + eps);
        var y: [dim]f32 = undefined;
        for (0..dim) |d| y[d] = bf16((x[d] - mean) / denom * weight[d] + bias[d]);
        for (0..rope_dim / 2) |i| {
            const c = cos[r * (rope_dim / 2) + i];
            const s = sin[r * (rope_dim / 2) + i];
            out[r * dim + 2 * i] = bf16(y[2 * i] * c - y[2 * i + 1] * s);
            out[r * dim + 2 * i + 1] = bf16(y[2 * i + 1] * c + y[2 * i] * s);
        }
        @memcpy(out[r * dim + rope_dim ..][0 .. dim - rope_dim], y[rope_dim..]);
    }
}

/// Apply interleaved RoPE once to qi's leading 64 columns per head; copy the remaining 64 unchanged.
pub fn queryRef(qi: []const f32, rows: usize, cos: []const f32, sin: []const f32, out: []f32) !void {
    if (qi.len < rows * heads * dim or out.len < rows * heads * dim or cos.len < rows * rope_dim / 2 or sin.len < rows * rope_dim / 2) return error.QueryShape;
    for (0..rows) |r| for (0..heads) |h| {
        const base = (r * heads + h) * dim;
        for (0..rope_dim / 2) |i| {
            const c = cos[r * rope_dim / 2 + i];
            const s = sin[r * rope_dim / 2 + i];
            const x0 = bf16(qi[base + 2 * i]);
            const x1 = bf16(qi[base + 2 * i + 1]);
            out[base + 2 * i] = bf16(x0 * c - x1 * s);
            out[base + 2 * i + 1] = bf16(x1 * c + x0 * s);
        }
        for (rope_dim..dim) |d| out[base + d] = bf16(qi[base + d]);
    };
}

/// Scalar oracle over an already-rotated qi. Tensor-core score parity is gated by top-k set overlap per row and ties; Python's
/// sequential reduction and CUDA's tensor-core/tree reduction are not expected to produce bit-identical scores.
/// q and keys contain bf16-rounded values as fp32.
pub fn scoresRef(qi: []const f32, iw: []const f32, iw_stride: usize, iw_col: usize, row: usize, keys: []const f32, pos: usize, out: []f32, rank: usize, world: usize) !void {
    if (qi.len < (row + 1) * heads * dim or iw_col + heads > iw_stride or iw.len < row * iw_stride + iw_col + heads or keys.len % dim != 0) return error.ScoreShape;
    if (world == 0 or rank >= world) return error.ScoreShape;
    const nkeys = keys.len / dim;
    if (out.len < keys.len / dim) return error.ScoreShape;
    for (0..nkeys) |k| {
        if (k * world + rank > pos) {
            out[k] = -std.math.inf(f32);
            continue;
        }
        var total: f32 = 0;
        for (0..heads) |h| {
            var dot: f32 = 0;
            for (0..dim) |d| {
                dot = @mulAdd(f32, qi[(row * heads + h) * dim + d], keys[k * dim + d], dot);
            }
            const w = iw[row * iw_stride + iw_col + h] * 0.1767766952966369;
            total = @mulAdd(f32, w, @max(dot * 0.08838834764831845, 0), total);
        }
        out[k] = total;
    }
}

fn better(a: Pick, b: Pick) bool {
    const a_score = canonicalScore(a.score);
    const b_score = canonicalScore(b.score);
    var a_bits: u32 = @bitCast(a_score);
    var b_bits: u32 = @bitCast(b_score);
    a_bits = if (a_bits & 0x80000000 != 0) ~a_bits else a_bits ^ 0x80000000;
    b_bits = if (b_bits & 0x80000000 != 0) ~b_bits else b_bits ^ 0x80000000;
    if (a_bits != b_bits) return a_bits > b_bits;
    return a.token < b.token;
}

fn canonicalScore(score: f32) f32 {
    if (std.math.isNan(score)) return @bitCast(@as(u32, 0x7fffffff));
    return score + 0.0;
}

/// The DCP2 exchange is one contiguous 8-byte record per local pick.
pub fn packedExchangeBytes(rows: usize, k: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, rows, k), @sizeOf(PackedCandidate));
}

pub fn packedCandidateRef(scores: []const f32, positions: []const i32, out: []PackedCandidate) !void {
    if (positions.len != out.len) return error.DcpShape;
    for (positions, out) |position, *candidate| {
        if (position < 0) {
            candidate.* = .{ .score = -std.math.inf(f32), .token = -1 };
        } else {
            const slot: usize = @intCast(position);
            if (slot >= scores.len) return error.DcpShape;
            candidate.* = .{ .score = canonicalScore(scores[slot]), .token = position };
        }
    }
}

/// Exact top-k reference: score descending, ties to the lower global token, output ascending by token.
pub fn topkRef(allocator: std.mem.Allocator, scores: []const f32, pos: usize, k: usize) ![]i32 {
    const visible = @min(scores.len, pos + 1);
    const picks = try allocator.alloc(Pick, visible);
    defer allocator.free(picks);
    var valid: usize = 0;
    for (scores[0..visible], 0..) |score, i| {
        if (score == -std.math.inf(f32)) continue;
        picks[valid] = .{ .score = score, .token = @intCast(i) };
        valid += 1;
    }
    std.mem.sort(Pick, picks[0..valid], {}, struct {
        fn less(_: void, a: Pick, b: Pick) bool {
            return better(a, b);
        }
    }.less);
    const n = @min(k, valid);
    const out = try allocator.alloc(i32, k);
    @memset(out, -1);
    const chosen = out[0..n];
    for (picks[0..n], chosen[0..n]) |pick, *token| token.* = pick.token;
    std.mem.sort(i32, chosen, {}, std.sort.asc(i32));
    return out;
}

/// Rank-local exact top-k oracle with global positions in its output, matching DCP2 kernel semantics.
pub fn topkDcpRef(allocator: std.mem.Allocator, scores: []const f32, pos: usize, k: usize, rank: usize, world: usize) ![]i32 {
    if (world == 0 or rank >= world) return error.DcpShape;
    const visible = if (pos < rank) 0 else @min(scores.len, (pos - rank) / world + 1);
    const tokens = try topkRef(allocator, scores[0..visible], if (visible == 0) 0 else visible - 1, k);
    for (tokens) |*token| {
        if (token.* >= 0) token.* = @intCast(try slotToGlobal(@intCast(token.*), rank, world));
    }
    std.mem.sort(i32, tokens, {}, std.sort.asc(i32));
    return tokens;
}

pub const SelectionRef = struct { tokens: []i32, count: usize };

pub fn selectRef(allocator: std.mem.Allocator, scores: []const f32, pos: usize, k: usize) !SelectionRef {
    const tokens = try topkRef(allocator, scores, pos, k);
    var count: usize = 0;
    for (tokens) |token| count += @intFromBool(token >= 0);
    return .{
        .tokens = tokens,
        .count = if (pos + 1 <= dense_limit) 0 else count,
    };
}

pub const DcpResult = struct { global: []i32, local: []i32, count: usize };
pub const FusedDcpResult = struct { global: []i32, global_count: usize, local: []i32, count: usize };

/// Host mirror of the 512-thread fused filter: scan every selected token in 512-entry passes.
pub fn fusedFilterRef(global: []const i32, world: usize, rank: usize, position: usize, local: []i32) !usize {
    if (world == 0 or rank >= world or local.len < global.len) return error.DcpShape;
    @memset(local[0..global.len], -1);
    if (position + 1 <= dense_limit) return 0;
    var count: usize = 0;
    var base: usize = 0;
    while (base < global.len) : (base += 512) {
        const end = @min(global.len, base + 512);
        for (global[base..end]) |token| {
            if (token < 0) continue;
            const value: usize = @intCast(token);
            if (value % world == rank) {
                local[count] = @intCast(value / world);
                count += 1;
            }
        }
    }
    return count;
}

/// Host oracle for the packed exchange's global selection plus fused local filter/remap.
pub fn mergePackedDcp(allocator: std.mem.Allocator, records: []const PackedCandidate, world: usize, rank: usize, k: usize, position: usize) !FusedDcpResult {
    if (world == 0 or rank >= world or k == 0 or records.len % k != 0) return error.DcpShape;
    const candidates = try allocator.alloc(Pick, records.len);
    defer allocator.free(candidates);
    var valid: usize = 0;
    for (records) |candidate| {
        if (candidate.token < 0 or candidate.score == -std.math.inf(f32)) continue;
        candidates[valid] = .{ .score = canonicalScore(candidate.score), .token = candidate.token };
        valid += 1;
    }
    std.mem.sort(Pick, candidates[0..valid], {}, struct {
        fn less(_: void, a: Pick, b: Pick) bool {
            return better(a, b);
        }
    }.less);
    const chosen = @min(k, valid);
    const global = try allocator.alloc(i32, k);
    @memset(global, -1);
    for (candidates[0..chosen], 0..) |candidate, i| global[i] = candidate.token;
    std.mem.sort(i32, global[0..chosen], {}, std.sort.asc(i32));
    const local = try allocator.alloc(i32, k);
    @memset(local, -1);
    const count = try fusedFilterRef(global, world, rank, position, local);
    return .{ .global = global, .global_count = chosen, .local = local, .count = count };
}

/// Gather each chosen global position's score from its owning DCP rank's local score slot.
pub fn candidateScoresRef(scores: []const f32, positions: []const i32, rank: usize, world: usize, out_scores: []f32, out_positions: []i32) !void {
    if (world == 0 or rank >= world or positions.len != out_scores.len or positions.len != out_positions.len) return error.DcpShape;
    for (positions, 0..) |position, i| {
        if (position < 0 or @as(usize, @intCast(position)) % world != rank) {
            out_scores[i] = -std.math.inf(f32);
            out_positions[i] = -1;
            continue;
        }
        const slot = @as(usize, @intCast(position)) / world;
        if (slot >= scores.len) return error.DcpShape;
        const score = scores[slot];
        out_scores[i] = score;
        out_positions[i] = if (score == -std.math.inf(f32)) -1 else position;
    }
}

/// Merge ranks' local candidates globally, then keep and remap this rank's positions to DCP slots p/world.
pub fn mergeDcp(allocator: std.mem.Allocator, scores: []const f32, positions: []const i32, world: usize, rank: usize, k: usize) !DcpResult {
    if (world == 0 or rank >= world or scores.len != positions.len) return error.DcpShape;
    const candidates = try allocator.alloc(Pick, positions.len);
    defer allocator.free(candidates);
    var valid: usize = 0;
    for (positions, scores) |position, score| {
        if (position < 0 or score == -std.math.inf(f32)) continue;
        candidates[valid] = .{ .score = score, .token = position };
        valid += 1;
    }
    std.mem.sort(Pick, candidates[0..valid], {}, struct {
        fn less(_: void, a: Pick, b: Pick) bool {
            return better(a, b);
        }
    }.less);
    const n = @min(k, valid);
    const global = try allocator.alloc(i32, k);
    const local = try allocator.alloc(i32, k);
    @memset(global, -1);
    @memset(local, -1);
    var count: usize = 0;
    for (candidates[0..n], 0..) |pick, i| {
        global[i] = pick.token;
        if (@as(usize, @intCast(pick.token)) % world == rank) {
            local[count] = @intCast(@divTrunc(pick.token, @as(i32, @intCast(world))));
            count += 1;
        }
    }
    std.mem.sort(i32, global[0..n], {}, std.sort.asc(i32));
    std.mem.sort(i32, local[0..count], {}, std.sort.asc(i32));
    return .{ .global = global, .local = local, .count = count };
}

/// Gather layout used by DCP is rank-major [world][rows][k].
pub fn mergeGatheredDcp(allocator: std.mem.Allocator, scores: []const f32, positions: []const i32, world: usize, rows: usize, row: usize, rank: usize, k: usize) !DcpResult {
    if (world == 0 or row >= rows or scores.len != world * rows * k or positions.len != scores.len) return error.DcpShape;
    const row_scores = try allocator.alloc(f32, world * k);
    defer allocator.free(row_scores);
    const row_positions = try allocator.alloc(i32, world * k);
    defer allocator.free(row_positions);
    for (0..world) |g| {
        const src = g * rows * k + row * k;
        @memcpy(row_scores[g * k ..][0..k], scores[src..][0..k]);
        @memcpy(row_positions[g * k ..][0..k], positions[src..][0..k]);
    }
    return mergeDcp(allocator, row_scores, row_positions, world, rank, k);
}

pub const KeyPlan = struct { grid: u32, block: u32, shared: u32 };
pub fn keyPlan(rows: usize) KeyPlan {
    return .{ .grid = @intCast(rows), .block = 128, .shared = key_smem };
}

/// Prompt length used by the 262k DCP2 timing harness; this is a target label, not a timing result.
pub const timing_context = 262144;

pub const Job = extern struct {
    scores: u64 = 0,
    score_stride: i64 = 0,
    positions: u64 = 0,
    position_stride: i64 = 0,
    tokens: u64 = 0,
    token_stride: i64 = 0,
    counts: u64 = 0,
    candidate_records: u64 = 0,
    nkeys: c_int = 0,
    k: c_int = 0,
    mode: c_int = 0,
    entries_per_thread: c_int = 0,
};
pub const TopkArgs = extern struct {
    jobs: [2]Job = .{ .{}, .{} },
    positions: u64 = 0,
    ratio: c_int = 1,
    block_size: c_int = 1,
    owner: c_int = 0,
};

comptime {
    std.debug.assert(@sizeOf(Job) == 80 and @offsetOf(TopkArgs, "positions") == 160 and @sizeOf(TopkArgs) == 184);
}

pub const Index = struct {
    key: cuda.Function,
    query: cuda.Function,
    score: cuda.Function,
    score_mma_4: cuda.Function,
    score_mma_8: cuda.Function,
    score_mma_16: cuda.Function,
    score_mma_rows: usize = 4,
    /// TF_GLM53_INDEX=scalar|mma (default mma); same bits (the mma kernel keeps the scalar one's dot and sum order).
    score_kernel: ScoreKernel = .mma,
    score_oracle: cuda.Function,
    select: cuda.Function,
    merge_fn: cuda.Function,
    merge2_fn: cuda.Function,

    pub fn load(module: cuda.Module, score_mma_rows: usize) !Index {
        if (score_mma_rows != 4 and score_mma_rows != 8 and score_mma_rows != 16) return error.BadIndexRows;
        var result = Index{
            .key = try module.function("tf_glm53_index_key"),
            .query = try module.function("tf_glm53_index_q"),
            .score = try module.function("tf_glm53_index_scores"),
            .score_mma_4 = try module.function("tf_glm53_index_scores_mma_4"),
            .score_mma_8 = try module.function("tf_glm53_index_scores_mma_8"),
            .score_mma_16 = try module.function("tf_glm53_index_scores_mma_16"),
            .score_mma_rows = score_mma_rows,
            .score_kernel = try scoreKernelFromEnv(),
            .score_oracle = try module.function("tf_glm53_index_scores_oracle"),
            .select = try module.function(topkSymbol()),
            .merge_fn = try module.function("tf_glm53_merge"),
            .merge2_fn = try module.function("tf_glm53_merge2"),
        };
        try result.select.allowDynamicShared(contract.topk_dynamic_smem);
        try result.score_mma_4.allowDynamicShared(score_mma_smem);
        try result.score_mma_8.allowDynamicShared(score_mma_smem);
        try result.score_mma_16.allowDynamicShared(score_mma_smem);
        return result;
    }

    /// Writes DCP rows by global position; projection has already been rounded to bf16.
    pub fn keys(k: *const Index, stream: cuda.Stream, projected: u64, weight: u64, bias: u64, cos: u64, sin: u64, positions: u64, out: u64, rows: usize, stride: usize, col: usize, rank: usize, world: usize, eps: f32) !void {
        if (rows == 0) return;
        if (stride < col + dim or world == 0 or rank >= world) return error.KeyShape;
        var a: cuda.Args = .{};
        a.add(projected);
        a.add(weight);
        a.add(bias);
        a.add(cos);
        a.add(sin);
        a.add(positions);
        a.add(out);
        a.add(@as(i32, @intCast(stride)));
        a.add(@as(i32, @intCast(col)));
        a.add(@as(i32, @intCast(rank)));
        a.add(@as(i32, @intCast(world)));
        a.add(eps);
        const p = keyPlan(rows);
        try cuda.launch.launch(k.key, .{ .grid = .{ .x = p.grid, .y = 1, .z = 1 }, .block = .{ .x = p.block } }, stream, &a);
    }

    /// Applies query RoPE exactly once. Score kernels accept the resulting qi and never rotate it again.
    pub fn q(k: *const Index, stream: cuda.Stream, input: u64, cos: u64, sin: u64, out: u64, rows: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(input);
        a.add(cos);
        a.add(sin);
        a.add(out);
        try cuda.launch.launch(k.query, .{ .grid = .{ .x = 32, .y = @intCast(rows), .z = 1 }, .block = .{ .x = 128 } }, stream, &a);
    }

    /// qi has already passed through `q`; this fast tensor-core kernel performs no RoPE.
    pub fn scores(k: *const Index, stream: cuda.Stream, qi: u64, iw: u64, iw_stride: usize, iw_col: usize, cache: u64, positions: u64, out: u64, rows: usize, nkeys: usize, score_stride: usize, rank: usize, world: usize) !void {
        if (rows == 0 or nkeys == 0) return;
        if (iw_col + heads > iw_stride or score_stride < nkeys or world == 0 or rank >= world) return error.ScoreShape;
        var a: cuda.Args = .{};
        a.add(qi);
        a.add(iw);
        a.add(@as(i32, @intCast(iw_stride)));
        a.add(@as(i32, @intCast(iw_col)));
        a.add(cache);
        a.add(positions);
        a.add(out);
        a.add(@as(i32, @intCast(nkeys)));
        a.add(@as(i32, @intCast(score_stride)));
        a.add(@as(i32, @intCast(rank)));
        a.add(@as(i32, @intCast(world)));
        if (k.score_kernel == .mma) {
            a.add(@as(i32, @intCast(rows)));
            const p = try scoreMmaPlan(rows, nkeys, k.score_mma_rows);
            const kernel = switch (p.rows) { 4 => k.score_mma_4, 8 => k.score_mma_8, 16 => k.score_mma_16, else => unreachable };
            try cuda.launch.launch(kernel, .{ .grid = .{ .x = p.grid_x, .y = p.grid_y, .z = 1 }, .block = .{ .x = 256 }, .shared = p.dynamic_smem }, stream, &a);
            return;
        }
        try cuda.launch.launch(k.score, .{ .grid = .{ .x = @intCast((nkeys + 15) / 16), .y = @intCast(rows), .z = 1 }, .block = .{ .x = score_tile }, .shared = score_smem }, stream, &a);
    }

    /// Local exact radix top-k; local cache slots are emitted as global positions (`slot * world + rank`).
    pub fn topk(k: *const Index, stream: cuda.Stream, scores_ptr: u64, score_stride: usize, tokens: u64, counts: u64, candidate_records: u64, positions: u64, rows: usize, nkeys: usize, topk_count: usize, rank: usize, world: usize) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world or nkeys > max_local_keys or topk_count == 0 or topk_count > top_k or score_stride < nkeys) return error.KeyCapacity;
        const p = try topkPlan(nkeys);
        const job: Job = .{ .scores = scores_ptr, .score_stride = @intCast(score_stride), .positions = 0, .position_stride = 0, .tokens = tokens, .token_stride = @intCast(topk_count), .counts = counts, .candidate_records = candidate_records, .nkeys = @intCast(nkeys), .k = @intCast(topk_count), .mode = 0, .entries_per_thread = contract.topk_entries_per_thread };
        const args: TopkArgs = .{ .jobs = .{ job, job }, .positions = positions, .ratio = @intCast(world), .block_size = 1, .owner = @intCast(rank) };
        var a: cuda.Args = .{};
        a.add(args);
        try cuda.launch.launch(k.select, .{ .grid = .{ .x = p.grid_x, .y = @intCast(rows), .z = 1 }, .block = .{ .x = p.block }, .shared = p.dynamic_smem, .cluster = .{ .x = p.grid_x, .y = 1, .z = 1 } }, stream, &a);
    }

    /// Merge rank-major packed candidates and fuse sparse filter plus owner-slot remap.
    pub fn merge(k: *const Index, stream: cuda.Stream, candidate_records: u64, global: u64, global_counts: u64, out: u64, counts: u64, positions: u64, rank: usize, world: usize, rows: usize, topk_count: usize) !void {
        if (rows == 0) return;
        if (topk_count == 0 or topk_count > top_k or (world != 1 and world != 2) or rank >= world) return error.DcpShape;
        var a: cuda.Args = .{};
        a.add(candidate_records);
        a.add(global);
        a.add(global_counts);
        a.add(out);
        a.add(counts);
        a.add(positions);
        a.add(@as(i32, @intCast(rank)));
        a.add(@as(i32, @intCast(world)));
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(topk_count)));
        try cuda.launch.launch(k.merge_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 512 }, .shared = 4096 * @sizeOf(Pick) }, stream, &a);
    }

    /// Exact linear merge. block_rows describes the per-gather row tile in block-major index-pipe storage.
    pub fn merge2(k: *const Index, stream: cuda.Stream, candidate_records: u64, global: u64, global_counts: u64, out: u64, counts: u64, positions: u64, rank: usize, world: usize, rows: usize, topk_count: usize, block_rows: usize) !void {
        if (rows == 0) return;
        if (topk_count == 0 or topk_count > top_k or (world != 1 and world != 2) or rank >= world or block_rows == 0) return error.DcpShape;
        var a: cuda.Args = .{};
        a.add(candidate_records);
        a.add(global);
        a.add(global_counts);
        a.add(out);
        a.add(counts);
        a.add(positions);
        a.add(@as(i32, @intCast(rank)));
        a.add(@as(i32, @intCast(world)));
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(topk_count)));
        a.add(@as(i32, @intCast(block_rows)));
        try cuda.launch.launch(k.merge2_fn, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 512 }, .shared = 4096 * @sizeOf(Pick) }, stream, &a);
    }

    /// GPU timing entry point for the 262k DCP2 case: score the local-key capacity and select the local top-k.
    pub fn timing262k(k: *const Index, stream: cuda.Stream, qi: u64, iw: u64, iw_stride: usize, iw_col: usize, cache: u64, positions: u64, scores_ptr: u64, tokens: u64, counts: u64, rows: usize, rank: usize) !void {
        if (rank >= 2) return error.DcpShape;
        try k.scores(stream, qi, iw, iw_stride, iw_col, cache, positions, scores_ptr, rows, max_local_keys, max_local_keys, rank, 2);
        try k.topk(stream, scores_ptr, max_local_keys, tokens, counts, 0, positions, rows, max_local_keys, top_k, rank, 2);
    }
};

pub fn topkSymbol() [:0]const u8 {
    return "_ZN10glm53_topk13tf_glm53_topkENS_4ArgsE";
}

test "GLM DSA IndexShare maps shared layers to the latest full layer" {
    var cfg: glm_config.Config = .{};
    cfg.full.set(0);
    cfg.full.set(2);
    cfg.full.set(6);
    try std.testing.expectEqual(@as(u32, 6), cfg.indexFrom(7));
    try std.testing.expectEqual(@as(u32, 2), cfg.indexFrom(5));
}

test "top-k ties go to lower tokens and output is ascending with -1 padding" {
    const got = try topkRef(std.testing.allocator, &.{ 2, 5, 5, 1 }, 3, 3);
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2 }, got);
    const padded = try topkRef(std.testing.allocator, &.{ 2, 1 }, 1, 4);
    defer std.testing.allocator.free(padded);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, -1, -1 }, padded);
    const dense = try selectRef(std.testing.allocator, &.{ 3, 1 }, 1, top_k);
    defer std.testing.allocator.free(dense.tokens);
    try std.testing.expectEqual(@as(usize, 0), dense.count);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, dense.tokens[0..2]);
}

test "index key reference rounds LayerNorm and applies interleaved RoPE" {
    var projected: [dim]f32 = undefined;
    for (&projected, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) / 128;
    const weight: [dim]f32 = @splat(1);
    const bias: [dim]f32 = @splat(0);
    const cos: [rope_dim / 2]f32 = @splat(@cos(0.03));
    const sin: [rope_dim / 2]f32 = @splat(@sin(0.03));
    var out: [dim]f32 = undefined;
    try keysRef(&projected, 1, dim, 0, &weight, &bias, &cos, &sin, 1e-6, &out);
    try std.testing.expectEqual(bf16(out[0]), out[0]);
    try std.testing.expectEqual(bf16(out[127]), out[127]);
    try std.testing.expect(out[0] != 0);
}

test "index query reference applies RoPE once and copies the non-rope tail" {
    const qi: [2 * heads * dim]f32 = @splat(1);
    const cos: [2 * rope_dim / 2]f32 = @splat(0.5);
    const sin: [2 * rope_dim / 2]f32 = @splat(0.5);
    var out: [2 * heads * dim]f32 = undefined;
    try queryRef(&qi, 2, &cos, &sin, &out);
    try std.testing.expectEqual(bf16(0), out[0]);
    try std.testing.expectEqual(bf16(1), out[rope_dim]);
    try std.testing.expectEqual(bf16(0), out[heads * dim]);
    try std.testing.expectEqual(bf16(1), out[heads * dim + rope_dim]);
    try std.testing.expectError(error.QueryShape, queryRef(&qi, 2, cos[0 .. rope_dim / 2], &sin, &out));
}

test "index score reference applies causal masking and head scale factors" {
    const qi: [2 * heads * dim]f32 = @splat(1);
    var iw: [2 * 48]f32 = @splat(7);
    for (0..heads) |h| iw[48 + 8 + h] = 1;
    const cache: [2 * dim]f32 = @splat(1);
    var out: [2]f32 = undefined;
    try scoresRef(&qi, &iw, 48, 8, 1, &cache, 0, &out, 0, 1);
    try std.testing.expect(out[0] > 0);
    try std.testing.expectEqual(-std.math.inf(f32), out[1]);
}

test "DCP merge selects globally then filters and remaps owned positions" {
    const got = try mergeDcp(std.testing.allocator, &.{ 8, 10, 9, 7, 11, 6 }, &.{ 0, 1, 2, 3, 4, 5 }, 2, 0, 3);
    defer std.testing.allocator.free(got.global);
    defer std.testing.allocator.free(got.local);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 4 }, got.global);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, -1 }, got.local);
    try std.testing.expectEqual(@as(usize, 2), got.count);
}

test "DCP gathered candidates use rank-major row layout and discard negative infinity" {
    // [G=2][R=2][k=2], so the target row is at offsets 2..3 and 6..7.
    const scores = [_]f32{ 0, 0, 100, 90, 0, 0, 80, -std.math.inf(f32) };
    const positions = [_]i32{ -1, -1, 2, 4, -1, -1, 1, 3 };
    const got = try mergeGatheredDcp(std.testing.allocator, &scores, &positions, 2, 2, 1, 0, 2);
    defer std.testing.allocator.free(got.global);
    defer std.testing.allocator.free(got.local);
    try std.testing.expectEqualSlices(i32, &.{ 2, 4 }, got.global);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, got.local);
    try std.testing.expectEqual(@as(usize, 2), got.count);
}

test "DCP candidate scoring remaps global positions to owned local slots" {
    const local_scores = [_]f32{ 10, 20, 30 };
    const picks = [_]i32{ 2, 1, 4, -1 };
    var scores: [4]f32 = undefined;
    var positions: [4]i32 = undefined;
    try candidateScoresRef(&local_scores, &picks, 0, 2, &scores, &positions);
    try std.testing.expectEqualSlices(f32, &.{ 20, -std.math.inf(f32), 30, -std.math.inf(f32) }, &scores);
    try std.testing.expectEqualSlices(i32, &.{ 2, -1, 4, -1 }, &positions);
}

test "DCP packed candidates keep score and position in one 8-byte record" {
    const source = [_]f32{ 1, -0.0, 3 };
    const positions = [_]i32{ 2, -1, 0 };
    var records: [3]PackedCandidate = undefined;
    try packedCandidateRef(&source, &positions, &records);
    try std.testing.expectEqual(@as(usize, 24), try packedExchangeBytes(1, 3));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(PackedCandidate));
    try std.testing.expectEqual(@as(i32, 2), records[0].token);
    try std.testing.expectEqual(@as(f32, 1), records[2].score);
    try std.testing.expectEqual(@as(i32, -1), records[1].token);
}

test "DCP comparator canonicalizes NaNs like GPU arithmetic and collapses signed zero" {
    const nan_a: f32 = @bitCast(@as(u32, 0x7fc00001));
    const nan_b: f32 = @bitCast(@as(u32, 0x7fc00002));
    try std.testing.expect(better(.{ .score = nan_b, .token = 1 }, .{ .score = nan_a, .token = 9 }));
    try std.testing.expect(better(.{ .score = 0.0, .token = 1 }, .{ .score = -0.0, .token = 2 }));
    try std.testing.expect(better(.{ .score = nan_b, .token = 8 }, .{ .score = std.math.inf(f32), .token = 8 }));
    try std.testing.expectEqual(@as(u32, 0x7fffffff), @as(u32, @bitCast(canonicalScore(nan_b))));
}

test "packed DCP merge matches separate reference selection and fused remap at dense boundary" {
    const scores = [_]f32{ 8, 10, 9, 7, 11, 6 };
    const positions = [_]i32{ 0, 1, 2, 3, 4, 5 };
    var records: [6]PackedCandidate = undefined;
    try packedCandidateRef(&scores, &positions, &records);
    const old = try mergeDcp(std.testing.allocator, &scores, &positions, 2, 0, 3);
    defer std.testing.allocator.free(old.global);
    defer std.testing.allocator.free(old.local);
    const fused = try mergePackedDcp(std.testing.allocator, &records, 2, 0, 3, 2048);
    defer std.testing.allocator.free(fused.global);
    defer std.testing.allocator.free(fused.local);
    try std.testing.expectEqualSlices(i32, old.global, &.{ 1, 2, 4 });
    try std.testing.expectEqualSlices(i32, old.global, fused.global);
    try std.testing.expectEqual(old.global.len, fused.global_count);
    try std.testing.expectEqualSlices(i32, old.local, fused.local);
    try std.testing.expectEqual(old.count, fused.count);
    const dense = try mergePackedDcp(std.testing.allocator, &records, 2, 0, 3, 2047);
    defer std.testing.allocator.free(dense.global);
    defer std.testing.allocator.free(dense.local);
    try std.testing.expectEqual(@as(usize, 0), dense.count);
    try std.testing.expectEqualSlices(i32, &.{ -1, -1, -1 }, dense.local);
}

test "packed exchange preserves Python tie, NaN, signed-zero, and negative-infinity decisions" {
    const nan0: f32 = @bitCast(@as(u32, 0x7fc00001));
    const nan1: f32 = @bitCast(@as(u32, 0x7fc00002));
    const scores = [_]f32{ 0.0, -0.0, nan0, nan1, 5, 5, -std.math.inf(f32), 1, 0, 0 };
    const positions = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -1 };
    var records: [10]PackedCandidate = undefined;
    try packedCandidateRef(&scores, &positions, &records);
    try std.testing.expectEqual(@as(usize, 16384), try packedExchangeBytes(1, top_k));
    for (0..2) |rank| {
        const split = try mergeDcp(std.testing.allocator, &scores, &positions, 2, rank, 5);
        defer std.testing.allocator.free(split.global);
        defer std.testing.allocator.free(split.local);
        const fused = try mergePackedDcp(std.testing.allocator, &records, 2, rank, 5, 2048);
        defer std.testing.allocator.free(fused.global);
        defer std.testing.allocator.free(fused.local);
        try std.testing.expectEqualSlices(i32, split.local, fused.local);
        try std.testing.expectEqual(split.count, fused.count);
    }
}

test "packed DCP merge matches the Python order when NaNs exceed K" {
    const scores = [_]f32{
        @bitCast(@as(u32, 0x7fc00001)),
        @bitCast(@as(u32, 0xffc00002)),
        @bitCast(@as(u32, 0x7fc00003)),
        @bitCast(@as(u32, 0x7fc00004)),
        0.0,
        -0.0,
    };
    const positions = [_]i32{ 0, 1, 2, 3, 4, 5 };
    var records: [6]PackedCandidate = undefined;
    try packedCandidateRef(&scores, &positions, &records);
    for (0..2) |rank| {
        const split = try mergeDcp(std.testing.allocator, &scores, &positions, 2, rank, 3);
        defer std.testing.allocator.free(split.global);
        defer std.testing.allocator.free(split.local);
        const fused = try mergePackedDcp(std.testing.allocator, &records, 2, rank, 3, 2048);
        defer std.testing.allocator.free(fused.global);
        defer std.testing.allocator.free(fused.local);
        try std.testing.expectEqualSlices(i32, split.local, fused.local);
        try std.testing.expectEqual(split.count, fused.count);
    }
}

test "packed merge scans all 2048 picks and restores the global prefill token list" {
    const records = try std.testing.allocator.alloc(PackedCandidate, 4096);
    defer std.testing.allocator.free(records);
    for (records, 0..) |*record, i| record.* = .{ .score = @floatFromInt(4096 - i), .token = @intCast(i) };
    const merged = try mergePackedDcp(std.testing.allocator, records, 2, 0, top_k, 4096);
    defer std.testing.allocator.free(merged.global);
    defer std.testing.allocator.free(merged.local);
    try std.testing.expectEqual(@as(usize, top_k), merged.global_count);
    try std.testing.expectEqual(@as(usize, 1024), merged.count);
    try std.testing.expectEqual(@as(i32, 0), merged.global[0]);
    try std.testing.expectEqual(@as(i32, 2047), merged.global[top_k - 1]);
    for (0..1024) |i| try std.testing.expectEqual(@as(i32, @intCast(i)), merged.local[i]);
}

test "512-thread fused filter visits every selected token across four scan passes" {
    var global: [top_k]i32 = undefined;
    for (&global, 0..) |*token, i| token.* = @intCast(i);
    var local: [top_k]i32 = undefined;
    const count = try fusedFilterRef(&global, 2, 1, 4096, &local);
    try std.testing.expectEqual(@as(usize, 1024), count);
    try std.testing.expectEqual(@as(i32, 0), local[0]);
    try std.testing.expectEqual(@as(i32, 1023), local[count - 1]);
    try std.testing.expectEqual(@as(i32, -1), local[count]);
}

test "single-rank dense selection returns zero sparse count" {
    const dense = try selectRef(std.testing.allocator, &.{ 9, 8, 7 }, 2047, top_k);
    defer std.testing.allocator.free(dense.tokens);
    try std.testing.expectEqual(@as(usize, 0), dense.count);
}

test "DCP2 top-k reference emits global positions rather than local slots" {
    const local_scores = [_]f32{ 1, 10, 2, 9 };
    const rank_one = try topkDcpRef(std.testing.allocator, &local_scores, 5, 2, 1, 2);
    defer std.testing.allocator.free(rank_one);
    try std.testing.expectEqualSlices(i32, &.{ 3, 5 }, rank_one);
    const rank_zero = try topkDcpRef(std.testing.allocator, &local_scores, 5, 2, 0, 2);
    defer std.testing.allocator.free(rank_zero);
    try std.testing.expectEqualSlices(i32, &.{ 2, 4 }, rank_zero);
}

test "K5 launch plans include the full DCP2 verify shard and smem budget" {
    const plan = try topkPlan(131072);
    try std.testing.expectEqual(@as(u32, 8), plan.grid_x);
    try std.testing.expectEqual(@as(u32, 67584), plan.dynamic_smem);
    try std.testing.expectEqual(@as(u32, 11392), score_smem);
    try std.testing.expectEqual(@as(u32, 85_504), score_mma_smem);
    try std.testing.expectEqual(@as(u32, 11_392), scoresPlan(4, 513).dynamic_smem);
    try std.testing.expectEqual(@as(u32, 33), scoresPlan(4, 513).grid_x);
    try std.testing.expectEqual(@as(u32, 8), (try topkPlan(262144 / 2 + 4)).grid_x);
    try std.testing.expectEqual(@as(u32, 8), (try topkPlan(max_local_keys)).grid_x);
    try std.testing.expectError(error.KeyCapacity, topkPlan(max_local_keys + 1));
    try std.testing.expectEqual(@as(usize, 135168), max_local_keys);
    try std.testing.expectEqual(@as(usize, 262144), max_global_keys);
    try std.testing.expectEqual(@as(usize, 67584), try topkSmem(33));
    try std.testing.expectEqualStrings("_ZN10glm53_topk13tf_glm53_topkENS_4ArgsE", topkSymbol());
    try std.testing.expectEqual(@as(usize, 262143), try slotToGlobal(131071, 1, 2));
    try std.testing.expectEqual(@as(usize, 262142), try slotToGlobal(131071, 0, 2));
    try std.testing.expectError(error.DcpShape, slotToGlobal(0, 2, 2));
}

test "index MMA launch selection uses configured row instance and grid" {
    for ([_]usize{4, 8, 16}) |rows_per_cta| {
        const p = try scoreMmaPlan(17, 513, rows_per_cta);
        try std.testing.expectEqual(@as(u32, 9), p.grid_x);
        try std.testing.expectEqual(@as(u32, @intCast((17 + rows_per_cta - 1) / rows_per_cta)), p.grid_y);
        try std.testing.expectEqual(@as(u32, 85_504), p.dynamic_smem);
        try std.testing.expectEqual(rows_per_cta, p.rows);
    }
    try std.testing.expectError(error.BadIndexRows, scoreMmaPlan(17, 513, 12));
    try std.testing.expectEqual(@as(usize, 4), try parseScoreRows(null));
    try std.testing.expectEqual(@as(usize, 16), try parseScoreRows("16"));
    try std.testing.expectError(error.BadIndexRows, parseScoreRows("12"));
}

test "Index CUDA launch methods accept empty host-only batches without a device" {
    var kernels: Index = undefined;
    try kernels.keys(undefined, 0, 0, 0, 0, 0, 0, 0, 0, dim, 0, 0, 1, 1e-6);
    try kernels.q(undefined, 0, 0, 0, 0, 0);
    try kernels.scores(undefined, 0, 0, 0, heads, 0, 0, 0, 0, 0, 0, 0, 1);
    try kernels.topk(undefined, 0, 0, 0, 0, 0, 0, 0, 0, top_k, 0, 1);
    try kernels.merge(undefined, 0, 0, 0, 0, 0, 0, 0, 2, 0, top_k);
}

test "indexer module declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Index);
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(Pick);
    std.testing.refAllDecls(TopkArgs);
    std.testing.refAllDecls(Job);
    std.testing.refAllDecls(PackedCandidate);
    std.testing.refAllDecls(SelectionRef);
    std.testing.refAllDecls(DcpResult);
    std.testing.refAllDecls(FusedDcpResult);
    std.testing.refAllDecls(KeyPlan);
}
