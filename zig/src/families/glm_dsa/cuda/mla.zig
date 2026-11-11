// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
// Adapted from bertholomus/bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/mla_pe.py (Apache-2.0).
// DCP coordination follows bertholomus/bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/dcp.py and ashhart/TensorFold PR #159 by drowzeys (Apache-2.0).
// Changes: Zig host references and CUDA launch plans for full GLM-5.3 MLA, including native FP8 kv_b access.
//! GLM-5.3 MLA host references, launch plans and CUDA entry points (K3/K4).
const std = @import("std");
const cuda = @import("cuda");
const types = @import("types");
extern "c" fn glm53_stage_plane_slot_host(position: i64, world: c_int) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;

pub const latent_dim = 512;
pub const rope_dim = 64;
pub const nope_dim = 192;
pub const value_dim = 256;
pub const max_decode_rows = 8;
/// Row limit for attend_verify only; decode absorb/expand kernels remain capped at max_decode_rows.
pub const verify_max_rows = 16;
pub const decode_mla_v2_threads = 128;
pub const kvb_rows_per_head = nope_dim + value_dim;
pub const heads = 64;
pub const key_tile = 16;
pub const partial_values_per_head = latent_dim / 2 + 1;
pub const partial_bytes_per_head = partial_values_per_head * @sizeOf(u32);
pub const partial_o_bytes_per_head = latent_dim * @sizeOf(u16);
pub const prefill_rows = 64;
pub const prefill_masked_mha_switch_default = 5120;
pub const prompt_mla_smem: u32 = (16 * 16 + 8 * 16) * @sizeOf(u16);
pub const prompt_mla_rows: usize = 16;
pub const prompt_mla_big_rows: usize = 64;
pub const prompt_mla_big_threads: u32 = 256;
pub const prompt_mla_big_smem: u32 = (64 * 16 + 64 * (16 + 2)) * @sizeOf(u16);
pub const PromptMlaMode = enum { off, tile16, tile64 };

pub const Plan = struct { grid: [3]u32, block: u32, dynamic_smem: u32 };

pub fn kvWritePlan(rows: usize) Plan {
    return .{ .grid = .{ @intCast(rows), 1, 1 }, .block = 256, .dynamic_smem = 0 };
}

pub fn absorbPlan(rows: usize, local_heads: usize) Plan {
    return .{ .grid = .{ @intCast((latent_dim + 127) / 128), @intCast(local_heads), @intCast(rows) }, .block = 128, .dynamic_smem = 0 };
}

pub const AttendKernel = enum { scalar, mma };

/// tf_glm53_attend_mma's dynamic shared memory (mla.cu MMA_SMEM): Q [32][584] and two 16-key K tiles [2][16][584] bf16 (cp.async double buffer),
/// the two K-half score partials [2][32][16] fp32, P [32][24] bf16, per-head max/denominator/alpha, both tiles' slots and planes.
/// Decode attend on the tensor cores splits a row's keys into fixed 64-key ranges (4 tiles of 16; 32 splits cover the
/// 2048 selected keys), one block each, merged by tf_glm53_attend_split_merge. Fixed, not rows-dependent: a row's
/// bits stay the same alone or in a verify window.
pub const decode_split_tiles = 4;
pub const decode_splits = 2048 / 16 / decode_split_tiles;

pub fn headGroups(local_heads: usize) usize {
    return (local_heads + 31) / 32;
}

/// The decode split workspace: [groups][rows][splits][32 heads][latent + max + denominator] fp32.
pub fn splitWorkspaceBytes(rows: usize, local_heads: usize) u64 {
    return @as(u64, rows) * headGroups(local_heads) * decode_splits * 32 * (latent_dim + 2) * 4;
}

pub const attend_mma_smem: u32 = 32 * 584 * 2 + 2 * 16 * 584 * 2 + 2 * 32 * 16 * 4 + 32 * 24 * 2 + 3 * 32 * 4 + 4 * 16 * 4;
/// tf_glm53_attend_verify: attend_mma's shared carve plus two 64-slot owner-list tables.
pub const decode_verify_chunks = 16;
pub const decode_verify_splits_per_cta = decode_splits / decode_verify_chunks;
pub const attend_verify_smem: u32 = attend_mma_smem + decode_verify_splits_per_cta * decode_split_tiles * key_tile * 4;

pub fn verifySplit(chunk: usize, k: usize) usize {
    return chunk + k * decode_verify_chunks;
}

pub fn splitBusy(local_keys: usize, split: usize) bool {
    return split * decode_split_tiles * key_tile < local_keys;
}

pub fn attendVerifyPlan(rows: usize, local_heads: usize) !Plan {
    if (rows == 0 or rows > verify_max_rows) return error.DecodeMlaRows;
    if (local_heads == 0 or (local_heads > 32 and local_heads % 32 != 0) or (local_heads < 32 and local_heads % 2 != 0)) return error.AttendShape;
    return .{ .grid = .{ @intCast(rows), @intCast(headGroups(local_heads)), decode_verify_chunks }, .block = 256, .dynamic_smem = attend_verify_smem };
}

/// tf_glm53_smla_p (mla.cu SMLA_*, TF_GLM53_SMLA_P=1): the prefill attend with query blocking. A block takes
/// smla_p_rows = 2 adjacent rows x one 32-head group (M = 64; 4 rows would need 256 KB of fp32 O registers) and steps
/// through the rows' own 16-entry key tiles in lockstep, loading each step's union of keys once. Dynamic shared memory:
/// two 32-key union buffers [2][32][584] bf16, a zero key row [584] bf16, the four K-quarter score partials
/// [4][64][16] fp32, P [64][24] bf16, per-query-head max/denominator/alpha [3][64] fp32, the rows' positions and list
/// lengths, and a 3-deep step ring of union tables (entry -> union row, union slots, planes, counts).
pub const smla_p_rows = 2;
pub const smla_p_threads = 512;
pub const smla_p_smem: u32 = 2 * 32 * 584 * 2 + 584 * 2 + 4 * 64 * 16 * 4 + 64 * 24 * 2 + 3 * 64 * 4 + smla_p_rows * 8 + 16 + 3 * 3 * 32 * 4 + 16;

/// tf_glm53_smla_p_fp8 (mla.cu SMF_*, TF_GLM53_SMLA_FP8=1, only with TF_GLM53_SMLA_P=1): SMLA-P with the score GEMM
/// in e4m3 (power-of-two scales per query row-head and per key row; the KV cache stays bf16, keys are converted in shared
/// memory each step). Same grid, block and arguments as SMLA-P. Dynamic shared memory: bf16 latent rows [2][32][520],
/// bf16 rope rows [2][32][72] (the step's [64][16] fp32 scores reuse the step's rope buffer), e4m3 key rows [32][592],
/// a zero latent row [520] bf16, P [64][24] bf16, max/denominator/alpha [3][64] fp32, Q scales [64] and key scales [32]
/// fp32, and SMLA-P's positions, list lengths and union ring.
pub const smla_fp8_smem: u32 = 2 * 32 * 520 * 2 + 2 * 32 * 72 * 2 + 32 * 592 + 520 * 2 + 64 * 24 * 2 + 3 * 64 * 4 + 64 * 4 + 32 * 4 + smla_p_rows * 8 + 16 + 3 * 3 * 32 * 4 + 16;

/// tf_glm53_smla_p3 (mla.cu SMLA3_*, TF_GLM53_SMLA_P3=1, only with TF_GLM53_SMLA_P=1): SMLA-P with the step union
/// tables precomputed by tf_glm53_smla_union into a workspace table and 1 full + 2 named barriers a step (bit-identical
/// to SMLA-P). Same grid and block as SMLA-P. Dynamic shared memory: SMLA-P's key buffers, zero row, score partials,
/// P, softmax state, positions and lengths, then a 3-deep ring of 176-byte step records.
pub const smla_p3_record_bytes: u64 = 176;
pub const smla_p3_steps: u64 = 2048 / key_tile;
pub const smla_p3_union_warps = 4;
pub const smla_p3_smem: u32 = 2 * 32 * 584 * 2 + 584 * 2 + 4 * 64 * 16 * 4 + 64 * 24 * 2 + 3 * 64 * 4 + smla_p_rows * 8 + 16 + 3 * @as(u32, @intCast(smla_p3_record_bytes));

/// The union table for `rows` prefill rows: [ceil(rows / 2) pairs][128 steps][176 B] (mla.cu SMLA3_REC).
pub fn smlaP3TableBytes(rows: usize) u64 {
    const pairs: u64 = @intCast((rows + smla_p_rows - 1) / smla_p_rows);
    return pairs * smla_p3_steps * smla_p3_record_bytes;
}

/// tf_glm53_smla_union: one warp a row pair, 4 pairs a block.
pub fn smlaUnionPlan(rows: usize) Plan {
    const pairs = (rows + smla_p_rows - 1) / smla_p_rows;
    return .{ .grid = .{ @intCast((pairs + smla_p3_union_warps - 1) / smla_p3_union_warps), 1, 1 }, .block = smla_p3_union_warps * 32, .dynamic_smem = 0 };
}

pub fn smlaP3Plan(rows: usize, local_heads: usize) !Plan {
    var p = try smlaPPlan(rows, local_heads);
    p.dynamic_smem = smla_p3_smem;
    return p;
}

pub fn smlaPFp8Plan(rows: usize, local_heads: usize) !Plan {
    var p = try smlaPPlan(rows, local_heads);
    p.dynamic_smem = smla_fp8_smem;
    return p;
}

pub fn smlaPPlan(rows: usize, local_heads: usize) !Plan {
    if (local_heads == 0 or (local_heads > 32 and local_heads % 32 != 0) or (local_heads < 32 and local_heads % 2 != 0)) return error.AttendShape;
    return .{ .grid = .{ @intCast((rows + smla_p_rows - 1) / smla_p_rows), @intCast(headGroups(local_heads)), 1 }, .block = smla_p_threads, .dynamic_smem = smla_p_smem };
}

pub fn attendPlan(rows: usize, local_heads: usize, prefill: bool, kernel: AttendKernel) !Plan {
    _ = prefill;
    if (kernel == .mma and (local_heads == 0 or (local_heads > 32 and local_heads % 32 != 0) or (local_heads < 32 and local_heads % 2 != 0))) return error.AttendShape;
    return .{ .grid = .{ @intCast(rows), @intCast(if (kernel == .mma) headGroups(local_heads) else local_heads), 1 }, .block = 256, .dynamic_smem = if (kernel == .mma) attend_mma_smem else 0 };
}

/// The scalar MLA kernel maps one block to one complete head for decode and prefill.
pub fn headForBlock(block_y: usize, local_heads: usize) ?usize {
    return if (block_y < local_heads) block_y else null;
}

pub fn denseLocalCount(position: usize, rank: usize, world: usize) !usize {
    if (world == 0 or rank >= world) return error.DcpShape;
    if (position < rank) return 0;
    return (position - rank) / world + 1;
}

pub fn denseLocalPosition(rank: usize, world: usize, local_index: usize) !usize {
    if (world == 0 or rank >= world or local_index > (std.math.maxInt(usize) - rank) / world) return error.DcpShape;
    return local_index * world + rank;
}

pub fn isDense(position: usize) bool {
    return position < 2048;
}

pub fn localSlot(position: usize, rank: usize, world: usize) !?usize {
    if (world == 0 or rank >= world) return error.DcpShape;
    if (position % world != rank) return null;
    return position / world;
}

pub fn globalPosition(slot: usize, rank: usize, world: usize) !usize {
    if (world == 0 or rank >= world) return error.DcpShape;
    return slot * world + rank;
}

test "decode verify plan: 16 chunks of 2 splits, attend_mma smem + slot table, rows 1..16" {
    try std.testing.expectEqual(decode_splits, decode_verify_chunks * decode_verify_splits_per_cta);
    try std.testing.expectEqual(@as(u32, 81536), attend_verify_smem); // mla.cu static_assert
    try std.testing.expect(attend_verify_smem <= 99 * 1024);
    for (1..17) |r| {
        const p = try attendVerifyPlan(r, 64);
        try std.testing.expectEqual([3]u32{ @intCast(r), 2, 16 }, p.grid);
        try std.testing.expectEqual(@as(u32, 256), p.block);
        try std.testing.expectEqual(@as(u32, 81536), p.dynamic_smem);
    }
    try std.testing.expectEqual(2 * splitWorkspaceBytes(8, 64), splitWorkspaceBytes(16, 64));
    try std.testing.expectEqual(@as(u64, 67_371_008), splitWorkspaceBytes(16, 64));
    try std.testing.expectEqual(@as(u32, 1), (try attendVerifyPlan(3, 32)).grid[1]);
    try std.testing.expectEqual(@as(u32, 1), (try attendVerifyPlan(3, 22)).grid[1]);
    try std.testing.expectEqual(@as(u32, 1), (try attendVerifyPlan(3, 20)).grid[1]);
    try std.testing.expectEqual(@as(u32, 1), (try smlaPPlan(3, 22)).grid[1]);
    try std.testing.expectEqual(@as(u32, 1), (try smlaPPlan(3, 20)).grid[1]);
    try std.testing.expectEqual(@as(u64, 33_685_504), splitWorkspaceBytes(16, 22));
    try std.testing.expectEqual(@as(u64, 33_685_504), splitWorkspaceBytes(16, 20));
    try std.testing.expectError(error.DecodeMlaRows, attendVerifyPlan(0, 64));
    try std.testing.expectError(error.DecodeMlaRows, attendVerifyPlan(17, 64));
    for ([_]usize{ 31, 48, 0 }) |h| try std.testing.expectError(error.AttendShape, attendVerifyPlan(1, h));
    const counts = [_]usize{ 0, 1, 15, 16, 17, 63, 64, 65, 640, 1000, 1024, 1025, 2047, 2048 };
    for (counts) |n| {
        const busy = (n + 63) / 64;
        var seen: [decode_splits]bool = @splat(false);
        for (0..decode_verify_chunks) |c| for (0..decode_verify_splits_per_cta) |k| {
            const s = verifySplit(c, k);
            const is_busy = splitBusy(n, s);
            try std.testing.expectEqual(s < busy, is_busy);
            if (is_busy) {
                try std.testing.expect(!seen[s]);
                seen[s] = true;
            }
            if (k == 1 and is_busy) try std.testing.expect(splitBusy(n, verifySplit(c, 0)));
        };
        for (0..decode_splits) |s| try std.testing.expectEqual(s < busy, seen[s]);
    }
}

/// Absolute cache slot used by the gather-on-prefill kernel for a peer key position.
pub fn stagePlaneSlot(position: usize, world: usize) !usize {
    if (world == 0) return error.DcpShape;
    return @intCast(glm53_stage_plane_slot_host(@intCast(position), @intCast(world)));
}

pub fn useMaskedMha(position: usize, switch_point: usize) bool {
    return position < switch_point;
}

fn validateAttendStage(world: usize, prefill: bool, stage_latent: u64, stage_rope: u64) !void {
    try validateKvWriteStage(world, prefill, stage_latent, stage_rope);
}

fn validateKvWriteStage(world: usize, prefill: bool, stage_latent: u64, stage_rope: u64) !void {
    if (world > 1 and prefill and (stage_latent == 0 or stage_rope == 0)) return error.MissingPrefillStage;
    if ((stage_latent == 0) != (stage_rope == 0)) return error.InvalidStagePair;
}

const StagePointers = struct { latent: u64, rope: u64 };

fn stagePointers(stage: ?types.StagePlanePair) StagePointers {
    if (stage) |planes| return .{ .latent = @intCast(planes.latent.data.ptr), .rope = @intCast(planes.rope.data.ptr) };
    return .{ .latent = 0, .rope = 0 };
}

fn validateAttendHeads(prefill: bool, local_heads: usize) !void {
    if (prefill and local_heads > 32) return error.PrefillHeadLimit;
}

/// Round fp32 to BF16, round-to-nearest ties-to-even.
pub fn bf16(x: f32) f32 {
    var bits: u32 = @bitCast(x);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @bitCast(bits & 0xffff0000);
}

/// Match rope.py:_apply_heads with --fmad=false: round products before the add/subtract.
pub fn ropePair(x: f32, y: f32, c: f32, s: f32) [2]f32 {
    const xc = x * c;
    const ys = y * s;
    const yc = y * c;
    const xs = x * s;
    return .{ bf16(xc - ys), bf16(yc + xs) };
}

/// DCP layout helper for [group][row][32] buffers.
pub fn groupRowHead(group: usize, row: usize, head: usize, rows: usize) usize {
    return (group * rows + row) * 32 + head;
}

/// Peer-only DCP2 payload: contiguous BF16 partial plane followed by its FP32 LSE plane.
pub fn peerPartialBytes(rows: usize, local_heads: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, rows, local_heads), partial_bytes_per_head);
}

/// Bytes in one contiguous DCP block: [rows, heads, BF16 o][rows, heads, FP32 LSE].
pub fn partialBlockBytes(rows: usize, local_heads: usize) !usize {
    return peerPartialBytes(rows, local_heads);
}

pub fn partialLseByteOffset(rows: usize, local_heads: usize) !usize {
    return std.math.mul(usize, try std.math.mul(usize, rows, local_heads), partial_o_bytes_per_head);
}

pub fn partialGroupByteOffset(group: usize, rows: usize, local_heads: usize) !usize {
    return std.math.mul(usize, group, try partialBlockBytes(rows, local_heads));
}

/// Whether kv_write must skip a peer row when decode passes null staging planes.
pub fn kvWriteMaySkip(owner: usize, rank: usize, stage_available: bool) bool {
    return owner != rank and !stage_available;
}

pub fn kvWriteSourceOffset(block_row: usize, stride: usize) usize {
    return block_row * stride;
}

pub fn kvWriteDestinationOffset(slot: usize, row_stride: usize) usize {
    return slot * row_stride;
}

/// Decode one finite E4M3FN byte to fp32. NaN encodings follow CUDA's E4M3FN conversion.
pub fn e4m3(code: u8) f32 {
    const sign: f32 = if (code & 0x80 == 0) 1 else -1;
    const exponent: u8 = (code >> 3) & 0x0f;
    const mantissa: u8 = code & 7;
    if (exponent == 0) return sign * @as(f32, @floatFromInt(mantissa)) * (1.0 / 512.0);
    if (exponent == 15 and mantissa == 7) return std.math.nan(f32);
    const m: f32 = 1.0 + @as(f32, @floatFromInt(mantissa)) * 0.125;
    return sign * m * std.math.pow(f32, 2, @as(f32, @floatFromInt(exponent)) - 7.0);
}

/// Dequantize a kv_b value using its native 128x128 inverse scale, with the required BF16 rounding.
pub fn kvbValue(codes: []const u8, scales: []const f32, row: usize, col: usize, ncols: usize) !f32 {
    const scale_cols = (ncols + 127) / 128;
    if (ncols == 0 or row * ncols + col >= codes.len or (row / 128) * scale_cols + col / 128 >= scales.len) return error.KvBShape;
    return bf16(e4m3(codes[row * ncols + col]) * scales[(row / 128) * scale_cols + col / 128]);
}

/// Reference absorb for tiny and full heads. q is BF16-rounded and kv_b is native E4M3 + 128x128 scales.
pub fn absorbRef(q: []const f32, rows: usize, head_count: usize, codes: []const u8, scales: []const f32, ncols: usize, out: []f32) !void {
    if (ncols != latent_dim or q.len < rows * head_count * (nope_dim + rope_dim) or codes.len < head_count * kvb_rows_per_head * ncols or out.len < rows * head_count * ncols) return error.AbsorbShape;
    for (0..rows) |r| for (0..head_count) |h| for (0..ncols) |c| {
        var acc: f64 = 0;
        for (0..nope_dim) |d| {
            const w = try kvbValue(codes, scales, h * kvb_rows_per_head + d, c, ncols);
            acc += @as(f64, bf16(q[(r * head_count + h) * (nope_dim + rope_dim) + d])) * w;
        }
        out[(r * head_count + h) * ncols + c] = bf16(@floatCast(acc));
    };
}

/// RoPE the q_b output's 64-wide q_pe tail into packed storage, using per-launch-row tables.
pub fn ropeQueryRef(q: []const f32, rows: usize, head_count: usize, positions: []const usize, cos: []const f32, sin: []const f32, out: []f32) !void {
    const stride = nope_dim + rope_dim;
    if (q.len < rows * head_count * stride or positions.len < rows or out.len < rows * head_count * rope_dim) return error.QueryRopeShape;
    for (0..rows) |r| for (0..head_count) |h| for (0..rope_dim / 2) |i| {
        _ = positions[r];
        if (r * (rope_dim / 2) + i >= cos.len or r * (rope_dim / 2) + i >= sin.len) return error.RopeTableShape;
        const base = (r * head_count + h) * stride + nope_dim + 2 * i;
        const c = cos[r * (rope_dim / 2) + i];
        const s = sin[r * (rope_dim / 2) + i];
        const x = bf16(q[base]);
        const y = bf16(q[base + 1]);
        const pair = ropePair(x, y, c, s);
        out[(r * head_count + h) * rope_dim + 2 * i] = pair[0];
        out[(r * head_count + h) * rope_dim + 2 * i + 1] = pair[1];
    };
}

/// Host reference for kv_a normalization, row-indexed RoPE, and cache-row construction.
pub fn kvWriteRef(projected: []const f32, rows: usize, stride: usize, weight: []const f32, positions: []const usize, cos: []const f32, sin: []const f32, latent: []f32, rope: []f32) !void {
    if (stride < latent_dim + rope_dim or projected.len < rows * stride or weight.len < latent_dim or positions.len < rows or cos.len < rows * rope_dim / 2 or sin.len < rows * rope_dim / 2 or latent.len < rows * latent_dim or rope.len < rows * rope_dim) return error.KvWriteShape;
    for (0..rows) |r| {
        var ss: f64 = 0;
        for (0..latent_dim) |d| {
            const x = bf16(projected[r * stride + d]);
            ss += @as(f64, x) * x;
        }
        const inv = 1.0 / @sqrt(@as(f32, @floatCast(ss / latent_dim)) + 1e-5);
        for (0..latent_dim) |d| latent[r * latent_dim + d] = bf16(bf16(bf16(projected[r * stride + d]) * inv) * bf16(weight[d]));
        for (0..rope_dim / 2) |i| {
            if (r * (rope_dim / 2) + i >= cos.len or r * (rope_dim / 2) + i >= sin.len) return error.RopeTableShape;
            const c = cos[r * (rope_dim / 2) + i];
            const s = sin[r * (rope_dim / 2) + i];
            const x = bf16(projected[r * stride + latent_dim + 2 * i]);
            const y = bf16(projected[r * stride + latent_dim + 2 * i + 1]);
            const pair = ropePair(x, y, c, s);
            rope[r * rope_dim + 2 * i] = pair[0];
            rope[r * rope_dim + 2 * i + 1] = pair[1];
        }
    }
}

pub const Online = struct { max: f64 = -std.math.inf(f64), sum: f64 = 0, value: []f64 };

/// Deterministic fp64 host attention reference. Keys are visited in ascending absolute-position tiles;
/// `query_position` and `key_positions` make chunk boundaries irrelevant.
pub fn attendRef(allocator: std.mem.Allocator, qa: []const f32, qpe: []const f32, latent: []const f32, rope: []const f32, key_positions: []const usize, query_position: usize, out: []f32) !void {
    if (qa.len != latent_dim or qpe.len != rope_dim or latent.len < key_positions.len * latent_dim or rope.len < key_positions.len * rope_dim or out.len < latent_dim) return error.AttendShape;
    const acc = try allocator.alloc(f64, latent_dim);
    defer allocator.free(acc);
    @memset(acc, 0);
    var maximum: f64 = -std.math.inf(f64);
    var denominator: f64 = 0;
    var tile_start: usize = 0;
    while (tile_start < key_positions.len) {
        const pos = key_positions[tile_start];
        if (pos > query_position) return error.FutureKey;
        const abs_tile = pos / key_tile;
        const tile_end = blk: {
            var e = tile_start + 1;
            while (e < key_positions.len and key_positions[e] / key_tile == abs_tile) : (e += 1) {}
            break :blk e;
        };
        for (tile_start..tile_end) |k| {
            if (k > tile_start and key_positions[k] < key_positions[k - 1]) return error.UnorderedKeys;
            var score: f64 = 0;
            for (0..latent_dim) |d| score += @as(f64, qa[d]) * latent[k * latent_dim + d];
            for (0..rope_dim) |d| score += @as(f64, qpe[d]) * rope[k * rope_dim + d];
            score /= 16.0;
            const next_max = @max(maximum, score);
            const alpha = if (maximum == -std.math.inf(f64)) 0 else @exp(maximum - next_max);
            const p = @exp(score - next_max);
            const p_bf16: f64 = bf16(@floatCast(p));
            denominator = denominator * alpha + p;
            for (0..latent_dim) |d| acc[d] = acc[d] * alpha + p_bf16 * latent[k * latent_dim + d];
            maximum = next_max;
        }
        tile_start = tile_end;
    }
    if (denominator == 0) return error.EmptyAttention;
    for (0..latent_dim) |d| out[d] = bf16(@floatCast(acc[d] / denominator));
}

/// Scalar CUDA-order emulation: fp32 FMA dots and sequential online softmax, with BF16 P and output.
pub fn attendCudaRef(qa: []const f32, qpe: []const f32, latent: []const f32, rope: []const f32, key_positions: []const usize, query_position: usize, out: []f32) !void {
    if (qa.len != latent_dim or qpe.len != rope_dim or latent.len < key_positions.len * latent_dim or rope.len < key_positions.len * rope_dim or out.len < latent_dim) return error.AttendShape;
    var maximum: f32 = -std.math.inf(f32);
    var denominator: f32 = 0;
    var accum: [latent_dim]f32 = @splat(0);
    for (key_positions, 0..) |position, k| {
        if (position > query_position) return error.FutureKey;
        if (k > 0 and position < key_positions[k - 1]) return error.UnorderedKeys;
        var score: f32 = 0;
        for (0..latent_dim) |d| score = @mulAdd(f32, qa[d], latent[k * latent_dim + d], score);
        for (0..rope_dim) |d| score = @mulAdd(f32, qpe[d], rope[k * rope_dim + d], score);
        score *= 0.0625;
        const next = @max(maximum, score);
        const alpha: f32 = if (std.math.isInf(maximum)) 0 else @exp(maximum - next);
        const p = @exp(score - next);
        const pb = bf16(p);
        denominator = denominator * alpha + p;
        for (0..latent_dim) |d| accum[d] = @mulAdd(f32, accum[d], alpha, pb * latent[k * latent_dim + d]);
        maximum = next;
    }
    if (denominator == 0) return error.EmptyAttention;
    for (0..latent_dim) |d| out[d] = bf16(accum[d] / denominator);
}

/// Host emulation of tensor-core score tiles and tile-wise online softmax.
pub fn attendMmaRef(qa: []const f32, qpe: []const f32, latent: []const f32, rope: []const f32, key_positions: []const usize, query_position: usize, out: []f32) !void {
    if (qa.len != latent_dim or qpe.len != rope_dim or latent.len < key_positions.len * latent_dim or rope.len < key_positions.len * rope_dim or out.len < latent_dim) return error.AttendShape;
    var maximum: f32 = -std.math.inf(f32);
    var denominator: f32 = 0;
    var accum: [latent_dim]f32 = @splat(0);
    var start: usize = 0;
    while (start < key_positions.len) {
        const end = @min(start + key_tile, key_positions.len);
        var tile_scores: [key_tile]f32 = @splat(-std.math.inf(f32));
        for (start..end) |k| {
            const pos = key_positions[k];
            if (pos > query_position) return error.FutureKey;
            if (k > 0 and pos < key_positions[k - 1]) return error.UnorderedKeys;
            // The kernel's two K halves of the 576 dims [latent | rope]: 0..287 and 288..575, summed in that order.
            var half0: f32 = 0;
            var half1: f32 = 0;
            for (0..288) |d| half0 = @mulAdd(f32, qa[d], latent[k * latent_dim + d], half0);
            for (288..latent_dim) |d| half1 = @mulAdd(f32, qa[d], latent[k * latent_dim + d], half1);
            for (0..rope_dim) |d| half1 = @mulAdd(f32, qpe[d], rope[k * rope_dim + d], half1);
            tile_scores[k - start] = (half0 + half1) * 0.0625;
        }
        var next = maximum;
        for (tile_scores) |s| next = @max(next, s);
        const alpha: f32 = if (maximum == -std.math.inf(f32)) 0 else @exp(maximum - next);
        var tile_sum: f32 = 0;
        for (0..end - start) |j| tile_sum += @exp(tile_scores[j] - next);
        denominator = denominator * alpha + tile_sum;
        for (&accum) |*a| a.* *= alpha; // once a tile, as the kernel rescales O before the tile's P.V
        for (0..end - start) |j| {
            const p = bf16(@exp(tile_scores[j] - next));
            for (0..latent_dim) |d| accum[d] = @mulAdd(f32, p, latent[(start + j) * latent_dim + d], accum[d]);
        }
        maximum = next;
        start = end;
    }
    if (denominator == 0) return error.EmptyAttention;
    for (0..latent_dim) |d| out[d] = bf16(accum[d] / denominator);
}

/// Fixed-order merge of DCP partials: each rank contributes a BF16-normalized output and its log-sum-exp.
pub fn mergeRef(lse: [2]f32, o: [2][]const f32, out: []f32) !f32 {
    if (o[0].len != o[1].len or out.len < o[0].len) return error.MergeShape;
    const lse_max = @max(lse[0], lse[1]);
    const a: f32 = if (lse[0] == -std.math.inf(f32)) 0 else @exp2((lse[0] - lse_max) * 1.4426950408889634);
    const b: f32 = if (lse[1] == -std.math.inf(f32)) 0 else @exp2((lse[1] - lse_max) * 1.4426950408889634);
    const denom = a + b;
    if (denom == 0) return error.EmptyAttention;
    for (0..o[0].len) |d| {
        var acc = @mulAdd(f32, bf16(o[0][d]), a, 0);
        acc = @mulAdd(f32, bf16(o[1][d]), b, acc);
        out[d] = bf16(acc / denom);
    }
    return lse_max + @log(denom);
}

/// Local/peer inputs mapped back to rank order before applying mergeRef's arithmetic.
pub fn mergePeerRef(rank: usize, lse_local: f32, lse_peer: f32, o_local: []const f32, o_peer: []const f32, out: []f32) !f32 {
    if (rank > 1) return error.DcpShape;
    if (rank == 0) return mergeRef(.{ lse_local, lse_peer }, .{ o_local, o_peer }, out);
    return mergeRef(.{ lse_peer, lse_local }, .{ o_peer, o_local }, out);
}

/// Expand BF16 latent output through one head's FP8 kv_b value rows.
pub fn expandVRef(allocator: std.mem.Allocator, ol: []const f32, codes: []const u8, scales: []const f32, head: usize, out: []f32) !void {
    if (ol.len < latent_dim or out.len < value_dim or codes.len < (head + 1) * kvb_rows_per_head * latent_dim) return error.ExpandShape;
    _ = allocator;
    for (0..value_dim) |j| {
        var acc: f64 = 0;
        for (0..latent_dim) |c| acc += @as(f64, bf16(ol[c])) * (try kvbValue(codes, scales, head * kvb_rows_per_head + nope_dim + j, c, latent_dim));
        out[j] = bf16(@floatCast(acc));
    }
}

pub const Mla = struct {
    prep_fn: cuda.Function,
    qpe_fn: cuda.Function,
    absorb_fn: cuda.Function,
    absorb_decode_fn: cuda.Function,
    absorb_decode_v2_fn: cuda.Function,
    absorb_decode_v3_fn: cuda.Function,
    attend_fn: cuda.Function,
    attend_mma_fn: cuda.Function,
    attend_split_merge_fn: cuda.Function,
    attend_kernel: AttendKernel = .mma,
    merge_fn: cuda.Function,
    expand_fn: cuda.Function,

    absorb_prompt_fn: cuda.Function,
    expand_prompt_fn: cuda.Function,
    smla_p_fn: cuda.Function,
    smla_fp8_fn: cuda.Function,
    absorb_prompt_big_fn: cuda.Function,
    expand_prompt_big_fn: cuda.Function,
    expand_decode_fn: cuda.Function,
    expand_decode_v2_fn: cuda.Function,
    /// TF_GLM53_SMLA_P3=1 only (loadSmlaP3 at setup): tf_glm53_smla_union and tf_glm53_smla_p3; null otherwise.
    smla_union_fn: ?cuda.Function = null,
    smla_p3_fn: ?cuda.Function = null,
    /// TF_GLM53_DECODE_UNION=1 only (loadAttendVerify at setup).
    attend_verify_fn: ?cuda.Function = null,


    /// TF_GLM53_ABSORB_ROWS=1: decode absorb v3 decodes each kv_b weight once for all of a thread's rows
    /// (tf_glm53_absorb_decode_v3r) instead of one d loop a row. Same FMA order per output: bit-identical.
    pub fn absorbRowsSwitch() bool {
        const v = getenv("TF_GLM53_ABSORB_ROWS") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    pub fn load(module: cuda.Module) !Mla {
        const selected = if (getenv("TF_GLM53_ATTEND")) |raw| blk: {
            const value = std.mem.span(raw);
            if (std.mem.eql(u8, value, "mma")) break :blk AttendKernel.mma;
            if (std.mem.eql(u8, value, "scalar")) break :blk AttendKernel.scalar;
            return error.InvalidAttendKernel;
        } else AttendKernel.mma; // default: tensor cores (decode with key splits); TF_GLM53_ATTEND=scalar: the reference kernel
        const attend_mma_fn = try module.function("tf_glm53_attend_mma");
        try attend_mma_fn.allowDynamicShared(attend_mma_smem);

        const absorb_prompt_fn = try module.function("tf_glm53_absorb_prompt_mma");
        const expand_prompt_fn = try module.function("tf_glm53_expand_v_prompt_mma");
        try absorb_prompt_fn.allowDynamicShared(prompt_mla_smem);
        try expand_prompt_fn.allowDynamicShared(prompt_mla_smem);
        const smla_p_fn = try module.function("tf_glm53_smla_p");
        try smla_p_fn.allowDynamicShared(smla_p_smem);
        const smla_fp8_fn = try module.function("tf_glm53_smla_p_fp8");
        try smla_fp8_fn.allowDynamicShared(smla_fp8_smem);
        const absorb_prompt_big_fn = try module.function("tf_glm53_absorb_prompt_mma_big");
        const expand_prompt_big_fn = try module.function("tf_glm53_expand_v_prompt_mma_big");
        try absorb_prompt_big_fn.allowDynamicShared(prompt_mla_big_smem);
        try expand_prompt_big_fn.allowDynamicShared(prompt_mla_big_smem);
        return .{ .prep_fn = try module.function("tf_glm53_kv_write"), .qpe_fn = try module.function("tf_glm53_qpe_pack"), .absorb_fn = try module.function("tf_glm53_absorb"), .absorb_decode_fn = try module.function("tf_glm53_absorb_decode"), .absorb_decode_v2_fn = try module.function("tf_glm53_absorb_decode_v2"), .absorb_decode_v3_fn = try module.function(if (absorbRowsSwitch()) "tf_glm53_absorb_decode_v3r" else "tf_glm53_absorb_decode_v3"), .attend_fn = try module.function("tf_glm53_attend"), .attend_mma_fn = attend_mma_fn, .attend_split_merge_fn = try module.function("tf_glm53_attend_split_merge"), .attend_kernel = selected, .merge_fn = try module.function("tf_glm53_mla_merge"), .expand_fn = try module.function("tf_glm53_expand_v"), .expand_decode_fn = try module.function("tf_glm53_expand_v_decode"), .expand_decode_v2_fn = try module.function("tf_glm53_expand_v_decode_v2"), .absorb_prompt_fn = absorb_prompt_fn, .expand_prompt_fn = expand_prompt_fn, .smla_p_fn = smla_p_fn, .smla_fp8_fn = smla_fp8_fn, .absorb_prompt_big_fn = absorb_prompt_big_fn, .expand_prompt_big_fn = expand_prompt_big_fn };

    }

    /// TF_GLM53_SMLA_P3=1: resolve the union pre-kernel and the P3 attend. Setup calls it only with the switch on (off:
    /// no extra symbol lookups), and fails here when the image lacks either kernel (a stale fatbin).
    pub fn loadSmlaP3(m: *Mla, module: cuda.Module) !void {
        const union_fn = module.function("tf_glm53_smla_union") catch return error.MissingSmlaP3Kernels;
        const p3_fn = module.function("tf_glm53_smla_p3") catch return error.MissingSmlaP3Kernels;
        try p3_fn.allowDynamicShared(smla_p3_smem);
        m.smla_union_fn = union_fn;
        m.smla_p3_fn = p3_fn;
    }

    /// Resolve tf_glm53_attend_verify only when TF_GLM53_DECODE_UNION=1 is selected.
    pub fn loadAttendVerify(m: *Mla, module: cuda.Module) !void {
        const verify_fn = module.function("tf_glm53_attend_verify") catch return error.MissingDecodeUnionKernel;
        try verify_fn.allowDynamicShared(attend_verify_smem);
        m.attend_verify_fn = verify_fn;
    }

    pub fn prep(m: *const Mla, stream: cuda.Stream, proj: u64, norm: u64, cos: u64, sin: u64, positions: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, rows: usize, stride: usize, rank: usize, world: usize, prefill: bool) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world or stride < latent_dim + rope_dim) return error.KvWriteShape;
        const stage_ptrs = stagePointers(stage);
        try validateKvWriteStage(world, prefill, stage_ptrs.latent, stage_ptrs.rope);
        var a: cuda.Args = .{};
        inline for (.{ proj, norm, cos, sin, positions, latent, rope, stage_ptrs.latent, stage_ptrs.rope, @as(i32, @intCast(rows)), @as(i32, @intCast(stride)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)) }) |v| a.add(v);
        try cuda.launch.launch(m.prep_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
    }

    pub fn packQpe(m: *const Mla, stream: cuda.Stream, q: u64, cos: u64, sin: u64, qpe: u64, rows: usize, local_heads: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        inline for (.{ q, cos, sin, qpe, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)) }) |v| a.add(v);
        try cuda.launch.launch(m.qpe_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(local_heads) }, .block = .{ .x = 32 } }, stream, &a);
    }

    pub fn absorb(m: *const Mla, stream: cuda.Stream, q: u64, kvb: u64, scales: u64, qa: u64, rows: usize, local_heads: usize, global_head0: usize, decode_mla: bool) !void {
        if (rows == 0) return;
        if (decode_mla and rows > max_decode_rows) return error.DecodeMlaRows;
        var a: cuda.Args = .{};
        inline for (.{ q, kvb, scales, qa, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        try cuda.launch.launch(if (decode_mla) m.absorb_decode_fn else m.absorb_fn, .{ .grid = .{ .x = 1, .y = @intCast(local_heads), .z = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
    }

    pub fn absorbV2(m: *const Mla, stream: cuda.Stream, q: u64, kvb: u64, scales: u64, qa: u64, rows: usize, local_heads: usize, global_head0: usize) !void {
        if (rows == 0) return;
        if (rows > max_decode_rows) return error.DecodeMlaRows;
        var a: cuda.Args = .{};
        inline for (.{ q, kvb, scales, qa, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        try cuda.launch.launch(m.absorb_decode_v2_fn, .{ .grid = .{ .x = 1, .y = @intCast(local_heads), .z = 1 }, .block = .{ .x = decode_mla_v2_threads } }, stream, &a);
    }

    pub fn absorbV3(m: *const Mla, stream: cuda.Stream, q: u64, kvb: u64, scales: u64, qa: u64, rows: usize, local_heads: usize, global_head0: usize) !void {
        if (rows == 0) return;
        if (rows > max_decode_rows) return error.DecodeMlaRows;
        var a: cuda.Args = .{};
        inline for (.{ q, kvb, scales, qa, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        try cuda.launch.launch(m.absorb_decode_v3_fn, .{ .grid = .{ .x = @intCast(latent_dim / 64), .y = @intCast(local_heads), .z = 1 }, .block = .{ .x = decode_mla_v2_threads } }, stream, &a);
    }

    pub fn absorbPrompt(m: *const Mla, stream: cuda.Stream, q: u64, kvb: u64, scales: u64, qa: u64, rows: usize, local_heads: usize, global_head0: usize, mode: PromptMlaMode) !void {
        if (rows == 0) return;
        if (mode == .off) return error.PromptMlaDisabled;
        var a: cuda.Args = .{};
        inline for (.{ q, kvb, scales, qa, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        if (mode == .tile16) {
            try cuda.launch.launch(m.absorb_prompt_fn, .{ .grid = .{ .x = @intCast(latent_dim / 8), .y = @intCast(local_heads), .z = @intCast((rows + prompt_mla_rows - 1) / prompt_mla_rows) }, .block = .{ .x = 32 }, .shared = prompt_mla_smem }, stream, &a);
        } else {
            try cuda.launch.launch(m.absorb_prompt_big_fn, .{ .grid = .{ .x = @intCast(latent_dim / 64), .y = @intCast(local_heads), .z = @intCast((rows + prompt_mla_big_rows - 1) / prompt_mla_big_rows) }, .block = .{ .x = prompt_mla_big_threads }, .shared = prompt_mla_big_smem }, stream, &a);
        }
    }

    /// Prefill uses gather-on-prefill for dense and sparse rows, so each rank produces complete attention for
    /// its own heads and must not call `merge`. Decode writes one contiguous partial/LSE block and uses `merge`.
    /// `split_ws`: the decode split workspace (splitWorkspaceBytes(rows, local_heads)); 0, or prefill: one pass a row.
    pub fn attend(m: *const Mla, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize, prefill: bool, split_ws: u64) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world) return error.DcpShape;
        const stage_ptrs = stagePointers(stage);
        try validateAttendStage(world, prefill, stage_ptrs.latent, stage_ptrs.rope);
        try validateAttendHeads(prefill, local_heads);
        const p = try attendPlan(rows, local_heads, prefill, m.attend_kernel);
        var a: cuda.Args = .{};
        inline for (.{ qa, qpe, latent, rope, stage_ptrs.latent, stage_ptrs.rope, positions, global_tokens, local_slots, global_counts, local_counts, dense_len, partial_block, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)), @as(i32, @intFromBool(prefill)) }) |v| a.add(v);
        if (m.attend_kernel == .scalar) {
            try cuda.launch.launch(m.attend_fn, .{ .grid = .{ .x = p.grid[0], .y = p.grid[1], .z = p.grid[2] }, .block = .{ .x = p.block }, .shared = p.dynamic_smem }, stream, &a);
            return;
        }
        const split = !prefill and split_ws != 0;
        a.add(split_ws);
        a.add(@as(i32, if (split) decode_split_tiles else 0));
        try cuda.launch.launch(m.attend_mma_fn, .{ .grid = .{ .x = p.grid[0], .y = p.grid[1], .z = if (split) decode_splits else 1 }, .block = .{ .x = p.block }, .shared = p.dynamic_smem }, stream, &a);
        if (split) {
            var b: cuda.Args = .{};
            inline for (.{ split_ws, partial_block, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, decode_splits) }) |v| b.add(v);
            try cuda.launch.launch(m.attend_split_merge_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(local_heads) }, .block = .{ .x = 256 } }, stream, &b);
        }
    }

    pub fn attendVerify(m: *const Mla, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, positions: u64, local_slots: u64, local_counts: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize, split_ws: u64) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world) return error.DcpShape;
        if (split_ws == 0) return error.DecodeUnionWorkspace;
        const verify_fn = m.attend_verify_fn orelse return error.MissingDecodeUnionKernel;
        const p = try attendVerifyPlan(rows, local_heads);
        var a: cuda.Args = .{};
        inline for (.{ qa, qpe, latent, rope, positions, local_slots, local_counts, split_ws, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)) }) |v| a.add(v);
        try cuda.launch.launch(verify_fn, .{ .grid = .{ .x = p.grid[0], .y = p.grid[1], .z = p.grid[2] }, .block = .{ .x = p.block }, .shared = p.dynamic_smem }, stream, &a);
        var b: cuda.Args = .{};
        inline for (.{ split_ws, partial_block, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, decode_splits) }) |v| b.add(v);
        try cuda.launch.launch(m.attend_split_merge_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(local_heads) }, .block = .{ .x = 256 } }, stream, &b);
    }

    /// SMLA-P: the gather-on-prefill attend (`attend` with prefill = true) with query blocking. Same arguments,
    /// [rows][32][512] BF16 output plane + [rows][32] FP32 LSE plane, empty-row and invalid-key rules; prefill only.
    /// A row's bits do not depend on which rows share its block (mla.cu SMLA-P comment).
    pub fn attendSmlaP(m: *const Mla, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize) !void {
        try m.launchSmla(false, stream, qa, qpe, latent, rope, stage, positions, global_tokens, local_slots, global_counts, local_counts, dense_len, partial_block, rows, local_heads, rank, world);
    }

    /// SMLA-FP8 (TF_GLM53_SMLA_FP8=1 with SMLA-P): `attendSmlaP` with the score GEMM in e4m3. Same arguments and output
    /// contract; lossy (KLD gate). A row's bits do not depend on which rows share its block.
    pub fn attendSmlaPFp8(m: *const Mla, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize) !void {
        try m.launchSmla(true, stream, qa, qpe, latent, rope, stage, positions, global_tokens, local_slots, global_counts, local_counts, dense_len, partial_block, rows, local_heads, rank, world);
    }

    /// SMLA-P3 (TF_GLM53_SMLA_P3=1 with SMLA-P): `attendSmlaP` in two launches on `stream`, tf_glm53_smla_union (the
    /// step union records of every row pair into `table`, smlaP3TableBytes(rows) <= `table_bytes`) then
    /// tf_glm53_smla_p3 (SMLA-P's arguments + the table). Bit-identical output to `attendSmlaP` (mla.cu SMLA-P3 comment).
    pub fn attendSmlaP3(m: *const Mla, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize, table: u64, table_bytes: u64) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world) return error.DcpShape;
        const union_fn = m.smla_union_fn orelse return error.MissingSmlaP3Kernels;
        const p3_fn = m.smla_p3_fn orelse return error.MissingSmlaP3Kernels;
        if (table == 0 or table % 16 != 0 or table_bytes < smlaP3TableBytes(rows)) return error.SmlaP3Table;
        const stage_ptrs = stagePointers(stage);
        try validateAttendStage(world, true, stage_ptrs.latent, stage_ptrs.rope);
        try validateAttendHeads(true, local_heads);
        const u = smlaUnionPlan(rows);
        var ua: cuda.Args = .{};
        inline for (.{ positions, global_tokens, global_counts, table, @as(i32, @intCast(rows)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)) }) |v| ua.add(v);
        try cuda.launch.launch(union_fn, .{ .grid = .{ .x = u.grid[0], .y = u.grid[1], .z = u.grid[2] }, .block = .{ .x = u.block }, .shared = u.dynamic_smem }, stream, &ua);
        const p = try smlaP3Plan(rows, local_heads);
        var a: cuda.Args = .{};
        inline for (.{ qa, qpe, latent, rope, stage_ptrs.latent, stage_ptrs.rope, positions, global_tokens, local_slots, global_counts, local_counts, dense_len, partial_block, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)), @as(i32, 1), table }) |v| a.add(v);
        try cuda.launch.launch(p3_fn, .{ .grid = .{ .x = p.grid[0], .y = p.grid[1], .z = p.grid[2] }, .block = .{ .x = p.block }, .shared = p.dynamic_smem }, stream, &a);
    }

    fn launchSmla(m: *const Mla, fp8: bool, stream: cuda.Stream, qa: u64, qpe: u64, latent: u64, rope: u64, stage: ?types.StagePlanePair, positions: u64, global_tokens: u64, local_slots: u64, global_counts: u64, local_counts: u64, dense_len: u64, partial_block: u64, rows: usize, local_heads: usize, rank: usize, world: usize) !void {
        if (rows == 0) return;
        if (world == 0 or rank >= world) return error.DcpShape;
        const stage_ptrs = stagePointers(stage);
        try validateAttendStage(world, true, stage_ptrs.latent, stage_ptrs.rope);
        try validateAttendHeads(true, local_heads);
        const p = if (fp8) try smlaPFp8Plan(rows, local_heads) else try smlaPPlan(rows, local_heads);
        var a: cuda.Args = .{};
        inline for (.{ qa, qpe, latent, rope, stage_ptrs.latent, stage_ptrs.rope, positions, global_tokens, local_slots, global_counts, local_counts, dense_len, partial_block, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(rank)), @as(i32, @intCast(world)), @as(i32, 1) }) |v| a.add(v);
        try cuda.launch.launch(if (fp8) m.smla_fp8_fn else m.smla_p_fn, .{ .grid = .{ .x = p.grid[0], .y = p.grid[1], .z = p.grid[2] }, .block = .{ .x = p.block }, .shared = p.dynamic_smem }, stream, &a);
    }

    pub fn merge(m: *const Mla, stream: cuda.Stream, partial_local: u64, partial_peer: u64, out: u64, lse_out: u64, rows: usize, local_heads: usize, rank: usize) !void {
        if (rows == 0) return;
        if (rank > 1) return error.DcpShape;
        var a: cuda.Args = .{};
        inline for (.{ partial_local, partial_peer, out, lse_out, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(rank)) }) |v| a.add(v);
        try cuda.launch.launch(m.merge_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(local_heads) }, .block = .{ .x = 128 } }, stream, &a);
    }

    pub fn expandV(m: *const Mla, stream: cuda.Stream, ol: u64, kvb: u64, scales: u64, out: u64, rows: usize, local_heads: usize, global_head0: usize, decode_mla: bool) !void {
        if (rows == 0) return;
        if (decode_mla and rows > max_decode_rows) return error.DecodeMlaRows;
        var a: cuda.Args = .{};
        inline for (.{ ol, kvb, scales, out, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        try cuda.launch.launch(if (decode_mla) m.expand_decode_fn else m.expand_fn, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(local_heads), .z = if (decode_mla) @intCast(value_dim / 32) else 1 }, .block = .{ .x = 128 } }, stream, &a);
    }

    pub fn expandVV2(m: *const Mla, stream: cuda.Stream, ol: u64, kvb: u64, scales: u64, out: u64, rows: usize, local_heads: usize, global_head0: usize) !void {
        if (rows == 0) return;
        if (rows > max_decode_rows) return error.DecodeMlaRows;
        var a: cuda.Args = .{};
        inline for (.{ ol, kvb, scales, out, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        try cuda.launch.launch(m.expand_decode_v2_fn, .{ .grid = .{ .x = 1, .y = @intCast(local_heads), .z = @intCast(value_dim / decode_mla_v2_threads) }, .block = .{ .x = decode_mla_v2_threads } }, stream, &a);
    }

    pub fn expandVPrompt(m: *const Mla, stream: cuda.Stream, ol: u64, kvb: u64, scales: u64, out: u64, rows: usize, local_heads: usize, global_head0: usize, mode: PromptMlaMode) !void {
        if (rows == 0) return;
        if (mode == .off) return error.PromptMlaDisabled;
        var a: cuda.Args = .{};
        inline for (.{ ol, kvb, scales, out, @as(i32, @intCast(rows)), @as(i32, @intCast(local_heads)), @as(i32, @intCast(global_head0)) }) |v| a.add(v);
        if (mode == .tile16) {
            try cuda.launch.launch(m.expand_prompt_fn, .{ .grid = .{ .x = @intCast(value_dim / 8), .y = @intCast(local_heads), .z = @intCast((rows + prompt_mla_rows - 1) / prompt_mla_rows) }, .block = .{ .x = 32 }, .shared = prompt_mla_smem }, stream, &a);
        } else {
            try cuda.launch.launch(m.expand_prompt_big_fn, .{ .grid = .{ .x = @intCast(value_dim / 64), .y = @intCast(local_heads), .z = @intCast((rows + prompt_mla_big_rows - 1) / prompt_mla_big_rows) }, .block = .{ .x = prompt_mla_big_threads }, .shared = prompt_mla_big_smem }, stream, &a);
        }
    }
};

test "MLA launch plans stay under the sm_121 shared-memory budget" {
    try std.testing.expectEqual(@as(u32, 0), (try attendPlan(64, 64, false, .scalar)).dynamic_smem);
    try std.testing.expectEqual(@as(u32, 0), (try attendPlan(64, 64, true, .scalar)).dynamic_smem);
    try std.testing.expectEqual(@as(u32, 64), (try attendPlan(64, 64, false, .scalar)).grid[1]);
    try std.testing.expectEqual(@as(u32, 2), (try attendPlan(64, 64, true, .mma)).grid[1]);
    try std.testing.expect((try attendPlan(64, 64, true, .mma)).dynamic_smem <= 99 * 1024);
    try std.testing.expectError(error.AttendShape, attendPlan(1, 31, false, .mma));
    try std.testing.expectEqual(@as(u32, 97344), smla_p_smem); // mla.cu static_assert(SMLA_SMEM == 97344)
    try std.testing.expect(smla_p_smem <= 99 * 1024);
    try std.testing.expectEqual([3]u32{ 4, 1, 1 }, (try smlaPPlan(7, 32)).grid);
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, (try smlaPPlan(1, 32)).grid);
    try std.testing.expectEqual(@as(u32, 512), (try smlaPPlan(64, 32)).block);
    try std.testing.expectError(error.AttendShape, smlaPPlan(4, 31));
    try std.testing.expectEqual(@as(u32, 101184), smla_fp8_smem); // mla.cu static_assert(SMF_SMEM == 101184)
    try std.testing.expect(smla_fp8_smem <= 99 * 1024);
    try std.testing.expectEqual((try smlaPPlan(7, 64)).grid, (try smlaPFp8Plan(7, 64)).grid);
    try std.testing.expectEqual(@as(u32, 512), (try smlaPFp8Plan(64, 32)).block);
    try std.testing.expectEqual(smla_fp8_smem, (try smlaPFp8Plan(64, 32)).dynamic_smem);
    try std.testing.expectError(error.AttendShape, smlaPFp8Plan(4, 31));
    try std.testing.expectEqual(@as(u32, 96704), smla_p3_smem); // mla.cu static_assert(SMLA3_SMEM == 96704)
    try std.testing.expect(smla_p3_smem <= 99 * 1024);
    try std.testing.expectEqual((try smlaPPlan(7, 32)).grid, (try smlaP3Plan(7, 32)).grid);
    try std.testing.expectEqual(@as(u32, 512), (try smlaP3Plan(513, 32)).block);
    try std.testing.expectEqual(@as(u32, 257), (try smlaP3Plan(513, 32)).grid[0]);
    try std.testing.expectError(error.AttendShape, smlaP3Plan(4, 31));
    try std.testing.expectEqual(@as(u64, 128), smla_p3_steps);
    try std.testing.expectEqual(@as(u64, 0), smla_p3_record_bytes % 16); // cp.async pieces
    try std.testing.expectEqual(@as(u64, 256 * 128 * 176), smlaP3TableBytes(512)); // 5.77 MB a 512-row piece
    try std.testing.expectEqual(@as(u64, 257 * 128 * 176), smlaP3TableBytes(513));
    try std.testing.expectEqual(@as(u64, 128 * 176), smlaP3TableBytes(1));
    try std.testing.expectEqual(@as(u64, 0), smlaP3TableBytes(0));
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, smlaUnionPlan(512).grid);
    try std.testing.expectEqual([3]u32{ 65, 1, 1 }, smlaUnionPlan(513).grid); // 257 pairs, 4 a block
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, smlaUnionPlan(1).grid);
    try std.testing.expectEqual(@as(u32, 128), smlaUnionPlan(7).block);
    for (0..64) |head| {
        try std.testing.expectEqual(@as(?usize, head), headForBlock(head, 64));
    }
    try std.testing.expectEqual(@as(?usize, null), headForBlock(64, 64));
    try std.testing.expectEqual(@as(usize, 1024), try denseLocalCount(2047, 1, 2));
    try std.testing.expect(isDense(2047));
    try std.testing.expect(!isDense(2048));
    try std.testing.expectEqual(@as(?usize, 2), try localSlot(5, 1, 2));
    try std.testing.expectEqual(@as(?usize, null), try localSlot(5, 0, 2));
    try std.testing.expectEqual(@as(usize, 5), try globalPosition(2, 1, 2));
    try std.testing.expect(useMaskedMha(5119, prefill_masked_mha_switch_default));
    try std.testing.expect(!useMaskedMha(5120, prefill_masked_mha_switch_default));
}

test "kv write row and slot addressing is applied exactly once for decode and prefill batches" {
    const stride = latent_dim + rope_dim;
    inline for (.{ 1, 3, 8, 64 }) |rows| {
        for (0..rows) |r| {
            const slot = r * 3 + 1;
            const source_element = kvWriteSourceOffset(r, stride) + latent_dim - 1;
            const destination_element = kvWriteDestinationOffset(slot, latent_dim);
            try std.testing.expectEqual(r * stride + latent_dim - 1, source_element);
            try std.testing.expectEqual((r * 3 + 1) * latent_dim, destination_element);
        }
    }
}

test "decode skips peer cache rows when prefill staging is absent" {
    try std.testing.expect(!kvWriteMaySkip(1, 1, false));
    try std.testing.expect(kvWriteMaySkip(0, 1, false));
    try std.testing.expect(!kvWriteMaySkip(0, 1, true));
}

test "dense prefill rank-local key spans merge to every global key" {
    for (1984..2048) |position| {
        const a = try denseLocalCount(position, 0, 2);
        const b = try denseLocalCount(position, 1, 2);
        for (0..a) |i| try std.testing.expect(try denseLocalPosition(0, 2, i) <= position);
        for (0..b) |i| try std.testing.expect(try denseLocalPosition(1, 2, i) <= position);
        try std.testing.expectEqual(position + 1, a + b);
    }
}

fn attendDensePrefillKernelRef(qa: []const f32, qpe: []const f32, own_latent: []const f32, own_rope: []const f32, stage_latent: []const f32, stage_rope: []const f32, query_position: usize, rank: usize, world: usize, out: []f64) !void {
    if (world != 2 or rank >= world or query_position >= 2048 or out.len < latent_dim) return error.AttendShape;
    var maximum: f64 = -std.math.inf(f64);
    var denom: f64 = 0;
    @memset(out[0..latent_dim], 0);
    for (0..query_position + 1) |j| {
        const slot = try stagePlaneSlot(j, world);
        const owned = j % world == rank;
        const kl = if (owned) own_latent else stage_latent;
        const kr = if (owned) own_rope else stage_rope;
        var score: f64 = 0;
        for (0..latent_dim) |k| score += @as(f64, bf16(qa[k])) * bf16(kl[slot * latent_dim + k]);
        for (0..rope_dim) |k| score += @as(f64, bf16(qpe[k])) * bf16(kr[slot * rope_dim + k]);
        score *= 1.0 / 16.0;
        const next = @max(maximum, score);
        const alpha: f64 = if (std.math.isInf(maximum)) 0 else @exp(maximum - next);
        const p = @exp(score - next);
        denom = denom * alpha + p;
        for (0..latent_dim) |k| out[k] = out[k] * alpha + p * bf16(kl[slot * latent_dim + k]);
        maximum = next;
    }
    for (out[0..latent_dim]) |*v| v.* /= denom;
}

test "gather-on-prefill dense MLA follows kernel owner and staging indices through position 2047" {
    const allocator = std.testing.allocator;
    const rows = 1024;
    const own0 = try allocator.alloc(f32, rows * latent_dim);
    defer allocator.free(own0);
    const own1 = try allocator.alloc(f32, rows * latent_dim);
    defer allocator.free(own1);
    const stage0 = try allocator.alloc(f32, rows * latent_dim);
    defer allocator.free(stage0);
    const stage1 = try allocator.alloc(f32, rows * latent_dim);
    defer allocator.free(stage1);
    const rope0 = try allocator.alloc(f32, rows * rope_dim);
    defer allocator.free(rope0);
    const rope1 = try allocator.alloc(f32, rows * rope_dim);
    defer allocator.free(rope1);
    const stage_rope0 = try allocator.alloc(f32, rows * rope_dim);
    defer allocator.free(stage_rope0);
    const stage_rope1 = try allocator.alloc(f32, rows * rope_dim);
    defer allocator.free(stage_rope1);
    for (0..rows) |s| for (0..latent_dim) |k| {
        const x = @as(f32, @floatFromInt(@as(isize, @intCast((s * 17 + k * 3) % 97)) - 48)) / 32;
        own0[s * latent_dim + k] = bf16(x);
        own1[s * latent_dim + k] = bf16(-x / 2);
        stage0[s * latent_dim + k] = own1[s * latent_dim + k];
        stage1[s * latent_dim + k] = own0[s * latent_dim + k];
    };
    for (0..rows) |s| for (0..rope_dim) |k| {
        const x = @as(f32, @floatFromInt(@as(isize, @intCast((s * 7 + k) % 31)) - 15)) / 16;
        rope0[s * rope_dim + k] = bf16(x);
        rope1[s * rope_dim + k] = bf16(-x / 2);
        stage_rope0[s * rope_dim + k] = rope1[s * rope_dim + k];
        stage_rope1[s * rope_dim + k] = rope0[s * rope_dim + k];
    };
    var qa: [latent_dim]f32 = undefined;
    var qpe: [rope_dim]f32 = undefined;
    for (&qa, 0..) |*v, k| v.* = bf16(@as(f32, @floatFromInt(@as(isize, @intCast(k % 23)) - 11)) / 8);
    for (&qpe, 0..) |*v, k| v.* = bf16(@as(f32, @floatFromInt(@as(isize, @intCast(k % 13)) - 6)) / 8);
    var actual: [latent_dim]f64 = undefined;
    var expected: [latent_dim]f64 = undefined;
    // 1024 models a prompt continuation: stage contains fetched peer history below
    // chunk_start as well as peer rows written by the current chunk.
    const query_positions = [_]usize{ 1024, 2047 };
    for (query_positions) |query_position| {
        for (0..2) |rank| {
            try attendDensePrefillKernelRef(&qa, &qpe, if (rank == 0) own0 else own1, if (rank == 0) rope0 else rope1, if (rank == 0) stage0 else stage1, if (rank == 0) stage_rope0 else stage_rope1, query_position, rank, 2, &actual);
            var max_score: f64 = -std.math.inf(f64);
            for (0..query_position + 1) |j| {
                const slot = try stagePlaneSlot(j, 2);
                const kl = if (j % 2 == 0) own0 else own1;
                const kr = if (j % 2 == 0) rope0 else rope1;
                var score: f64 = 0;
                for (0..latent_dim) |k| score += @as(f64, qa[k]) * kl[slot * latent_dim + k];
                for (0..rope_dim) |k| score += @as(f64, qpe[k]) * kr[slot * rope_dim + k];
                max_score = @max(max_score, score / 16);
            }
            var den: f64 = 0;
            @memset(&expected, 0);
            for (0..query_position + 1) |j| {
                const slot = try stagePlaneSlot(j, 2);
                const kl = if (j % 2 == 0) own0 else own1;
                const kr = if (j % 2 == 0) rope0 else rope1;
                var score: f64 = 0;
                for (0..latent_dim) |k| score += @as(f64, qa[k]) * kl[slot * latent_dim + k];
                for (0..rope_dim) |k| score += @as(f64, qpe[k]) * kr[slot * rope_dim + k];
                const p = @exp(score / 16 - max_score);
                den += p;
                for (0..latent_dim) |k| expected[k] += p * kl[slot * latent_dim + k];
            }
            for (0..latent_dim) |k| try std.testing.expectApproxEqAbs(expected[k] / den, actual[k], 1e-12);
        }
    }
}

test "prefill attention rejects either missing staging plane" {
    try std.testing.expectError(error.MissingPrefillStage, validateAttendStage(2, true, 1, 0));
    try std.testing.expectError(error.MissingPrefillStage, validateAttendStage(2, true, 0, 1));
    try validateAttendStage(2, true, 1, 1);
    try validateAttendStage(2, false, 0, 0);
    try validateAttendStage(1, true, 0, 0);
}

test "prefill kv write requires paired stage planes and attend limits local head output layout" {
    try std.testing.expectError(error.MissingPrefillStage, validateKvWriteStage(2, true, 0, 0));
    try std.testing.expectError(error.InvalidStagePair, validateKvWriteStage(2, false, 1, 0));
    try validateKvWriteStage(2, false, 0, 0);
    try validateKvWriteStage(1, true, 0, 0);
    try std.testing.expectError(error.PrefillHeadLimit, validateAttendHeads(true, 33));
    try validateAttendHeads(true, 32);
    try validateAttendHeads(false, 64);
}

test "MLA launcher stage pair selects the buffers owned by the shared stage contract" {
    const pair: types.StagePlanePair = .{
        .latent = .{ .data = .{ .d = undefined, .ptr = 0x1234, .len = 1 }, .dtype = .bf16, .rows = 1, .cols = latent_dim, .stride = latent_dim },
        .rope = .{ .data = .{ .d = undefined, .ptr = 0x5678, .len = 1 }, .dtype = .bf16, .rows = 1, .cols = rope_dim, .stride = rope_dim },
    };
    const pointers = stagePointers(pair);
    try std.testing.expectEqual(@as(u64, 0x1234), pointers.latent);
    try std.testing.expectEqual(@as(u64, 0x5678), pointers.rope);
    const decode = stagePointers(null);
    try std.testing.expectEqual(@as(u64, 0), decode.latent);
    try std.testing.expectEqual(@as(u64, 0), decode.rope);
}

test "DCP decode query and partial addresses preserve group and row for R 1 3 and 8" {
    inline for (.{ 1, 3, 8 }) |rows| {
        var seen: [2 * 8 * 32]bool = @splat(false);
        for (0..2) |g| for (0..rows) |r| for (0..32) |h| {
            const at = groupRowHead(g, r, h, rows);
            try std.testing.expect(!seen[at]);
            seen[at] = true;
            try std.testing.expectEqual(g * rows * 32 + r * 32 + h, at);
        };
    }
}

test "RoPE tables are indexed by row and arithmetic rounds products before addition" {
    const pos = [_]usize{ 1200, 8800, 16000 };
    var q: [3 * 256]f32 = @splat(0);
    for (0..3) |r| {
        q[r * 256 + nope_dim] = @as(f32, @floatFromInt(r + 1));
        q[r * 256 + nope_dim + 1] = 0.5;
    }
    var cos: [3 * 32]f32 = @splat(0);
    var sin: [3 * 32]f32 = @splat(0);
    for (0..3) |r| {
        cos[r * 32] = @as(f32, @floatFromInt(r + 1)) / 4;
        sin[r * 32] = 0.25;
    }
    var out: [3 * 64]f32 = undefined;
    try ropeQueryRef(&q, 3, 1, &pos, &cos, &sin, &out);
    for (0..3) |r| {
        const pair = ropePair(@as(f32, @floatFromInt(r + 1)), 0.5, cos[r * 32], sin[r * 32]);
        try std.testing.expectEqual(pair[0], out[r * 64]);
        try std.testing.expectEqual(pair[1], out[r * 64 + 1]);
    }
}

test "kv write and attention are bit-identical across absolute chunk boundaries" {
    const n = 17;
    var proj: [17 * (latent_dim + rope_dim)]f32 = @splat(0);
    const weight: [latent_dim]f32 = @splat(1);
    const cos: [17 * rope_dim / 2]f32 = @splat(1);
    const sin: [17 * rope_dim / 2]f32 = @splat(0);
    for (&proj, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast(i % 31)) - 15)) / 16;
    const pos = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    var lat_a: [17 * latent_dim]f32 = @splat(0);
    var rope_a: [17 * rope_dim]f32 = @splat(0);
    var lat_b: [17 * latent_dim]f32 = @splat(0);
    var rope_b: [17 * rope_dim]f32 = @splat(0);
    try kvWriteRef(&proj, n, latent_dim + rope_dim, &weight, &pos, &cos, &sin, &lat_a, &rope_a);
    try kvWriteRef(proj[0 .. 5 * (latent_dim + rope_dim)], 5, latent_dim + rope_dim, &weight, pos[0..5], &cos, &sin, lat_b[0 .. 5 * latent_dim], rope_b[0 .. 5 * rope_dim]);
    try kvWriteRef(proj[5 * (latent_dim + rope_dim) .. 12 * (latent_dim + rope_dim)], 7, latent_dim + rope_dim, &weight, pos[5..12], &cos, &sin, lat_b[5 * latent_dim .. 12 * latent_dim], rope_b[5 * rope_dim .. 12 * rope_dim]);
    try kvWriteRef(proj[12 * (latent_dim + rope_dim) ..], 5, latent_dim + rope_dim, &weight, pos[12..], &cos, &sin, lat_b[12 * latent_dim ..], rope_b[12 * rope_dim ..]);
    try std.testing.expectEqualSlices(f32, &lat_a, &lat_b);
    try std.testing.expectEqualSlices(f32, &rope_a, &rope_b);

    var q: [latent_dim]f32 = @splat(0);
    var qp: [rope_dim]f32 = @splat(0);
    for (&q, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 32;
    for (&qp, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 7)) / 8;
    var out_whole: [latent_dim]f32 = @splat(0);
    try attendCudaRef(&q, &qp, &lat_a, &rope_a, &pos, 16, &out_whole);
    var out_chunked: [latent_dim]f32 = @splat(0);
    try attendCudaRef(&q, &qp, &lat_b, &rope_b, &pos, 16, &out_chunked);
    try std.testing.expectEqualSlices(f32, &out_whole, &out_chunked);
}

test "all MLA declarations analyze recursively" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(Online);
    std.testing.refAllDecls(Mla);
}

test "FP8 kv_b single-copy lookup applies native scale before BF16 rounding" {
    const codes: [128]u8 = @splat(0x38);
    const scales = [_]f32{0.5};
    try std.testing.expectEqual(@as(f32, 0.5), try kvbValue(&codes, &scales, 0, 0, 128));
    try std.testing.expectEqual(@as(f32, 1), e4m3(0x38));
}

test "absorb consumes q_b's 256-wide per-head layout and ropes q_pe" {
    const q: [256]f32 = @splat(1);
    const codes: [448 * 512]u8 = @splat(0x38);
    const scales: [16]f32 = @splat(1);
    var qa: [512]f32 = undefined;
    try absorbRef(&q, 1, 1, &codes, &scales, 512, &qa);
    for (qa) |v| try std.testing.expectEqual(@as(f32, 192), v);

    var cos: [32]f32 = @splat(0);
    var sin: [32]f32 = @splat(0);
    cos[0] = 0;
    sin[0] = 1;
    const positions = [_]usize{0};
    var qpe: [64]f32 = undefined;
    try ropeQueryRef(&q, 1, 1, &positions, &cos, &sin, &qpe);
    try std.testing.expectEqual(@as(f32, -1), qpe[0]);
    try std.testing.expectEqual(@as(f32, 1), qpe[1]);
}

test "DCP merge uses fixed rank order and returns logsumexp" {
    var out: [2]f32 = undefined;
    const o0 = [_]f32{ 1, 2 };
    const o1 = [_]f32{ 3, 4 };
    const lse = try mergeRef(.{ 0, 0 }, .{ &o0, &o1 }, &out);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, @log(@as(f32, 2))), lse, 1e-6);
}

test "DCP peer-only partial exchange preserves rank-order output and payload size" {
    try std.testing.expectEqual(@as(usize, 32896), partial_bytes_per_head * 32);
    try std.testing.expectEqual(@as(usize, 32896), try peerPartialBytes(1, 32));
    const peer_bytes = try peerPartialBytes(1, 32);
    try std.testing.expectEqual(@as(usize, 65792), peer_bytes * 2);
    try std.testing.expectEqual(@as(usize, 2565888), (peer_bytes * 2 - peer_bytes) * 78);
    var rank0: [2]f32 = undefined;
    var rank1: [2]f32 = undefined;
    const low = [_]f32{ 1, 2 };
    const high = [_]f32{ 3, 4 };
    _ = try mergePeerRef(0, 0, 0, &low, &high, &rank0);
    _ = try mergePeerRef(1, 0, 0, &high, &low, &rank1);
    try std.testing.expectEqualSlices(u32, @as(*const [2]u32, @ptrCast(&rank0)), @as(*const [2]u32, @ptrCast(&rank1)));
    try std.testing.expectEqualSlices(f32, &.{ 2, 3 }, &rank0);
    try std.testing.expectError(error.DcpShape, mergePeerRef(2, 0, 0, &low, &high, &rank0));
}

test "packed DCP partial and LSE offsets form one peer gather block" {
    try std.testing.expectEqual(@as(usize, 32896), try partialBlockBytes(1, 32));
    try std.testing.expectEqual(@as(usize, 32768), try partialLseByteOffset(1, 32));
    try std.testing.expectEqual(@as(usize, 32896), try partialGroupByteOffset(1, 1, 32));
    try std.testing.expectEqual(@as(usize, 65792), try partialBlockBytes(2, 32));
    try std.testing.expectEqual(@as(usize, 65536), try partialLseByteOffset(2, 32));
    try std.testing.expectEqual(@as(usize, 65792), try partialGroupByteOffset(2, 1, 32));
}

test "DCP merge reference covers unequal, separated, empty, infinite and NaN LSEs" {
    var out: [1]f32 = undefined;
    const one = [_]f32{1};
    const zero = [_]f32{0};
    _ = try mergeRef(.{ 0, -1 }, .{ &one, &zero }, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), out[0], 0.001);
    _ = try mergeRef(.{ 0, -4 }, .{ &one, &zero }, &out);
    try std.testing.expectEqual(bf16(0.9820138), out[0]);
    try std.testing.expectError(error.EmptyAttention, mergeRef(.{ -std.math.inf(f32), -std.math.inf(f32) }, .{ &one, &one }, &out));
    _ = try mergeRef(.{ std.math.inf(f32), 0 }, .{ &one, &one }, &out);
    try std.testing.expect(std.math.isNan(out[0]));
    _ = try mergeRef(.{ std.math.nan(f32), 0 }, .{ &one, &one }, &out);
    try std.testing.expect(std.math.isNan(out[0]));
}

test "DCP merge treats only negative infinity LSE as empty" {
    var out: [1]f32 = undefined;
    const one = [_]f32{1};
    _ = try mergePeerRef(0, std.math.inf(f32), 0, &one, &one, &out);
    try std.testing.expect(std.math.isNan(out[0]));
}

test "MLA declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(Online);
    std.testing.refAllDecls(Mla);
}

test "tensor-core attend host order stays within tolerance of the fp64 reference" {
    const n = 64;
    var qa: [latent_dim]f32 = undefined;
    var qpe: [rope_dim]f32 = undefined;
    var latent: [n * latent_dim]f32 = undefined;
    var rope: [n * rope_dim]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(0x6d6d61);
    const rnd = prng.random();
    for (&qa) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (&qpe) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (&latent) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    for (&rope) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    var keys: [n]usize = undefined;
    for (&keys, 0..) |*k, i| k.* = i * 3;
    var want: [latent_dim]f32 = undefined;
    var got: [latent_dim]f32 = undefined;
    var cuda_order: [latent_dim]f32 = undefined;
    try attendRef(std.testing.allocator, &qa, &qpe, &latent, &rope, &keys, 1000, &want);
    try attendMmaRef(&qa, &qpe, &latent, &rope, &keys, 1000, &got);
    try attendCudaRef(&qa, &qpe, &latent, &rope, &keys, 1000, &cuda_order);
    var scale: f32 = 0;
    for (want) |v| scale = @max(scale, @abs(v));
    for (want, got, cuda_order) |w, g, c| {
        try std.testing.expect(@abs(w - g) <= 2.0 / 128.0 * scale + 1e-6); // 2 bf16 ulps: P rounds against a tile max, not a running one
        try std.testing.expect(@abs(c - g) <= 2.0 / 128.0 * scale + 1e-6);
    }
}
