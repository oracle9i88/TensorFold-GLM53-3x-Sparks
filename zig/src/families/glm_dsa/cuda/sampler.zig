// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! GLM-5.3 fp32 candidate sampling and rank-local logsumexp merging (K9).
//! Adapted from ashhart/TensorFold@d1018b31:zig/kernels/cuda/sample.cu (Apache-2.0).
//! Sampling order and keyed doubles follow bertholomus/bertholomus/TensorFold@ddfad35:src/tensorfold/engine/exact_sampling.py and glm5_next/cuda/decode.py (Apache-2.0).
//! Pairwise reduction order follows NumPy 2.0.2/2.5.4: loops_utils.h.src and NPY_BUFSIZE in ndarraytypes.h (BSD-3-Clause).
//! Changes: fp32 GLM head candidates, TP merge, and Python-compatible keyed sampling.
const std = @import("std");
const cuda = @import("cuda");

pub const Candidate = struct { value: f32, id: u32 };
pub const Lse = struct { max: f32, sum: f32 };
pub const Sampling = struct { temperature: f64, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) };
pub const GateRow = struct { logp: []f32, mass_outside: f32, lse: f32, argmax: u32 };
pub const RowWindow = struct { start: usize, count: usize };

/// Maximum row count sent to one candidate-kernel launch.
pub const candidate_rows_per_launch = 64;

/// Validate candidate launch tiling and return the maximum rows covered by the supplied scratch.
pub fn candidateLaunchRows(rows: usize, k: usize, scratch_bytes: usize) !usize {
    if (rows == 0) return 0;
    const tile_rows = @min(rows, candidate_rows_per_launch);
    if (scratch_bytes < try Head.candidateScratchBytes(tile_rows, k)) return error.CandidateWorkspaceTooSmall;
    return tile_rows;
}

/// Head rows needed by dump mode. Gate capture always evaluates every prefill row.
pub fn headWindow(prefill_rows: usize, gate_mode: bool) RowWindow {
    if (gate_mode or prefill_rows == 0) return .{ .start = 0, .count = prefill_rows };
    return .{ .start = prefill_rows - 1, .count = 1 };
}

/// Layer 77 prefill pruning follows the head's requested row window; decode and verify retain every row.
pub fn layer77Window(rows: usize, gate_mode: bool, prefill: bool) RowWindow {
    return if (prefill) headWindow(rows, gate_mode) else .{ .start = 0, .count = rows };
}

/// Gather 64 probe logits across rank shards and compute Python gate fields from merged full-vocabulary LSE.
pub fn gateRow(allocator: std.mem.Allocator, rank_values: []const []const f32, rank_offsets: []const usize, ids: []const u32) !GateRow {
    if (ids.len != 64 or rank_values.len != rank_offsets.len or rank_values.len == 0) return error.InvalidGateShape;
    const logp = try allocator.alloc(f32, ids.len);
    var lse_parts = try allocator.alloc(Lse, rank_values.len);
    defer allocator.free(lse_parts);
    var argmax: u32 = 0xffffffff;
    var maxv: f32 = -std.math.inf(f32);
    for (rank_values, rank_offsets, 0..) |values, offset, r| {
        lse_parts[r] = partialLse(values);
        for (values, 0..) |v, i| if (!std.math.isNan(v) and (v > maxv or (v == maxv and offset + i < argmax))) {
            maxv = v;
            argmax = @intCast(offset + i);
        };
    }
    const total_lse = mergeLse(lse_parts);
    var mass_sum: f32 = 0;
    for (ids, 0..) |id, j| {
        var value: f32 = 0;
        var found = false;
        for (rank_values, rank_offsets) |values, offset| if (id >= offset and id - offset < values.len) {
            value = values[id - offset];
            found = true;
            break;
        };
        if (!found) return error.ProbeIdOutOfRange;
        logp[j] = value - total_lse;
        mass_sum += @exp(logp[j]);
    }
    return .{ .logp = logp, .mass_outside = @max(0, 1 - mass_sum), .lse = total_lse, .argmax = argmax };
}

/// fp32 head kernels, loaded from `glm53_head` by the CUDA forward once per engine.
/// Candidate selection shards each row over 32 CTAs, each retaining up to k exact keys.
/// At k=2048 scratch is 512 KiB/row. Merge stages searched lists in shared memory
/// and exits each candidate's search at rank k. Based on the review's estimated
/// 3-5 searched lists per candidate, 65,536 candidate keys need about 2.16-3.60M
/// shared-memory compare steps per row, versus 22.35M in the prior 31-list search.
/// Staging each list once per source CTA is 16 MiB of coalesced reads per row,
/// versus about 179 MB of logical dependent global reads before. Estimated
/// merge latency is 10-30 us/row, unmeasured on sm_121.
pub const Head = struct {
    top1_fn: cuda.Function,
    mask_fn: cuda.Function,
    candidates_fill_fn: cuda.Function,
    candidates_part_fn: cuda.Function,
    candidates_merge_fn: cuda.Function,
    lse_fn: cuda.Function,
    gate_gather_fn: cuda.Function,
    gate_mass_fn: cuda.Function,

    pub fn resolve(mod: cuda.Module) !Head {
        return .{
            .top1_fn = try mod.function("tf_glm53_top1_f32"),
            .mask_fn = try mod.function("tf_glm53_mask_f32"),
            .candidates_fill_fn = try mod.function("tf_glm53_cand_fill_f32"),
            .candidates_part_fn = try mod.function("tf_glm53_cand_part_f32"),
            .candidates_merge_fn = try mod.function("tf_glm53_cand_merge_f32"),
            .lse_fn = try mod.function("tf_glm53_lse_f32"),
            .gate_gather_fn = try mod.function("tf_glm53_gate_gather_f32"),
            .gate_mass_fn = try mod.function("tf_glm53_gate_mass_f32"),
        };
    }

    pub fn mask(h: *const Head, stream: cuda.Stream, logits: u64, vocab: usize, vocab_offset: usize, bits: u64, words: u32, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(vocab_offset)));
        a.add(bits);
        a.add(words);
        try cuda.launch.launch(h.mask_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    pub fn top1(h: *const Head, stream: cuda.Stream, logits: u64, vocab: usize, vocab_offset: usize, values: u64, ids: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(vocab_offset)));
        a.add(values);
        a.add(ids);
        try cuda.launch.launch(h.top1_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// Workspace for 32 independently selected lists of at most k 64-bit keys per row.
    pub fn candidateScratchBytes(rows: usize, k: usize) !usize {
        if (k == 0 or k > 2048) return error.InvalidCandidateCount;
        const per_row = try std.math.mul(usize, 32 * k, @sizeOf(u64));
        return std.math.mul(usize, rows, per_row) catch error.CandidateWorkspaceOverflow;
    }

    /// Rank-local top-k rows, in value-descending/id-ascending order. Scratch is reused per 64-row tile.
    pub fn candidates(h: *const Head, stream: cuda.Stream, logits: u64, vocab: usize, vocab_offset: usize, k: usize, values: u64, ids: u64, scratch: u64, scratch_bytes: usize, rows: usize) !void {
        if (rows == 0) return;
        if (k == 0 or k > 2048 or vocab == 0 or scratch == 0) return error.InvalidCandidateCount;
        _ = try candidateLaunchRows(rows, k, scratch_bytes);
        var row_start: usize = 0;
        while (row_start < rows) {
            const tile_rows = @min(candidate_rows_per_launch, rows - row_start);
            const logits_tile = try offsetDevicePtr(logits, row_start, vocab, @sizeOf(f32));
            const values_tile = try offsetDevicePtr(values, row_start, k, @sizeOf(f32));
            const ids_tile = try offsetDevicePtr(ids, row_start, k, @sizeOf(u32));
            var fill_args: cuda.Args = .{};
            fill_args.add(values_tile);
            fill_args.add(ids_tile);
            fill_args.add(@as(u32, @intCast(tile_rows * k)));
            try cuda.launch.launch(h.candidates_fill_fn, .{ .grid = .{ .x = @intCast(try std.math.divCeil(usize, tile_rows * k, 256)) }, .block = .{ .x = 256 } }, stream, &fill_args);
            var a: cuda.Args = .{};
            a.add(logits_tile);
            a.add(@as(u32, @intCast(vocab)));
            a.add(@as(u32, @intCast(vocab_offset)));
            a.add(@as(u32, @intCast(k)));
            a.add(scratch);
            try cuda.launch.launch(h.candidates_part_fn, .{ .grid = .{ .x = @intCast(tile_rows), .y = 32 }, .block = .{ .x = 256 }, .shared = 19_484 }, stream, &a);
            var merge_args: cuda.Args = .{};
            merge_args.add(scratch);
            merge_args.add(@as(u32, @intCast(vocab)));
            merge_args.add(@as(u32, @intCast(k)));
            merge_args.add(values_tile);
            merge_args.add(ids_tile);
            try cuda.launch.launch(h.candidates_merge_fn, .{ .grid = .{ .x = @intCast(tile_rows), .y = 32 }, .block = .{ .x = 256 }, .shared = 16_384 }, stream, &merge_args);
            row_start += tile_rows;
        }
    }

    /// Stable local logsumexp partials `[max,sum]` for mergeLse.
    pub fn lse(h: *const Head, stream: cuda.Stream, logits: u64, vocab: usize, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(u32, @intCast(vocab)));
        a.add(out);
        try cuda.launch.launch(h.lse_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// Gather each row's probe logits from this rank's vocab shard; missing ids write zero for the TP sum.
    pub fn gateGather(h: *const Head, stream: cuda.Stream, logits: u64, ids: u64, vocab: usize, offset: usize, gathered: u64, probes: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(ids);
        a.add(@as(u32, @intCast(vocab)));
        a.add(@as(u32, @intCast(offset)));
        a.add(gathered);
        a.add(@as(u32, @intCast(probes)));
        a.add(@as(u32, @intCast(rows)));
        const items = try std.math.mul(usize, rows, probes);
        const blocks = try std.math.divCeil(usize, items, 256);
        try cuda.launch.launch(h.gate_gather_fn, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, stream, &a);
    }

    /// Convert gathered global probe logits and merged LSE into Python gate `logp` and `mass_outside` fields.
    pub fn gateMass(h: *const Head, stream: cuda.Stream, probe_logits: u64, lse_ptr: u64, logp: u64, mass_outside: u64, probes: usize, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(probe_logits);
        a.add(lse_ptr);
        a.add(logp);
        a.add(mass_outside);
        a.add(@as(u32, @intCast(probes)));
        a.add(@as(u32, @intCast(rows)));
        try cuda.launch.launch(h.gate_mass_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1 } }, stream, &a);
    }
};

fn offsetDevicePtr(ptr: u64, row_start: usize, width: usize, element_size: usize) !u64 {
    const elements = std.math.mul(usize, row_start, width) catch return error.CandidateWorkspaceOverflow;
    const bytes = std.math.mul(usize, elements, element_size) catch return error.CandidateWorkspaceOverflow;
    return std.math.add(u64, ptr, @intCast(bytes)) catch error.CandidateWorkspaceOverflow;
}

/// Sort a rank's local candidates into global Python ordering: value descending, id ascending.
pub fn less(_: void, a: Candidate, b: Candidate) bool {
    if (a.value != b.value) return a.value > b.value;
    return a.id < b.id;
}

/// Host oracle for tf_glm53_top1_f32; NaNs never win and ties select the lower global token id.
pub fn top1Ref(values: []const f32, offset: u32) Candidate {
    var best = Candidate{ .value = -std.math.inf(f32), .id = 0xffffffff };
    for (values, 0..) |value, i| {
        const id = offset + @as(u32, @intCast(i));
        if (!std.math.isNan(value) and (best.id == 0xffffffff or value > best.value or (value == best.value and id < best.id))) best = .{ .value = value, .id = id };
    }
    return best;
}

/// Host oracle for the CUDA local candidate kernel, including global-id tie ordering.
pub fn topCandidatesRef(allocator: std.mem.Allocator, values: []const f32, offset: u32, k: usize) ![]Candidate {
    var all: std.ArrayList(Candidate) = .empty;
    defer all.deinit(allocator);
    for (values, 0..) |value, i| {
        if (!std.math.isNan(value)) try all.append(allocator, .{ .value = value, .id = offset + @as(u32, @intCast(i)) });
    }
    std.mem.sort(Candidate, all.items, {}, less);
    return try allocator.dupe(Candidate, all.items[0..@min(k, all.items.len)]);
}

fn sortableKey(c: Candidate) u64 {
    var bits: u32 = @bitCast(c.value);
    if ((bits & 0x7fffffff) == 0) bits = 0;
    const ordered = if ((bits & 0x80000000) != 0) ~bits else bits ^ 0x80000000;
    return (@as(u64, ~ordered) << 32) | c.id;
}

/// Host emulation of the CUDA eight-byte radix selection, used to validate tie and large-k behavior.
pub fn topCandidatesRadixRef(allocator: std.mem.Allocator, values: []const f32, offset: u32, k: usize) ![]Candidate {
    const n = @min(k, 2048);
    if (n == 0) return try allocator.alloc(Candidate, 0);
    var all: std.ArrayList(Candidate) = .empty;
    defer all.deinit(allocator);
    for (values, 0..) |value, i| if (!std.math.isNan(value)) {
        try all.append(allocator, .{ .value = value, .id = offset + @as(u32, @intCast(i)) });
    };
    if (all.items.len == 0) return try allocator.alloc(Candidate, 0);
    var prefix: u64 = 0;
    var mask: u64 = 0;
    var target: usize = @min(n, all.items.len) - 1;
    for (0..8) |pass| {
        const shift: u6 = @intCast(56 - pass * 8);
        var counts: [256]usize = @splat(0);
        for (all.items) |candidate| {
            const key = sortableKey(candidate);
            if ((key & mask) == prefix) counts[@intCast((key >> shift) & 255)] += 1;
        }
        var skipped: usize = 0;
        var bucket: usize = 0;
        while (bucket < counts.len and skipped + counts[bucket] <= target) : (bucket += 1) skipped += counts[bucket];
        if (bucket == counts.len) break;
        target -= skipped;
        prefix |= @as(u64, @intCast(bucket)) << shift;
        mask |= @as(u64, 0xff) << shift;
    }
    var result: std.ArrayList(Candidate) = .empty;
    defer result.deinit(allocator);
    for (all.items) |candidate| if (sortableKey(candidate) <= prefix) try result.append(allocator, candidate);
    std.mem.sort(Candidate, result.items, {}, less);
    return try allocator.dupe(Candidate, result.items[0..@min(n, result.items.len)]);
}

/// Host emulation of the kernel's 32-shard candidate reduction and final shared-list ordering.
pub fn topCandidatesShardedRef(allocator: std.mem.Allocator, values: []const f32, offset: u32, k: usize) ![]Candidate {
    const n = @min(k, 2048);
    if (n == 0) return try allocator.alloc(Candidate, 0);
    const local_k = n;
    var merged: std.ArrayList(Candidate) = .empty;
    defer merged.deinit(allocator);
    var starts: [32]usize = undefined;
    var ends: [32]usize = undefined;
    for (0..32) |part| {
        const begin = values.len * part / 32;
        const end = values.len * (part + 1) / 32;
        const local = try topCandidatesRadixRef(allocator, values[begin..end], offset + @as(u32, @intCast(begin)), local_k);
        defer allocator.free(local);
        starts[part] = merged.items.len;
        try merged.appendSlice(allocator, local);
        ends[part] = merged.items.len;
    }
    const out_len = @min(n, merged.items.len);
    const result = try allocator.alloc(Candidate, out_len);
    for (0..32) |part| for (starts[part]..ends[part]) |at| {
        const key = merged.items[at];
        var rank = at - starts[part];
        for (0..32) |other| {
            if (other == part) continue;
            var lo = starts[other];
            var hi = ends[other];
            while (lo < hi) {
                const mid = lo + (hi - lo) / 2;
                if (less({}, merged.items[mid], key)) lo = mid + 1 else hi = mid;
            }
            rank += lo - starts[other];
        }
        if (rank < out_len) result[rank] = key;
    };
    return result;
}

/// Host reference for the fixed-width CUDA output, including unwritten-tail sentinels.
pub fn candidateOutputRef(allocator: std.mem.Allocator, values: []const f32, offset: u32, k: usize) ![]Candidate {
    const out = try allocator.alloc(Candidate, k);
    @memset(out, .{ .value = -std.math.inf(f32), .id = std.math.maxInt(u32) });
    const selected = try topCandidatesShardedRef(allocator, values, offset, k);
    defer allocator.free(selected);
    @memcpy(out[0..selected.len], selected);
    return out;
}

/// Merge each rank's local top-k. A token outside its rank's top-k cannot be in the global top-k.
pub fn mergeCandidates(allocator: std.mem.Allocator, ranks: []const []const Candidate, k: usize) ![]Candidate {
    var all: std.ArrayList(Candidate) = .empty;
    defer all.deinit(allocator);
    for (ranks) |rank| try all.appendSlice(allocator, rank);
    std.mem.sort(Candidate, all.items, {}, less);
    const n = @min(k, all.items.len);
    return try allocator.dupe(Candidate, all.items[0..n]);
}

/// Stable rank-local logsumexp state, suitable for a fp32 head partial.
pub fn partialLse(values: []const f32) Lse {
    const lanes = 256;
    var m: f32 = -std.math.inf(f32);
    var lane_max: [lanes]f32 = @splat(-std.math.inf(f32));
    for (0..lanes) |lane| {
        var i = lane;
        while (i < values.len) : (i += lanes) {
            if (values[i] > lane_max[lane]) lane_max[lane] = values[i];
        }
    }
    reduceTree(&lane_max, .max);
    m = lane_max[0];
    if (m == -std.math.inf(f32)) return .{ .max = m, .sum = 0 };
    var lane_sum: [lanes]f32 = @splat(0);
    for (0..lanes) |lane| {
        var i = lane;
        while (i < values.len) : (i += lanes) lane_sum[lane] += @exp(values[i] - m);
    }
    reduceTree(&lane_sum, .sum);
    return .{ .max = m, .sum = lane_sum[0] };
}

const LseReduce = enum { max, sum };

fn reduceTree(values: *[256]f32, op: LseReduce) void {
    var width: usize = 128;
    while (width != 0) : (width >>= 1) for (0..width) |i| {
        values[i] = switch (op) {
            .max => if (values[i + width] > values[i]) values[i + width] else values[i],
            .sum => values[i] + values[i + width],
        };
    };
}

/// Merge partials in rank order, equivalent to logsumexp over concatenated vocabulary shards.
pub fn mergeLse(ranks: []const Lse) f32 {
    var m: f32 = -std.math.inf(f32);
    for (ranks) |x| if (x.max > m) {
        m = x.max;
    };
    if (m == -std.math.inf(f32)) return m;
    var s: f32 = 0;
    for (ranks) |x| s += x.sum * @exp(x.max - m);
    return m + @log(s);
}

fn mix(x_: u64) u64 {
    var x = x_;
    x ^= x >> 30;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 27;
    x *%= 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

/// Python exact_sampling.uniform_rows 53-bit double from the splitmix hash of seed, position and token id.
pub fn keyedUniform(seed: u64, position: u64, id: u32) f64 {
    var x = mix(seed +% 0x9E3779B97F4A7C15);
    x = mix(x ^ (position *% 0xD1B54A32D192ED03));
    x = mix(x ^ id);
    return @as(f64, @floatFromInt(x >> 11)) * 0x1p-53 + 0x1p-54;
}

test "sampling randomness is stable for seed and absolute position" {
    const seed: u64 = 0x4b1d_78c2_91a0_ef35;
    const position: u64 = 262143;
    const uniform = keyedUniform(seed, position, 17);
    try std.testing.expectEqual(uniform, keyedUniform(seed, position, 17));
    try std.testing.expect(uniform != keyedUniform(seed, position + 1, 17));

    const allocator = std.testing.allocator;
    const candidates = [_]Candidate{ .{ .value = 0.0, .id = 17 }, .{ .value = -0.1, .id = 29 }, .{ .value = -0.2, .id = 41 } };
    const rule: Sampling = .{ .temperature = 0.8, .top_k = 3, .top_p = 1 };
    const first = try drawCandidates(allocator, &candidates, seed, position, rule);
    const repeated = try drawCandidates(allocator, &candidates, seed, position, rule);
    try std.testing.expectEqual(first, repeated);
}

/// Apply Python choose_rows top-k/top-p/min-p ordering and 53-bit keyed Gumbel noise.
pub fn drawCandidates(allocator: std.mem.Allocator, candidates: []const Candidate, seed: u64, position: u64, rule: Sampling) !u32 {
    if (candidates.len == 0 or !(rule.temperature > 0)) return error.InvalidCandidates;
    const ordered = try allocator.dupe(Candidate, candidates);
    defer allocator.free(ordered);
    std.mem.sort(Candidate, ordered, {}, less);
    const k = @min(ordered.len, if (rule.top_k == 0) ordered.len else @max(1, rule.top_k));
    const scaled = try allocator.alloc(f64, k);
    defer allocator.free(scaled);
    const temperature = @max(rule.temperature, 1e-6);
    for (ordered[0..k], scaled) |c, *v| v.* = @as(f64, c.value) / temperature;
    const norm = numpyPairwiseSumExp(scaled);
    if (!(norm > 0) or !std.math.isFinite(norm)) return error.InvalidCandidates;
    var keep = k;
    if (rule.top_p > 0 and rule.top_p < 1) {
        keep = topPCount(scaled, norm, rule.top_p);
    }
    if (rule.min_log != -std.math.inf(f64)) {
        var min_keep: usize = 0;
        while (min_keep < keep and scaled[min_keep] >= scaled[0] + rule.min_log) : (min_keep += 1) {}
        keep = @max(min_keep, 1);
    }
    var winner = ordered[0];
    var best = -std.math.inf(f64);
    for (ordered[0..keep]) |c| {
        const u = keyedUniform(seed, position, c.id);
        const g = -@log(-@log(u));
        const score = @as(f64, c.value) / temperature + g;
        if (score > best or (score == best and c.id < winner.id)) {
            best = score;
            winner = c;
        }
    }
    return winner.id;
}

/// Whether drawCandidates' top-p nucleus over `candidates` (the merged global top K) at a temperature T < 1 is the one
/// the full vocabulary gives, for every total the merged T = 1 logsumexp `log_z` allows. Every other token scores at
/// most the smallest candidate logit c (each rank sent its own top K), so at T < 1 its mass relative to the top
/// candidate is at most exp((c - top)(1/T - 1)) times its T = 1 mass, which log_z minus the candidates' mass bounds.
/// The same nucleus size at both ends of that range means the same tokens, so the same Gumbel keys and winner.
pub fn temperedNucleusFits(allocator: std.mem.Allocator, candidates: []const Candidate, log_z: f32, rule: Sampling) !bool {
    if (!(rule.temperature > 0 and rule.temperature < 1) or rule.top_k != 0 or !(rule.top_p > 0 and rule.top_p < 1)) return false;
    if (candidates.len == 0 or !std.math.isFinite(log_z)) return false;
    const ordered = try allocator.dupe(Candidate, candidates);
    defer allocator.free(ordered);
    std.mem.sort(Candidate, ordered, {}, less);
    const scaled = try allocator.alloc(f64, ordered.len);
    defer allocator.free(scaled);
    const temperature = @max(rule.temperature, 1e-6);
    for (ordered, scaled) |c, *v| v.* = @as(f64, c.value) / temperature;
    const norm = numpyPairwiseSumExp(scaled);
    if (!(norm > 0) or !std.math.isFinite(norm)) return false;
    const top: f64 = ordered[0].value;
    const smallest: f64 = ordered[ordered.len - 1].value;
    var mass: f64 = 0;
    for (ordered) |c| mass += @exp(@as(f64, c.value) - top);
    // The GPU logsumexp is f32: keep 1e-5 of the total as slack so rounding never hides tail mass.
    const total = @exp(@as(f64, log_z) - top);
    const tail = @max(total - mass, 0) + 1e-5 * total;
    const wide = norm + tail * @exp((smallest - top) * (1 / temperature - 1));
    const keep = topPCount(scaled, norm, rule.top_p);
    var cumulative: f64 = 0;
    for (scaled, 0..) |v, i| {
        cumulative += @exp(v - scaled[0]) / wide;
        if (cumulative >= rule.top_p) return i + 1 == keep;
    }
    return false;
}

test "tempered nucleus fits a peaked row and falls back on a flat or truncated one" {
    const a = std.testing.allocator;
    const peaked = [_]Candidate{ .{ .value = 30, .id = 1 }, .{ .value = 29, .id = 2 }, .{ .value = 20, .id = 3 }, .{ .value = 10, .id = 4 } };
    // log_z at T = 1 a hair above the candidates' own mass: the tail is negligible.
    var m: f64 = 0;
    for (peaked) |c| m += @exp(@as(f64, c.value) - 30);
    const rule: Sampling = .{ .temperature = 0.6, .top_p = 0.95 };
    try std.testing.expect(try temperedNucleusFits(a, &peaked, @floatCast(30 + @log(m * 1.0001)), rule));
    // A tail as heavy as the candidates themselves, close to the top: the boundary can move, so fall back.
    const flat = [_]Candidate{ .{ .value = 30, .id = 1 }, .{ .value = 29.9, .id = 2 }, .{ .value = 29.8, .id = 3 } };
    try std.testing.expect(!try temperedNucleusFits(a, &flat, 30 + @log(@as(f32, 400)), rule));
    // T = 1 and T > 1 keep the exact paths they had.
    try std.testing.expect(!try temperedNucleusFits(a, &peaked, 31, .{ .temperature = 1, .top_p = 0.95 }));
    try std.testing.expect(!try temperedNucleusFits(a, &peaked, 31, .{ .temperature = 1.2, .top_p = 0.95 }));
}

fn topPCount(scaled: []const f64, norm: f64, top_p: f64) usize {
    var count: usize = 0;
    var cumulative: f64 = 0;
    while (count < scaled.len) : (count += 1) {
        cumulative += @exp(scaled[count] - scaled[0]) / norm;
        if (cumulative >= top_p) return @max(count + 1, 1);
    }
    return @max(count, 1);
}

/// top_k=0 nucleus sampling needs the entire global vocabulary; reject a caller's truncated candidate set.
pub fn drawNucleusCandidates(allocator: std.mem.Allocator, candidates: []const Candidate, global_vocab: usize, seed: u64, position: u64, rule: Sampling) !u32 {
    if (rule.top_k != 0 or !(rule.top_p > 0 and rule.top_p < 1)) return error.InvalidNucleusRule;
    if (candidates.len != global_vocab) return error.IncompleteNucleusCandidates;
    const seen = try allocator.alloc(bool, global_vocab);
    defer allocator.free(seen);
    @memset(seen, false);
    for (candidates) |candidate| {
        if (candidate.id >= global_vocab or seen[candidate.id]) return error.IncompleteNucleusCandidates;
        seen[candidate.id] = true;
    }
    return drawCandidates(allocator, candidates, seed, position, rule);
}

/// Draw from a truncated ranked list only after the full-vocabulary normalizer proves that its nucleus fits.
/// The caller supplies logsumexp(logits) for the same row. This is the exact top-k candidate path at T=1.
pub fn drawNucleusWithLse(allocator: std.mem.Allocator, candidates: []const Candidate, log_z: f32, seed: u64, position: u64, rule: Sampling) !u32 {
    return drawNucleusWithLseImpl(allocator, candidates, log_z, seed, position, rule, false);
}

/// Full-vocabulary fallback: tolerate a rounded-down GPU logsumexp after exhausting every available token.
pub fn drawNucleusWithFullLse(allocator: std.mem.Allocator, candidates: []const Candidate, log_z: f32, seed: u64, position: u64, rule: Sampling) !u32 {
    return drawNucleusWithLseImpl(allocator, candidates, log_z, seed, position, rule, true);
}

fn drawNucleusWithLseImpl(allocator: std.mem.Allocator, candidates: []const Candidate, log_z: f32, seed: u64, position: u64, rule: Sampling, allow_exhausted: bool) !u32 {
    if (rule.top_k != 0 or rule.temperature != 1 or !(rule.top_p > 0 and rule.top_p < 1)) return error.UnsupportedExactNucleusRule;
    if (candidates.len == 0 or !std.math.isFinite(log_z)) return error.InvalidCandidates;
    const ordered = try allocator.dupe(Candidate, candidates);
    defer allocator.free(ordered);
    std.mem.sort(Candidate, ordered, {}, less);
    var keep: usize = 0;
    var mass: f64 = 0;
    while (keep < ordered.len) : (keep += 1) {
        mass += @exp(@as(f64, ordered[keep].value) - @as(f64, log_z));
        if (mass >= rule.top_p) {
            keep += 1;
            break;
        }
    }
    if (mass < rule.top_p and !allow_exhausted) return error.NucleusExceedsCandidateK;
    if (rule.min_log != -std.math.inf(f64)) {
        var min_keep: usize = 0;
        while (min_keep < keep and @as(f64, ordered[min_keep].value) >= @as(f64, ordered[0].value) + rule.min_log) : (min_keep += 1) {}
        keep = @max(min_keep, 1);
    }
    var winner = ordered[0];
    var best = -std.math.inf(f64);
    for (ordered[0..keep]) |c| {
        const u = keyedUniform(seed, position, c.id);
        const g = -@log(-@log(u));
        const score = c.value + g;
        if (score > best or (score == best and c.id < winner.id)) {
            best = score;
            winner = c;
        }
    }
    return winner.id;
}

/// NumPy's pairwise sum order: sequential below 8, eight accumulators through 128, then aligned recursive halves.
fn numpyPairwiseSumExp(values: []const f64) f64 {
    if (values.len == 0) return 0;
    return numpyBufferedPairwise(values, values[0]);
}

fn numpyPairwiseSum(values: []const f64) f64 {
    return numpyBufferedPairwise(values, null);
}

/// NumPy's reduction iterator calls the pairwise loop on 8192-element buffers and adds buffer results in order.
fn numpyBufferedPairwise(values: []const f64, origin: ?f64) f64 {
    var total: f64 = -0.0;
    var start: usize = 0;
    while (start < values.len) {
        const len = @min(8192, values.len - start);
        total += pairwiseRange(values, origin, start, len);
        start += len;
    }
    return total;
}

fn pairwiseRange(values: []const f64, origin: ?f64, start: usize, len: usize) f64 {
    if (len == 0) return -0.0;
    const term = struct {
        fn get(xs: []const f64, base: ?f64, i: usize) f64 {
            return if (base) |b| @exp(xs[i] - b) else xs[i];
        }
    }.get;
    if (len < 8) {
        var sum: f64 = -0.0;
        for (start..start + len) |i| sum += term(values, origin, i);
        return sum;
    }
    if (len <= 128) {
        var r: [8]f64 = undefined;
        for (0..8) |i| r[i] = term(values, origin, start + i);
        var i: usize = 8;
        while (i < len - (len % 8)) : (i += 8) {
            inline for (0..8) |lane| r[lane] += term(values, origin, start + i + lane);
        }
        var result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while (i < len) : (i += 1) result += term(values, origin, start + i);
        return result;
    }
    var left_len = len / 2;
    left_len -= left_len % 8;
    return pairwiseRange(values, origin, start, left_len) + pairwiseRange(values, origin, start + left_len, len - left_len);
}

test "exact truncated nucleus accepts only when candidate mass reaches top_p" {
    const candidates = [_]Candidate{ .{ .value = 0, .id = 4 }, .{ .value = -2, .id = 9 } };
    const rule: Sampling = .{ .temperature = 1, .top_p = 0.95 };
    try std.testing.expectEqual(@as(u32, 4), try drawNucleusWithLse(std.testing.allocator, &candidates, 0, 7, 2, rule));
    try std.testing.expectError(error.NucleusExceedsCandidateK, drawNucleusWithLse(std.testing.allocator, &candidates, 2, 7, 2, rule));
}

test "NumPy pairwise exp sum matches Python generated boundary fixtures" {
    const fixture = @import("numpy_pairwise_fixture.zig");
    for (fixture.cases) |case| {
        try std.testing.expectEqual(case.expected_bits, @as(u64, @bitCast(numpyPairwiseSum(case.values))));
    }
    const equal_logits = [_]f64{ 0, 0, 0 };
    const norm = numpyPairwiseSumExp(&equal_logits);
    try std.testing.expectEqual(@as(usize, 1), topPCount(&equal_logits, norm, fixture.top_p_below));
    try std.testing.expectEqual(@as(usize, 2), topPCount(&equal_logits, norm, fixture.top_p_above));
}

test "candidate workspace scales with k and 64-row tiles" {
    try std.testing.expectEqual(@as(usize, 16_384), try Head.candidateScratchBytes(1, 64));
    try std.testing.expectEqual(@as(usize, 524_288), try Head.candidateScratchBytes(1, 2048));
    try std.testing.expectEqual(@as(usize, 33_554_432), try Head.candidateScratchBytes(64, 2048));
    try std.testing.expectError(error.CandidateWorkspaceOverflow, Head.candidateScratchBytes(std.math.maxInt(usize), 2048));
    try std.testing.expectError(error.InvalidCandidateCount, Head.candidateScratchBytes(1, 2049));
    try std.testing.expectEqual(@as(usize, 64), candidate_rows_per_launch);
    try std.testing.expectEqual(@as(usize, 64), try candidateLaunchRows(65, 64, try Head.candidateScratchBytes(64, 64)));
    try std.testing.expectError(error.CandidateWorkspaceTooSmall, candidateLaunchRows(65, 64, try Head.candidateScratchBytes(63, 64)));
}

test "candidate output initializes missing candidates to Python sort sentinels" {
    const allocator = std.testing.allocator;
    const values = [_]f32{ std.math.nan(f32), 2, std.math.nan(f32), -1, std.math.nan(f32) };
    const out = try candidateOutputRef(allocator, &values, 10, 4);
    defer allocator.free(out);
    try std.testing.expectEqualSlices(Candidate, &.{
        .{ .value = 2, .id = 11 },
        .{ .value = -1, .id = 13 },
        .{ .value = -std.math.inf(f32), .id = std.math.maxInt(u32) },
        .{ .value = -std.math.inf(f32), .id = std.math.maxInt(u32) },
    }, out);
    const all_nan = [_]f32{ std.math.nan(f32), std.math.nan(f32) };
    const empty = try candidateOutputRef(allocator, &all_nan, 0, 3);
    defer allocator.free(empty);
    for (empty) |candidate| {
        try std.testing.expectEqual(-std.math.inf(f32), candidate.value);
        try std.testing.expectEqual(std.math.maxInt(u32), candidate.id);
    }
}

test "32-shard candidate reduction matches the exact global value and id order" {
    const allocator = std.testing.allocator;
    const ks = [_]usize{ 1, 2, 8, 9, 64, 72, 1000, 2047, 2048 };
    for (ks) |k| {
        const values = try allocator.alloc(f32, 3073);
        defer allocator.free(values);
        for (values, 0..) |*v, i| {
            const bucket = (i * 17 + i / 11) % 73;
            v.* = @as(f32, @floatFromInt(@as(isize, @intCast(bucket)) - 36)) / 8;
        }
        values[0] = -0.0;
        values[1] = 0.0;
        values[2] = std.math.nan(f32);
        const expected = try topCandidatesRef(allocator, values, 73, k);
        defer allocator.free(expected);
        const actual = try topCandidatesShardedRef(allocator, values, 73, k);
        defer allocator.free(actual);
        try std.testing.expectEqualSlices(Candidate, expected, actual);
    }
    const edge = [_]f32{ -std.math.inf(f32), -0.0, 0.0, std.math.nan(f32), 1, 1, 1 };
    const expected = try topCandidatesRef(allocator, &edge, 0, 8);
    defer allocator.free(expected);
    const actual = try topCandidatesShardedRef(allocator, &edge, 0, 8);
    defer allocator.free(actual);
    try std.testing.expectEqualSlices(Candidate, expected, actual);
}

test "clustered low-id top-k retains multiple winners from a single shard" {
    const allocator = std.testing.allocator;
    const values = try allocator.alloc(f32, 65_536);
    defer allocator.free(values);
    for (values, 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i));
    values[10] = 10;
    values[11] = 9;
    values[12] = 8;
    const expected = try topCandidatesRef(allocator, values, 0, 3);
    defer allocator.free(expected);
    const actual = try topCandidatesShardedRef(allocator, values, 0, 3);
    defer allocator.free(actual);
    try std.testing.expectEqualSlices(Candidate, expected, actual);
    try std.testing.expectEqual(@as(u32, 11), actual[1].id);

    // Zipf-like descending preference by low token id exercises the reviewer's clustered case.
    for (values, 0..) |*v, i| v.* = -@as(f32, @floatFromInt(i / 7));
    const zipf_expected = try topCandidatesRef(allocator, values, 0, 64);
    defer allocator.free(zipf_expected);
    const zipf_actual = try topCandidatesShardedRef(allocator, values, 0, 64);
    defer allocator.free(zipf_actual);
    try std.testing.expectEqualSlices(Candidate, zipf_expected, zipf_actual);
}

test "51 top-k edge cases preserve NumPy value order and lower-id ties" {
    const allocator = std.testing.allocator;
    const ks = [_]usize{ 1, 2, 8, 9, 64, 72, 1000, 2047, 2048 };
    for (0..51) |case| {
        const len: usize = if (case % 7 == 0) 19 else 257 + case * 3;
        const values = try allocator.alloc(f32, len);
        defer allocator.free(values);
        for (values, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(isize, @intCast((i * 37 + case * 13) % 29)) - 14)) / 4;
        switch (case % 7) {
            0 => @memset(values, std.math.nan(f32)),
            1 => @memset(values, 3.0),
            2 => @memset(values, -std.math.inf(f32)),
            3 => {
                values[0] = 100;
                values[1] = 100;
            },
            4 => {
                values[0] = -0.0;
                values[1] = 0.0;
                values[2] = std.math.nan(f32);
            },
            5 => {
                for (values, 0..) |*v, i| if (i % 5 != 0) {
                    v.* = std.math.nan(f32);
                };
            },
            6 => {
                values[0] = std.math.inf(f32);
                values[len - 1] = -std.math.inf(f32);
            },
            else => unreachable,
        }
        const k = ks[case % ks.len];
        const expected = try topCandidatesRef(allocator, values, 129, k);
        defer allocator.free(expected);
        const actual = try topCandidatesShardedRef(allocator, values, 129, k);
        defer allocator.free(actual);
        try std.testing.expectEqualSlices(Candidate, expected, actual);
    }
}

test "NumPy reduction iterator uses exact 8192 element pairwise buffers" {
    const cases = [_]struct { len: usize, bits: u64 }{
        .{ .len = 8193, .bits = 13841726847038077702 },
        .{ .len = 10000, .bits = 13839215224168967211 },
        .{ .len = 16385, .bits = 13841207200927227253 },
        .{ .len = 25000, .bits = 13838002716576982825 },
    };
    for (cases) |case| {
        const values = try std.testing.allocator.alloc(f64, case.len);
        defer std.testing.allocator.free(values);
        for (values, 0..) |*v, i| v.* = (@as(f64, @floatFromInt(@as(isize, @intCast((i * 37) % 101)) - 50))) / 13;
        try std.testing.expectEqual(case.bits, @as(u64, @bitCast(numpyPairwiseSum(values))));
    }
}

test "sampler recursively analyzes all public nested declarations" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Head);
    std.testing.refAllDecls(Candidate);
    std.testing.refAllDecls(Lse);
    std.testing.refAllDecls(Sampling);
    std.testing.refAllDecls(RowWindow);
    std.testing.refAllDecls(GateRow);
}
