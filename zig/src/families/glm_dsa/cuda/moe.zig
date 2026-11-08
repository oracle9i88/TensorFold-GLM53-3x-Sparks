// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 DSA router, slot contracts, and rank-local low-rank correction (WP8).
const std = @import("std");
const cuda = @import("cuda");
const mlp = @import("mlp.zig");

pub const hidden = 6144;
pub const experts = 256;
pub const top_k = 8;
pub const shared_slot = 8;
pub const slots = 9;
pub const lowrank = 64;
pub const decode_splits = 128;

pub fn decodeSwitch() bool {
    const value = std.c.getenv("TF_GLM53_LOWRANK_DECODE") orelse return true;
    return decodeSelected(std.mem.span(value));
}

pub fn decodeSelected(value: ?[]const u8) bool {
    const raw = value orelse return true;
    return !std.mem.eql(u8, raw, "0");
}

pub const Plan = struct {
    rows: usize,
    router_part_floats: usize,
    pick_i32: usize,
    weights_f32: usize,
    expert_slots: usize,
    lowrank_mid_floats: usize,

    pub fn of(rows: usize, rank_width: usize) !Plan {
        if (rows == 0 or (rank_width != lowrank and rank_width != lowrank / 2)) return error.BadMoePlan;
        return .{
            .rows = rows,
            .router_part_floats = try std.math.mul(usize, try std.math.mul(usize, rows, experts), 8),
            .pick_i32 = try std.math.mul(usize, rows, slots),
            .weights_f32 = try std.math.mul(usize, rows, slots),
            .expert_slots = try std.math.mul(usize, try std.math.mul(usize, rows, slots), hidden),
            .lowrank_mid_floats = try lowRankMidFloats(rows, rank_width),
        };
    }
};

fn lowRankMidFloats(rows: usize, rank_width: usize) !usize {
    return try std.math.mul(usize, try std.math.mul(usize, rows, rank_width), 8);
}

fn mockLowRankPartialWrite(buffer: []f32, rows: usize, rank_width: usize) !void {
    const required = try lowRankMidFloats(rows, rank_width);
    if (buffer.len < required) return error.LowRankScratchTooSmall;
    for (0..required) |i| buffer[i] = @floatFromInt(i);
}

pub const Route = struct { pick: [top_k]u16, weights: [top_k]f32 };

pub const RouterKernels = struct { router_part: cuda.Function, router_tile: cuda.Function, topk: cuda.Function, slice_sum: cuda.Function };

/// Reuse Flash's captured router kernels; GLM-5.3 uses the same sigmoid,
/// correction-bias choice, unbiased normalized scores and lower-id tie rule.
pub fn route(k: RouterKernels, stream: cuda.Stream, x: u64, x_stride: usize, router: u64, bias: u64, logits: u64, partials: u64, pick: u64, weights: u64, rows: usize) !void {
    var dot: cuda.Args = .{};
    inline for (.{ x, router, partials, @as(i64, @intCast(rows)), @as(i64, @intCast(x_stride)), @as(i64, hidden), @as(i64, experts), @as(c_int, 8) }) |v| dot.add(v);
    const tile = x % 16 == 0 and x_stride % 8 == 0;
    const rf = if (tile) k.router_tile else k.router_part;
    const gx: u32 = if (tile) @intCast((rows + 63) / 64) else @intCast((rows + 15) / 16);
    try cuda.launch.launch(rf, .{ .grid = .{ .x = gx, .y = 8, .z = 8 }, .block = .{ .x = 128 }, .shared = if (tile) 27_648 else 0 }, stream, &dot);
    var reduce: cuda.Args = .{};
    inline for (.{ partials, logits, @as(i64, @intCast(rows * experts)), @as(c_int, 8), @as(c_int, 1) }) |v| reduce.add(v);
    try cuda.launch.launch(k.slice_sum, .{ .grid = .{ .x = @intCast((rows * experts + 1023) / 1024) }, .block = .{ .x = 128 } }, stream, &reduce);
    var top: cuda.Args = .{};
    inline for (.{ logits, bias, pick, weights, @as(f32, 2.5), @as(c_int, experts), @as(c_int, top_k), @as(c_int, 1) }) |v| top.add(v);
    try cuda.launch.launch(k.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &top);
}

/// Host reference for tf_glm_topk: choice uses sigmoid(logit)+bias, but weights
/// use the unbiased sigmoid scores and are scaled after normalization.
pub fn routeRef(logits: []const f32, bias: []const f32, scale: f32) !Route {
    if (logits.len != experts or bias.len != experts) return error.RouterShape;
    var scores: [experts]f32 = undefined;
    for (logits, 0..) |v, i| scores[i] = 1.0 / (1.0 + @exp(-v));
    var result: Route = undefined;
    var selected: [experts]bool = @splat(false);
    var total: f32 = 0;
    for (0..top_k) |k| {
        var best: usize = 0;
        var best_score: f32 = -std.math.inf(f32);
        for (0..experts) |i| {
            if (selected[i]) continue;
            const choice = scores[i] + bias[i];
            if (choice > best_score or (choice == best_score and i < best)) {
                best = i;
                best_score = choice;
            }
        }
        selected[best] = true;
        result.pick[k] = @intCast(best);
        result.weights[k] = scores[best];
        total += scores[best];
    }
    for (&result.weights) |*w| w.* = (w.* / (total + 1e-20)) * scale;
    return result;
}

pub const LowRankKernels = struct { partial: cuda.Function, reduce: cuda.Function, finish: cuda.Function, decode: cuda.Function };
pub const Kernels = struct {
    lowrank: LowRankKernels,
    decode_fast: bool = true,
    embed_pick: cuda.Function,
    lowrank_combine_bf16: cuda.Function,
    lowrank_combine_bf16_decode: cuda.Function,
    moe_post: ?cuda.Function = null,
    residual_add_bf16: cuda.Function,

    pub fn load(module: cuda.Module) !Kernels {
        return .{
            .lowrank = .{ .partial = try module.function("tf_glm53_lowrank_partial"), .reduce = try module.function("tf_glm53_lowrank_reduce"), .finish = try module.function("tf_glm53_lowrank_finish"), .decode = try module.function("tf_glm53_lowrank_decode") },
            .embed_pick = try module.function("tf_glm53_embed_pick"),
            .lowrank_combine_bf16 = try module.function("tf_glm53_lowrank_combine_bf16"),
            .lowrank_combine_bf16_decode = try module.function("tf_glm53_lowrank_combine_bf16_decode"),
            .moe_post = try module.function("tf_glm53_moe_post"),
            .residual_add_bf16 = try module.function("tf_glm53_moe_residual_add"),
        };
    }
};

/// Add this rank's `(x.float() @ A.float()) @ B.float()` partial into the
/// existing routed-plus-shared fp32 accumulator. A and B carry this rank's R/2 slice.
pub fn lowRankAdd(k: LowRankKernels, stream: cuda.Stream, x: u64, a: u64, b: u64, acc: u64, mid: u64, rows: usize, r_local: usize, x_stride: usize, a_stride: usize, b_stride: usize) !void {
    if (rows == 0 or r_local == 0 or r_local > lowrank or x_stride != hidden or a_stride != r_local or b_stride != hidden) return error.LowRankShape;
    var args: cuda.Args = .{};
    inline for (.{ x, a, mid, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(r_local)), @as(c_int, @intCast(x_stride)), @as(c_int, @intCast(a_stride)), @as(c_int, 8) }) |v| args.add(v);
    try cuda.launch.launch(k.partial, .{ .grid = .{ .x = @intCast((rows * r_local * 8 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
    var reduce: cuda.Args = .{};
    // in place: the 8 split partials are in mid; thread i reads slots i, total+i, ... and writes slot i
    inline for (.{ mid, mid, @as(c_int, @intCast(rows * r_local)), @as(c_int, 8) }) |v| reduce.add(v);
    try cuda.launch.launch(k.reduce, .{ .grid = .{ .x = @intCast((rows * r_local + 255) / 256) }, .block = .{ .x = 256 } }, stream, &reduce);
    var finish: cuda.Args = .{};
    inline for (.{ b, mid, acc, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(r_local)), @as(c_int, @intCast(b_stride)) }) |v| finish.add(v);
    try cuda.launch.launch(k.finish, .{ .grid = .{ .x = @intCast((rows * hidden + 255) / 256) }, .block = .{ .x = 256 } }, stream, &finish);
}

pub const LowRankWeights = struct { a: u64, b: u64, mid: u64, rank_local: usize };
pub const RoutedRunner = *const fn (ctx: *anyopaque, stream: cuda.Stream, x: u64, x_stride: usize, picks: u64, weights: u64, rows: usize, out: u64) anyerror!void;
pub const ProfileHooks = struct {
    ctx: *anyopaque,
    begin_fn: *const fn (*anyopaque, u8) anyerror!void,
    end_fn: *const fn (*anyopaque, u8) anyerror!void,
};

/// Compose route -> WP4 routed slots -> shared expert -> low-rank into one BF16 rank partial.
/// The callback owns the WP4 expert plan and must write routed outputs in slot order.
pub fn block(k: Kernels, route_k: RouterKernels, mlp_k: mlp.Kernels, stream: cuda.Stream, shared_weights: mlp.Weights, shared_scratch: mlp.Scratch, routed_ctx: *anyopaque, run_routed: RoutedRunner, x: u64, x_stride: usize, router: u64, bias: u64, logits: u64, router_part: u64, picks: u64, weights: u64, shared_out: u64, rows: usize, routed_out: u64, moe_out: u64, lowrank_weights: ?LowRankWeights) !void {
    try blockWithHooks(k, route_k, mlp_k, stream, shared_weights, shared_scratch, routed_ctx, run_routed, x, x_stride, router, bias, logits, router_part, picks, weights, shared_out, rows, routed_out, moe_out, lowrank_weights, null);
}

/// Decode path variant with optional timing hooks; null preserves the original launch sequence exactly.
pub fn blockWithHooks(k: Kernels, route_k: RouterKernels, mlp_k: mlp.Kernels, stream: cuda.Stream, shared_weights: mlp.Weights, shared_scratch: mlp.Scratch, routed_ctx: *anyopaque, run_routed: RoutedRunner, x: u64, x_stride: usize, router: u64, bias: u64, logits: u64, router_part: u64, picks: u64, weights: u64, shared_out: u64, rows: usize, routed_out: u64, moe_out: u64, lowrank_weights: ?LowRankWeights, hooks: ?ProfileHooks) !void {
    if (hooks) |h| try h.begin_fn(h.ctx, 0);
    try route(route_k, stream, x, x_stride, router, bias, logits, router_part, picks, weights, rows);
    if (hooks) |h| try h.end_fn(h.ctx, 0);
    if (hooks) |h| try h.begin_fn(h.ctx, 1);
    try run_routed(routed_ctx, stream, x, x_stride, picks, weights, rows, routed_out);
    if (hooks) |h| try h.end_fn(h.ctx, 1);
    if (hooks) |h| try h.begin_fn(h.ctx, 2);
    try mlp.block(mlp_k, stream, shared_weights, shared_scratch, x, rows, shared_weights.down.k, shared_out);
    if (hooks) |h| try h.end_fn(h.ctx, 2);
    var b: u64 = 0;
    var mid: u64 = 0;
    var rank: usize = 0;
    if (lowrank_weights) |lr| {
        if (hooks) |h| try h.begin_fn(h.ctx, 3);
        b = lr.b;
        mid = lr.mid;
        rank = lr.rank_local;
        try lowrankPreMode(k, stream, x, x_stride, rows, lr, true);
        if (hooks) |h| try h.end_fn(h.ctx, 3);
    }
    const count = rows * hidden;
    var combine: cuda.Args = .{};
    inline for (.{ routed_out, shared_out, b, mid, moe_out, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(rank)), @as(c_int, hidden) }) |v| combine.add(v);
    if (rows <= 16 and k.decode_fast) {
        try cuda.launch.launch(k.lowrank_combine_bf16_decode, .{ .grid = .{ .x = @intCast((count + 31) / 32) }, .block = .{ .x = 32 } }, stream, &combine);
    } else {
        try cuda.launch.launch(k.lowrank_combine_bf16, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &combine);
    }
}

/// Run the per-piece work used by deferred routed prefill. The route, shared MLP and low-rank
/// partial launches retain the same row shapes as `block`; the caller runs routed experts later.
pub fn blockPre(k: Kernels, route_k: RouterKernels, mlp_k: mlp.Kernels, stream: cuda.Stream, shared_weights: mlp.Weights, shared_scratch: mlp.Scratch, x: u64, x_stride: usize, router: u64, bias: u64, logits: u64, router_part: u64, picks: u64, weights: u64, shared_out: u64, rows: usize, lowrank_weights: ?LowRankWeights) !void {
    try route(route_k, stream, x, x_stride, router, bias, logits, router_part, picks, weights, rows);
    try mlp.block(mlp_k, stream, shared_weights, shared_scratch, x, rows, shared_weights.down.k, shared_out);
    if (lowrank_weights) |lr| try lowrankPreMode(k, stream, x, x_stride, rows, lr, false);
}

/// The low-rank partial (8 K-splits into `lr.mid`) and its in-place reduce, which blockPost's combine reads. Every
/// MoE path that ends in blockPost must run this first for a low-rank layer (the decode side stream included).
pub fn lowrankPre(k: Kernels, stream: cuda.Stream, x: u64, x_stride: usize, rows: usize, lr: LowRankWeights) !void {
    try lowrankPreMode(k, stream, x, x_stride, rows, lr, true);
}

fn lowrankPreMode(k: Kernels, stream: cuda.Stream, x: u64, x_stride: usize, rows: usize, lr: LowRankWeights, decode_mode: bool) !void {
    if (rows == 0 or lr.rank_local == 0 or lr.rank_local > lowrank or x_stride != hidden) return error.LowRankShape;
    if (decode_mode and rows <= 16 and k.decode_fast) {
        const splits: c_int = decode_splits;
        var partial: cuda.Args = .{};
        inline for (.{ x, lr.a, lr.mid, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(lr.rank_local)), @as(c_int, hidden), @as(c_int, @intCast(lr.rank_local)), splits }) |v| partial.add(v);
        const rank_tiles = (lr.rank_local + 31) / 32;
        try cuda.launch.launch(k.lowrank.decode, .{ .grid = .{ .x = @intCast(rows * decode_splits * rank_tiles) }, .block = .{ .x = 256 }, .shared = 8 * 32 * 4 }, stream, &partial);
        var reduce: cuda.Args = .{};
        inline for (.{ lr.mid, lr.mid, @as(c_int, @intCast(rows * lr.rank_local)), splits }) |v| reduce.add(v);
        try cuda.launch.launch(k.lowrank.reduce, .{ .grid = .{ .x = @intCast((rows * lr.rank_local + 255) / 256) }, .block = .{ .x = 256 } }, stream, &reduce);
        return;
    }
    var partial: cuda.Args = .{};
    inline for (.{ x, lr.a, lr.mid, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(lr.rank_local)), @as(c_int, hidden), @as(c_int, @intCast(lr.rank_local)), @as(c_int, 8) }) |v| partial.add(v);
    try cuda.launch.launch(k.lowrank.partial, .{ .grid = .{ .x = @intCast((rows * lr.rank_local * 8 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &partial);
    var reduce: cuda.Args = .{};
    inline for (.{ lr.mid, lr.mid, @as(c_int, @intCast(rows * lr.rank_local)), @as(c_int, 8) }) |v| reduce.add(v);
    try cuda.launch.launch(k.lowrank.reduce, .{ .grid = .{ .x = @intCast((rows * lr.rank_local + 255) / 256) }, .block = .{ .x = 256 } }, stream, &reduce);
}

/// Combine a previously computed routed result with shared and low-rank outputs.
pub fn blockPost(k: Kernels, stream: cuda.Stream, routed_out: u64, shared_out: u64, lowrank_weights: ?LowRankWeights, rows: usize, moe_out: u64) !void {
    var b: u64 = 0;
    var mid: u64 = 0;
    var rank: usize = 0;
    if (lowrank_weights) |lr| {
        if (rows == 0 or lr.rank_local == 0 or lr.rank_local > lowrank) return error.LowRankShape;
        b = lr.b;
        mid = lr.mid;
        rank = lr.rank_local;
    }
    const count = rows * hidden;
    var combine: cuda.Args = .{};
    inline for (.{ routed_out, shared_out, b, mid, moe_out, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(rank)), @as(c_int, hidden) }) |v| combine.add(v);
    if (rows <= 16 and k.decode_fast) {
        try cuda.launch.launch(k.lowrank_combine_bf16_decode, .{ .grid = .{ .x = @intCast((count + 31) / 32) }, .block = .{ .x = 32 } }, stream, &combine);
    } else {
        try cuda.launch.launch(k.lowrank_combine_bf16, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &combine);
    }
}

/// Fused routed slot reduction and block post for grouped prompt experts.
pub fn postPrompt(k: Kernels, stream: cuda.Stream, y: u64, weights: u64, shared_out: u64, lowrank_weights: ?LowRankWeights, rows: usize, moe_out: u64) !void {
    var b: u64 = 0;
    var mid: u64 = 0;
    var rank: usize = 0;
    if (lowrank_weights) |lr| {
        if (rows == 0 or lr.rank_local == 0 or lr.rank_local > lowrank) return error.LowRankShape;
        b = lr.b;
        mid = lr.mid;
        rank = lr.rank_local;
    }
    const kernel = k.moe_post orelse return error.MoePostKernelMissing;
    var args: cuda.Args = .{};
    inline for (.{ y, weights, shared_out, b, mid, moe_out, @as(c_int, @intCast(rows)), @as(c_int, hidden), @as(c_int, @intCast(rank)), @as(c_int, hidden) }) |v| args.add(v);
    try cuda.launch.launch(kernel, .{ .grid = .{ .x = @intCast((rows * hidden + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
}

/// Round-to-nearest-even BF16, matching CUDA's `__float2bfloat16_rn`.
pub fn bf16Bits(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/// Fixed combine contract: one fp32 add for shared + low-rank, then routed + that result,
/// followed by the single BF16 output rounding.
pub fn combineBf16Ref(routed: f32, shared: f32, low_rank: f32) u16 {
    return bf16Bits(routed + (shared + low_rank));
}

/// Launch the existing MoE residual boundary for rank-major BF16 partials.
pub fn residualAddBf16(f: cuda.Function, stream: cuda.Stream, x: u64, parts: u64, rows: usize, world: usize) !void {
    const count = rows * hidden;
    if (rows == 0) return;
    var args: cuda.Args = .{};
    const count_arg: c_int = @intCast(count | (world << 28));
    inline for (.{ x, x, parts, count_arg }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
}

/// Pick the owner contribution after vocab-sharded BF16 embeds have been gathered.
pub fn embedPick(f: cuda.Function, stream: cuda.Stream, gathered: u64, owners: u64, out: u64, rows: usize, width: usize, world: usize) !void {
    if (rows == 0 or width == 0 or world == 0) return error.BadEmbedPick;
    var args: cuda.Args = .{};
    inline for (.{ gathered, owners, out, @as(c_int, @intCast(rows)), @as(c_int, @intCast(width)), @as(c_int, @intCast(world)) }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * width + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
}

/// Host reference for the CUDA low-rank path: fp32 resident factors, 8 K-splits,
/// 64-wide ordered tiles, split reduction in order, then ordered rank contraction.
pub fn lowRankRef(allocator: std.mem.Allocator, x: []const f32, a: []const f32, b: []const f32, rows: usize, d: usize, r: usize) ![]f32 {
    if (x.len != rows * d or a.len != d * r or b.len != r * d or d == 0 or d % 512 != 0) return error.LowRankShape;
    const mid = try allocator.alloc(f32, rows * r);
    defer allocator.free(mid);
    const out = try allocator.alloc(f32, rows * d);
    errdefer allocator.free(out);
    for (0..rows) |m| for (0..r) |j| {
        var split_sums: [8]f32 = @splat(0);
        for (0..8) |split| {
            var tile = split * (d / 8);
            const end = tile + d / 8;
            while (tile < end) : (tile += 64) {
                for (tile..tile + 64) |k| split_sums[split] = @mulAdd(f32, x[m * d + k], a[k * r + j], split_sums[split]);
            }
        }
        var accum = split_sums[0];
        for (1..8) |split| accum += split_sums[split];
        mid[m * r + j] = accum;
    };
    for (0..rows) |m| for (0..d) |j| {
        var accum: f32 = 0;
        for (0..r) |k| accum = @mulAdd(f32, mid[m * r + k], b[k * d + j], accum);
        out[m * d + j] = accum;
    };
    return out;
}

/// Python MoE partial order: fp32 low-rank FMA chain seeded with shared, then routed + shared-plus-low-rank.
pub fn partialRef(routed: f32, shared: f32, low_rank: f32) f32 {
    return routed + (shared + low_rank);
}

const Recording = struct {
    const abi = cuda.abi;
    const Entry = struct { function: usize, grid: [3]u32, block: [3]u32, len: usize, bytes: [128]u8 };
    var calls: [32]Entry = undefined;
    var count: usize = 0;

    fn sizes(id: usize) []const usize {
        return switch (id) {
            1 => &.{ 8, 8, 8, 8, 8, 8, 8, 4 }, // router tile
            2 => &.{ 8, 8, 8, 4, 4 }, // router slice sum
            3 => &.{ 8, 8, 8, 8, 4, 4, 4, 4 }, // top-k
            4 => &.{ 8, 8, 4, 4 }, // SwiGLU
            5 => &.{ 8, 8, 8, 8, 8, 4, 4, 4, 4 }, // combine
            6 => &.{ 8, 8, 8, 8, 8, 8 }, // stand-in routed launch
            else => unreachable,
        };
    }

    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const id = @intFromPtr(f.?);
        var entry: Entry = .{ .function = id, .grid = .{ gx, gy, gz }, .block = .{ bx, by, bz }, .len = 0, .bytes = @splat(0) };
        for (sizes(id), 0..) |n, i| {
            @memcpy(entry.bytes[entry.len..][0..n], @as([*]const u8, @ptrCast(params.?[i].?))[0..n]);
            entry.len += n;
        }
        calls[count] = entry;
        count += 1;
        return abi.success;
    }

    fn function(d: *cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};

const RecordingProject = struct {
    fn run(_: ?*anyopaque, _: cuda.Stream, _: u64, _: usize, _: usize, _: cuda.qmmf.Weight, _: u64, _: cuda.qmmf.Out, _: u64) anyerror!void {}
};

const RecordingRouted = struct {
    d: *cuda.Driver,
    fn run(ctx: *anyopaque, stream: cuda.Stream, x: u64, stride: usize, picks: u64, weights: u64, rows: usize, out: u64) anyerror!void {
        const r: *RecordingRouted = @ptrCast(@alignCast(ctx));
        var args: cuda.Args = .{};
        inline for (.{ x, picks, weights, out, @as(i64, @intCast(rows)), @as(i64, @intCast(stride)) }) |v| args.add(v);
        try cuda.launch.launch(Recording.function(r.d, 6), .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 128, .y = 1, .z = 1 } }, stream, &args);
    }
};

test "MoE null profile hooks preserve launches around routed work" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Recording.launch;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const route_k: RouterKernels = .{ .router_part = Recording.function(&d, 1), .router_tile = Recording.function(&d, 1), .topk = Recording.function(&d, 3), .slice_sum = Recording.function(&d, 2) };
    var lane: cuda.qmmf.Lane = undefined;
    const mlp_k: mlp.Kernels = .{ .linear = &lane, .swiglu_plain = Recording.function(&d, 4), .project_ctx = null, .project_fn = RecordingProject.run };
    const w: mlp.Weights = .{
        .gate_up = .{ .mode = .fp8b, .codes = 0x8000, .scales = 0x9000, .n = 2048, .k = hidden, .npad = 2048 },
        .down = .{ .mode = .fp8b, .codes = 0xa000, .scales = 0xb000, .n = hidden, .k = 1024, .npad = hidden },
    };
    const scratch: mlp.Scratch = .{ .gate_up = 0xc000, .activation = 0xd000, .mm_part = 0xe000 };
    const k: Kernels = .{ .lowrank = .{ .partial = Recording.function(&d, 1), .reduce = Recording.function(&d, 2), .finish = Recording.function(&d, 3), .decode = Recording.function(&d, 1) }, .embed_pick = Recording.function(&d, 3), .lowrank_combine_bf16 = Recording.function(&d, 5), .lowrank_combine_bf16_decode = Recording.function(&d, 5), .residual_add_bf16 = Recording.function(&d, 5) };
    var routed: RecordingRouted = .{ .d = &d };
    const common = .{ .x = @as(u64, 0x10000), .x_stride = hidden, .router = @as(u64, 0x20000), .bias = @as(u64, 0x30000), .logits = @as(u64, 0x40000), .router_part = @as(u64, 0x50000), .picks = @as(u64, 0x60000), .weights = @as(u64, 0x70000), .shared_out = @as(u64, 0x80000), .rows = @as(usize, 32), .routed_out = @as(u64, 0x90000), .moe_out = @as(u64, 0xa0000) };
    Recording.count = 0;
    try block(k, route_k, mlp_k, stream, w, scratch, &routed, RecordingRouted.run, common.x, common.x_stride, common.router, common.bias, common.logits, common.router_part, common.picks, common.weights, common.shared_out, common.rows, common.routed_out, common.moe_out, null);
    const whole = Recording.calls[0..Recording.count];
    var reference: [32]Recording.Entry = undefined;
    @memcpy(reference[0..whole.len], whole);
    const whole_len = whole.len;

    Recording.count = 0;
    try blockWithHooks(k, route_k, mlp_k, stream, w, scratch, &routed, RecordingRouted.run, common.x, common.x_stride, common.router, common.bias, common.logits, common.router_part, common.picks, common.weights, common.shared_out, common.rows, common.routed_out, common.moe_out, null, null);
    try std.testing.expectEqual(whole_len, Recording.count);
    for (reference[0..whole_len]) |expected| {
        var matches: usize = 0;
        for (Recording.calls[0..Recording.count]) |actual| {
            if (expected.function == actual.function and std.mem.eql(u32, &expected.grid, &actual.grid) and std.mem.eql(u32, &expected.block, &actual.block) and expected.len == actual.len and std.mem.eql(u8, expected.bytes[0..expected.len], actual.bytes[0..actual.len])) matches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), matches);
    }

    Recording.count = 0;
    try blockPre(k, route_k, mlp_k, stream, w, scratch, common.x, common.x_stride, common.router, common.bias, common.logits, common.router_part, common.picks, common.weights, common.shared_out, common.rows, null);
    const pre_len = Recording.count;
    try RecordingRouted.run(&routed, stream, common.x, common.x_stride, common.picks, common.weights, common.rows, common.routed_out);
    const routed_at = Recording.count - 1;
    try blockPost(k, stream, common.routed_out, common.shared_out, null, common.rows, common.moe_out);
    try std.testing.expect(routed_at >= pre_len and routed_at < Recording.count - 1);
    try std.testing.expectEqual(whole_len, Recording.count);
    for (reference[0..whole_len]) |expected| {
        var matches: usize = 0;
        for (Recording.calls[0..Recording.count]) |actual| {
            if (expected.function == actual.function and std.mem.eql(u32, &expected.grid, &actual.grid) and std.mem.eql(u32, &expected.block, &actual.block) and expected.len == actual.len and std.mem.eql(u8, expected.bytes[0..expected.len], actual.bytes[0..actual.len])) matches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), matches);
    }
}

test "MoE combines routed plus shared and low-rank in Python order" {
    try std.testing.expectEqual(@as(f32, 0), partialRef(1.0e20, -1.0e20, 3));
    try std.testing.expectEqual(bf16Bits(0), combineBf16Ref(1.0e20, -1.0e20, 3));
    try std.testing.expect(bf16Bits(3) != combineBf16Ref(1.0e20, -1.0e20, 3));
}

test "MoE plan uses 8 router slices and a ninth shared slot" {
    const p = try Plan.of(3, lowrank / 2);
    try std.testing.expectEqual(@as(usize, 3 * 256 * 8), p.router_part_floats);
    try std.testing.expectEqual(@as(usize, 27), p.pick_i32);
    try std.testing.expectEqual(@as(usize, 3 * 9 * hidden), p.expert_slots);
    try std.testing.expectEqual(@as(usize, 3 * (lowrank / 2) * 8), p.lowrank_mid_floats);
    const writes = 3 * (lowrank / 2) * 8;
    var too_small: [3 * (lowrank / 2)]f32 = undefined;
    try std.testing.expectError(error.LowRankScratchTooSmall, mockLowRankPartialWrite(&too_small, 3, lowrank / 2));
    var bounded_mid: [3 * (lowrank / 2) * 8]f32 = undefined;
    try mockLowRankPartialWrite(&bounded_mid, 3, lowrank / 2);
    try std.testing.expectEqual(p.lowrank_mid_floats, bounded_mid.len);
    try std.testing.expectEqual(@as(f32, @floatFromInt(writes - 1)), bounded_mid[writes - 1]);
    try std.testing.expectEqual(@as(usize, 8), shared_slot);
}

test "low-rank plan rejects overflow and supports only the CUDA split shape" {
    try std.testing.expectError(error.Overflow, Plan.of(std.math.maxInt(usize), lowrank));
    try std.testing.expectError(error.LowRankShape, lowRankAdd(undefined, undefined, 0, 0, 0, 0, 0, 1, lowrank + 1, hidden, lowrank + 1, hidden));
}

test "force semantic analysis of every MoE declaration, including route and block" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plan);
    std.testing.refAllDecls(Route);
    std.testing.refAllDecls(RouterKernels);
    std.testing.refAllDecls(LowRankKernels);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(LowRankWeights);
}

test "router ties go to lower ids and weights use unbiased sigmoid scores" {
    var logits: [experts]f32 = @splat(0);
    var bias: [experts]f32 = @splat(0);
    bias[3] = 1;
    const got = try routeRef(&logits, &bias, 2.5);
    try std.testing.expectEqual(@as(u16, 3), got.pick[0]);
    try std.testing.expectEqual(@as(u16, 0), got.pick[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3125), got.weights[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), sumVals(&got.weights), 1e-6);
}

fn sumVals(values: []const f32) f32 {
    var total: f32 = 0;
    for (values) |v| total += v;
    return total;
}

test "two TP2 low-rank slices sum to the full fp32 product" {
    const d = 512;
    const r = 4;
    const x = try std.testing.allocator.alloc(f32, d);
    defer std.testing.allocator.free(x);
    const a = try std.testing.allocator.alloc(f32, d * r);
    defer std.testing.allocator.free(a);
    const b = try std.testing.allocator.alloc(f32, r * d);
    defer std.testing.allocator.free(b);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.125;
    for (a, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.0625;
    for (b, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.25;
    const full = try lowRankRef(std.testing.allocator, x, a, b, 1, d, r);
    defer std.testing.allocator.free(full);
    const sum_parts = try std.testing.allocator.alloc(f32, d);
    defer std.testing.allocator.free(sum_parts);
    @memset(sum_parts, 0);
    for (0..2) |rank| {
        var ar: [d * 2]f32 = undefined;
        var br: [2 * d]f32 = undefined;
        for (0..d) |i| for (0..2) |j| {
            ar[i * 2 + j] = a[i * r + rank * 2 + j];
        };
        for (0..2) |i| for (0..d) |j| {
            br[i * d + j] = b[(rank * 2 + i) * d + j];
        };
        const part = try lowRankRef(std.testing.allocator, x, &ar, &br, 1, d, 2);
        defer std.testing.allocator.free(part);
        for (0..d) |i| sum_parts[i] += part[i];
    }
    for (0..d) |i| try std.testing.expectApproxEqAbs(full[i], sum_parts[i], 1e-6);
}
