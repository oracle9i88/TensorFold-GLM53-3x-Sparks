// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
// Adapted from bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/forward.py (Apache-2.0). Changes: Zig FP8B launch path and host reference for TP2 dense MLP slices.
//! GLM-5.3 DSA shared expert and dense MLP on the FP8B linear path (WP8).
const std = @import("std");
const cuda = @import("cuda");

pub const hidden = 6144;
pub const dense_width = 6144; // half of 12,288 at TP2
pub const shared_width = 1024; // half of 2,048 at TP2

pub const Plan = struct {
    rows: usize,
    gate_up_bf16: usize,
    activation_bf16: usize,
    output_f32: usize,

    pub fn dense(rows: usize) !Plan {
        if (rows == 0) return error.BadMlpPlan;
        return .{ .rows = rows, .gate_up_bf16 = rows * 2 * dense_width, .activation_bf16 = rows * dense_width, .output_f32 = rows * hidden };
    }

    pub fn shared(rows: usize) !Plan {
        if (rows == 0) return error.BadMlpPlan;
        return .{ .rows = rows, .gate_up_bf16 = rows * 2 * shared_width, .activation_bf16 = rows * shared_width, .output_f32 = rows * hidden };
    }
};

pub const Scratch = struct { gate_up: u64, activation: u64, mm_part: u64 };
pub const Weights = struct { gate_up: cuda.qmmf.Weight, down: cuda.qmmf.Weight };
pub const Kernels = struct {
    linear: *const cuda.qmmf.Lane,
    swiglu_plain: cuda.Function,
    project_ctx: ?*anyopaque = null,
    project_fn: ?*const fn (?*anyopaque, cuda.Stream, u64, usize, usize, cuda.qmmf.Weight, u64, cuda.qmmf.Out, u64) anyerror!void = null,

    fn project(k: Kernels, stream: cuda.Stream, x: u64, ldx: usize, rows: usize, w: cuda.qmmf.Weight, y: u64, out: cuda.qmmf.Out, part: u64) !void {
        if (k.project_fn) |run| return run(k.project_ctx, stream, x, ldx, rows, w, y, out, part);
        return k.linear.matmulOut(stream, x, ldx, rows, w, y, out, part);
    }
};

/// Plain SwiGLU with no clamp: bf16(silu(g) rounded to bf16 times bf16(u)).
pub fn swigluPlain(f: cuda.Function, stream: cuda.Stream, gu: u64, out: u64, rows: usize, width: usize) !void {
    if (rows == 0 or width == 0 or gu % 16 != 0 or out % 16 != 0) return error.BadSwiGLUShape;
    var args: cuda.Args = .{};
    inline for (.{ gu, out, @as(c_int, @intCast(rows)), @as(c_int, @intCast(width)) }) |v| args.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * width + 255) / 256) }, .block = .{ .x = 256 } }, stream, &args);
}

/// Execute a rank's dense MLP split: [gate | up] -> plain SwiGLU -> fp32 down partial.
pub fn block(k: Kernels, stream: cuda.Stream, w: Weights, s: Scratch, x: u64, rows: usize, width: usize, out: u64) !void {
    if (width % 128 != 0 or width == 0) return error.BadMlpWidth;
    if ((w.gate_up.mode != .fp8b and w.gate_up.mode != .fp4) or (w.down.mode != .fp8b and w.down.mode != .fp4) or w.gate_up.k != hidden or w.down.n != hidden or w.down.k != width or w.gate_up.n != width * 2) return error.BadMlpWeights;
    try k.project(stream, x, hidden, rows, w.gate_up, s.gate_up, .bf16, s.mm_part);
    try swigluPlain(k.swiglu_plain, stream, s.gate_up, s.activation, rows, width);
    try k.project(stream, s.activation, width, rows, w.down, out, .f32, s.mm_part);
}

/// Host reference for one activation value, with the contract's bf16 rounding after SiLU and product.
pub fn swigluRef(gate: f32, up: f32) f32 {
    const silu_bf16 = bf16(tritonSilu(gate));
    return bf16(silu_bf16 * up);
}

/// Host operation sequence for Triton `tl.exp` and `/`: ex2.approx is modeled
/// with the host exp2 implementation; the CUDA kernel uses the exact PTX ops.
pub fn tritonSilu(gate: f32) f32 {
    const exp = @exp2(-gate * 1.4426950408889634074);
    return gate / (1.0 + exp);
}

pub fn bf16(value: f32) f32 {
    var bits: u32 = @bitCast(value);
    bits +%= 0x7fff + ((bits >> 16) & 1);
    return @bitCast(bits & 0xffff0000);
}

/// Host reference for one intermediate slice. Matrices use gate/up [width, hidden]
/// and down [hidden, width], with the split applied to the intermediate axis.
pub fn denseSplitRef(allocator: std.mem.Allocator, x: []const f32, gate: []const f32, up: []const f32, down: []const f32, d: usize, width: usize, lo: usize, hi: usize) ![]f32 {
    if (x.len != d or gate.len != width * d or up.len != width * d or down.len != d * width or lo >= hi or hi > width) return error.DenseSplitShape;
    const output = try allocator.alloc(f32, d);
    errdefer allocator.free(output);
    const act = try allocator.alloc(f32, hi - lo);
    defer allocator.free(act);
    for (lo..hi) |r| {
        var g: f32 = 0;
        var u: f32 = 0;
        for (0..d) |k| {
            g = @mulAdd(f32, x[k], gate[r * d + k], g);
            u = @mulAdd(f32, x[k], up[r * d + k], u);
        }
        act[r - lo] = swigluRef(bf16(g), bf16(u));
    }
    for (0..d) |j| {
        var accum: f32 = 0;
        for (lo..hi) |r| accum = @mulAdd(f32, act[r - lo], down[j * width + r], accum);
        output[j] = accum;
    }
    return output;
}

test "dense and shared MLP plans use TP2 intermediate halves" {
    const d = try Plan.dense(8);
    const s = try Plan.shared(8);
    try std.testing.expectEqual(@as(usize, 8 * 12288), d.gate_up_bf16);
    try std.testing.expectEqual(@as(usize, 8 * 6144), d.activation_bf16);
    try std.testing.expectEqual(@as(usize, 8 * 2048), s.gate_up_bf16);
    try std.testing.expectEqual(@as(usize, 8 * 6144), s.output_f32);
}

test "plain SwiGLU rounds SiLU before multiplying and leaves the up value unclamped" {
    const expected = bf16(bf16(2.0 / (1.0 + @exp2(-2.0 * 1.4426950408889634074))) * 8.0);
    try std.testing.expectEqual(expected, swigluRef(2.0, 8.0));
    try std.testing.expect(swigluRef(2.0, 100.0) > 8.0);
    try std.testing.expectEqual(@as(f32, 0), tritonSilu(0));
}

test "force semantic analysis of every MLP declaration" {
    std.testing.refAllDecls(@This());
}

test "two dense intermediate halves sum to the full MLP partial" {
    const d = 3;
    const width = 4;
    const x = [_]f32{ 1, -0.5, 2 };
    const gate = [_]f32{ 0.5, 1, 0, 1, -1, 0.5, 1, 0, 0.25, -0.5, 1, 1 };
    const up = [_]f32{ 1, 0, 0.5, 1, 0.25, 1, 1, -0.5, 1, 1, 0, 0.5 };
    const down = [_]f32{ 1, 0.5, -0.5, 2, 0, 1, 1, -1, 0.5, 0, 2, 1 };
    try std.testing.expectEqual(@as(usize, d), x.len);
    try std.testing.expectEqual(@as(usize, width * d), gate.len);
    try std.testing.expectEqual(@as(usize, width * d), up.len);
    try std.testing.expectEqual(@as(usize, d * width), down.len);
    const full = try denseSplitRef(std.testing.allocator, &x, &gate, &up, &down, d, width, 0, width);
    defer std.testing.allocator.free(full);
    const left = try denseSplitRef(std.testing.allocator, &x, &gate, &up, &down, d, width, 0, width / 2);
    defer std.testing.allocator.free(left);
    const right = try denseSplitRef(std.testing.allocator, &x, &gate, &up, &down, d, width, width / 2, width);
    defer std.testing.allocator.free(right);
    for (0..d) |i| try std.testing.expectApproxEqAbs(full[i], left[i] + right[i], 1e-6);
}
