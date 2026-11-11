// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! The common EXL3 prompt path, after @BobClawblaw's #976 design and MiaAI-Lab's transposed kernels.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const launch = @import("launch.zig");
const grouped = @import("grouped.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const pmx_mod = @import("exl3_pmx.zig");
const p5a_mod = @import("exl3_p5a.zig");

pub const tile = 64;
/// Stream order: route writes the plan; rotate writes xg/xu; =1 reads xg/xu and writes xd/y.
/// =2 small GU reads xg/xu and writes xd/y, then wide up reads xu and writes y as uu, wide gate reads xg/y
/// and writes xd, and down reads xd and writes y. combinePrompt reads y and writes its caller ids/out; ids is also
/// cleared before that kernel. All operations use the same stream.
pub const Scratch = struct { xg: u64, xu: u64, xd: u64, y: u64, uu: u64, ids: u64, rows: usize, slots: usize };
pub fn eligibleWidths(k2: []const i32) bool {
    if (k2.len == 0) return false;
    for (k2) |v| if (v < 2 or v > 8) return false;
    return true;
}
pub fn eligible(l: ex.Layer) bool {
    return eligibleTiles(l) and l.width <= l.dims;
}

/// Shapes accepted by the original 64-member path. The wide path has an additional scratch layout constraint.
pub fn eligibleTiles(l: ex.Layer) bool {
    return (l.codebook == .mcg or l.codebook == .mcg_e4m3 or l.codebook == .mul1) and l.widths_ok and l.gu_stride == 0 and l.dims > 0 and l.width > 0 and l.dims % 256 == 0 and l.width % 128 == 0 and (l.act_mode == 0 or l.act_mode == 1);
}

/// TF_GLM53_PROMPT_EXPERTS=2 (exl3_prompt.cuh's wide path): a 128-member plan. Items of 64 members or fewer run the
/// 64-member kernels unchanged (the _s entries); larger items run 8-warp kernels with one 16x16 weight tile a warp over
/// all of the item's rows, one 128-column block of one matrix a CTA (up, then gate, then down). Same output bits.
pub const wide_tile = 128;
pub const wide_threads: u32 = 256; // 8 warps; 64 fp32 accumulators a thread; 2 CTAs an SM
/// Dynamic shared memory of wup/wgate/wdown: 3 stages of 128 X rows and 8 tiles' trellis words
/// (exl3_prompt.cuh wsmem_bytes).
pub const wide_smem: u32 = 3 * (wide_tile * 40 * 2 + 8 * 2 * 32 * 4);
/// The wide path's kernels of the exl3_experts image, [0] mcg, [1] mcg_e4m3, [2] mul1.
pub const Wide = struct {
    gu_small: [3]Function,
    down_small: [3]Function,
    up: [3]Function,
    gate: [3]Function,
    down: [3]Function,

    pub fn resolve(mod: Module) !Wide {
        const w: Wide = .{
            .gu_small = .{ try mod.function("tf_exl3_prompt_gu_s"), try mod.function("tf_exl3_prompt_gu_s_e4m3"), try mod.function("tf_exl3_prompt_gu_s_mul1") },
            .down_small = .{ try mod.function("tf_exl3_prompt_down_s"), try mod.function("tf_exl3_prompt_down_s_e4m3"), try mod.function("tf_exl3_prompt_down_s_mul1") },
            .up = .{ try mod.function("tf_exl3_prompt_wup"), try mod.function("tf_exl3_prompt_wup_e4m3"), try mod.function("tf_exl3_prompt_wup_mul1") },
            .gate = .{ try mod.function("tf_exl3_prompt_wgate"), try mod.function("tf_exl3_prompt_wgate_e4m3"), try mod.function("tf_exl3_prompt_wgate_mul1") },
            .down = .{ try mod.function("tf_exl3_prompt_wdown"), try mod.function("tf_exl3_prompt_wdown_e4m3"), try mod.function("tf_exl3_prompt_wdown_mul1") },
        };
        for ([_][3]Function{ w.up, w.gate, w.down }) |triple| {
            for (triple) |f| try f.allowDynamicShared(wide_smem);
        }
        return w;
    }
};

fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.Exl3Shape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.Exl3Shape;
}
fn aligned(n: usize) !usize {
    return (try add(n, 255)) / 256 * 256;
}
fn sizes(l: ex.Layer, rows: usize, slots: usize, overlay: bool) ![6]usize {
    if (!eligibleTiles(l) or rows == 0 or slots == 0 or slots > 32 or l.experts == 0 or l.experts > grouped.max_experts) return error.Exl3Shape;
    const pairs = try mul(rows, slots);
    if (pairs > std.math.maxInt(i32) or l.dims > std.math.maxInt(i32) or l.width > std.math.maxInt(i32)) return error.Exl3Shape;
    const rotated = try aligned(try mul(try mul(if (l.shared_input) rows else pairs, l.dims), 2));
    const xu = if (l.shared_input) 0 else rotated;
    const xd = try aligned(try mul(try mul(pairs, l.width), 2));
    const y = try aligned(try mul(try mul(pairs, l.dims), 4));
    _ = overlay;
    return .{ rotated, xu, xd, y, 0, 0 };
}
pub fn bytes(l: ex.Layer, rows: usize, slots: usize, overlay: bool) !usize {
    const ns = try sizes(l, rows, slots, overlay);
    var sum: usize = 0;
    for (ns) |n| sum = try add(sum, n);
    return sum;
}
pub fn scratch(base: u64, l: ex.Layer, rows: usize, slots: usize, overlay: bool) !Scratch {
    const ns = try sizes(l, rows, slots, overlay);
    const total = try bytes(l, rows, slots, overlay);
    _ = std.math.add(u64, base, total) catch return error.Exl3Shape;
    const xg = base;
    const xu = if (l.shared_input) base else base + ns[0];
    const xd = base + ns[0] + ns[1];
    const y = base + ns[0] + ns[1] + ns[2];
    return .{ .xg = xg, .xu = xu, .xd = xd, .y = y, .uu = y, .ids = xg, .rows = rows, .slots = slots };
}
fn go(f: Function, s: Stream, grid: [3]usize, threads: u32, values: anytype) !void {
    var args: launch.Args = .{};
    inline for (values) |v| args.add(v);
    try launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads } }, s, &args);
}
fn goShared(f: Function, s: Stream, grid: [3]usize, threads: u32, shared: u32, values: anytype) !void {
    var args: launch.Args = .{};
    inline for (values) |v| args.add(v);
    try launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = shared }, s, &args);
}
fn i(n: usize) i32 {
    return @intCast(n);
}
pub const Kernels = struct {
    rotate: Function,
    gateup: Function,
    down: Function,
    gateup_e4m3: Function,
    down_e4m3: Function,
    gateup_mul1: Function,
    down_mul1: Function,
    /// The TF_GLM53_PROMPT_EXPERTS=2 kernels; null where a caller builds Kernels by hand without them.
    wide: ?Wide = null,
    pub fn resolve(mod: Module) !Kernels {
        return .{
            .rotate = try mod.function("tf_exl3_prompt_rotate"),
            .gateup = try mod.function("tf_exl3_prompt_gu"),
            .down = try mod.function("tf_exl3_prompt_down"),
            .gateup_e4m3 = try mod.function("tf_exl3_prompt_gu_e4m3"),
            .down_e4m3 = try mod.function("tf_exl3_prompt_down_e4m3"),
            .gateup_mul1 = try mod.function("tf_exl3_prompt_gu_mul1"),
            .down_mul1 = try mod.function("tf_exl3_prompt_down_mul1"),
            .wide = Wide.resolve(mod) catch null,
        };
    }
    pub fn run(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        return k.runTile(e, router, s, l, x, stride, picks, rows, sc, p, tile);
    }
    /// Run with a family-selected plan tile; GLM-5.3's tensor-core kernels use a 64-pair tile.
    pub fn runTile(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan, route_tile: usize) !void {
        try k.prepare(e, router, s, l, x, stride, picks, rows, sc, p, route_tile);
        try k.gateUpTiles(s, l, rows, sc, p, route_tile);
        try k.downTiles(s, l, rows, sc, p, route_tile);
    }

    /// TF_GLM53_PROMPT_EXPERTS=2: the same rotation over a 128-member plan, then the wide gate/up and down. y gets
    /// runTile's bits: equal decoded tiles, X rows and k order, and an output never depends on its item.
    pub fn runWide(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (l.width > l.dims) return error.Exl3Shape;
        if (k.wide == null) return error.Exl3PromptWideMissing;
        try k.prepare(e, router, s, l, x, stride, picks, rows, sc, p, wide_tile);
        try k.gateUpWide(s, l, rows, sc, p);
        try k.downWide(s, l, rows, sc, p);
    }

    /// =3 uses the same 128-member plan and rotation as =2. Small items retain =1 gate/up;
    /// larger items use fused PMX gate/up, followed by =2 down and combine.
    pub fn runPmx(k: Kernels, pmx: *const pmx_mod.Pmx, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (l.widths != 0) return error.Exl3PerExpertWidthsUnsupported; // pmx.cu reads one width for every expert
        if (l.width > l.dims) return error.Exl3Shape;
        if (k.wide == null) return error.Exl3PromptWideMissing;
        try k.prepare(e, router, s, l, x, stride, picks, rows, sc, p, wide_tile);
        const w = k.wide.?;
        const c: usize = if (l.codebook == .mcg_e4m3) 1 else 0;
        const items = grouped.maxItems(rows * sc.slots, l.experts, wide_tile);
        try go(w.gu_small[c], s, .{ l.width / 128, items, 1 }, 256, guArgs(l, sc, p));
        var args: launch.Args = .{};
        const a: pmx_mod.PmxArgs = .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = l.gate, .tp1 = l.up, .k2_0 = l.gate_k2, .k2_1 = l.up_k2, .items = p.items, .counts = p.counts, .members = p.members, .sv0 = l.svh_g, .sv1 = l.svh_u, .sd = l.suh_d, .xd = sc.xd, .y = sc.y, .K = i(l.dims), .N = i(l.width), .E = i(l.experts), .limit = l.limit, .act_mode = l.act_mode };
        args.add(a);
        const pmx_fn = pmx.gu[c];
        try launch.launch(pmx_fn, .{ .grid = .{ .x = @intCast(l.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @intCast(pmx_mod.gu_smem) }, s, &args);
        try k.downWide(s, l, rows, sc, p);
    }

    /// TF_GLM53_PROMPT_EXPERTS=10 gate/up: =3's 128-member plan; items <= 64 members on p5a, larger ones on =3 PMX.
    pub fn gateUpP5a(k: Kernels, pmx: *const pmx_mod.Pmx, small: *const p5a_mod.P5a, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        _ = k;
        if (l.widths != 0) return error.Exl3PerExpertWidthsUnsupported; // p5a.cu and pmx.cu read one width
        const c: usize = if (l.codebook == .mcg_e4m3) 1 else 0;
        const items = grouped.maxItems(rows * sc.slots, l.experts, wide_tile);
        try go(small.gu[c], s, .{ l.width / 128, items, 1 }, p5a_mod.threads, guArgs(l, sc, p));
        var args: launch.Args = .{};
        const a: pmx_mod.PmxArgs = .{ .x0 = sc.xg, .x1 = sc.xu, .tp0 = l.gate, .tp1 = l.up, .k2_0 = l.gate_k2, .k2_1 = l.up_k2, .items = p.items, .counts = p.counts, .members = p.members, .sv0 = l.svh_g, .sv1 = l.svh_u, .sd = l.suh_d, .xd = sc.xd, .y = sc.y, .K = i(l.dims), .N = i(l.width), .E = i(l.experts), .limit = l.limit, .act_mode = l.act_mode };
        args.add(a);
        try launch.launch(pmx.gu[c], .{ .grid = .{ .x = @intCast(l.width / 128), .y = @intCast(items), .z = 1 }, .block = .{ .x = 512 }, .shared = @intCast(pmx_mod.gu_smem) }, s, &args);
    }

    /// =10 down: items <= 64 members on p5a, larger ones on =2's wide down.
    pub fn downP5a(k: Kernels, small: *const p5a_mod.P5a, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (l.widths != 0) return error.Exl3PerExpertWidthsUnsupported; // p5a.cu reads one width
        const w = k.wide orelse return error.Exl3PromptWideMissing;
        const c: usize = if (l.codebook == .mcg_e4m3) 1 else 0;
        const items = grouped.maxItems(rows * sc.slots, l.experts, wide_tile);
        try go(small.down[c], s, .{ l.dims / 256, items, 1 }, p5a_mod.threads, downArgs(l, sc, p));
        try goShared(w.down[c], s, .{ l.dims / 128, items, 1 }, wide_threads, wide_smem, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), l.down_k2, l.widths });
    }

    /// =10 is =3 (same plan, rotation, PMX gate/up and wide down) with p5a for its small items; every output has =3's
    /// bits, which are =1's (pmx.cu rounds uu as tgateup does).
    pub fn runP5a(k: Kernels, pmx: *const pmx_mod.Pmx, small: *const p5a_mod.P5a, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (l.width > l.dims or l.shared_input) return error.Exl3Shape;
        if (k.wide == null) return error.Exl3PromptWideMissing;
        if ((try mul(try mul(rows, sc.slots), l.dims)) > std.math.maxInt(i32)) return error.Exl3Shape;
        try k.prepare(e, router, s, l, x, stride, picks, rows, sc, p, wide_tile);
        try k.gateUpP5a(pmx, small, s, l, rows, sc, p);
        try k.downP5a(small, s, l, rows, sc, p);
    }

    /// Phase 1 of both paths: shape checks, the plan at `route_tile`, the rotated gate/up inputs.
    pub fn prepare(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan, route_tile: usize) !void {
        if (!eligibleTiles(l) or rows == 0 or rows > sc.rows or stride < l.dims or stride > std.math.maxInt(i32)) return error.Exl3Shape;
        if (route_tile == 0) return error.Exl3Shape;
        _ = try bytes(l, rows, sc.slots, false); // shape check only; the overlay changes sizes, not shapes
        const pairs = rows * sc.slots;
        try router.route(s, picks, pairs, l.experts, route_tile, p);
        if (l.shared_input) {
            try go(k.rotate, s, .{ rows, l.dims / 128, 1 }, 32, .{ x, i(stride), l.suh_g, sc.xg, i(l.dims) });
        } else {
            var args: launch.Args = .{};
            inline for (.{ x, i(stride), @as(i32, @backingInt(ex.DType.bf16)), picks, l.suh_g, l.suh_u, sc.xg, sc.xu, i(l.dims), i(sc.slots), i(l.experts) }) |v| args.add(v);
            try launch.launch(e.rot_in, .{ .grid = .{ .x = @intCast(pairs), .y = @intCast(l.dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &args);
        }
    }

    /// =1's gate/up (SwiGLU and the down rotation in the epilogue) over a plan made at `route_tile`.
    pub fn gateUpTiles(k: Kernels, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan, route_tile: usize) !void {
        const items = grouped.maxItems(rows * sc.slots, l.experts, route_tile);
        const gateup = switch (l.codebook) {
            .mcg_e4m3 => k.gateup_e4m3,
            .mul1 => k.gateup_mul1,
            .mcg => k.gateup,
            else => return error.Exl3Shape,
        };
        try go(gateup, s, .{ l.width / 128, items, 1 }, 256, guArgs(l, sc, p));
    }

    /// =1's down over a plan made at `route_tile`.
    pub fn downTiles(k: Kernels, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan, route_tile: usize) !void {
        const items = grouped.maxItems(rows * sc.slots, l.experts, route_tile);
        const down = switch (l.codebook) {
            .mcg_e4m3 => k.down_e4m3,
            .mul1 => k.down_mul1,
            .mcg => k.down,
            else => return error.Exl3Shape,
        };
        try go(down, s, .{ l.dims / 256, items, 1 }, 256, downArgs(l, sc, p));
    }

    /// =2's gate/up: items <= 64 members on =1's kernel, larger ones up (uu into the y region) then gate (xd).
    pub fn gateUpWide(k: Kernels, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        const w = k.wide orelse return error.Exl3PromptWideMissing;
        const c: usize = switch (l.codebook) {
            .mcg => 0,
            .mcg_e4m3 => 1,
            .mul1 => 2,
            else => return error.Exl3Shape,
        };
        const items = grouped.maxItems(rows * sc.slots, l.experts, wide_tile);
        const xdiv = i(if (l.shared_input) sc.slots else 1);
        try go(w.gu_small[c], s, .{ l.width / 128, items, 1 }, 256, guArgs(l, sc, p));
        try goShared(w.up[c], s, .{ l.width / 128, items, 1 }, wide_threads, wide_smem, .{ sc.xu, l.up, p.items, p.counts, p.members, l.svh_u, sc.uu, i(l.dims), i(l.width), xdiv, l.act_mode, l.up_k2, l.widths });
        try goShared(w.gate[c], s, .{ l.width / 128, items, 1 }, wide_threads, wide_smem, .{ sc.xg, l.gate, p.items, p.counts, p.members, l.svh_g, l.suh_d, sc.uu, sc.xd, i(l.dims), i(l.width), xdiv, l.limit, l.act_mode, l.gate_k2, l.widths });
    }

    /// =2's down: items <= 64 members on =1's kernel, larger ones on 128-column wide blocks.
    pub fn downWide(k: Kernels, s: Stream, l: ex.Layer, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        const w = k.wide orelse return error.Exl3PromptWideMissing;
        const c: usize = switch (l.codebook) {
            .mcg => 0,
            .mcg_e4m3 => 1,
            .mul1 => 2,
            else => return error.Exl3Shape,
        };
        const items = grouped.maxItems(rows * sc.slots, l.experts, wide_tile);
        try go(w.down_small[c], s, .{ l.dims / 256, items, 1 }, 256, downArgs(l, sc, p));
        try goShared(w.down[c], s, .{ l.dims / 128, items, 1 }, wide_threads, wide_smem, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), l.down_k2, l.widths });
    }
};

fn guArgs(l: ex.Layer, sc: Scratch, p: grouped.Plan) GuArgs {
    return .{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, i(l.dims), i(l.width), i(if (l.shared_input) sc.slots else 1), l.limit, l.act_mode, l.gate_k2, l.up_k2, l.widths };
}
// The last entry is the span table (exl3_prompt kernels' `widths`); p5a's small gate/up does not declare it and the
// driver passes only the parameters a kernel has.
const GuArgs = struct { u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, u64, i32, i32, i32, f32, i32, u64, u64, u64 };

fn downArgs(l: ex.Layer, sc: Scratch, p: grouped.Plan) DownArgs {
    return .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), i(sc.slots), l.down_k2, l.widths };
}
const DownArgs = struct { u64, u64, u64, u64, u64, u64, u64, i32, i32, i32, u64, u64 };

test "wide prompt kernels: tiles, threads and dynamic shared memory match exl3_prompt.cuh; 2 CTAs an SM fit" {
    try std.testing.expectEqual(@as(u32, 36_864), wide_smem);
    // The epilogue's fp32 sums (32 rows of 128 + 4 columns) reuse the stages.
    try std.testing.expect(32 * 132 * 4 <= wide_smem);
    // Two CTAs an SM with their static 128 member rows and virtual-block coordinates (100 KB an SM on sm_86/89/120/121).
    try std.testing.expect(2 * (wide_smem + wide_tile * 4 + 64) <= 102_400);
    try std.testing.expectEqual(@as(u32, 256), wide_threads);
    try std.testing.expect(grouped.maxItems(16384, 256, wide_tile) <= grouped.maxItems(16384, 256, tile)); // the 64-tile plan carve holds it
}

test "p5a prompt member tile matches =1" {
    try std.testing.expectEqual(tile, p5a_mod.small_max);
}
