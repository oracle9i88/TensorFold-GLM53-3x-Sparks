// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! The lane matmul (qmmf.cu) for FP4, FP8, MXFP8, FP8G, FP8B and BF16 weights, with its kernel, tile and slices.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

pub const group = 64; // inputs a kernel stage reads
pub const fused_rows = 256; // rows from which a projection's K slices meet in one block (linear.py FUSED_ROWS)
const bn = 64;
const threads = 128; // WM 1 x WN 4 warps

/// The packed formats (qmmf.cu Mode): FP4, e4m3, MXFP8, FP8G, native block-FP8 and bf16.
pub const Mode = enum(u3) { fp4 = 0, fp8 = 1, mxfp8 = 2, fp8g = 3, bf16 = 4, fp8b = 5 };

const modes = [_]Mode{ .fp4, .fp8, .mxfp8, .fp8g, .bf16, .fp8b };

fn modeIndex(mode: Mode) usize {
    return std.mem.indexOfScalar(Mode, &modes, mode).?;
}

/// The dynamic shared bytes of a BM-row tile (qmmf.cu Tile<MODE, BM, 64, 1, 4, 4>::SMEM).
pub fn sharedBytes(mode: Mode, bm: u32) u32 {
    const w: u32 = switch (mode) {
        .fp4 => bn * group / 2,
        .fp8, .mxfp8, .fp8g, .fp8b => bn * group,
        .bf16 => bn * group * 2,
    };
    const s: u32 = if (mode == .fp4 or mode == .fp8g) bn * 4 else if (mode == .mxfp8) bn * 2 else 0;
    const stage = (bm * group * 2 + w + s + 127) / 128 * 128;
    const partials = (bm / 16) * 2 * 4 * threads * 4;
    return @max(4 * stage, partials);
}

/// K slices for an (n, k) weight, from the shape alone so a row's bits never depend on the row count (qmm.split_k).
pub fn splitK(n: usize, k: usize) u32 {
    const tiles = (n + bn - 1) / bn;
    const groups = k / group;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return @intCast(sk);
}

/// GLM's Python `b16_split_k`: 64-column tiles and at least four 64-wide K steps per slice.
pub fn b16SplitK(n: usize, k: usize) u32 {
    const tiles = (n + bn - 1) / bn;
    const steps = k / group;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and steps % (sk * 2) == 0 and steps / (sk * 2) >= 4) sk *= 2;
    return @intCast(sk);
}

/// The row tile: 16 or 32 rows, else 64-row tiles side by side (qmm.bucket).
pub fn bucket(m: usize) u32 {
    return if (m <= 16) 16 else if (m <= 32) 32 else 64;
}

/// A packed projection on the device: codes or words, scale tiles, the tensor scale, its shape, its input scale (act).
pub const Weight = struct { mode: Mode, codes: u64, scales: u64, scale: f32 = 1.0, n: u32, k: u32, npad: u32, act: f32 = 0 };

/// What one call launches, from the shapes alone: tile, slices, cluster or reduce (qmmf_cuda's choice).
pub const Plan = struct { bm: u32, sk: u32, fused: bool, cluster: bool, reduce: bool };

pub fn plan(m: usize, n: usize, k: usize, clusters: bool) Plan {
    return planWithSplit(m, splitK(n, k), clusters);
}

/// Launch plan using the Python GLM rule for `.fp8b`; other modes retain qmmf's own split rule.
pub fn planForMode(m: usize, n: usize, k: usize, clusters: bool, mode: Mode) Plan {
    return planWithSplit(m, if (mode == .fp8b) b16SplitK(n, k) else splitK(n, k), clusters);
}

fn planWithSplit(m: usize, sk: u32, clusters: bool) Plan {
    const fused = sk > 1 and m >= fused_rows;
    const cluster = !fused and sk > 1 and sk <= 8 and clusters;
    return .{ .bm = if (fused) 64 else bucket(m), .sk = sk, .fused = fused, .cluster = cluster, .reduce = !fused and sk > 1 and !cluster };
}

/// qmmf.cu's instance names, per mode: row tiles 16, 32, 64 (each plain, then with a cluster), then the fused kernel.
fn symbol(
    comptime mode: Mode,
    comptime bm: u32,
    comptime cluster: bool,
    comptime fused: bool,
    comptime wide: bool,
) [:0]const u8 {
    const name = "_ZN7tf_qmmf11qmmf_kernelILi{d}ELi{d}ELi64ELi1ELi4ELi4ELb{d}ELb{d}ELb{d}" ++
        "EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    const flags = .{ @intFromBool(wide), @intFromBool(cluster), @intFromBool(fused) };
    return std.fmt.comptimePrint(name, .{ @backingInt(mode), bm, flags[0], flags[1], flags[2] });
}

pub const reduce_symbols = [2][:0]const u8{
    "_ZN7tf_qmmf13reduce_kernelILb0EEEvPKfPvxif",
    "_ZN7tf_qmmf13reduce_kernelILb1EEEvPKfPvxif",
};
const row_tiles = [_]u32{ 16, 32, 64 };

/// The BF16 64-row instance uses 64 KiB dynamic shared memory and opts in during Lane.load.
pub const launch_contract = struct {
    pub const max_dynamic_shared: u32 = sharedBytes(.bf16, 64);
    pub const opt_in_required = max_dynamic_shared > 48 * 1024;
};

/// What a call writes: bf16 rows, or the fp32 sums unrounded (a tensor-parallel rank's partial rows).
pub const Out = enum { bf16, f32 };

/// One output of Lane.matmulGroup: an FP8B weight over the shared input rows, written to `y` [m, n] bf16 (row stride n).
pub const GroupOut = struct { w: Weight, y: u64 };
pub const max_group = 4;
/// split: splitPlan instead of planForMode. prefetch/prefetch_bytes: device bytes each block pulls into L2 after its work.
pub const GroupOptions = struct { split: bool = false, prefetch: u64 = 0, prefetch_bytes: usize = 0 };
/// The launch of one grouped call: the shared plan, each weight's blocks in grid.x order, the kernel (group_symbols index).
pub const GroupPlan = struct { plan: Plan, blocks: [max_group]u32 = @splat(0), total: u32 = 0, band: u32 = 1, count: u32 = 0, kernel: usize = 0 };
pub const group_symbols = [4][:0]const u8{ "tf_qmmf_group_fp8b_16c", "tf_qmmf_group_fp8b_32c", "tf_qmmf_group_fp8b_64c", "tf_qmmf_group_fp8b_64f" };
const group_tiles = [4]u32{ 16, 32, 64, 64 };
/// qmmf.cu GroupSeg / GroupArgs (static_asserts there).
pub const GroupSegAbi = extern struct { w: u64 = 0, bs: u64 = 0, out: u64 = 0, scale: f32 = 0, n: i32 = 0, npad: i32 = 0, sk: i32 = 0, blocks: i32 = 0, pad: u32 = 0 };
pub const GroupArgsAbi = extern struct { x: u64, prefetch: u64, prefetch_bytes: i64, m: i32, k: i32, ldx: i32, band: i32, count: i32, pad: u32 = 0, seg: [max_group]GroupSegAbi };
comptime { std.debug.assert(@sizeOf(GroupSegAbi) == 48 and @sizeOf(GroupArgsAbi) == 240 and @offsetOf(GroupArgsAbi, "seg") == 48); }

pub fn splitPlan(m: usize, n: usize, k: usize, clusters: bool, mode: Mode) Plan {
    const sk = if (mode == .fp8b) b16SplitK(n, k) else splitK(n, k);
    return .{ .bm = bucket(m), .sk = sk, .fused = false, .cluster = sk > 1 and clusters, .reduce = sk > 1 and !clusters };
}

fn groupKernel(p: Plan) !usize {
    if (p.fused) return if (p.bm == 64) 3 else error.GroupPlanUnsupported;
    if (!p.cluster) return error.GroupPlanUnsupported;
    return switch (p.bm) { 16 => 0, 32 => 1, 64 => 2, else => error.GroupPlanUnsupported };
}

pub fn groupPlan(m: usize, k: usize, outs: []const GroupOut, clusters: bool, split: bool) !GroupPlan {
    if (outs.len == 0 or outs.len > max_group) return error.GroupSize;
    if (m == 0 or k == 0 or k % group != 0) return error.GroupShape;
    var g: GroupPlan = .{ .plan = undefined, .count = @intCast(outs.len) };
    for (outs, 0..) |o, i| {
        if (o.w.mode != .fp8b) return error.GroupMode;
        if (o.w.k != k or o.w.n == 0 or o.w.npad < o.w.n or o.y == 0 or o.w.codes == 0 or o.w.scales == 0) return error.GroupShape;
        const p = if (split) splitPlan(m, o.w.n, k, clusters, .fp8b) else planForMode(m, o.w.n, k, clusters, .fp8b);
        if (i == 0) g.plan = p else if (!std.meta.eql(p, g.plan)) return error.GroupPlanMismatch;
        g.blocks[i] = @intCast(((m + p.bm - 1) / p.bm) * ((o.w.n + bn - 1) / bn));
        g.total += g.blocks[i];
    }
    g.kernel = try groupKernel(g.plan);
    const rows_t = (m + g.plan.bm - 1) / g.plan.bm;
    g.band = @intCast(@max(1, @min(rows_t, (12 << 20) / (@as(usize, g.plan.bm) * k * 2))));
    return g;
}

fn loadTiles(mod: Module, comptime mode: Mode, comptime wide: bool) ![7]Function {
    var set: [7]Function = undefined;
    inline for (row_tiles, 0..) |bm, ti| inline for (.{ false, true }, 0..) |cl, ci| {
        const f = try mod.function(symbol(mode, bm, cl, false, wide));
        try f.allowDynamicShared(sharedBytes(mode, bm));
        set[ti * 2 + ci] = f;
    };
    set[6] = try mod.function(symbol(mode, 64, false, true, wide));
    try set[6].allowDynamicShared(sharedBytes(mode, 64));
    return set;
}

pub const Lane = struct {
    mod: Module,
    fns: [modes.len][7]Function, // per mode: lane16, lane16c, lane32, lane32c, lane64, lane64c, fused
    f32_fns: [modes.len][7]Function,
    reduce: [2]Function, // by Out
    dequant_fp8b_bf16: Function = undefined,
    dequant_fp8b_bf16_rowmajor: Function = undefined,
    group_fns: [4]?Function = .{ null, null, null, null },
    clusters: bool, // sm_90 on: slices meet in a cluster's shared memory

    pub fn load(d: *const Driver, capability_major: u32) !Lane {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.qmmf);
        errdefer mod.unload();
        var l: Lane = .{
            .mod = mod,
            .fns = undefined,
            .f32_fns = undefined,
            .reduce = undefined,
            .dequant_fp8b_bf16 = undefined,
            .dequant_fp8b_bf16_rowmajor = undefined,
            .clusters = capability_major >= 9,
        };
        inline for (modes, 0..) |mode, mi| {
            l.fns[mi] = try loadTiles(mod, mode, false);
            l.f32_fns[mi] = try loadTiles(mod, mode, true);
        }
        for (&l.reduce, reduce_symbols) |*f, name| f.* = try mod.function(name);
        l.dequant_fp8b_bf16 = try mod.function("tf_qmmf_fp8b_dequant_bf16");
        l.dequant_fp8b_bf16_rowmajor = try mod.function("tf_qmmf_fp8b_dequant_bf16_rowmajor");
        return l;
    }

    pub fn unload(l: *Lane) void {
        l.mod.unload();
    }

    /// Convert one FP8B projection to the packed BF16 layout for the W8A16 prefill path.
    /// The kernel applies the native block scale in fp32 and rounds once to BF16.
    pub fn dequantizeFp8bBf16(l: Lane, s: Stream, w: Weight, out: u64) !void {
        if (w.mode != .fp8b or w.n == 0 or w.k == 0 or w.k % group != 0 or out == 0) return error.InvalidWeight;
        var args: launch_.Args = .{};
        args.add(w.codes);
        args.add(w.scales);
        args.add(out);
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(w.npad)));
        const total = @as(usize, w.npad) * w.k;
        try launch_.launch(l.dequant_fp8b_bf16, .{
            .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// Convert a projection to row-major BF16 for cuBLASLt's prefill path (the default `.fp8b` lane stays packed).
    pub fn dequantizeFp8bBf16RowMajor(l: Lane, s: Stream, w: Weight, out: u64) !void {
        if (w.mode != .fp8b or w.n == 0 or w.k == 0 or w.k % group != 0 or out == 0) return error.InvalidWeight;
        var args: launch_.Args = .{};
        args.add(w.codes);
        args.add(w.scales);
        args.add(out);
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(w.npad)));
        const total = @as(usize, w.n) * w.k;
        try launch_.launch(l.dequant_fp8b_bf16_rowmajor, .{
            .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
            .block = .{ .x = 256, .y = 1, .z = 1 },
        }, s, &args);
    }

    /// Slice partials a call needs (fp32 [sk][m][n]) when it reduces them, else 0.
    pub fn partBytes(l: Lane, m: usize, w: Weight) usize {
        const p = planForMode(m, w.n, w.k, l.clusters, w.mode);
        return if (p.reduce) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// Scratch sizing when the split configuration is pinned to another output width.
    pub fn partBytesPinned(l: Lane, m: usize, w: Weight, pin_n: usize) usize {
        const p = planForMode(m, pin_n, w.k, l.clusters, w.mode);
        return if (p.reduce) @as(usize, p.sk) * m * w.n * 4 else 0;
    }

    /// y [m, n] bf16 = x [m, k] bf16 (rows `ldx` apart) times the weight, fp32 sums; `part` holds partBytes.
    pub fn matmul(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        return l.matmulOut(s, x, ldx, m, w, y, .bf16, part);
    }

    /// Preserve FP32 outputs for a later residual or tensor-rank sum.
    pub fn matmulF32(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, part: u64) !void {
        return l.matmulOut(s, x, ldx, m, w, y, .f32, part);
    }

    /// matmul() writing y as `out` says: bf16, or fp32.
    pub fn matmulOut(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64) !void {
        const p = planForMode(m, w.n, w.k, l.clusters, w.mode);
        return l.matmulOutWithPlan(s, x, ldx, m, w, y, out, part, p);
    }

    /// Use the plan selected for `pin_n` output columns while computing the actual `w.n` columns.
    pub fn matmulOutPinned(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64, pin_n: usize) !void {
        const p = planForMode(m, pin_n, w.k, l.clusters, w.mode);
        return l.matmulOutWithPlan(s, x, ldx, m, w, y, out, part, p);
    }

    pub fn loadGroup(l: *Lane) !void {
        for (&l.group_fns, group_symbols, group_tiles) |*slot, name, bm| {
            const f = try l.mod.function(name);
            try f.allowDynamicShared(sharedBytes(.fp8b, bm));
            slot.* = f;
        }
    }

    pub fn matmulGroup(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, outs: []const GroupOut, opt: GroupOptions) !void {
        if (outs.len == 0) return error.GroupSize;
        const k: usize = outs[0].w.k;
        const g = try groupPlan(m, k, outs, l.clusters, opt.split);
        const f = l.group_fns[g.kernel] orelse return error.MissingGroupKernels;
        var a: GroupArgsAbi = .{ .x = x, .prefetch = opt.prefetch, .prefetch_bytes = if (opt.prefetch == 0) 0 else @intCast(opt.prefetch_bytes), .m = @intCast(m), .k = @intCast(k), .ldx = @intCast(if (m == 1) k else ldx), .band = @intCast(g.band), .count = @intCast(outs.len), .seg = @splat(.{}) };
        for (outs, 0..) |o, i| {
            const own_plan = if (opt.split) splitPlan(m, o.w.n, k, l.clusters, o.w.mode) else planForMode(m, o.w.n, k, l.clusters, o.w.mode);
            a.seg[i] = .{ .w = o.w.codes, .bs = o.w.scales, .out = o.y, .scale = o.w.scale, .n = @intCast(o.w.n), .npad = @intCast(o.w.npad), .sk = @intCast(own_plan.sk), .blocks = @intCast(g.blocks[i]) };
        }
        var args: launch_.Args = .{};
        args.add(a);
        try launch_.launch(f, .{ .grid = .{ .x = g.total, .y = 1, .z = if (g.plan.fused) 1 else g.plan.sk }, .block = .{ .x = threads, .y = 1, .z = 1 }, .shared = sharedBytes(.fp8b, g.plan.bm), .cluster = if (g.plan.cluster) .{ .x = 1, .y = 1, .z = g.plan.sk } else null }, s, &args);
    }

    pub fn matmulOutSplit(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64) !void {
        if (!l.clusters) return error.SplitNeedsClusters;
        return l.matmulOutWithPlan(s, x, ldx, m, w, y, out, part, splitPlan(m, w.n, w.k, l.clusters, w.mode));
    }

    fn matmulOutWithPlan(l: Lane, s: Stream, x: u64, ldx: usize, m: usize, w: Weight, y: u64, out: Out, part: u64, p: Plan) !void {
        if (p.reduce and part == 0) return error.MissingLanePart;
        const set = (if (out == .f32) l.f32_fns else l.fns)[modeIndex(w.mode)];
        const ti: usize = if (p.bm == 16) 0 else if (p.bm == 32) 1 else 2;
        const f = if (p.fused) set[6] else set[ti * 2 + @intFromBool(p.cluster)];
        const rows_t = (m + p.bm - 1) / p.bm;
        const band = @max(1, @min(rows_t, (12 << 20) / (@as(usize, p.bm) * w.k * 2)));
        var args: launch_.Args = .{};
        args.add(x);
        args.add(w.codes);
        args.add(w.scales);
        args.add(w.scale);
        args.add(y);
        args.add(if (p.reduce) part else @as(u64, 0));
        args.add(@as(i32, @intCast(m)));
        args.add(@as(i32, @intCast(w.n)));
        args.add(@as(i32, @intCast(w.k)));
        args.add(@as(i32, @intCast(p.sk)));
        args.add(@as(i32, @intCast(w.npad)));
        args.add(@as(i32, @intCast(if (m == 1) w.k else ldx)));
        args.add(@as(i32, @intCast(band)));
        try launch_.launch(f, .{
            .grid = .{ .x = @intCast(rows_t * ((w.n + bn - 1) / bn)), .y = 1, .z = if (p.fused) 1 else p.sk },
            .block = .{ .x = threads, .y = 1, .z = 1 },
            .shared = sharedBytes(w.mode, p.bm),
            .cluster = if (p.cluster) .{ .x = 1, .y = 1, .z = p.sk } else null,
        }, s, &args);
        if (p.reduce) {
            const total: usize = m * w.n;
            var r: launch_.Args = .{};
            r.add(part);
            r.add(y);
            r.add(@as(i64, @intCast(total)));
            r.add(@as(i32, @intCast(p.sk)));
            r.add(w.scale);
            try launch_.launch(l.reduce[@backingInt(out)], .{
                .grid = .{ .x = @intCast((total + 255) / 256), .y = 1, .z = 1 },
                .block = .{ .x = 256, .y = 1, .z = 1 },
            }, s, &r);
        }
    }
};

/// bf16 [n, k] in the BF16 mode's order [npad/64][k/64][8][4][32][4]: n8 tile, k16 step, lane, b0 then b1 halves.
pub fn packBf16(out: []u16, w: []const u16, n: usize, k: usize) void {
    const npad = (n + 127) / 128 * 128;
    std.debug.assert(out.len == npad * k and w.len == n * k and k % group == 0);
    for (out, 0..) |*o, i| {
        const e = i % 4;
        const lane = (i / 4) % 32;
        const kt = (i / 128) % 4;
        const jj = (i / 512) % 8;
        const tg = i / 4096;
        const row = (tg / (k / group)) * 64 + jj * 8 + lane / 4;
        const col = (tg % (k / group)) * group + kt * 16 + 2 * (lane % 4) + (e & 1) + 8 * (e >> 1);
        o.* = if (row < n) w[row * k + col] else 0;
    }
}

test "bf16 packing puts each lane's b0 and b1 pairs where the m16n8k16 B fragment reads them" {
    const gpa = std.testing.allocator;
    const n = 70;
    const k = 128;
    const w = try gpa.alloc(u16, n * k);
    defer gpa.free(w);
    for (w, 0..) |*x, i| x.* = @truncate(i * 7 + 3);
    const out = try gpa.alloc(u16, 128 * k);
    defer gpa.free(out);
    packBf16(out, w, n, k);
    // tile 0, group 1, n8 tile 5, k16 step 2, lane 13: row 40 + 3 = 43, k0 = 64 + 32 + 2 * 1 = 98
    const at = (((1 * 8 + 5) * 4 + 2) * 32 + 13) * 4;
    try std.testing.expectEqualSlices(u16, &.{ w[43 * k + 98], w[43 * k + 99], w[43 * k + 106], w[43 * k + 107] }, out[at..][0..4]);
    try std.testing.expectEqual(w[63 * k + 6], out[(((0 * 8 + 7) * 4 + 0) * 32 + 31) * 4]); // row 56 + 7 = 63
    try std.testing.expectEqual(@as(u16, 0), out[(((2 * 8 + 1) * 4 + 0) * 32 + 31) * 4]); // row 64 + 8 + 7: padding
}

test "tile shared bytes, slices and row tiles as the Python host picks them" {
    try std.testing.expectEqual(@as(u32, 25600), sharedBytes(.fp8g, 16));
    try std.testing.expectEqual(@as(u32, 33792), sharedBytes(.fp8g, 32));
    try std.testing.expectEqual(@as(u32, 50176), sharedBytes(.fp8g, 64));
    try std.testing.expectEqual(@as(u32, 25088), sharedBytes(.mxfp8, 16));
    try std.testing.expectEqual(@as(u32, 33280), sharedBytes(.mxfp8, 32));
    try std.testing.expectEqual(@as(u32, 49664), sharedBytes(.mxfp8, 64));
    try std.testing.expectEqual(@as(u32, 17408), sharedBytes(.fp4, 16));
    try std.testing.expectEqual(@as(u32, 25600), sharedBytes(.fp4, 32));
    try std.testing.expectEqual(@as(u32, 41984), sharedBytes(.fp4, 64));
    try std.testing.expectEqual(@as(u32, 24576), sharedBytes(.fp8, 16)); // no scale tile: 4 x (2048 + 4096)
    try std.testing.expectEqual(@as(u32, 24576), sharedBytes(.fp8b, 16)); // FP8B keeps its block scale table outside tile smem
    try std.testing.expectEqual(@as(u32, 40960), sharedBytes(.bf16, 16)); // 4 x (2048 + 8192)
    try std.testing.expectEqual(@as(u32, 65536), sharedBytes(.bf16, 64));
    try std.testing.expect(launch_contract.opt_in_required);
    try std.testing.expectEqual(@as(u32, 65536), launch_contract.max_dynamic_shared);
    try std.testing.expectEqual(@as(u32, 2), splitK(7168, 2560)); // values from qmm.split_k itself
    try std.testing.expectEqual(@as(u32, 8), splitK(2560, 6144));
    try std.testing.expectEqual(@as(u32, 4), splitK(512, 2560));
    try std.testing.expectEqual(@as(u32, 1), splitK(2560, 512));
    try std.testing.expectEqual(@as(u32, 1), splitK(32768, 2560));
    try std.testing.expectEqual(@as(u32, 16), bucket(1));
    try std.testing.expectEqual(@as(u32, 32), bucket(17));
    try std.testing.expectEqual(@as(u32, 64), bucket(300));
    const p = plan(300, 2560, 6144, true);
    try std.testing.expect(p.fused and p.bm == 64 and !p.cluster and !p.reduce);
    const q = plan(3, 2560, 6144, true);
    try std.testing.expect(!q.fused and q.cluster and q.bm == 16 and q.sk == 8);
    const r = plan(3, 2560, 6144, false);
    try std.testing.expect(r.reduce and !r.cluster);
}

test "instance names are qmmf.cu's" {
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi4ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.bf16, 64, false, true, true));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi2ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.mxfp8, 16, true, false, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi3ELi16ELi64ELi1ELi4ELi4ELb0ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp8g, 16, true, false, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi64ELi64ELi1ELi4ELi4ELb0ELb0ELb1EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp4, 64, false, true, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi0ELi32ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp4, 32, true, false, true));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi5ELi16ELi64ELi1ELi4ELi4ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp8b, 16, false, false, false));
    try std.testing.expectEqualStrings("_ZN7tf_qmmf11qmmf_kernelILi5ELi64ELi64ELi1ELi4ELi4ELb1ELb1ELb0EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii", symbol(.fp8b, 64, true, false, true));
}

fn groupTestWeight(n: u32) Weight { return .{ .mode = .fp8b, .codes = 0x1000, .scales = 0x2000, .n = n, .k = 6144, .npad = (n + 127) / 128 * 128 }; }

test "group plans preserve projection plans and ABI" {
    @setEvalBranchQuota(10000);
    const ws = [_]GroupOut{ .{ .w = groupTestWeight(2048), .y = 1 }, .{ .w = groupTestWeight(576), .y = 2 }, .{ .w = groupTestWeight(128), .y = 3 } };
    for ([_]usize{ 1, 4, 8, 16 }) |m| {
        const g = try groupPlan(m, 6144, &ws, true, false);
        try std.testing.expect(g.plan.bm == 16 and g.plan.sk == 8 and g.plan.cluster and g.kernel == 0 and g.blocks[0] == 32 and g.blocks[1] == 9 and g.blocks[2] == 2 and g.total == 43 and g.band == 1);
    }
    for ([_]usize{ 17, 32 }) |m| { const g = try groupPlan(m, 6144, &ws, true, false); try std.testing.expect(g.plan.bm == 32 and g.kernel == 1 and g.total == 43); }
    const p200 = try groupPlan(200, 6144, &ws, true, false);
    try std.testing.expect(p200.plan.bm == 64 and p200.kernel == 2 and p200.blocks[0] == 128 and p200.blocks[1] == 36 and p200.blocks[2] == 8 and p200.total == 172 and p200.band == 4);
    inline for ([_]usize{ 512, 2048 }) |m| {
        const normal = try groupPlan(m, 6144, &ws, true, false);
        const forced = try groupPlan(m, 6144, &ws, true, true);
        try std.testing.expect(normal.plan.fused and normal.kernel == 3);
        try std.testing.expect(normal.blocks[0] == (if (m == 512) 256 else 1024) and normal.blocks[1] == (if (m == 512) 72 else 288) and normal.blocks[2] == (if (m == 512) 16 else 64));
        try std.testing.expect(forced.plan.cluster and forced.plan.sk == 8 and forced.kernel == 2 and forced.total == normal.total);
    }
    inline for (.{ .fp8b, .bf16 }) |mode| inline for ([_]usize{ 32, 128, 576, 2048 }) |n| inline for (1..256) |m| {
        try std.testing.expectEqual(planForMode(m, n, 6144, true, mode), splitPlan(m, n, 6144, true, mode));
    };
    const bf16 = splitPlan(2048, 32, 6144, true, .bf16);
    try std.testing.expect(bf16.bm == 64 and bf16.sk == 8 and bf16.cluster and !bf16.fused);
    try std.testing.expectEqual(@as(usize, 48), @sizeOf(GroupSegAbi));
    try std.testing.expectEqual(@as(usize, 240), @sizeOf(GroupArgsAbi));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(GroupArgsAbi, "seg"));
}

test "group plans reject incompatible modes, shapes, sizes and plans" {
    const ws = [_]GroupOut{ .{ .w = groupTestWeight(2048), .y = 1 }, .{ .w = groupTestWeight(576), .y = 2 }, .{ .w = groupTestWeight(128), .y = 3 } };
    try std.testing.expectError(error.GroupSize, groupPlan(4, 6144, &.{}, true, false));
    try std.testing.expectError(error.GroupSize, groupPlan(4, 6144, &.{ ws[0], ws[1], ws[2], ws[0], ws[1] }, true, false));
    var bad = ws[0]; bad.w.mode = .bf16;
    try std.testing.expectError(error.GroupMode, groupPlan(4, 6144, &.{bad}, true, false));
    bad = ws[0]; bad.w.k = 2048;
    try std.testing.expectError(error.GroupShape, groupPlan(4, 6144, &.{bad}, true, false));
    const extra = GroupOut{ .w = groupTestWeight(12288), .y = 4 };
    try std.testing.expectError(error.GroupPlanMismatch, groupPlan(4, 6144, &.{ ws[0], ws[1], ws[2], extra }, true, false));
    try std.testing.expectError(error.GroupPlanUnsupported, groupPlan(4, 6144, &ws, false, false));
}

test {
    std.testing.refAllDecls(@This());
}
