// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! DFlash2's device passes: the context ingest (taps through fc into the rings) and the block (embed to candidates).
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const mm = @import("cuda_mm.zig");
const q4 = @import("cuda_q4.zig");
const df = @import("dflash.zig");
const Drafter = df.Drafter;

pub const Plan = struct {
    rows: usize,
    hidden: usize,
    qkv_width: usize,
    query_heads: usize,
    kv_heads: usize,
    query_heads_per_kv: usize,
    ff: usize,

    pub fn init(g: df.Geometry, streams: usize) !Plan {
        if (streams == 0 or streams > df.MAX_S or (!g.flashKernelAbi() and g.tap_streams != 1) or g.hidden == 0 or g.kv_heads == 0 or g.heads % g.kv_heads != 0 or g.heads / g.kv_heads > 8) return error.DrafterGeometryUnsupported;
        if (g.block_size == 0 or g.block_size > df.BLOCK or g.head_dim != (if (g.kind == .dspark) @as(usize, 64) else df.HD) or g.conv_group == 0 or g.hidden % g.conv_group != 0) return error.DrafterGeometryUnsupported;
        return .{ .rows = streams * g.block_size, .hidden = g.hidden, .qkv_width = g.qkvWidth(), .query_heads = g.heads, .kv_heads = g.kv_heads, .query_heads_per_kv = g.heads / g.kv_heads, .ff = g.ff };
    }
};

fn go(f: cuda.Function, grid: cuda.Dim3, threads: u32, e: anytype, a: *cuda.Args) !void {
    try cuda.launch.launch(f, .{ .grid = grid, .block = .{ .x = threads } }, e.stream, a);
}

fn set(e: anytype) glue.Set {
    return .{ .glue = &e.m.glue, .mm = &e.m.mm };
}

/// out[m, n] = x[m, k] . w^T, bf16 out (`wide`: fp32, glm_mm only): 4 bits 16 rows a launch, or glm_mm.
fn lin(dr: *const Drafter, e: anytype, x: u64, w: df.Lin, out: u64, m: usize, wide: bool) !void {
    switch (w) {
        .bf16 => |t| try mm.matmul(&e.m.mm, e.stream, x, t.k, t, out, dr.part, m, wide, false),
        .q4 => |q| {
            if (wide) return error.WideFromFourBits;
            var a: cuda.Args = .{};
            a.add(x);
            a.add(@as(i32, @intCast(q.k)));
            a.add(@as(i32, @intCast(q.k)));
            a.add(dr.gs);
            try go(dr.k.gsum, .{ .x = @intCast(m) }, 128, e, &a);
            var r0: usize = 0;
            while (r0 < m) : (r0 += 16) {
                const rows = @min(16, m - r0);
                try q4.matmul(&e.m.q4, e.stream, x + r0 * q.k * 2, q.k, dr.gs + r0 * (q.k / 64) * 4, q, out + r0 * q.n * 2, rows);
            }
        },
    }
}

fn linB(dr: *const Drafter, e: anytype, x: u64, w: mm.Tiled, out: u64, m: usize, wide: bool) !void {
    try lin(dr, e, x, .{ .bf16 = w }, out, m, wide);
}

pub fn copy(e: anytype, dst: u64, src: u64, bytes: usize) !void {
    try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
}

pub fn upload(dr: *const Drafter, e: anytype, dst: u64, bytes: []const u8) !void {
    _ = dr;
    try e.d.check(e.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, e.stream.handle), "cuMemcpyHtoDAsync");
}

/// After the pass: the stream idle, a failed RoCE gather's error raised, `src` read into `dst`.
pub fn download(dr: *const Drafter, e: anytype, dst: []u8, src: u64) !void {
    _ = dr;
    try e.stream.synchronize();
    if (e.group) |g| try g.check();
    try e.d.check(e.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "cuMemcpyDtoH");
}

/// Every rank's `bytes` at buf + rank * bytes into every rank's buf, rank order (the engine's gathers' route).
fn gather(e: anytype, buf: u64, bytes: usize) !void {
    if (e.spans.len == 1) return;
    if (e.group) |g| return g.allGather(e.stream, buf + e.share.rank * bytes, buf, bytes);
    try e.lib.check(e.lib.api.ncclAllGather(buf + e.share.rank * bytes, buf, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
}

/// `N` context rows (cin, cpos, crings uploaded) through fc, the hidden norm and each layer's k, v into the rings.
pub fn ingestPass(dr: *Drafter, e: anytype, N: usize) !void {
    const g = dr.geometry;
    const D = g.hidden;
    const KVW = g.kvWidth();
    try lin(dr, e, dr.cin, dr.fc, dr.c1, N, false);
    try glue.rmsnorm(set(e), e.stream, dr.c1, D, dr.hnorm, dr.ctx, D, null, g.eps, N, D);
    for (dr.layers, 0..) |l, li| {
        try lin(dr, e, dr.ctx, l.kv, dr.ckv, N, false);
        var a: cuda.Args = .{};
        a.add(dr.ckv);
        a.add(l.kn);
        a.add(dr.cpos);
        a.add(dr.crings);
        a.add(@as(i32, @intCast(li)));
        a.add(g.eps);
        a.add(g.rope_theta);
        if (!g.flashKernelAbi()) {
            a.add(@as(i32, @intCast(g.kv_heads)));
            a.add(@as(i32, @intCast(g.head_dim)));
            a.add(@as(i32, @intCast(KVW)));
            a.add(@as(i32, @intCast(g.window)));
        }
        try go(dr.k.ctx_kv, .{ .x = @intCast(N), .y = @intCast(g.kv_heads) }, @intCast(g.head_dim / 2), e, &a);
    }
}

fn conv(dr: *const Drafter, e: anytype, z: u64, which: i32, base: u64, out: u64, add: bool, M: usize) !void {
    const g = dr.geometry;
    var a: cuda.Args = .{};
    a.add(z);
    a.add(dr.dyn);
    a.add(which);
    a.add(base);
    a.add(out);
    a.add(@as(i32, @intFromBool(add)));
    if (!g.flashKernelAbi()) {
        a.add(@as(i32, @intCast(g.hidden)));
        a.add(@as(i32, @intCast(g.conv_group)));
        a.add(@as(i32, @intCast(g.conv_kernel)));
        a.add(@as(i32, @intCast(g.block_size)));
    }
    try go(dr.k.conv, .{ .x = @intCast(M), .y = @intCast((g.hidden + 1023) / 1024) }, 256, e, &a);
}

/// Embed draft ids. For vocab-sharded GLM, the hook is collective: ids are device i32s,
/// each rank embeds its local rows, all ranks gather rank-major [world][M][D], and
/// the output copies the row from owner = id / (vocab / world), never sums masked rows.
pub fn embedTarget(dr: *Drafter, e: anytype, M: usize, D: usize) !void {
    if (comptime @hasDecl(@TypeOf(e.*), "embedDraft")) {
        try e.embedDraft(dr.ids, dr.x, M, D);
    } else {
        if (!dr.geometry.flashKernelAbi()) return error.DrafterEmbedNeedsShardHook;
        try glue.embedB16(set(e), e.stream, dr.ids, e.w.embed, dr.x, M, D, 1);
    }
}

/// The block pass for `S` streams (tables uploaded): embed, 5 layers, norm, each rank's 16 best a row, gathered.
pub fn pass(dr: *Drafter, e: anytype, S: usize, rank0: bool) !void {
    const g = dr.geometry;
    const D = g.hidden;
    const KVW = g.kvWidth();
    const QKV = g.qkvWidth();
    const shape = try Plan.init(g, S);
    const BLOCK = g.block_size;
    const M = shape.rows;
    try embedTarget(dr, e, M, D);
    for (dr.layers, 0..) |l, li| {
        // attention, wrapped in the attention conv
        try glue.rmsnorm(set(e), e.stream, dr.x, D, l.in_norm, dr.y, D, null, g.eps, M, D);
        const attn_in = if (g.conv) blk: {
            try lin(dr, e, dr.y, l.a_proj, dr.dyn, M, false);
            try conv(dr, e, dr.y, 0, l.a_base, dr.y2, false, M);
            break :blk dr.y2;
        } else dr.y;
        try lin(dr, e, attn_in, l.qkv, dr.qkv, M, false);
        {
            var a: cuda.Args = .{};
            a.add(dr.qkv);
            a.add(@as(i32, @intCast(QKV)));
            a.add(l.qn);
            a.add(l.kn);
            a.add(dr.bpos);
            a.add(g.eps);
            a.add(g.rope_theta);
            if (!g.flashKernelAbi()) {
                a.add(@as(i32, @intCast(g.heads)));
                a.add(@as(i32, @intCast(g.kv_heads)));
                a.add(@as(i32, @intCast(g.head_dim)));
                a.add(@as(i32, @intCast(KVW)));
            }
            a.add(dr.q);
            a.add(dr.kb);
            a.add(dr.vb);
            try go(dr.k.block_qkv, .{ .x = @intCast(M), .y = @intCast(g.heads + g.kv_heads) }, @intCast(g.head_dim / 2), e, &a);
        }
        {
            var a: cuda.Args = .{};
            for ([_]u64{ dr.q, dr.kb, dr.vb, dr.rings, dr.starts, dr.froms }) |p| a.add(p);
            a.add(@as(i32, @intCast(li)));
            a.add(g.scale());
            if (!g.flashKernelAbi()) {
                a.add(@as(i32, @intCast(g.window)));
                a.add(@as(i32, @intCast(BLOCK)));
                a.add(@as(i32, @intCast(g.heads)));
                a.add(@as(i32, @intCast(g.kv_heads)));
                a.add(@as(i32, @intCast(g.head_dim)));
                a.add(@as(i32, @intCast(KVW)));
            }
            for ([_]u64{ dr.pm, dr.pl, dr.pacc }) |p| a.add(p);
            try go(dr.k.attn_part, .{ .x = @intCast(S), .y = @intCast(g.kv_heads), .z = df.CH }, 256, e, &a);
            var b: cuda.Args = .{};
            for ([_]u64{ dr.pm, dr.pl, dr.pacc, dr.att }) |p| b.add(p);
            if (!g.flashKernelAbi()) {
                b.add(@as(i32, @intCast(g.heads)));
                b.add(@as(i32, @intCast(g.head_dim)));
            }
            try go(dr.k.attn_merge, .{ .x = @intCast(M), .y = @intCast(g.heads) }, @intCast(g.head_dim), e, &b);
        }
        try lin(dr, e, dr.att, l.o, dr.a, M, false);
        if (g.conv) {
            try conv(dr, e, dr.a, 1, l.a_base + g.conv_kernel * D * 2, dr.x, true, M);
        } else {
            var a: cuda.Args = .{};
            a.add(dr.x);
            a.add(dr.a);
            a.add(@as(i32, @intCast(M)));
            a.add(@as(i32, @intCast(D)));
            try go(dr.k.add, .{ .x = @intCast(M), .y = @intCast((D + 255) / 256) }, 256, e, &a);
        }
        // the MLP, wrapped in the MLP conv
        try glue.rmsnorm(set(e), e.stream, dr.x, D, l.post_norm, dr.y, D, null, g.eps, M, D);
        const mlp_in = if (g.conv) blk: {
            try lin(dr, e, dr.y, l.m_proj, dr.dyn, M, false);
            try conv(dr, e, dr.y, 0, l.m_base, dr.y2, false, M);
            break :blk dr.y2;
        } else dr.y;
        try lin(dr, e, mlp_in, l.gu, dr.gu, M, false);
        {
            var a: cuda.Args = .{};
            a.add(dr.gu);
            a.add(dr.act);
            a.add(@as(i32, @intCast(dr.ff)));
            try go(dr.k.swiglu, .{ .x = @intCast(M), .y = @intCast((dr.ff + 1023) / 1024) }, 256, e, &a);
        }
        if (dr.split) { // each rank's share of down's K as fp32 partials, gathered and summed
            const bytes = M * D * 4;
            try linB(dr, e, dr.act, l.down, dr.sums + e.share.rank * bytes, M, true);
            try gather(e, dr.sums, bytes);
            var a: cuda.Args = .{};
            a.add(dr.sums);
            a.add(dr.a);
            a.add(@as(i32, @intCast(M)));
            if (!g.flashKernelAbi()) a.add(@as(i32, @intCast(D)));
            try go(if (dr.world == 3) dr.k.sum3 else dr.k.sum2, .{ .x = @intCast(M), .y = @intCast((D + 1023) / 1024) }, 256, e, &a);
        } else try linB(dr, e, dr.act, l.down, dr.a, M, false);
        if (g.conv) {
            try conv(dr, e, dr.a, 1, l.m_base + g.conv_kernel * D * 2, dr.x, true, M);
        } else {
            var a: cuda.Args = .{};
            a.add(dr.x);
            a.add(dr.a);
            a.add(@as(i32, @intCast(M)));
            a.add(@as(i32, @intCast(D)));
            try go(dr.k.add, .{ .x = @intCast(M), .y = @intCast((D + 255) / 256) }, 256, e, &a);
        }
    }
    // every row through the drafts' 4-bit head (this rank's vocabulary), its 16 best, gathered; rank 0's projections
    try glue.rmsnorm(set(e), e.stream, dr.x, D, dr.norm, dr.hn, D, dr.xs, g.eps, M, D);
    if (g.confidence and rank0) try download(dr, e, std.mem.sliceAsBytes(dr.hrows[0 .. M * D]), dr.hn);
    const head = dr.head orelse (e.w.mtp orelse return error.DrafterNeedsDraftHead).draft;
    var r0: usize = 0;
    while (r0 < M) : (r0 += 16) {
        const rows = @min(16, M - r0);
        try q4.matmul(&e.m.q4, e.stream, dr.hn + r0 * D * 2, D, dr.xs + r0 * (D / 64) * 4, head, dr.logits + r0 * head.n * 2, rows);
    }
    const chunks = (head.n + 4095) / 4096;
    const nw = M * df.cw(g);
    {
        var a: cuda.Args = .{};
        a.add(dr.logits);
        a.add(@as(i32, @intCast(head.n)));
        a.add(@as(i32, @intCast(e.share.vocab.lo)));
        if (!g.flashKernelAbi()) a.add(@as(i32, @intCast(g.selector_top_k)));
        a.add(dr.pv);
        a.add(dr.pi);
        try go(dr.k.top16_part, .{ .x = @intCast(M), .y = @intCast(chunks) }, 256, e, &a);
        var b: cuda.Args = .{};
        b.add(dr.pv);
        b.add(dr.pi);
        b.add(@as(i32, @intCast(chunks)));
        if (!g.flashKernelAbi()) b.add(@as(i32, @intCast(g.selector_top_k)));
        b.add(dr.gtk + e.share.rank * nw * 4);
        try go(dr.k.top16_merge, .{ .x = @intCast(M) }, 256, e, &b);
    }
    try gather(e, dr.gtk, nw * 4);
    if (rank0 and g.kind != .dspark) try linB(dr, e, dr.hn, dr.hproj, dr.g, M, true); // DSpark has no hidden projection
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Plan);
}
