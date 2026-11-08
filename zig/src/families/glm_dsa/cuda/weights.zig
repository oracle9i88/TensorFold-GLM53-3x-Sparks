// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 CUDA weights for one TP2 rank (WP9): the FP8 backbone on qmmf `.fp8b` with its native 128x128 scales,
//! kv_b once in checkpoint form (absorb and expand-V both read it), EXL3 routed experts through WP4's loader.
//! Loader shape adapted from ashhart/TensorFold@d1018b31:zig/src/families/glm/cuda_weights.zig (Apache-2.0).
//! Tensor names and TP2 split rules follow bertholomus/bertholomus/TensorFold@ddfad35:src/tensorfold/families/glm_moe_dsa/cuda/weights.py
//! and glm5_next/cuda/split.py (Apache-2.0). Changes: Zig loader for this engine's kernels, fail-closed admission.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const core = @import("core");
const glm_config = @import("glm_config");
const share_mod = @import("share.zig");
const experts = @import("experts.zig");
const mlp = @import("mlp.zig");
const state = @import("state.zig");
const uma = @import("uma.zig");
const manifest_mod = @import("manifest.zig");
const weight_plan = @import("weight_plan.zig");
pub const manifest = manifest_mod;

const qmmf = cuda.qmmf;
const fp8 = cuda.fp8;
const CkTensor = core.checkpoint.Tensor;
pub const Config = glm_config.Config;

pub const DType = enum { bf16, f32, e4m3, raw };

/// Host view of a safetensors tensor; its native block scales remain paired with its E4M3 codes.
pub const Tensor = struct {
    bytes: []const u8,
    dtype: DType,
    rows: usize,
    cols: usize,
    scale_inv: []const f32 = &.{},
};

/// Exactly one FP8B projection on a rank, in row-major code and [128-row][128-column] scale order.
pub const Fp8B = struct {
    codes: []u8,
    scales: []f32,
    rows: usize,
    cols: usize,
};

/// TP2 cut used by q_b/o_proj/head and intermediate projections.
pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub fn packFp8B(allocator: std.mem.Allocator, src: Tensor, row_span: Span, col_span: Span) !Fp8B {
    if (src.dtype != .e4m3 or src.bytes.len != src.rows * src.cols or
        row_span.lo >= row_span.hi or row_span.hi > src.rows or
        col_span.lo >= col_span.hi or col_span.hi > src.cols or
        row_span.lo % 128 != 0 or col_span.lo % 128 != 0 or
        (row_span.hi != src.rows and row_span.hi % 128 != 0) or
        (col_span.hi != src.cols and col_span.hi % 128 != 0)) return error.BadFp8BShape;
    const rows = row_span.hi - row_span.lo;
    const cols = col_span.hi - col_span.lo;
    const scale_cols = (src.cols + 127) / 128;
    const expected_scales = ((src.rows + 127) / 128) * scale_cols;
    if (src.scale_inv.len != expected_scales) return error.BadScaleShape;
    const out_scales = ((rows + 127) / 128) * ((cols + 127) / 128);
    const codes = try allocator.alloc(u8, rows * cols);
    errdefer allocator.free(codes);
    const scales = try allocator.alloc(f32, out_scales);
    errdefer allocator.free(scales);
    for (0..rows) |r| @memcpy(codes[r * cols ..][0..cols], src.bytes[(row_span.lo + r) * src.cols + col_span.lo ..][0..cols]);
    for (0..(rows + 127) / 128) |r| for (0..(cols + 127) / 128) |c| {
        scales[r * ((cols + 127) / 128) + c] = src.scale_inv[(row_span.lo / 128 + r) * scale_cols + col_span.lo / 128 + c];
    };
    return .{ .codes = codes, .scales = scales, .rows = rows, .cols = cols };
}

/// Host-side layer of FP8B slices (WP9 round 1 layout contract); kv_b is intentionally represented once.
pub const Layer = struct {
    q_a: Fp8B,
    q_b: Fp8B,
    kv_a: Fp8B,
    kv_b: Fp8B,
    o_proj: Fp8B,
    wq_b: Fp8B,
    wk: Fp8B,
    weights_proj: []u16,
    lowrank_a: ?[]f32 = null,
    lowrank_b: ?[]f32 = null,

    pub fn deinit(layer: *Layer, allocator: std.mem.Allocator) void {
        inline for (.{ "q_a", "q_b", "kv_a", "kv_b", "o_proj", "wq_b", "wk" }) |field| {
            allocator.free(@field(layer, field).codes);
            allocator.free(@field(layer, field).scales);
        }
        allocator.free(layer.weights_proj);
        if (layer.lowrank_a) |v| allocator.free(v);
        if (layer.lowrank_b) |v| allocator.free(v);
        layer.* = undefined;
    }
};

/// Where uploads go: production allocates device memory and copies host bytes in (synchronously); host tests put
/// real host memory behind cuda.memory's hook. Large tensors are written in row blocks at offsets.
pub const Device = struct {
    ctx: ?*anyopaque = null,
    alloc_fn: *const fn (?*anyopaque, *const cuda.Driver, usize) anyerror!cuda.DeviceBuffer = deviceAlloc,
    write_fn: *const fn (?*anyopaque, cuda.DeviceBuffer, usize, []const u8) anyerror!void = deviceWrite,

    fn deviceAlloc(_: ?*anyopaque, d: *const cuda.Driver, len: usize) anyerror!cuda.DeviceBuffer {
        return cuda.DeviceBuffer.alloc(d, len);
    }

    fn deviceWrite(_: ?*anyopaque, buf: cuda.DeviceBuffer, offset: usize, bytes: []const u8) anyerror!void {
        try buf.upload(offset, bytes);
    }
};

/// Input services for a load. Admission happened once, before this (state.admitRank on planRank's exact bytes);
/// the loader verifies its plan against `admission` before it uploads anything.
pub const TraceHook = struct {
    ctx: *anyopaque,
    call: *const fn (*anyopaque, []const u8, []const u8, u64) void,

    pub fn step(h: ?TraceHook, label: []const u8, category: []const u8, bytes: u64) void {
        if (h) |hook| hook.call(hook.ctx, label, category, bytes);
    }
};

pub const Budget = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    driver: *const cuda.Driver,
    device_ordinal: cuda.abi.Device = 0,
    admission: *const state.Admission,
    /// Request the rank-local VMM weight arena; false preserves Device.alloc_fn's legacy path.
    weights_vmm: bool = false,
    device: Device = .{},
    /// Read tensor bytes with O_DIRECT through four 4 MiB staging slots instead of the mmap; null reads
    /// TF_GLM53_ODIRECT (default off). Device bytes are identical either way.
    odirect: ?bool = null,
    /// Staging slots for O_DIRECT reads; null allocates pinned, mlocked slots on `driver`.
    staging: ?*const uma.Staging = null,
    trace: ?TraceHook = null,
    /// After the load: unmap, then posix_fadvise(DONTNEED) every shard (regular files only).
    drop_page_cache: bool = true,
};

/// A pre-mapped, rank-local VMM range. The hook hands out planned class spans in load order.
const WeightVmm = struct {
    gpa: std.mem.Allocator,
    vmm: cuda.arena.Vmm,
    plan: weight_plan.Plan,
    sizes: []u64,
    next: usize = 0,
    bad_alloc: bool = false,
    base: u64,

    fn init(gpa: std.mem.Allocator, driver: *const cuda.Driver, device: cuda.abi.Device, sizes: []const u64, trace: ?TraceHook) !*WeightVmm {
        var reserve_bytes: u64 = 2 << 20;
        for (sizes) |size| reserve_bytes = try std.math.add(u64, reserve_bytes, try std.math.add(u64, size, 256));
        const result = try gpa.create(WeightVmm);
        errdefer gpa.destroy(result);
        var vmm = if (weightVmmRdmaOff()) try cuda.arena.Vmm.openMinimumNoRdma(driver, device, @intCast(reserve_bytes)) else try cuda.arena.Vmm.openMinimum(driver, device, @intCast(reserve_bytes));
        var reserve_label_buf: [96]u8 = undefined;
        const reserve_label: []const u8 = std.fmt.bufPrint(&reserve_label_buf, "weight VMM reserve (rdma {d}, granularity {d})", .{ vmm.prop.rdma, vmm.granularity }) catch "weight VMM reserve";
        TraceHook.step(trace, reserve_label, "vmm", reserve_bytes);
        errdefer vmm.close();
        vmm.retain_handles = true;
        vmm.close_unmap_all = true;
        const plan = try weight_plan.build(gpa, processEnvLookup, sizes, @max(state.tensor_alignment, 256), @intCast(vmm.granularity));
        errdefer gpa.free(plan.offsets);
        if (plan.mode != .vmm) return error.WeightVmmModeMismatch;
        const backend = vmm.backend();
        const base = backend.chunk(backend.ctx, @intCast(plan.mapped_bytes)) orelse {
            if (vmm.failed) |failure| {
                std.log.err("GLM-5.3 CUDA VMM step {s} failed with CUresult {d}", .{ failure.call, failure.code });
            }
            return error.WeightVmmMapFailed;
        };
        TraceHook.step(trace, "weight VMM create+map+access", "vmm", plan.mapped_bytes);
        result.* = .{ .gpa = gpa, .vmm = vmm, .plan = plan, .sizes = try gpa.dupe(u64, sizes), .base = base };
        return result;
    }

    fn deinit(self: *WeightVmm) void {
        self.gpa.free(self.plan.offsets);
        self.gpa.free(self.sizes);
        self.vmm.close();
        self.gpa.destroy(self);
    }

    fn alloc(ctx: *anyopaque, len: usize) ?cuda.abi.DevicePtr {
        const self: *WeightVmm = @ptrCast(@alignCast(ctx));
        if (self.next >= self.sizes.len or self.sizes[self.next] != @as(u64, @intCast(len))) {
            self.bad_alloc = true;
            return null;
        }
        const ptr = self.base + self.plan.offsets[self.next];
        self.next += 1;
        return ptr;
    }

    fn free(ctx: *anyopaque, ptr: cuda.abi.DevicePtr) bool {
        const self: *WeightVmm = @ptrCast(@alignCast(ctx));
        return ptr >= self.base and ptr < self.base + self.plan.mapped_bytes;
    }

    fn hook(self: *WeightVmm) cuda.memory.Hook {
        return .{ .ctx = self, .alloc = alloc, .free = free };
    }
};

pub fn processEnvLookup(name: [:0]const u8) ?[]const u8 {
    const p = std.c.getenv(name.ptr) orelse return null;
    return std.mem.span(p);
}

pub const weightsVmmFromLookup = weight_plan.weightsVmmFromLookup;

pub fn weightsVmmFromEnv() bool {
    return weight_plan.weightsVmmFromLookup(processEnvLookup);
}

/// TF_GLM53_WEIGHT_VMM_RDMA=0: weight arenas without gpuDirectRDMACapable (weights are never RDMA-registered: RoCE registers
/// its own host region). Unset or any other value keeps today's prop. Both ranks take it from the same launcher env.
fn weightVmmRdmaOff() bool {
    const raw = std.c.getenv("TF_GLM53_WEIGHT_VMM_RDMA") orelse return false;
    return std.mem.eql(u8, std.mem.span(raw), "0");
}

/// A resident named tensor. FP8 tensors keep code and block-scale allocations separate; kv_b is one entry.
pub const ResidentTensor = struct {
    name: []u8,
    dtype: DType,
    data: cuda.DeviceBuffer,
    scales: ?cuda.DeviceBuffer = null,
    rows: usize,
    cols: usize,
};

/// kv_b in checkpoint form: row-major E4M3 codes [local heads x 448][512] and scale_inv [rows/128][4].
/// The MLA kernels index it with local heads (head0 = 0) because the rows are this rank's head slice.
pub const RawFp8 = struct { codes: u64, scales: u64, rows: usize, cols: usize };

/// One full-indexer layer: wq_b (on q_resid), wk and weights_proj (on the attention input), k_norm weight and bias.
pub const Indexer = struct { wq_b: qmmf.Weight, wk: qmmf.Weight, weights_proj: qmmf.Weight, k_norm: u64, k_bias: u64 };

pub const Attention = struct {
    q_a: qmmf.Weight,
    q_b: qmmf.Weight,
    kv_a: qmmf.Weight,
    kv_b: RawFp8,
    o_proj: qmmf.Weight,
    q_a_norm: u64,
    kv_a_norm: u64,
    indexer: ?Indexer = null,
    local_heads: usize,
    head0: usize,
};

/// This rank's R/2 slice of mlp.lowrank_{a,b}, fp32 resident: a [hidden][r_local], b [r_local][hidden].
pub const LowRank = struct { a: u64, b: u64, rank_local: usize, rank_full: usize, span: Span };

pub const Moe = struct { router: u64, bias: u64, routed: experts.Loaded, shared: mlp.Weights, lowrank: ?LowRank = null };
pub const Ffn = union(enum) { dense: mlp.Weights, moe: Moe };

/// One layer's device weights, as the forward binds them.
pub const DeviceLayer = struct { index: usize, in_norm: u64, post_norm: u64, attn: Attention, ffn: Ffn };

/// One row range of a stacked linear (gate rows then up rows), read from tensor `prefix ++ "weight"`.
const Part = struct { prefix: []const u8, rows: Span };
const To = enum { same, bf16, f32 };
/// Rows per block for the big bf16 tensors (embed, lm_head): 8,192 rows x 6,144 columns = 96 MiB of bf16.
const block_rows = 8192;
/// Host staging above this is returned to the allocator after its upload (no capacity is retained).
const host_keep_bytes = 64 << 20;

/// Breakdown class of a resident tensor; planRank and ModelWeights.classes share it.
pub const Class = enum { routed, backbone_fp8, embed_head, router_norms, lowrank };

pub fn classOf(name: []const u8, dtype: DType) Class {
    if (std.mem.indexOf(u8, name, ".moe.exl3.") != null) return .routed;
    if (std.mem.indexOf(u8, name, "embed_tokens") != null or std.mem.startsWith(u8, name, "lm_head")) return .embed_head;
    if (std.mem.indexOf(u8, name, "lowrank") != null) return .lowrank;
    return if (dtype == .e4m3) .backbone_fp8 else .router_norms;
}

pub const class_count = @typeInfo(Class).@"enum".field_names.len;

/// Bytes a tensor (or its scales) takes inside its class arena: placed at state.tensor_alignment.
fn placed(n: usize) u64 {
    return std.mem.alignForward(u64, n, state.tensor_alignment);
}

fn alignWeight(n: u64, multiple: u64) !u64 {
    if (multiple == 0) return error.WeightVmmGranularity;
    const rem = n % multiple;
    if (rem == 0) return n;
    return std.math.add(u64, n, multiple - rem) catch error.WeightVmmOverflow;
}

/// These class sizes come from roundedClasses(plan.placed), so each includes state.arenaBytes granularity padding.
fn classBytes(w: state.WeightClasses, c: Class) u64 {
    return switch (c) {
        .routed => w.routed,
        .backbone_fp8 => w.backbone_fp8,
        .embed_head => w.embed_head,
        .router_norms => w.router_norms,
        .lowrank => w.lowrank,
    };
}

/// Each class arena rounded once to the allocation granularity: what CUDA allocates for the weights.
pub fn roundedClasses(w: state.WeightClasses) state.WeightClasses {
    return .{ .routed = state.arenaBytes(w.routed), .backbone_fp8 = state.arenaBytes(w.backbone_fp8), .embed_head = state.arenaBytes(w.embed_head), .router_norms = state.arenaBytes(w.router_norms), .lowrank = state.arenaBytes(w.lowrank), .drafter = w.drafter };
}

fn addClass(w: *state.WeightClasses, c: Class, bytes: u64) void {
    switch (c) {
        .routed => w.routed += bytes,
        .backbone_fp8 => w.backbone_fp8 += bytes,
        .embed_head => w.embed_head += bytes,
        .router_norms => w.router_norms += bytes,
        .lowrank => w.lowrank += bytes,
    }
}

/// O_DIRECT reads land in a ring of the last `ring_len` tensors (no tensor is used after 8 later reads), so at
/// most a few tensors are resident on the host at once.
const Direct = struct {
    const ring_len = 8;
    files: []uma.DirectFile,
    staging: *const uma.Staging,
    ring: [ring_len][]u8 = @splat(&.{}),
    next: usize = 0,

    fn hold(d: *Direct, gpa: std.mem.Allocator, len: usize) ![]u8 {
        gpa.free(d.ring[d.next]);
        d.ring[d.next] = &.{};
        const buf = try gpa.alloc(u8, len);
        d.ring[d.next] = buf;
        d.next = (d.next + 1) % ring_len;
        return buf;
    }

    fn deinit(d: *Direct, gpa: std.mem.Allocator) void {
        for (d.ring) |b| gpa.free(b);
    }
};

/// Checkpoint reader and uploader for one rank (the glm/cuda_weights.zig get/put contract that WP4's
/// experts.load also takes). In plan mode (`plan` set) it walks the same names from the headers only and
/// accounts each upload's exact device bytes instead of reading or uploading.
const Loader = struct {
    gpa: std.mem.Allocator,
    ck: *core.Checkpoint,
    budget: ?Budget,
    model: ?*ModelWeights,
    select: Select = .{},
    /// Plan mode: per-class bytes as placed in the arenas (256 B aligned), and the raw tensor bytes.
    plan: ?*state.WeightClasses = null,
    plan_logical: ?*state.WeightClasses = null,
    host: std.ArrayList(u8) = .empty,
    name_buf: [256]u8 = undefined,
    full_buf: [288]u8 = undefined,
    direct: ?*Direct = null,
    lowrank_r: ?usize = null,
    /// Keep the exact WP4 staging reservation across one layer's three projections.
    hold_host: bool = false,

    /// weights.py name(): "model." + raw, then "model.language_model." + raw, then the bare name.
    fn full(L: *Loader, raw: []const u8) ?[]const u8 {
        for ([_][]const u8{ "model.", "model.language_model." }) |prefix| {
            const name = std.fmt.bufPrint(&L.full_buf, "{s}{s}", .{ prefix, raw }) catch return null;
            if (L.ck.has(name)) return name;
        }
        if (L.ck.has(raw)) return raw;
        return null;
    }

    fn hasRaw(L: *Loader, raw: []const u8) bool {
        return L.full(raw) != null;
    }

    /// Whether a checkpoint has a model tensor under any accepted prefix, without touching its payload.
    pub fn has(L: *Loader, comptime fmt: []const u8, args: anytype) bool {
        return L.hasRaw(std.fmt.bufPrint(&L.name_buf, fmt, args) catch return false);
    }

    /// Shape and dtype only (the mapping's pages are not touched); marks the tensor used.
    fn header(L: *Loader, raw: []const u8) !CkTensor {
        const name = L.full(raw) orelse return error.MissingTensor;
        return L.ck.get(name);
    }

    /// The whole tensor. A missing name is error.MissingTensor without a log line (WP4's marker probe).
    fn tensor(L: *Loader, raw: []const u8) !CkTensor {
        var t = try L.header(raw);
        if (L.direct) |d| t.bytes = try L.readDirect(d, t.bytes);
        return t;
    }

    /// Rows [rows.lo, rows.hi) of the outermost axis (contiguous), as a tensor of rows.len() rows: only these bytes
    /// are read (O_DIRECT) or touched (mmap).
    fn tensorRows(L: *Loader, raw: []const u8, rows: Span) !CkTensor {
        var t = try L.header(raw);
        const n = if (t.rank == 0) 1 else t.dim(0);
        if (rows.hi > n or rows.lo >= rows.hi) return error.BadTensorSlice;
        const row_bytes = t.bytes.len / n;
        const span = t.bytes[rows.lo * row_bytes .. rows.hi * row_bytes];
        t.bytes = if (L.direct) |d| try L.readDirect(d, span) else span;
        if (t.rank > 0) t.shape[0] = rows.len();
        return t;
    }

    fn readDirect(L: *Loader, d: *Direct, mapped: []const u8) ![]const u8 {
        for (L.ck.files.items, d.files) |*f, df| {
            const base = @intFromPtr(f.map.memory.ptr);
            const at = @intFromPtr(mapped.ptr);
            if (at < base or at + mapped.len > base + f.map.memory.len) continue;
            const buf = try d.hold(L.gpa, mapped.len);
            try df.read(d.staging, at - base, buf);
            return buf;
        }
        return error.TensorOutsideShards;
    }

    pub fn get(L: *Loader, comptime fmt: []const u8, args: anytype) !CkTensor {
        return L.tensor(try std.fmt.bufPrint(&L.name_buf, fmt, args));
    }

    /// The host bytes gathered so far uploaded as one tensor named `fmt` (WP4 experts.load's put).
    pub fn put(L: *Loader, comptime fmt: []const u8, args: anytype) !u64 {
        const name = try std.fmt.allocPrint(L.gpa, fmt, args);
        const ptr = try L.keep(name, .raw, L.host.items, null, L.host.items.len, 1);
        L.trimHost();
        return ptr;
    }

    fn trimHost(L: *Loader) void {
        if (L.host.capacity > host_keep_bytes and !L.hold_host) L.host.clearAndFree(L.gpa) else L.host.clearRetainingCapacity();
    }

    /// Account (plan) or allocate an entry of `len` bytes (+ scales) named `name` (owned on success).
    fn reserve(L: *Loader, name: []u8, dtype: DType, len: usize, scale_len: ?usize, rows: usize, cols: usize) !?*ResidentTensor {
        const cls = classOf(name, dtype);
        if (L.plan) |w| {
            defer L.gpa.free(name);
            addClass(w, cls, placed(len) + if (scale_len) |n| placed(n) else 0);
            if (L.plan_logical) |lw| addClass(lw, cls, len + (scale_len orelse 0));
            return null;
        }
        errdefer L.gpa.free(name);
        const m = L.model.?;
        // Carved from the class arena (review N3): no per-tensor cuMemAlloc, so the admitted bytes are exact.
        const data = try m.carve(cls, len);
        const scales: ?cuda.DeviceBuffer = if (scale_len) |n| try m.carve(cls, n) else null;
        try m.tensors.append(L.gpa, .{ .name = name, .dtype = dtype, .data = data, .scales = scales, .rows = rows, .cols = cols });
        return &m.tensors.items[m.tensors.items.len - 1];
    }

    fn write(L: *Loader, buf: cuda.DeviceBuffer, offset: usize, bytes: []const u8) !void {
        const b = L.budget.?;
        if (bytes.len != 0) try b.device.write_fn(b.device.ctx, buf, offset, bytes);
    }

    /// Upload `bytes` (and optional scale bytes) as one entry; 0 in plan mode.
    fn keep(L: *Loader, name: []u8, dtype: DType, bytes: []const u8, scale_bytes: ?[]const u8, rows: usize, cols: usize) !u64 {
        const t = (try L.reserve(name, dtype, bytes.len, if (scale_bytes) |s| s.len else null, rows, cols)) orelse return 0;
        try L.write(t.data, 0, bytes);
        if (scale_bytes) |s| try L.write(t.scales.?, 0, s);
        return t.data.ptr;
    }

    fn label(L: *Loader, comptime fmt: []const u8, args: anytype) ![]u8 {
        return std.fmt.allocPrint(L.gpa, fmt, args);
    }
};

/// A linear on the qmmf lane: FP8 with weight_scale_inv becomes `.fp8b` (codes in fragment order, native scales
/// expanded to 64x64 lookups); a bf16/f32 weight becomes `.bf16`. Parts stack along output rows. Each part's
/// weight is read once, and only its own rows when the columns are whole.
fn linear(L: *Loader, parts: []const Part, cols: Span) !qmmf.Weight {
    var nb: [320]u8 = undefined;
    const first = try L.header(try std.fmt.bufPrint(&nb, "{s}weight", .{parts[0].prefix}));
    const is_f8 = first.dtype == .f8_e4m3;
    const full_cols = first.dim(1);
    var n: usize = 0;
    for (parts) |p| {
        const t = try L.header(try std.fmt.bufPrint(&nb, "{s}weight", .{p.prefix}));
        const scale_raw = try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{p.prefix});
        const scaled = L.hasRaw(scale_raw);
        if (scaled) _ = try L.header(scale_raw); // marked used in plan mode too
        if ((t.dtype == .f8_e4m3) != is_f8 or scaled != is_f8 or t.rank != 2 or t.dim(1) != full_cols or p.rows.hi > t.dim(0)) return error.MixedLinearParts;
        if (is_f8 and (p.rows.lo % fp8.group != 0 or (parts.len > 1 and (p.rows.lo % fp8.block != 0 or p.rows.len() % fp8.block != 0)))) return error.BadLinearShape;
        n += p.rows.len();
    }
    const k = cols.len();
    if (k == 0 or n == 0 or k % fp8.group != 0 or cols.lo % fp8.group != 0 or cols.hi > full_cols) return error.BadLinearShape;
    const name = try L.label("{s}weight", .{parts[0].prefix});
    if (is_f8 and backboneFp4() and fp4Eligible(parts[0].prefix)) return linearFp8AsFp4(L, name, parts, cols, n, k, full_cols);
    if (is_f8) return linearFp8(L, name, parts, cols, n, k, full_cols);
    return linearBf16(L, name, parts, cols, n, k, full_cols);
}

fn rowsOf(L: *Loader, raw: []const u8, rows: Span, whole_cols: bool) !CkTensor {
    return if (whole_cols) L.tensorRows(raw, rows) else L.tensor(raw);
}

fn linearFp8(L: *Loader, name: []u8, parts: []const Part, cols: Span, n: usize, k: usize, full_cols: usize) !qmmf.Weight {
    var nb: [320]u8 = undefined;
    const packed_len = fp8.padded(n) * k;
    const scales_len = fp8.padded(n) / fp8.group * (k / fp8.group) * 4;
    const t_out = (try L.reserve(name, .e4m3, packed_len, scales_len, n, k)) orelse return fp8.blockWeight(0, 0, n, k);
    const whole_cols = cols.lo == 0 and cols.hi == full_cols;
    const kb = (full_cols + fp8.block - 1) / fp8.block;
    const codes = try L.gpa.alloc(u8, n * k);
    defer L.gpa.free(codes);
    var scale: std.ArrayList(f32) = .empty;
    defer scale.deinit(L.gpa);
    var at: usize = 0;
    var src_rows: usize = 0;
    for (parts) |p| {
        const raw = try std.fmt.bufPrint(&nb, "{s}weight", .{p.prefix});
        const t = try rowsOf(L, raw, p.rows, whole_cols);
        const r0 = if (whole_cols) 0 else p.rows.lo;
        for (0..p.rows.len()) |r| {
            @memcpy(codes[at..][0..k], t.bytes[(r0 + r) * full_cols + cols.lo ..][0..k]);
            at += k;
        }
        const full_rows = (try L.header(raw)).dim(0);
        src_rows = full_rows;
        const s = try L.tensor(try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{p.prefix}));
        if (s.dtype != .f32 or s.numel() != ((full_rows + fp8.block - 1) / fp8.block) * kb) return error.BadScaleShape;
        const b0 = if (parts.len == 1) 0 else p.rows.lo / fp8.block;
        const b1 = if (parts.len == 1) s.numel() / kb else (p.rows.hi + fp8.block - 1) / fp8.block;
        for (b0 * kb..b1 * kb) |i| try scale.append(L.gpa, std.mem.bytesToValue(f32, s.bytes[i * 4 ..][0..4]));
    }
    const packed_codes = try L.gpa.alloc(u8, packed_len);
    defer L.gpa.free(packed_codes);
    fp8.packCodes(packed_codes, codes, n, k);
    const packed_scales = try L.gpa.alloc(f32, scales_len / 4);
    defer L.gpa.free(packed_scales);
    try fp8.packBlockScales(packed_scales, scale.items, if (parts.len == 1) src_rows else n, full_cols, if (parts.len == 1) parts[0].rows.lo else 0, cols.lo, n, k);
    try L.write(t_out.data, 0, packed_codes);
    try L.write(t_out.scales.?, 0, std.mem.sliceAsBytes(packed_scales));
    return fp8.blockWeight(t_out.data.ptr, t_out.scales.?.ptr, n, k);
}

/// bf16 lane weights, packed and written in blocks of `block_rows` rows (multiples of 128, so each block is a
/// contiguous span of packBf16's row-group-major layout): lm_head is never resident on the host whole.
fn linearBf16(L: *Loader, name: []u8, parts: []const Part, cols: Span, n: usize, k: usize, full_cols: usize) !qmmf.Weight {
    var nb: [320]u8 = undefined;
    const npad = (n + 127) / 128 * 128;
    const t_out = (try L.reserve(name, .bf16, npad * k * 2, null, n, k)) orelse return .{ .mode = .bf16, .codes = 0, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
    const whole_cols = cols.lo == 0 and cols.hi == full_cols;
    var out_row: usize = 0; // rows written so far, across parts
    for (parts) |p| {
        if (parts.len > 1 and p.rows.len() % 128 != 0) return error.BadLinearShape;
        const raw = try std.fmt.bufPrint(&nb, "{s}weight", .{p.prefix});
        var lo = p.rows.lo;
        while (lo < p.rows.hi) {
            const hi = @min(lo + block_rows, p.rows.hi);
            const rows = hi - lo;
            const t = try rowsOf(L, raw, .{ .lo = lo, .hi = hi }, whole_cols);
            const r0 = if (whole_cols) 0 else lo;
            const words = try L.gpa.alloc(u16, rows * k);
            defer L.gpa.free(words);
            for (0..rows) |r| for (0..k) |c| {
                words[r * k + c] = try bf16Word(t, (r0 + r) * full_cols + cols.lo + c);
            };
            const bpad = (rows + 127) / 128 * 128;
            const out = try L.gpa.alloc(u16, bpad * k);
            defer L.gpa.free(out);
            qmmf.packBf16(out, words, rows, k);
            try L.write(t_out.data, out_row * k * 2, std.mem.sliceAsBytes(out));
            out_row += rows;
            lo = hi;
        }
    }
    return .{ .mode = .bf16, .codes = t_out.data.ptr, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(npad) };
}

/// TF_GLM53_BACKBONE_FP4=1: the block-FP8 projections only the lane matmul reads (o_proj, q_b_proj, the shared and
/// dense MLPs) load as NVFP4 instead (e2m1 codes, an e4m3 scale every 16 inputs, one fp32 global), so every window
/// reads about half their bytes. kv_b (read by the MLA kernels) and the grouped q_a/kv_a/indexer projections stay FP8.
pub fn backboneFp4() bool {
    const v = std.c.getenv("TF_GLM53_BACKBONE_FP4") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn fp4Eligible(prefix: []const u8) bool {
    const has = struct {
        fn f(s: []const u8, part: []const u8) bool {
            return std.mem.indexOf(u8, s, part) != null;
        }
    }.f;
    if (has(prefix, "self_attn.o_proj.") or has(prefix, "self_attn.q_b_proj.") or has(prefix, "mlp.shared_experts.")) return true;
    // The first dense layers' MLP (the MoE layers' routed experts are EXL3 and never reach linear()).
    return has(prefix, ".mlp.") and !has(prefix, "experts") and (has(prefix, "gate_proj.") or has(prefix, "up_proj.") or has(prefix, "down_proj."));
}

/// Every e4m3 code's value (fp8bWeight at scale 1: exact, since bf16 holds any e4m3 value).
fn e4m3Table() [256]f32 {
    var t: [256]f32 = undefined;
    for (&t, 0..) |*v, i| v.* = fp8.fp8bWeight(@intCast(i), 1);
    return t;
}

/// The e4m3 byte (positive, finite) at or below `x`, and the one above it: a block tries both.
fn e4m3Around(x: f32, lut: *const [256]f32) [2]u8 {
    var lo: u8 = 0;
    var hi: u8 = 0x7e; // 448, the largest finite e4m3
    if (x >= lut[hi]) return .{ hi, hi };
    while (hi - lo > 1) {
        const mid = lo + (hi - lo) / 2;
        if (lut[mid] <= x) lo = mid else hi = mid;
    }
    return .{ lo, hi };
}

/// The e2m1 nibble nearest `x` (sign in bit 3), saturating at 6.
fn e2m1Code(x: f32) u8 {
    const a = @abs(x);
    const mag: u8 = if (a <= 0.25) 0 else if (a <= 0.75) 1 else if (a <= 1.25) 2 else if (a <= 1.75) 3 else if (a <= 2.5) 4 else if (a <= 3.5) 5 else if (a <= 5) 6 else 7;
    return if (x < 0 and mag != 0) mag | 8 else mag;
}

const e2m1_values = [8]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };

fn e2m1Value(code: u8) f32 {
    const v = e2m1_values[code & 7];
    return if (code & 8 != 0) -v else v;
}

/// Quantize one block of 16 values at the global scale `g`: the e4m3 scale (of the two around the ideal one) whose
/// codes give the smaller squared error, written into `codes` (nibbles, unpacked) and returned.
fn fp4Block(values: []const f32, g: f32, codes: []u8, lut: *const [256]f32) u8 {
    var amax: f32 = 0;
    for (values) |v| amax = @max(amax, @abs(v));
    if (amax == 0) {
        @memset(codes, 0);
        return 0;
    }
    const around = e4m3Around(amax / (6 * g), lut);
    var best: u8 = around[0];
    var best_err = std.math.inf(f32);
    for (around) |b| {
        const s = lut[b] * g;
        if (s == 0) continue;
        var err: f32 = 0;
        for (values) |v| {
            const d = v - e2m1Value(e2m1Code(v / s)) * s;
            err += d * d;
        }
        if (err < best_err) {
            best_err = err;
            best = b;
        }
    }
    const s = lut[best] * g;
    for (values, codes) |v, *c| c.* = if (s == 0) 0 else e2m1Code(v / s);
    return best;
}

/// Host threads for the NVFP4 conversion: on one thread it added about 4 minutes to every load.
const fp4_workers = 12;

/// One worker's rows (r0..r1) and, for packing, 64-row word tiles (t0..t1) of linearFp8AsFp4.
const Fp4Job = struct {
    codes: []const u8,
    scale: []const f32,
    lut: *const [256]f32,
    values: []f32,
    nibbles: []u8,
    block_scales: []u8,
    packed_codes: []u8,
    words: []u32,
    n: usize,
    k: usize,
    kb: usize,
    row0: usize,
    col0: usize,
    r0: usize,
    r1: usize,
    t0: usize = 0,
    t1: usize = 0,
    global: f32 = 1,
    amax: f32 = 0,
    err2: f64 = 0,
    ref2: f64 = 0,

    /// fp8bWeight without its per-call pow: the code's value times scale_inv, rounded to bf16.
    fn dequant(j: *Fp4Job) void {
        for (j.r0..j.r1) |r| for (0..j.k) |c| {
            const v = fp8.roundBf16(j.lut[j.codes[r * j.k + c]] * j.scale[((j.row0 + r) / fp8.block) * j.kb + (j.col0 + c) / fp8.block]);
            j.values[r * j.k + c] = v;
            j.amax = @max(j.amax, @abs(v));
        };
    }

    /// The rows' blocks of 16 (fp4Block), their error against the dequantized values, and their nibbles packed in
    /// pairs (even column in the low nibble).
    fn quantize(j: *Fp4Job) void {
        const k = j.k;
        for (j.r0..j.r1) |r| {
            for (0..k / 16) |b| {
                const at = r * k + b * 16;
                const sb = fp4Block(j.values[at..][0..16], j.global, j.nibbles[at..][0..16], j.lut);
                j.block_scales[r * (k / 16) + b] = sb;
                const s = j.lut[sb] * j.global;
                for (j.values[at..][0..16], j.nibbles[at..][0..16]) |v, c| {
                    const d: f64 = v - e2m1Value(c) * s;
                    j.err2 += d * d;
                    j.ref2 += @as(f64, v) * v;
                }
            }
            for (0..k / 2) |i| j.packed_codes[r * (k / 2) + i] = j.nibbles[r * k + 2 * i] | (j.nibbles[r * k + 2 * i + 1] << 4);
        }
    }

    /// cuda.nvfp4.packWords over tiles t0..t1 only (int32 words [npad/64][k/64][8][32][2], rows past n zero).
    fn packTiles(j: *Fp4Job) void {
        const kg = j.k / 64;
        const offs = [8]usize{ 0, 8, 16, 24, 1, 9, 17, 25 };
        for (j.t0 * kg * 512..j.t1 * kg * 512) |idx| {
            const v = idx & 1;
            const lane = (idx >> 1) & 31;
            const jj = (idx >> 6) & 7;
            const tg = idx >> 9;
            const g = tg % kg;
            const row = (tg / kg) * 64 + jj * 8 + (lane >> 2);
            var word: u32 = 0;
            if (row < j.n) for (offs, 0..) |off, p| {
                const input = g * 64 + 32 * v + 2 * (lane & 3) + off;
                const byte = j.packed_codes[row * (j.k / 2) + input / 2];
                const nib: u32 = if (input & 1 == 0) byte & 0xF else byte >> 4;
                word |= nib << @intCast(4 * p);
            };
            j.words[idx] = word;
        }
    }
};

/// Run `step` over every job on its own thread and wait for them; a thread that cannot start runs its job inline.
fn fp4Run(jobs: []Fp4Job, comptime step: fn (*Fp4Job) void) void {
    var threads: [fp4_workers]?std.Thread = @splat(null);
    for (jobs, 0..) |*j, i| threads[i] = std.Thread.spawn(.{}, step, .{j}) catch null;
    for (jobs, 0..) |*j, i| {
        if (threads[i]) |t| t.join() else step(j);
    }
}

/// linearFp8's tensor, dequantized exactly as the FP8 lane reads it (fp8bWeight, 128x128 scale_inv blocks) and
/// re-quantized to NVFP4 for the lane matmul (cuda.nvfp4 layout).
fn linearFp8AsFp4(L: *Loader, name: []u8, parts: []const Part, cols: Span, n: usize, k: usize, full_cols: usize) !qmmf.Weight {
    var nb: [320]u8 = undefined;
    if (k % 64 != 0) return error.BadLinearShape;
    const npad = cuda.nvfp4.padded(n);
    const words_len = npad / 64 * (k / 64) * 512;
    const scales_len = npad * (k / 16);
    const t_out = (try L.reserve(name, .e4m3, words_len * 4, scales_len, n, k)) orelse return cuda.nvfp4.weight(0, 0, 1, n, k);
    const whole_cols = cols.lo == 0 and cols.hi == full_cols;
    const kb = (full_cols + fp8.block - 1) / fp8.block;
    const codes = try L.gpa.alloc(u8, n * k);
    defer L.gpa.free(codes);
    var scale: std.ArrayList(f32) = .empty;
    defer scale.deinit(L.gpa);
    var at: usize = 0;
    for (parts) |p| {
        const raw = try std.fmt.bufPrint(&nb, "{s}weight", .{p.prefix});
        const t = try rowsOf(L, raw, p.rows, whole_cols);
        const r0 = if (whole_cols) 0 else p.rows.lo;
        for (0..p.rows.len()) |r| {
            @memcpy(codes[at..][0..k], t.bytes[(r0 + r) * full_cols + cols.lo ..][0..k]);
            at += k;
        }
        const full_rows = (try L.header(raw)).dim(0);
        const s = try L.tensor(try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{p.prefix}));
        if (s.dtype != .f32 or s.numel() != ((full_rows + fp8.block - 1) / fp8.block) * kb) return error.BadScaleShape;
        const b0 = if (parts.len == 1) 0 else p.rows.lo / fp8.block;
        const b1 = if (parts.len == 1) s.numel() / kb else (p.rows.hi + fp8.block - 1) / fp8.block;
        for (b0 * kb..b1 * kb) |i| try scale.append(L.gpa, std.mem.bytesToValue(f32, s.bytes[i * 4 ..][0..4]));
    }
    // One part keeps the source tensor's block rows (row0 its first row); stacked parts are 128-row aligned and their
    // block rows were appended in order, as fp8.packBlockScales reads them.
    const row0 = if (parts.len == 1) parts[0].rows.lo else 0;
    const lut = e4m3Table();
    const values = try L.gpa.alloc(f32, n * k);
    defer L.gpa.free(values);
    const nibbles = try L.gpa.alloc(u8, n * k);
    defer L.gpa.free(nibbles);
    const block_scales = try L.gpa.alloc(u8, n * (k / 16));
    defer L.gpa.free(block_scales);
    const packed_codes = try L.gpa.alloc(u8, n * k / 2);
    defer L.gpa.free(packed_codes);
    const words = try L.gpa.alloc(u32, words_len);
    defer L.gpa.free(words);
    // The rows split over fp4_workers threads: dequantize (and each worker's amax), then, with the global scale,
    // quantize and pack nibbles; then the lane's word tiles.
    var jobs: [fp4_workers]Fp4Job = undefined;
    const per = (n + fp4_workers - 1) / fp4_workers;
    for (&jobs, 0..) |*j, i| j.* = .{ .codes = codes, .scale = scale.items, .lut = &lut, .values = values, .nibbles = nibbles, .block_scales = block_scales, .packed_codes = packed_codes, .words = words, .n = n, .k = k, .kb = kb, .row0 = row0, .col0 = cols.lo, .r0 = @min(n, i * per), .r1 = @min(n, (i + 1) * per) };
    fp4Run(&jobs, Fp4Job.dequant);
    var amax: f32 = 0;
    for (jobs) |j| amax = @max(amax, j.amax);
    const global: f32 = if (amax > 0) amax / (6 * 448) else 1;
    for (&jobs) |*j| j.global = global;
    fp4Run(&jobs, Fp4Job.quantize);
    var err2: f64 = 0;
    var ref2: f64 = 0;
    for (jobs) |j| {
        err2 += j.err2;
        ref2 += j.ref2;
    }
    const tiles = npad / 64;
    const tper = (tiles + fp4_workers - 1) / fp4_workers;
    for (&jobs, 0..) |*j, i| {
        j.t0 = @min(tiles, i * tper);
        j.t1 = @min(tiles, (i + 1) * tper);
    }
    fp4Run(&jobs, Fp4Job.packTiles);
    const packed_scales = try L.gpa.alloc(u8, scales_len);
    defer L.gpa.free(packed_scales);
    cuda.nvfp4.packScales(packed_scales, block_scales, n, k);
    try L.write(t_out.data, 0, std.mem.sliceAsBytes(words));
    try L.write(t_out.scales.?, 0, packed_scales);
    if (std.mem.indexOf(u8, parts[0].prefix, "layers.3.") != null)
        std.log.info("glm53 backbone fp4: {s} {d}x{d} relative rms error {d:.4}", .{ parts[0].prefix, n, k, if (ref2 > 0) @sqrt(err2 / ref2) else 0 });
    return cuda.nvfp4.weight(t_out.data.ptr, t_out.scales.?.ptr, global, n, k);
}

/// kv_b as stored: row-major E4M3 rows of this rank's heads and their scale_inv rows (128-row aligned).
fn rawFp8(L: *Loader, prefix: []const u8, rows: Span) !RawFp8 {
    var nb: [320]u8 = undefined;
    const raw = try std.fmt.bufPrint(&nb, "{s}weight", .{prefix});
    const h = try L.header(raw);
    const cols = h.dim(1);
    const kb = (cols + fp8.block - 1) / fp8.block;
    const s = try L.header(try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{prefix}));
    if (h.dtype != .f8_e4m3 or h.rank != 2 or s.dtype != .f32 or rows.lo % fp8.block != 0 or rows.hi > h.dim(0) or
        (rows.hi != h.dim(0) and rows.hi % fp8.block != 0) or s.numel() != ((h.dim(0) + fp8.block - 1) / fp8.block) * kb)
        return error.BadKvbShape;
    const b0 = rows.lo / fp8.block;
    const b1 = (rows.hi + fp8.block - 1) / fp8.block;
    const name = try L.label("{s}weight", .{prefix});
    const t_out = (try L.reserve(name, .e4m3, rows.len() * cols, (b1 - b0) * kb * 4, rows.len(), cols)) orelse return .{ .codes = 0, .scales = 0, .rows = rows.len(), .cols = cols };
    const codes = try L.tensorRows(raw, rows);
    try L.write(t_out.data, 0, codes.bytes);
    const sc = try L.tensorRows(try std.fmt.bufPrint(&nb, "{s}weight_scale_inv", .{prefix}), .{ .lo = b0, .hi = b1 });
    try L.write(t_out.scales.?, 0, sc.bytes);
    return .{ .codes = t_out.data.ptr, .scales = t_out.scales.?.ptr, .rows = rows.len(), .cols = cols };
}

/// Rows x columns of a bf16/f32 tensor (a vector: rows of one element), converted to `to`, written in row blocks.
fn plain(L: *Loader, raw: []const u8, rows: ?Span, cols: ?Span, to: To) !u64 {
    const h = try L.header(raw);
    const nrows = if (h.rank >= 2) h.dim(0) else h.numel();
    const ncols = if (h.rank >= 2) h.numel() / nrows else 1;
    const r = rows orelse Span{ .lo = 0, .hi = nrows };
    const c = cols orelse Span{ .lo = 0, .hi = ncols };
    if (r.hi > nrows or c.hi > ncols or r.lo >= r.hi or c.lo >= c.hi) return error.BadTensorSlice;
    const out: DType = switch (to) {
        .bf16 => .bf16,
        .f32 => .f32,
        .same => switch (h.dtype) {
            .bf16 => .bf16,
            .f32 => .f32,
            .f16 => .bf16,
            else => return error.UnsupportedCheckpointDtype,
        },
    };
    const elem: usize = if (out == .bf16) 2 else 4;
    const t_out = (try L.reserve(try L.gpa.dupe(u8, raw), out, r.len() * c.len() * elem, null, r.len(), c.len())) orelse return 0;
    var lo = r.lo;
    while (lo < r.hi) {
        const hi = @min(lo + block_rows, r.hi);
        const t = try L.tensorRows(raw, .{ .lo = lo, .hi = hi });
        L.host.clearRetainingCapacity();
        for (0..hi - lo) |row| for (c.lo..c.hi) |col| {
            const i = row * ncols + col;
            switch (out) {
                .bf16 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&(try bf16Word(t, i)))),
                .f32 => try L.host.appendSlice(L.gpa, std.mem.asBytes(&(try f32Value(t, i)))),
                else => unreachable,
            }
        };
        try L.write(t_out.data, (lo - r.lo) * c.len() * elem, L.host.items);
        lo = hi;
    }
    L.trimHost();
    return t_out.data.ptr;
}

/// Element `i` of a bf16/f32/f16 tensor as bf16 bits (fp32/fp16 rounds to nearest even, as torch's cast does).
fn bf16Word(t: CkTensor, i: usize) !u16 {
    return switch (t.dtype) {
        .bf16 => std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little),
        .f32 => toBf16(std.mem.bytesToValue(f32, t.bytes[i * 4 ..][0..4])),
        .f16 => toBf16(@floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little))))),
        else => error.UnsupportedCheckpointDtype,
    };
}

fn f32Value(t: CkTensor, i: usize) !f32 {
    return switch (t.dtype) {
        .bf16 => @bitCast(@as(u32, std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)) << 16),
        .f32 => std.mem.bytesToValue(f32, t.bytes[i * 4 ..][0..4]),
        .f16 => @floatCast(@as(f16, @bitCast(std.mem.readInt(u16, t.bytes[i * 2 ..][0..2], .little)))),
        else => error.UnsupportedCheckpointDtype,
    };
}

/// Host bf16 from fp32: round to nearest even, NaN kept quiet (glm/cuda_weights.zig toBf16).
pub fn toBf16(v: f32) u16 {
    const b: u32 = @bitCast(v);
    if (std.math.isNan(v)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn routedCodebook(c: *const Config) experts.Codebook {
    return switch (c.routed_codebook) {
        .mcg => .mcg,
        .mcg_e4m3 => .mcg_e4m3,
        .mul1 => .mul1,
    };
}

/// The largest per-projection host staging WP4's experts.load builds (trellis tiles of this rank), from headers.
fn expertsHostBytes(L: *Loader, layer: usize, count: usize, dims: usize, lo: usize, hi: usize) !usize {
    var most: usize = 0;
    var nb: [320]u8 = undefined;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        var bytes: usize = 0;
        for (0..count) |e| {
            const t = try L.header(try std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, proj }));
            const local_k = if (pi == 2) hi - lo else dims;
            const local_n = if (pi == 2) dims else hi - lo;
            bytes += (local_k / 16) * (local_n / 16) * 16 * (t.dim(2) / 8);
        }
        most = @max(most, bytes);
    }
    return most;
}

/// WP4 experts.load's exact uploads, from the headers only (plan mode): per projection the sliced trellis tiles,
/// the pointer table, the K2 table and the suh/svh slices; every tensor it would read is marked used.
fn planExperts(L: *Loader, layer: usize, count: usize, dims: usize, full_width: usize, lo: usize, hi: usize) !void {
    const w = L.plan.?;
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        const down = pi == 2;
        var trellis: u64 = 0;
        var nb: [320]u8 = undefined;
        for (0..count) |e| {
            const t = try L.header(try std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, e, proj }));
            if (t.rank != 3 or t.dim(2) % 8 != 0) return error.BadExpertTensor;
            const k2 = t.dim(2) / 8;
            const local_k = if (down) hi - lo else dims;
            const local_n = if (down) dims else hi - lo;
            trellis += (local_k / 16) * (local_n / 16) * 16 * k2;
            for ([_][]const u8{ "suh", "svh", "mcg", "mcg_e4m3", "mul1" }) |part| {
                const raw = try std.fmt.bufPrint(&nb, "layers.{d}.mlp.experts.{d}.{s}.{s}", .{ layer, e, proj, part });
                if (L.hasRaw(raw)) _ = try L.header(raw);
            }
        }
        const suh: u64 = count * (if (down) hi - lo else dims) * 2;
        const svh: u64 = count * (if (down) dims else hi - lo) * 2;
        // Five puts a projection (trellis, pointers, K2, suh, svh), each placed in the routed arena.
        w.routed += placed(trellis) + placed(count * 8) + placed(count * 4) + placed(suh) + placed(svh);
        if (L.plan_logical) |lw| lw.routed += trellis + count * 8 + count * 4 + suh + svh;
    }
    _ = full_width;
}

fn loadLayer(L: *Loader, c: *const Config, sh: share_mod.Share, i: usize) !DeviceLayer {
    var pb: [64]u8 = undefined;
    var nb: [320]u8 = undefined;
    const p = try std.fmt.bufPrint(&pb, "layers.{d}.", .{i});
    const hidden: Span = .{ .lo = 0, .hi = c.hidden };
    if (L.select.shard_qkv_a and (c.q_lora != 2048 or c.kv_lora + c.rope != 576)) return error.UnsupportedShardQkvAShape;
    const q_a_rows = if (L.select.shard_qkv_a) try shardQkvARows(sh.rank, true) else Span{ .lo = 0, .hi = c.q_lora };
    const kv_a_rows = if (L.select.shard_qkv_a) try shardQkvARows(sh.rank, false) else Span{ .lo = 0, .hi = c.kv_lora + c.rope };
    const q_head = c.nope + c.rope;
    const kv_head = c.nope + c.v_dim;
    const heads: Span = .{ .lo = sh.heads.lo, .hi = sh.heads.hi };
    const in_norm = try plain(L, try std.fmt.bufPrint(&nb, "{s}input_layernorm.weight", .{p}), null, null, .bf16);
    const post_norm = try plain(L, try std.fmt.bufPrint(&nb, "{s}post_attention_layernorm.weight", .{p}), null, null, .bf16);
    var a: Attention = undefined;
    a.local_heads = heads.len();
    a.head0 = heads.lo;
    a.q_a = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.q_a_proj.", .{p}), .rows = q_a_rows }}, hidden);
    a.q_b = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.q_b_proj.", .{p}), .rows = .{ .lo = heads.lo * q_head, .hi = heads.hi * q_head } }}, .{ .lo = 0, .hi = c.q_lora });
    a.kv_a = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.kv_a_proj_with_mqa.", .{p}), .rows = kv_a_rows }}, hidden);
    a.kv_b = try rawFp8(L, try std.fmt.bufPrint(&nb, "{s}self_attn.kv_b_proj.", .{p}), .{ .lo = heads.lo * kv_head, .hi = heads.hi * kv_head });
    a.o_proj = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.o_proj.", .{p}), .rows = hidden }}, .{ .lo = heads.lo * c.v_dim, .hi = heads.hi * c.v_dim });
    a.q_a_norm = try plain(L, try std.fmt.bufPrint(&nb, "{s}self_attn.q_a_layernorm.weight", .{p}), null, null, .bf16);
    a.kv_a_norm = try plain(L, try std.fmt.bufPrint(&nb, "{s}self_attn.kv_a_layernorm.weight", .{p}), null, null, .bf16);
    a.indexer = if (c.full.isSet(i)) .{
        .wq_b = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.indexer.wq_b.", .{p}), .rows = .{ .lo = 0, .hi = c.i_heads * c.i_dim } }}, .{ .lo = 0, .hi = c.q_lora }),
        .wk = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.indexer.wk.", .{p}), .rows = .{ .lo = 0, .hi = c.i_dim } }}, hidden),
        .weights_proj = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}self_attn.indexer.weights_proj.", .{p}), .rows = .{ .lo = 0, .hi = c.i_heads } }}, hidden),
        .k_norm = try plain(L, try std.fmt.bufPrint(&nb, "{s}self_attn.indexer.k_norm.weight", .{p}), null, null, .bf16),
        .k_bias = try plain(L, try std.fmt.bufPrint(&nb, "{s}self_attn.indexer.k_norm.bias", .{p}), null, null, .bf16),
    } else null;
    var gb: [320]u8 = undefined;
    var ub: [320]u8 = undefined;
    const a_name = try std.fmt.bufPrint(&gb, "{s}mlp.lowrank_a", .{p});
    const b_name = try std.fmt.bufPrint(&ub, "{s}mlp.lowrank_b", .{p});
    const has_a = L.hasRaw(a_name);
    if (has_a != L.hasRaw(b_name)) return error.LowRankPairIncomplete;
    if (!c.isMoe(@intCast(i))) {
        if (has_a) return error.LowRankOnDenseLayer;
        const span: Span = .{ .lo = sh.dense.lo, .hi = sh.dense.hi };
        const gate_up = try linear(L, &.{
            .{ .prefix = try std.fmt.bufPrint(&gb, "{s}mlp.gate_proj.", .{p}), .rows = span },
            .{ .prefix = try std.fmt.bufPrint(&ub, "{s}mlp.up_proj.", .{p}), .rows = span },
        }, hidden);
        const down = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}mlp.down_proj.", .{p}), .rows = hidden }}, span);
        return .{ .index = i, .in_norm = in_norm, .post_norm = post_norm, .attn = a, .ffn = .{ .dense = .{ .gate_up = gate_up, .down = down } } };
    }
    var m: Moe = undefined;
    m.lowrank = null;
    if (has_a) {
        const at = try L.header(a_name);
        const bt = try L.header(b_name);
        if (at.dtype != .bf16 or bt.dtype != .bf16 or at.rank != 2 or bt.rank != 2 or at.dim(0) != c.hidden or bt.dim(1) != c.hidden or
            at.dim(1) != bt.dim(0) or at.dim(1) % sh.world != 0) return error.LowRankShape;
        const full_r = at.dim(1);
        // weights.py refuses low-rank ranks that differ across layers; the workspace holds R/2 <= 64 (review L4).
        if (L.lowrank_r) |r| if (r != full_r) return error.LowRankRankMismatch;
        L.lowrank_r = full_r;
        const r_span: Span = .{ .lo = full_r * sh.rank / sh.world, .hi = full_r * (sh.rank + 1) / sh.world };
        if (r_span.len() > state.lowrank_max) return error.LowRankTooWide;
        const la = try plain(L, a_name, null, r_span, .f32);
        const lb = try plain(L, b_name, r_span, null, .f32);
        m.lowrank = .{ .a = la, .b = lb, .rank_local = r_span.len(), .rank_full = full_r, .span = r_span };
    }
    m.router = try plain(L, try std.fmt.bufPrint(&nb, "{s}mlp.gate.weight", .{p}), null, null, .bf16);
    m.bias = try plain(L, try std.fmt.bufPrint(&nb, "{s}mlp.gate.e_score_correction_bias", .{p}), null, null, .f32);
    const moe_span = sh.moeForLayer(i);
    if (L.plan != null) {
        try planExperts(L, i, c.experts, c.hidden, c.moe_inter, moe_span.lo, moe_span.hi);
        m.routed = undefined;
    } else {
        // One exact reservation for WP4's host staging (the largest projection's rank half), so the ArrayList never
        // grows by doubling through about 0.4 GiB (review N9). experts.load reads at most 5 tensors per expert before
        // it consumes them; the O_DIRECT ring holds Direct.ring_len = 8.
        try L.host.ensureTotalCapacityPrecise(L.gpa, try expertsHostBytes(L, i, c.experts, c.hidden, moe_span.lo, moe_span.hi));
        L.hold_host = true;
        defer {
            L.hold_host = false;
            L.trimHost();
        }
        m.routed = try experts.load(L, i, c.experts, c.hidden, c.moe_inter, moe_span.lo, moe_span.hi, routedCodebook(c));
    }
    const shared_span = sh.sharedForLayer(i);
    const shared: Span = .{ .lo = shared_span.lo, .hi = shared_span.hi };
    m.shared = .{
        .gate_up = try linear(L, &.{
            .{ .prefix = try std.fmt.bufPrint(&gb, "{s}mlp.shared_experts.gate_proj.", .{p}), .rows = shared },
            .{ .prefix = try std.fmt.bufPrint(&ub, "{s}mlp.shared_experts.up_proj.", .{p}), .rows = shared },
        }, hidden),
        .down = try linear(L, &.{.{ .prefix = try std.fmt.bufPrint(&nb, "{s}mlp.shared_experts.down_proj.", .{p}), .rows = hidden }}, shared),
    };
    return .{ .index = i, .in_norm = in_norm, .post_norm = post_norm, .attn = a, .ffn = .{ .moe = m } };
}

pub const Select = struct {
    /// Layers to load; null loads all of them. A contiguous prefix (bringUp) runs the TP2 forward on a subset.
    layers: ?[]const usize = null,
    /// Embedding half, final norm and the rank's lm_head rows.
    globals: bool = true,
    manifest: ?*const manifest_mod.Manifest = null,
    /// Load only this TP rank's q_a and kv_a output rows.
    shard_qkv_a: bool = false,

    /// Bring-up (design §7 L1 at TP2 + DCP2): the first `count` layers and the globals.
    pub fn bringUp(store: []usize, count: usize) Select {
        for (store[0..count], 0..) |*l, i| l.* = i;
        return .{ .layers = store[0..count] };
    }
};

/// Output row spans for the opt-in q_a / kv_a TP split. Input columns remain whole in loadLayer.
fn shardQkvARows(rank: usize, is_q: bool) !Span {
    if (rank > 1) return error.BadRank;
    if (is_q) return .{ .lo = rank * 1024, .hi = (rank + 1) * 1024 };
    return if (rank == 0) .{ .lo = 0, .hi = 256 } else .{ .lo = 256, .hi = 576 };
}

test "sharded q_a and kv_a loader spans cover the exact rows per rank" {
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 1024 }, try shardQkvARows(0, true));
    try std.testing.expectEqual(Span{ .lo = 1024, .hi = 2048 }, try shardQkvARows(1, true));
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 256 }, try shardQkvARows(0, false));
    try std.testing.expectEqual(Span{ .lo = 256, .hi = 576 }, try shardQkvARows(1, false));
}

const Selector = struct {
    m: *const manifest_mod.Manifest,
    layers: ?[]const usize,
    stray: usize = 0,
    fn keep(raw: *anyopaque, layer: u32, width: u32, tensor: []const u8) bool {
        const s: *Selector = @ptrCast(@alignCast(raw));
        const rest = if (std.mem.startsWith(u8, tensor, "model.language_model.layers.")) tensor["model.language_model.layers.".len..] else if (std.mem.startsWith(u8, tensor, "model.layers.")) tensor["model.layers.".len..] else return false;
        var it = std.mem.splitScalar(u8, rest, '.');
        const l = std.fmt.parseInt(usize, it.next() orelse return false, 10) catch return false;
        if (!std.mem.eql(u8, it.next() orelse return false, "mlp")) return false;
        if (!std.mem.eql(u8, it.next() orelse return false, "experts")) return false;
        const e = std.fmt.parseInt(usize, it.next() orelse return false, 10) catch return false;
        const proj_name = it.next() orelse return false;
        const proj: manifest_mod.Proj = if (std.mem.eql(u8, proj_name, "gate_proj")) .gate_proj else if (std.mem.eql(u8, proj_name, "up_proj")) .up_proj else if (std.mem.eql(u8, proj_name, "down_proj")) .down_proj else return false;
        if (it.next() == null) return false;
        if (width > manifest_mod.max_k) return false;
        if (l != layer) { s.stray += 1; return false; }
        if (s.layers) |layers| { var found = false; for (layers) |v| if (v == l) { found = true; }; if (!found) return false; }
        return s.m.width(l, e, proj) == @as(?u8, @intCast(width));
    }
};

fn openCheckpoint(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: *const Config, select: Select) !core.Checkpoint {
    const m = select.manifest orelse return core.Checkpoint.openModel(gpa, io, dir);
    var selector: Selector = .{ .m = m, .layers = select.layers };
    var ck = try core.Checkpoint.openModelPacks(gpa, io, dir, .{ .ctx = &selector, .keep = Selector.keep });
    errdefer ck.close();
    if (selector.stray != 0) return error.PackLayerMismatch;
    try m.check(c.layers, c.dense_layers, c.experts, select.layers);
    const count = if (select.layers) |l| l.len else c.layers;
    for (0..count) |j| {
        const layer = if (select.layers) |l| l[j] else j;
        if (layer < c.dense_layers) continue;
        inline for (.{ manifest_mod.Proj.gate_proj, manifest_mod.Proj.up_proj, manifest_mod.Proj.down_proj }) |proj| for (0..c.experts) |expert| {
            const p = @tagName(proj);
            var raw_buf: [160]u8 = undefined;
            const raw = try std.fmt.bufPrint(&raw_buf, "layers.{d}.mlp.experts.{d}.{s}.trellis", .{ layer, expert, p });
            var chosen: ?[]const u8 = null;
            var chosen_buf: [256]u8 = undefined;
            for ([_][]const u8{ "model.", "model.language_model.", "" }) |prefix| {
                const name = try std.fmt.bufPrint(&chosen_buf, "{s}{s}", .{ prefix, raw });
                if (ck.has(name)) { chosen = name; break; }
            }
            const name = chosen orelse {
                // `bufPrint` above is stack backed; this diagnostic name is owned
                // because the checkpoint API returns borrowed tensor keys.
                const diagnostic_name = try std.fmt.allocPrint(gpa, "{d}.{d}.{s}", .{ layer, expert, p });
                defer gpa.free(diagnostic_name);
                if (!builtin.is_test) std.log.err("manifest {s}: {s} wants K{d}: no pack of that width holds it", .{ m.name, diagnostic_name, m.width(layer, expert, proj).? });
                return error.ManifestMatrixMissing;
            };
            const t = ck.get(name) catch return error.ManifestMatrixMissing;
            if (t.dim(2) != 16 * m.width(layer, expert, proj).?) return error.PackWidthMismatch;
        };
    }
    return ck;
}

fn checkShape(c: *const Config, rank: usize, world: usize) !void {
    if ((world != 2 and world != 3) or rank >= world) return error.BadTensorParallelRank;
    // This build has no MTP layer: a config that runs an MTP layer is refused, not half-loaded.
    if (c.mtp != 0) return error.MtpLayerNotInBuild;
    // The MLA kv_write kernel hard-codes RMSNorm eps 1e-5; the indexer takes config.indexer_eps (review L5).
    if (c.eps != 1e-5) return error.UnsupportedRmsEps;
    // Vocabulary splits in blocks of 128 rows (TP=2: equal halves; TP=3: 51712, 51584, 51584).
    if (world == 3) {
        if (c.vocab != 154880 or c.vocab % 128 != 0) return error.UnsupportedVocabSplit;
    } else {
        if (c.vocab % (128 * world) != 0) return error.UnsupportedVocabSplit;
    }
}

/// The globals first (embed, final norm, lm_head: the largest host transients, done while device use is lowest),
/// then the layers in order, releasing each layer's pages.
fn walk(L: *Loader, c: *const Config, rank: usize, world: usize, select: Select, model: ?*ModelWeights) !void {
    const sh = try share_mod.Share.of(c, rank, world);
    if (select.globals) {
        const vocab: Span = .{ .lo = sh.vocab.lo, .hi = sh.vocab.hi };
        const embed = try plain(L, "embed_tokens.weight", vocab, null, .bf16);
        const norm = try plain(L, "norm.weight", null, null, .bf16);
        const head = try linear(L, &.{.{ .prefix = "lm_head.", .rows = vocab }}, .{ .lo = 0, .hi = c.hidden });
        if (model) |m| {
            m.vocab = vocab;
            m.embed = embed;
            m.norm = norm;
            m.head = head;
        }
        L.ck.release();
        if (L.budget) |trace_budget| TraceHook.step(trace_budget.trace, "weight upload globals", "other", 0);
    }
    const count = if (select.layers) |l| l.len else c.layers;
    for (0..count) |j| {
        const i = if (select.layers) |l| l[j] else j;
        if (i >= c.layers) return error.BadLayer;
        const layer = try loadLayer(L, c, sh, i);
        if (model) |m| m.layers[j] = layer;
        L.ck.release(); // a rank reads about half the checkpoint: its pages would crowd the GPU's allocations out
        if (L.budget) |trace_budget| {
            var trace_label_buf: [48]u8 = undefined;
            const trace_label: []const u8 = std.fmt.bufPrint(&trace_label_buf, "weight upload layer {d}", .{i}) catch "weight upload layer";
            TraceHook.step(trace_budget.trace, trace_label, "other", 0);
        }
    }
}

fn isIgnoredUnused(k: []const u8) bool {
    // MTP layers (e.g. layers.78)
    if (std.mem.indexOf(u8, k, "layers.78.") != null) return true;
    // Superseded non-expert EXL3 weights when backbone is converted to FP8
    if (std.mem.indexOf(u8, k, ".mlp.experts.") == null) {
        if (std.mem.endsWith(u8, k, ".trellis") or
            std.mem.endsWith(u8, k, ".suh") or
            std.mem.endsWith(u8, k, ".svh") or
            std.mem.endsWith(u8, k, ".su") or
            std.mem.endsWith(u8, k, ".sv") or
            std.mem.endsWith(u8, k, ".mul1") or
            std.mem.endsWith(u8, k, ".mcg") or
            std.mem.endsWith(u8, k, ".mcg_e4m3") or
            std.mem.endsWith(u8, k, ".bias")) return true;
    }
    return false;
}

/// Names no get took, counted without a log line; `names` (if given) receives up to its length of them.
fn unusedNames(ck: *const core.Checkpoint, names: ?*std.ArrayList([]const u8), gpa: std.mem.Allocator) !usize {
    var n: usize = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |k| {
        if (ck.used.contains(k)) continue;
        if (isIgnoredUnused(k)) continue;
        if (names) |list| try list.append(gpa, try gpa.dupe(u8, k));
        n += 1;
    };
    return n;
}

/// What a rank would load, from the shard headers only (bring-up step 1, and admission's exact weight bytes).
pub const Plan = struct {
    /// Exact device bytes by class: each class arena rounded once (what admission takes).
    weights: state.WeightClasses,
    /// The raw tensor bytes (before 256 B placement and arena rounding), for the rounded-vs-logical print.
    logical: state.WeightClasses,
    /// Checkpoint tensors the selection does not read (a full load refuses any).
    unused: [][]const u8,

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        for (p.unused) |n| gpa.free(n);
        gpa.free(p.unused);
        p.* = undefined;
    }
};

const Planned = struct { placed: state.WeightClasses, logical: state.WeightClasses };

fn planOpen(gpa: std.mem.Allocator, ck: *core.Checkpoint, c: *const Config, rank: usize, world: usize, select: Select) !Planned {
    var out: Planned = .{ .placed = .{}, .logical = .{} };
    var L: Loader = .{ .gpa = gpa, .ck = ck, .budget = null, .model = null, .select = select, .plan = &out.placed, .plan_logical = &out.logical };
    defer L.host.deinit(gpa);
    try walk(&L, c, rank, world, select, null);
    return out;
}

/// Exact per-rank device bytes by class for `select`, and the tensors it leaves unread. No GPU, no data reads.
pub fn planRank(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: Config, rank: usize, world: usize, select: Select) !Plan {
    try checkShape(&c, rank, world);
    var ck = try openCheckpoint(gpa, io, dir, &c, select);
    defer ck.close();
    const w = try planOpen(gpa, &ck, &c, rank, world, select);
    var names: std.ArrayList([]const u8) = .empty;
    errdefer {
        for (names.items) |n| gpa.free(n);
        names.deinit(gpa);
    }
    _ = try unusedNames(&ck, &names, gpa);
    return .{ .weights = roundedClasses(w.placed), .logical = w.logical, .unused = try names.toOwnedSlice(gpa) };
}

fn sameClasses(a: state.WeightClasses, b: state.WeightClasses) bool {
    return a.routed == b.routed and a.backbone_fp8 == b.backbone_fp8 and a.embed_head == b.embed_head and a.router_norms == b.router_norms and a.lowrank == b.lowrank;
}

pub const LayerWeights = struct {
    layer: usize,
    tensors: []ResidentTensor,
    device: DeviceLayer,
    /// The class arenas the tensors are carved from (owned).
    arenas: [class_count]cuda.DeviceBuffer,

    /// One layer of the checkpoint in `dir` (its config.json); `budget.admission` must be for that layer's plan.
    pub fn load(dir: []const u8, layer: usize, rank: usize, world: usize, budget: Budget) !LayerWeights {
        const c = try readConfig(budget.allocator, budget.io, dir);
        var model = try ModelWeights.loadConfig(dir, c, rank, world, budget, .{ .layers = &.{layer}, .globals = false });
        defer {
            budget.allocator.free(model.layers);
            model.tensors.deinit(model.allocator);
        }
        return .{ .layer = layer, .tensors = try model.tensors.toOwnedSlice(budget.allocator), .device = model.layers[0], .arenas = model.arenas };
    }

    pub fn deinit(weights: *LayerWeights, allocator: std.mem.Allocator) void {
        for (weights.tensors) |*tensor| freeResident(tensor, allocator);
        allocator.free(weights.tensors);
        for (&weights.arenas) |*a| a.free();
        weights.* = undefined;
    }
};

/// One TP rank's checkpoint tensors. Tensor bytes are staged from mapped safetensors and uploaded before unmapping.
pub const ModelWeights = struct {
    allocator: std.mem.Allocator,
    rank: usize,
    world: usize,
    tensors: std.ArrayList(ResidentTensor) = .empty,
    config: Config = .{},
    layers: []DeviceLayer = &.{},
    /// bf16 [vocab rows of this rank][hidden]; rows `vocab` of the global table.
    embed: u64 = 0,
    vocab: Span = .{ .lo = 0, .hi = 0 },
    norm: u64 = 0,
    /// This rank's lm_head rows on the `.bf16` lane, read with fp32 output (TF_GLM_F32_LOGITS=1).
    head: ?qmmf.Weight = null,
    /// One arena per weight class (zero-length when the class is empty); tensors are spans of them.
    arenas: [class_count]cuda.DeviceBuffer = @splat(.{ .d = undefined, .ptr = 0, .len = 0 }),
    cursors: [class_count]u64 = @splat(0),
    weight_vmm: ?*WeightVmm = null,

    /// The next `len` bytes of class `c`'s arena (placed at state.tensor_alignment), as a borrowed span.
    fn carve(m: *ModelWeights, c: Class, len: usize) !cuda.DeviceBuffer {
        const i = @backingInt(c);
        const a = m.arenas[i];
        if (m.cursors[i] + len > a.len) return error.ArenaOverflow;
        const span: cuda.DeviceBuffer = .{ .d = a.d, .ptr = a.ptr + m.cursors[i], .len = len, .borrowed = true };
        m.cursors[i] += placed(len);
        return span;
    }

    pub fn load(dir: []const u8, rank: usize, world: usize, budget: Budget) !ModelWeights {
        const c = try readConfig(budget.allocator, budget.io, dir);
        return loadConfig(dir, c, rank, world, budget, .{});
    }

    /// Load with an explicit shape (a geometry override for synthetic checkpoints; production passes config.json's).
    /// The header plan must equal the admitted weights before anything is uploaded; the loaded bytes must equal
    /// the plan afterwards. Weights before State (State.init after this returns).
    pub fn loadConfig(dir: []const u8, c: Config, rank: usize, world: usize, budget: Budget, select: Select) !ModelWeights {
        try checkShape(&c, rank, world);
        if (select.manifest) |m| {
            const h = m.histogram();
            std.log.info("manifest {s}: max_context {d}, K1..K8 matrices {any}, sha256 {x}", .{ m.name, m.max_context, h[1..], m.digest });
        }
        const gpa = budget.allocator;
        var ck = try openCheckpoint(gpa, budget.io, dir, &c, select);
        var ck_open = true;
        defer if (ck_open) ck.close();
        const plan = try planOpen(gpa, &ck, &c, rank, world, select);
        const planned = roundedClasses(plan.placed);
        if (!sameClasses(planned, budget.admission.breakdown.weights)) return error.AdmissionMismatch;
        if (select.layers == null and select.globals and try unusedNames(&ck, null, gpa) != 0) return error.UnusedCheckpointTensors;
        ck.used.clearRetainingCapacity();
        var model: ModelWeights = .{ .allocator = gpa, .rank = rank, .world = world, .config = c };
        errdefer model.deinit();
        // A few large arenas (review N3): one per weight class, each its rounded admitted size.
        var manager_opt: ?*WeightVmm = null;
        if (budget.weights_vmm) {
            // This loader's allocation units are non-empty weight classes; tensors retain their existing
            // intra-class 256-byte carve offsets and upload calls.
            var sizes_buf: [class_count]u64 = undefined;
            var sizes_len: usize = 0;
            inline for (@typeInfo(Class).@"enum".field_names) |n| {
                const bytes = classBytes(planned, @field(Class, n));
                if (bytes != 0) {
                    sizes_buf[sizes_len] = bytes;
                    sizes_len += 1;
                }
            }
            const manager: ?*WeightVmm = WeightVmm.init(gpa, budget.driver, budget.device_ordinal, sizes_buf[0..sizes_len], budget.trace) catch |err| blk: {
                if (err != error.Unsupported) return err;
                std.log.warn("glm53 weights: VMM arenas unavailable ({s}), falling back to legacy", .{@errorName(err)});
                break :blk null;
            };
            if (manager) |vmm_manager| {
                manager_opt = vmm_manager;
                model.weight_vmm = vmm_manager;
                const previous_hook = cuda.memory.currentHook();
                cuda.memory.setHook(vmm_manager.hook());
                defer cuda.memory.setHook(previous_hook);
                inline for (@typeInfo(Class).@"enum".field_names, 0..) |n, i| {
                    const bytes = classBytes(planned, @field(Class, n));
                    if (bytes != 0) {
                        model.arenas[i] = try cuda.DeviceBuffer.alloc(budget.driver, @intCast(bytes));
                        if (vmm_manager.bad_alloc) {
                            // A null hook result fell through to cuMemAlloc; release that standalone allocation here.
                            cuda.memory.setHook(previous_hook);
                            model.arenas[i].free();
                            model.arenas[i] = .{ .d = undefined, .ptr = 0, .len = 0 };
                            return error.WeightVmmAllocationMismatch;
                        }
                    }
                }
                if (vmm_manager.next != sizes_len) return error.WeightVmmAllocationCount;
            }
        }
        if (manager_opt == null) {
            inline for (@typeInfo(Class).@"enum".field_names, 0..) |n, i| {
                const bytes = classBytes(planned, @field(Class, n));
                if (bytes != 0) model.arenas[i] = try budget.device.alloc_fn(budget.device.ctx, budget.driver, @intCast(bytes));
            }
        }
        var L: Loader = .{ .gpa = gpa, .ck = &ck, .budget = budget, .model = &model, .select = select };
        defer L.host.deinit(gpa);
        var own_staging: ?uma.Staging = null;
        defer {
            if (own_staging) |*st| {
                st.deinit();
                TraceHook.step(budget.trace, "O_DIRECT staging free", "pinned", uma.stage_slots * uma.stage_bytes);
            }
        }
        var direct: Direct = undefined;
        var opened: usize = 0;
        const files: []uma.DirectFile = if (budget.odirect orelse uma.odirectFromEnv()) try gpa.alloc(uma.DirectFile, ck.files.items.len) else &.{};
        defer {
            for (files[0..opened]) |*f| f.close();
            if (files.len != 0) gpa.free(files);
        }
        if (files.len != 0) {
            for (ck.files.items, files) |*sf, *df| {
                df.* = try uma.DirectFile.open(sf.path);
                opened += 1;
            }
            const staging = if (budget.staging) |given| given else blk: {
                own_staging = try uma.Staging.initPinned(budget.driver);
                TraceHook.step(budget.trace, "O_DIRECT staging alloc", "pinned", uma.stage_slots * uma.stage_bytes);
                break :blk &own_staging.?;
            };
            direct = .{ .files = files, .staging = staging };
            L.direct = &direct;
        }
        defer if (L.direct) |d| d.deinit(gpa);
        // DeviceLayer holds addresses only: the tensors list owns every allocation, so a partial load frees cleanly.
        model.layers = try gpa.alloc(DeviceLayer, if (select.layers) |l| l.len else c.layers);
        try walk(&L, &c, rank, world, select, &model);
        TraceHook.step(budget.trace, "weight upload fill", "other", planned.total());
        // The load used exactly the placement the plan computed, class by class.
        inline for (@typeInfo(Class).@"enum".field_names, 0..) |n, i| if (model.cursors[i] != classBytes(plan.placed, @field(Class, n))) return error.PlanMismatch;
        if (!sameClasses(model.classes(), planned)) return error.PlanMismatch;
        // Unmap first (a live mapping keeps its pages), then drop each shard's clean pages.
        ck.close();
        ck_open = false;
        TraceHook.step(budget.trace, "checkpoint unmapped", "other", 0);
        if (budget.drop_page_cache) dropShardPages(gpa, budget.io, dir, select.manifest != null);
        TraceHook.step(budget.trace, "shard pages dropped", "other", 0);
        return model;
    }

    pub fn deinit(model: *ModelWeights) void {
        for (model.tensors.items) |*tensor| freeResident(tensor, model.allocator);
        model.tensors.deinit(model.allocator);
        if (model.weight_vmm) |manager| {
            for (&model.arenas) |*a| a.* = .{ .d = undefined, .ptr = 0, .len = 0 };
            manager.deinit();
            model.weight_vmm = null;
        } else {
            for (&model.arenas) |*a| if (a.len != 0) a.free();
        }
        model.allocator.free(model.layers);
        model.* = undefined;
    }

    /// Allocated device bytes by class (the arenas), for the memory breakdown; equal to planRank's weights.
    pub fn classes(model: *const ModelWeights) state.WeightClasses {
        return .{ .routed = model.arenas[@backingInt(Class.routed)].len, .backbone_fp8 = model.arenas[@backingInt(Class.backbone_fp8)].len, .embed_head = model.arenas[@backingInt(Class.embed_head)].len, .router_norms = model.arenas[@backingInt(Class.router_norms)].len, .lowrank = model.arenas[@backingInt(Class.lowrank)].len };
    }

    /// The raw tensor bytes by class (before placement and rounding).
    pub fn logicalClasses(model: *const ModelWeights) state.WeightClasses {
        var w: state.WeightClasses = .{};
        for (model.tensors.items) |t| addClass(&w, classOf(t.name, t.dtype), @intCast(t.data.len + if (t.scales) |sc| sc.len else 0));
        return w;
    }

    /// Device allocations this rank's weights take (one per non-empty class).
    pub fn allocationCount(model: *const ModelWeights) usize {
        var n: usize = 0;
        for (model.arenas) |a| n += @intFromBool(a.len != 0);
        return n;
    }

    pub const VmmStats = struct { mapped_bytes: u64, granularity: u64 };

    pub fn vmmStats(model: *const ModelWeights) ?VmmStats {
        const manager = model.weight_vmm orelse return null;
        return .{ .mapped_bytes = manager.plan.mapped_bytes, .granularity = manager.plan.granularity };
    }

    /// Driver request bytes: one VMM mapping or the legacy per-class arena requests.
    pub fn arenaAllocationBytes(model: *const ModelWeights) u64 {
        if (model.weight_vmm) |manager| return manager.plan.mapped_bytes;
        var total: u64 = 0;
        for (model.arenas) |arena| total += arena.len;
        return total;
    }

    pub fn find(model: *const ModelWeights, name: []const u8) ?ResidentTensor {
        for (model.tensors.items) |tensor| if (std.mem.eql(u8, tensor.name, name)) return tensor;
        return null;
    }
};

/// posix_fadvise(DONTNEED) on every shard of `dir` (fail soft: a shard that cannot be advised keeps its pages).
fn dropShardPages(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, manifest_used: bool) void {
    if (core.checkpoint.shardFiles(gpa, io, dir)) |paths| {
        defer core.checkpoint.freeShardFiles(gpa, paths);
        for (paths) |p| _ = uma.dropPageCache(p) catch false;
    } else |_| {}
    if (manifest_used) {
        const packs = core.checkpoint.packFiles(gpa, io, dir) catch return;
        defer core.checkpoint.freePackFiles(gpa, packs);
        for (packs) |p| _ = uma.dropPageCache(p.path) catch false;
    }
}

/// Retry the serving gate after advising both checkpoint shard and pack-file cache pages away.
pub fn dropLoadedShardPages(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) void {
    dropShardPages(gpa, io, dir, true);
}

pub fn readConfig(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    var c = try glm_config.parse(gpa, text);
    try c.admitCuda();
    return c;
}

fn freeResident(tensor: *ResidentTensor, allocator: std.mem.Allocator) void {
    tensor.data.free();
    if (tensor.scales) |*scales| scales.free();
    allocator.free(tensor.name);
    tensor.* = undefined;
}

test "block FP8 TP slice keeps source scale indexing and one kv_b allocation" {
    const codes = try std.testing.allocator.alloc(u8, 256 * 256);
    defer std.testing.allocator.free(codes);
    for (codes, 0..) |*v, i| v.* = @truncate(i);
    const scales = try std.testing.allocator.alloc(f32, 4);
    defer std.testing.allocator.free(scales);
    const source_scales = [_]f32{ 1, 2, 3, 4 };
    @memcpy(scales, &source_scales);
    const result = try packFp8B(std.testing.allocator, .{ .bytes = codes, .dtype = .e4m3, .rows = 256, .cols = 256, .scale_inv = scales }, .{ .lo = 128, .hi = 256 }, .{ .lo = 0, .hi = 256 });
    defer std.testing.allocator.free(result.codes);
    defer std.testing.allocator.free(result.scales);
    try std.testing.expectEqual(@as(usize, 32768), result.codes.len);
    try std.testing.expectEqualSlices(f32, &.{ 3, 4 }, result.scales);
}

test "fp32 to bf16 load casts round to nearest even" {
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), toBf16(@bitCast(@as(u32, 0x3f808000))));
    try std.testing.expectEqual(@as(u16, 0x3f82), toBf16(@bitCast(@as(u32, 0x3f818000))));
}

test "weight loader declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Tensor);
    std.testing.refAllDecls(Fp8B);
    std.testing.refAllDecls(Span);
    std.testing.refAllDecls(Layer);
    std.testing.refAllDecls(Budget);
    std.testing.refAllDecls(ResidentTensor);
    std.testing.refAllDecls(RawFp8);
    std.testing.refAllDecls(Indexer);
    std.testing.refAllDecls(Attention);
    std.testing.refAllDecls(LowRank);
    std.testing.refAllDecls(Moe);
    std.testing.refAllDecls(DeviceLayer);
    std.testing.refAllDecls(LayerWeights);
    std.testing.refAllDecls(Select);
    std.testing.refAllDecls(ModelWeights);
    std.testing.refAllDecls(Loader);
    std.testing.refAllDecls(Device);
    std.testing.refAllDecls(Plan);
}

test {
    _ = weight_plan;
}
