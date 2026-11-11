// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! Two-rank all-gathers and rank-order all-reduces as one-shot RoCE writes after b12x's RoCEnante
//! (`flashinfer/experimental/b12x/comm/roce/_oneshot_cute.py`, local-inference-lab/flashinfer@4092b28, Apache-2.0).

const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const memory = @import("memory.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const launch = @import("launch.zig");
const tp_group = @import("tp_group.zig");
const Stream = @import("stream.zig").Stream;
const kernels = @import("kernels.zig");

const max_hcas = 2;
const slots = 2;
const flag_stride = 128;
const send_depth = 256;
const port_num: u8 = 1;
const threads = 512;
const max_blocks = 8; // the largest grid; smaller shards launch fewer (a power of two)
const classes = 4; // grid size classes 1, 2, 4, 8 (each its own arrival counters)
const idle_spins: u64 = 20_000_000; // doorbell polls before the proxy naps between them (a model's gaps are short)
const glm_timeout_ms: u32 = 2_000;
const default_timeout_ms: u32 = 120_000;
const poison_bits: u32 = 0xffff_ffff;

pub fn sameStream(bound: ?abi.Stream, s: abi.Stream) bool {
    return if (bound) |b| b == s else false;
}

/// Plain TF_ROCE gathers retain the prior long wait; GLM's opt-in defaults to the reviewed two-second bound.
pub fn timeoutMs(value: ?[*:0]const u8, glm: bool) !u32 {
    const default = if (glm) glm_timeout_ms else default_timeout_ms;
    const raw = value orelse return default;
    const text = std.mem.span(raw);
    const parsed = std.fmt.parseInt(u32, text, 10) catch return error.InvalidRoceTimeout;
    if (parsed == 0 or parsed > 3_600_000) return error.InvalidRoceTimeout;
    return parsed;
}

pub fn poisonBits() u32 {
    return poison_bits;
}

/// Model the zeroed epoch and poison words written during a complete Gather setup.
pub fn resetPoisonState(words: []u32) void {
    @memset(words, 0);
}

// -- libibverbs (the ABI of rdma-core's verbs.h; offsets checked against its headers) ---------------------------------
const Device = opaque {};
const Context = opaque {};
const Pd = opaque {};
const Cq = extern struct { context: *Context };
const Mr = extern struct { context: *Context, pd: *Pd, addr: ?*anyopaque, length: usize, handle: u32, lkey: u32, rkey: u32 };
const Qp = extern struct { context: *Context, qp_context: ?*anyopaque, pd: *Pd, send_cq: *Cq, recv_cq: *Cq, srq: ?*anyopaque, handle: u32, qp_num: u32 };
const QpCap = extern struct { max_send_wr: u32 = 0, max_recv_wr: u32 = 0, max_send_sge: u32 = 0, max_recv_sge: u32 = 0, max_inline_data: u32 = 0 };
const QpInitAttr = extern struct { qp_context: ?*anyopaque = null, send_cq: ?*Cq = null, recv_cq: ?*Cq = null, srq: ?*anyopaque = null, cap: QpCap = .{}, qp_type: c_int = 0, sq_sig_all: c_int = 0 };
const GlobalRoute = extern struct { dgid: [16]u8 = @splat(0), flow_label: u32 = 0, sgid_index: u8 = 0, hop_limit: u8 = 0, traffic_class: u8 = 0 };
const AhAttr = extern struct { grh: GlobalRoute = .{}, dlid: u16 = 0, sl: u8 = 0, src_path_bits: u8 = 0, static_rate: u8 = 0, is_global: u8 = 0, port_num: u8 = 0 };
const QpAttr = extern struct {
    qp_state: u32 = 0,
    cur_qp_state: u32 = 0,
    path_mtu: u32 = 0,
    path_mig_state: u32 = 0,
    qkey: u32 = 0,
    rq_psn: u32 = 0,
    sq_psn: u32 = 0,
    dest_qp_num: u32 = 0,
    qp_access_flags: u32 = 0,
    cap: QpCap = .{},
    ah_attr: AhAttr = .{},
    alt_ah_attr: AhAttr = .{},
    pkey_index: u16 = 0,
    alt_pkey_index: u16 = 0,
    en_sqd_async_notify: u8 = 0,
    sq_draining: u8 = 0,
    max_rd_atomic: u8 = 0,
    max_dest_rd_atomic: u8 = 0,
    min_rnr_timer: u8 = 0,
    port_num: u8 = 0,
    timeout: u8 = 0,
    retry_cnt: u8 = 0,
    rnr_retry: u8 = 0,
    alt_port_num: u8 = 0,
    alt_timeout: u8 = 0,
    rate_limit: u32 = 0,
    _tail: u32 = 0,
};
const Sge = extern struct { addr: u64, length: u32, lkey: u32 };
const SendWr = extern struct {
    wr_id: u64 = 0,
    next: ?*SendWr = null,
    sg_list: ?*Sge = null,
    num_sge: c_int = 0,
    opcode: u32 = 0,
    send_flags: u32 = 0,
    imm_data: u32 = 0,
    remote_addr: u64 = 0,
    rkey: u32 = 0,
    _tail: [128 - 52]u8 = @splat(0),
};
const Wc = extern struct { wr_id: u64, status: u32, opcode: u32, vendor_err: u32, _rest: [48 - 20]u8 };
const PortAttr = extern struct { bytes: [56]u8 align(8) };

comptime {
    std.debug.assert(@sizeOf(QpInitAttr) == 64 and @offsetOf(QpInitAttr, "cap") == 32 and @offsetOf(QpInitAttr, "qp_type") == 52);
    std.debug.assert(@sizeOf(AhAttr) == 32 and @offsetOf(AhAttr, "dlid") == 24 and @offsetOf(AhAttr, "port_num") == 30);
    std.debug.assert(@sizeOf(QpAttr) == 144 and @offsetOf(QpAttr, "ah_attr") == 56 and @offsetOf(QpAttr, "pkey_index") == 120);
    std.debug.assert(@offsetOf(QpAttr, "rnr_retry") == 132 and @offsetOf(QpAttr, "qp_access_flags") == 32);
    std.debug.assert(@sizeOf(SendWr) == 128 and @offsetOf(SendWr, "remote_addr") == 40 and @offsetOf(SendWr, "rkey") == 48);
    std.debug.assert(@sizeOf(Wc) == 48 and @offsetOf(Mr, "lkey") == 36 and @offsetOf(Qp, "qp_num") == 52);
}

// struct ibv_context's ops: the two calls verbs.h inlines
const ops_poll_cq = 96;
const ops_post_send = 208;
const PollCq = *const fn (*Cq, c_int, [*]Wc) callconv(.c) c_int;
const PostSend = *const fn (*Qp, *SendWr, *?*SendWr) callconv(.c) c_int;

const wr_rdma_write = 0;
const send_signaled = 2;
const send_inline = 8;
const qpt_rc = 2;
const qps_init = 1;
const qps_rtr = 2;
const qps_rts = 3;
const port_active = 4;
const access_local_write = 1;
const access_remote_write = 2;
const qp_state = 1;
const qp_access_flags = 8;
const qp_pkey_index = 16;
const qp_port = 32;
const qp_av = 128;
const qp_path_mtu = 256;
const qp_timeout = 512;
const qp_retry_cnt = 1024;
const qp_rnr_retry = 2048;
const qp_rq_psn = 4096;
const qp_max_qp_rd_atomic = 8192;
const qp_min_rnr_timer = 32768;
const qp_sq_psn = 65536;
const qp_max_dest_rd_atomic = 131072;
const qp_dest_qpn = 1048576;

const Api = struct {
    ibv_get_device_list: *const fn (?*c_int) callconv(.c) ?[*]?*Device,
    ibv_free_device_list: *const fn ([*]?*Device) callconv(.c) void,
    ibv_get_device_name: *const fn (*Device) callconv(.c) ?[*:0]const u8,
    ibv_open_device: *const fn (*Device) callconv(.c) ?*Context,
    ibv_close_device: *const fn (*Context) callconv(.c) c_int,
    ibv_query_port: *const fn (*Context, u8, *PortAttr) callconv(.c) c_int,
    ibv_query_gid: *const fn (*Context, u8, c_int, *[16]u8) callconv(.c) c_int,
    ibv_alloc_pd: *const fn (*Context) callconv(.c) ?*Pd,
    ibv_dealloc_pd: *const fn (*Pd) callconv(.c) c_int,
    ibv_reg_mr: *const fn (*Pd, ?*anyopaque, usize, c_int) callconv(.c) ?*Mr,
    ibv_dereg_mr: *const fn (*Mr) callconv(.c) c_int,
    ibv_create_cq: *const fn (*Context, c_int, ?*anyopaque, ?*anyopaque, c_int) callconv(.c) ?*Cq,
    ibv_destroy_cq: *const fn (*Cq) callconv(.c) c_int,
    ibv_create_qp: *const fn (*Pd, *QpInitAttr) callconv(.c) ?*Qp,
    ibv_destroy_qp: *const fn (*Qp) callconv(.c) c_int,
    ibv_modify_qp: *const fn (*Qp, *QpAttr, c_int) callconv(.c) c_int,
    ibv_wc_status_str: *const fn (u32) callconv(.c) ?[*:0]const u8,
};

fn opsCall(comptime T: type, ctx: *Context, offset: usize) T {
    const slot: *const T = @ptrFromInt(@intFromPtr(ctx) + offset);
    return slot.*;
}

pub const Error = error{ VerbsUnavailable, NoHca, HcaFailed, PeerRefused, ProxyFailed, ExchangeTimedOut, Misaligned, StreamMismatch };

/// Host oracle for the reducer's BF16 conversion; round to nearest, ties to even.
pub fn f32ToBf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((bits >> 16) | 0x40);
    return @intCast((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}

/// TF_ROCE=0 keeps every gather on NCCL; unset or any other value asks for this path (both ranks must ask).
pub fn wanted() bool {
    const v = std.c.getenv("TF_ROCE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// The peer's `mine`: both ranks' values cross the group's communicator, rank order.
fn swap(comptime T: type, group: *const tp_group.Group, d: *const Driver, s: Stream, mine: T) !T {
    var both: [2]T = undefined;
    try group.gather(T, d, s, mine, &both);
    return both[1 - group.rank];
}

// -- the region's layout (the same on both ranks) ---------------------------------------------------------------------
const Layout = struct {
    slot_bytes: usize,
    recv: usize,
    flags: usize,
    send: usize,
    ctrl: usize,
    total: usize,

    fn of(max_bytes: usize) Layout {
        const s = std.mem.alignForward(usize, max_bytes, 4096);
        const flags = slots * s;
        const send = flags + slots * max_hcas * flag_stride;
        const ctrl = send + slots * s;
        return .{ .slot_bytes = s, .recv = 0, .flags = flags, .send = send, .ctrl = ctrl, .total = ctrl + flag_stride };
    }
};
// ctrl words
const c_seq = 0;
const c_nbytes = 1; // + slot
const c_error = 3;
const c_hca = 4;
const c_error_seq = 5;
const c_abort = 6;
const c_error_kind = 7;

pub const SeqState = struct { next: u32, pending: u32 };

/// Sequence arithmetic deliberately wraps; zero is a valid sequence, never an error flag.
pub fn sequenceState(last: u32, published: u32) ?SeqState {
    const pending = published -% last;
    if (pending == 0 or pending > slots) return null;
    return .{ .next = last +% 1, .pending = pending };
}

pub fn errorFlag(_: u32) u32 {
    return 1;
}

pub fn aligned16(ptr: abi.DevicePtr) bool {
    return ptr % 16 == 0;
}

pub fn completionMatches(expected_seq: u32, expected_bytes: u32, received_seq: u32, received_bytes: u32) bool {
    return expected_seq == received_seq and expected_bytes == received_bytes;
}

pub fn reduceF32(a: f32, b: f32, poisoned: bool) f32 {
    return if (poisoned) @bitCast(poison_bits) else a + b;
}

pub fn reduceBf16(a: u16, b: u16, poisoned: bool) u16 {
    if (poisoned) return 0xffff;
    return f32ToBf16(@as(f32, @bitCast(@as(u32, a) << 16)) + @as(f32, @bitCast(@as(u32, b) << 16)));
}

pub const TokenStatus = struct { token: u32, status: u32 };

pub fn gateToken(token: u32, poisoned: bool) TokenStatus {
    return if (poisoned) .{ .token = 0xffff_ffff, .status = 1 } else .{ .token = token, .status = 0 };
}

/// What a rank tells its peer to connect: its region, and per HCA its key, address and queue pair.
const Blob = extern struct {
    ok: u32,
    n_hca: u32,
    slot_bytes: u64,
    max_bytes: u64, // the largest shard this rank would send this way (both ranks use the smaller)
    addr: u64,
    rkey: [max_hcas]u32,
    qpn: [max_hcas]u32,
    mtu: [max_hcas]u32,
    lid: [max_hcas]u16,
    gid: [max_hcas][16]u8,
};

const Hca = struct {
    ctx: ?*Context = null,
    pd: ?*Pd = null,
    mr: ?*Mr = null,
    cq: ?*Cq = null,
    qp: ?*Qp = null,
    gid: [16]u8 = @splat(0),
    gid_index: c_int = 0,
    lid: u16 = 0,
    mtu: u32 = 0,
    outstanding: u32 = 0,
};

pub const Gather = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    lib: std.DynLib,
    api: Api,
    rank: u8,
    hcas: [max_hcas]Hca = @splat(.{}),
    n_hca: u32 = 0,
    layout: Layout,
    region: memory.HostBuffer,
    region_dev: abi.DevicePtr,
    counters: memory.DeviceBuffer,
    module: Module,
    f: Function,
    reduce_f32: Function,
    reduce_bf16: Function,
    status_gate: Function,
    reduce_scratch: memory.DeviceBuffer,
    max_bytes: usize,
    timeout_ns: u64,
    peer_addr: u64 = 0,
    peer_rkey: [max_hcas]u32 = @splat(0),
    last_seq: u32 = 0,
    running: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,
    err: [256]u8 = @splat(0),
    gathers: u64 = 0,
    stream_handle: ?abi.Stream = null,
    /// A TP=3 triangle pair's neighbour: where its shard lands in roce_gather_tri's output (unused by two ranks).
    peer_slot: u8 = 0,
    f_tri: Function = undefined, // roce_gather_tri and its status gate (triangle pairs only)
    gate_tri: Function = undefined,

    /// Both ranks' exchange or null on both; setup crosses ``group``'s communicator on ``s``. ``max_bytes``: a shard.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, s: Stream, want: bool, max_bytes: usize, timeout_ms: u32) !?*Gather {
        if (group.world != 2) return error.PeerRefused;
        const g = try gpa.create(Gather);
        var mine: Blob = std.mem.zeroes(Blob);
        const ok = want and blk: {
            g.setup(gpa, io, d, @intCast(group.rank), max_bytes) catch |e| {
                std.log.warn("RoCE setup unavailable: {s} ({s})", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        if (ok) g.blob(&mine);
        const theirs = swap(Blob, group, d, s, mine) catch |e| {
            if (ok) g.teardown();
            gpa.destroy(g);
            return e;
        };
        if (!ok or theirs.ok == 0 or theirs.n_hca != mine.n_hca or theirs.slot_bytes != mine.slot_bytes) {
            if (ok) g.teardown();
            gpa.destroy(g);
            return null;
        }
        g.max_bytes = agreedCap(mine.max_bytes, theirs.max_bytes); // one cap for fits on both ranks
        g.timeout_ns = @as(u64, timeout_ms) * std.time.ns_per_ms;
        const connected = blk: {
            g.connect(&theirs) catch |e| {
                std.log.warn("RoCE peer setup unavailable: {s} ({s})", .{ @errorName(e), std.mem.sliceTo(&g.err, 0) });
                break :blk false;
            };
            break :blk true;
        };
        const peer_connected = swap(u32, group, d, s, @intFromBool(connected)) catch |e| {
            g.teardown();
            gpa.destroy(g);
            return e;
        };
        if (!connected or peer_connected == 0) {
            g.teardown();
            gpa.destroy(g);
            return null;
        }
        g.stream_handle = s.handle;
        std.log.info("RoCE all-gather: {d} HCA(s), shards up to {d} KiB by RDMA writes", .{ g.n_hca, g.max_bytes >> 10 });
        return g;
    }

    /// TP=3 over a cabled triangle (TF_ROCE_TRIANGLE=1): one two-rank Gather a neighbour, each on the HCAs of the port
    /// cabled to it. TF_ROCE_HCA (else NCCL_IB_HCA) names port 0's and port 1's HCAs, told apart by their f0/f1
    /// suffix; port 0 faces rank + 1 and port 1 rank + 2, or the reverse with TF_ROCE_TRIANGLE_F0=prev. Every rank
    /// sets both up, swaps blobs over the group's NCCL, connects, and keeps them only when all three ranks connected
    /// both; otherwise all three return none and stay on NCCL. [0] is the lower-ranked neighbour. A gather runs both
    /// pairs in one launch (triGather): run one after the other they would chain across the ranks ({0,1}, then
    /// {0,2}, then {1,2}) and cost three exchanges.
    pub fn initTriangle(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, group: *const tp_group.Group, s: Stream, max_bytes: usize, timeout_ms: u32) ![2]?*Gather {
        if (group.world != 3) return error.PeerRefused;
        const rank: u8 = @intCast(group.rank);
        const f0_prev = if (std.c.getenv("TF_ROCE_TRIANGLE_F0")) |v| std.mem.eql(u8, std.mem.span(v), "prev") else false;
        const peers = triPeers(rank);
        var gs: [2]?*Gather = .{ null, null };
        errdefer for (gs) |maybe| if (maybe) |g| g.deinit();
        var mine: Tri = std.mem.zeroes(Tri);
        for (peers, 0..) |peer, i| {
            const port = triPort(rank, peer, f0_prev);
            var names: [max_hcas][64]u8 = undefined;
            var lens: [max_hcas]usize = undefined;
            const n = portHcaNames(port, &names, &lens);
            if (n == 0) std.log.warn("RoCE triangle: TF_ROCE_HCA/NCCL_IB_HCA names no HCA ending f{c} (toward rank {d})", .{ port, peer });
            const g = try gpa.create(Gather);
            const ok = n > 0 and blk: {
                g.setupNamed(gpa, io, d, rank, max_bytes, &names, &lens, n) catch |e| {
                    std.log.warn("RoCE triangle: HCAs on port {c} toward rank {d} unavailable: {s} ({s})", .{ port, peer, @errorName(e), std.mem.sliceTo(&g.err, 0) });
                    break :blk false;
                };
                g.f_tri = g.module.function("roce_gather_tri") catch |e| {
                    std.log.warn("RoCE triangle: kernel roce_gather_tri missing: {s}", .{@errorName(e)});
                    g.teardown();
                    break :blk false;
                };
                g.gate_tri = g.module.function("roce_status_gate_tri") catch |e| {
                    std.log.warn("RoCE triangle: kernel roce_status_gate_tri missing: {s}", .{@errorName(e)});
                    g.teardown();
                    break :blk false;
                };
                break :blk true;
            };
            if (!ok) {
                gpa.destroy(g);
                continue;
            }
            gs[i] = g;
            g.blob(&mine.to[peer]);
        }
        var all: [3]Tri = undefined;
        try group.gather(Tri, d, s, mine, &all);
        var connected = Flags{};
        for (peers, 0..) |peer, i| {
            const g = gs[i] orelse continue;
            const theirs = all[peer].to[rank];
            if (theirs.ok == 0 or theirs.n_hca != g.n_hca or theirs.slot_bytes != g.layout.slot_bytes) continue;
            g.max_bytes = agreedCap(g.max_bytes, theirs.max_bytes);
            g.timeout_ns = @as(u64, timeout_ms) * std.time.ns_per_ms;
            g.connect(&theirs) catch |e| {
                std.log.warn("RoCE triangle: connecting rank {d} to rank {d} failed: {s} ({s})", .{ rank, peer, @errorName(e), std.mem.sliceTo(&g.err, 0) });
                continue;
            };
            g.stream_handle = s.handle;
            g.peer_slot = peer;
            connected.ok[i] = 1;
        }
        var every: [3]Flags = undefined;
        try group.gather(Flags, d, s, connected, &every);
        for (every) |f| if (f.ok[0] == 0 or f.ok[1] == 0) {
            for (&gs) |*maybe| if (maybe.*) |g| {
                g.deinit();
                maybe.* = null;
            };
            std.log.warn("RoCE triangle: not every rank connected both neighbours; gathers stay on NCCL", .{});
            return gs;
        };
        // One cap for all three pairs, so a gather of any size takes the same path on every rank.
        var caps: [3]Caps = undefined;
        try group.gather(Caps, d, s, .{ .bytes = @min(gs[0].?.max_bytes, gs[1].?.max_bytes) }, &caps);
        var cap: u64 = caps[0].bytes;
        for (caps) |c| cap = @min(cap, c.bytes);
        for (gs) |maybe| maybe.?.max_bytes = @intCast(cap);
        std.log.info("RoCE triangle all-gather: rank {d} to {d} and {d}, {d}+{d} HCA(s), shards up to {d} KiB by RDMA writes", .{ rank, peers[0], peers[1], gs[0].?.n_hca, gs[1].?.n_hca, cap >> 10 });
        return gs;
    }

    const Tri = extern struct { to: [3]Blob };
    const Flags = extern struct { ok: [2]u32 = .{ 0, 0 } };
    const Caps = extern struct { bytes: u64 };

    /// roce_gather_tri's RocePair: one pair's region, counters and neighbour.
    pub const Pair = extern struct { recv: u64, flags: u64, send: u64, ctrl: u64, counters: u64, slot_bytes: i64, peer_slot: i32, n_hca: i32 };

    fn pair(g: *const Gather) Pair {
        const base = g.region_dev;
        return .{ .recv = base + g.layout.recv, .flags = base + g.layout.flags, .send = base + g.layout.send, .ctrl = base + g.layout.ctrl, .counters = g.counters.ptr, .slot_bytes = @intCast(g.layout.slot_bytes), .peer_slot = g.peer_slot, .n_hca = @intCast(g.n_hca) };
    }

    /// Three ranks' ``bytes`` from ``send`` into ``recv`` in rank order (as ncclAllGather), over both of a triangle's
    /// pairs (initTriangle's [0] and [1]) in one launch.
    pub fn triGather(a: *Gather, b: *Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) !void {
        if (!a.boundTo(stream) or !b.boundTo(stream)) return error.StreamMismatch;
        if (!aligned16(send) or !aligned16(recv)) return error.Misaligned;
        const packs = bytes / 16;
        const need = @max(1, std.math.divCeil(usize, packs, 2 * threads) catch unreachable);
        const grid: u32 = @intCast(@min(std.math.ceilPowerOfTwo(usize, need) catch max_blocks, max_blocks));
        const klass: i32 = @intCast(std.math.log2_int(u32, grid));
        var args: launch.Args = .{};
        args.add(send);
        args.add(recv);
        args.add(@as(i32, @intCast(packs)));
        args.add(@as(i32, a.rank));
        args.add(a.pair());
        args.add(b.pair());
        args.add(klass);
        args.add(@min(a.timeout_ns, b.timeout_ns));
        try launch.launch(a.f_tri, .{ .grid = .{ .x = grid }, .block = .{ .x = threads } }, .{ .handle = stream, .d = a.d }, &args);
        a.gathers += 1;
        b.gathers += 1;
    }

    /// gateResult over both of a triangle's pairs.
    pub fn triGateResult(a: *Gather, b: *Gather, stream: abi.Stream, token: abi.DevicePtr, status: abi.DevicePtr) !void {
        var args: launch.Args = .{};
        args.add(a.region_dev + a.layout.ctrl);
        args.add(b.region_dev + b.layout.ctrl);
        args.add(token);
        args.add(status);
        try launch.launch(a.gate_tri, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, .{ .handle = stream, .d = a.d }, &args);
    }

    /// Whether a gather already waited past its deadline or its proxy failed, without logging (triangle probe).
    pub fn failedQuietly(g: *Gather) bool {
        return @atomicLoad(u32, &g.ctrl()[c_error], .acquire) != 0 or g.failed.load(.acquire);
    }

    /// Override the GPU wait bound (the triangle probe shortens it, then restores the configured one).
    pub fn setTimeoutMs(g: *Gather, timeout_ms: u32) void {
        g.timeout_ns = @as(u64, timeout_ms) * std.time.ns_per_ms;
    }

    fn fail(g: *Gather, comptime fmt: []const u8, args: anytype) void {
        _ = std.fmt.bufPrint(g.err[0 .. g.err.len - 1], fmt, args) catch {};
    }

    fn setup(g: *Gather, gpa: std.mem.Allocator, io: std.Io, d: *const Driver, rank: u8, max_bytes: usize) !void {
        var names: [max_hcas][64]u8 = undefined;
        var lens: [max_hcas]usize = undefined;
        const n = try hcaNames(io, gpa, &names, &lens);
        try g.setupNamed(gpa, io, d, rank, max_bytes, &names, &lens, n);
    }

    fn setupNamed(g: *Gather, gpa: std.mem.Allocator, io: std.Io, d: *const Driver, rank: u8, max_bytes: usize, names: *const [max_hcas][64]u8, lens: *const [max_hcas]usize, n_hca: u32) !void {
        g.* = .{ .gpa = gpa, .d = d, .lib = undefined, .api = undefined, .rank = rank, .layout = .of(max_bytes), .region = undefined, .region_dev = 0, .counters = undefined, .module = undefined, .f = undefined, .reduce_f32 = undefined, .reduce_bf16 = undefined, .status_gate = undefined, .reduce_scratch = undefined, .max_bytes = max_bytes, .timeout_ns = 0 };
        g.lib = std.DynLib.open("libibverbs.so.1") catch return @import("sys").optional_libraries.unavailable(.verbs);
        errdefer g.lib.close();
        const info = @typeInfo(Api).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            @field(g.api, name) = g.lib.lookup(T, name) orelse return @import("sys").optional_libraries.unavailable(.verbs);
        }
        g.n_hca = n_hca;
        g.region = try memory.HostBuffer.allocMapped(d, g.layout.total);
        errdefer g.region.free();
        @memset(g.region.bytes, 0);
        g.region_dev = try g.region.device();
        g.counters = try memory.DeviceBuffer.alloc(d, 64 * 4);
        errdefer g.counters.free();
        var counter_state: [64]u32 = undefined;
        resetPoisonState(&counter_state);
        try g.counters.upload(0, std.mem.sliceAsBytes(&counter_state));
        g.module = try Module.load(d, kernels.roce_gather);
        errdefer g.module.unload();
        g.f = try g.module.function("roce_gather");
        g.reduce_f32 = try g.module.function("roce_reduce_f32");
        g.reduce_bf16 = try g.module.function("roce_reduce_bf16");
        g.status_gate = try g.module.function("roce_status_gate");
        g.reduce_scratch = try memory.DeviceBuffer.alloc(d, 2 * max_bytes);
        errdefer g.reduce_scratch.free();
        errdefer g.closeHcas();
        for (0..g.n_hca) |h| try g.openHca(io, gpa, h, names[h][0..lens[h]]);
    }

    fn closeHcas(g: *Gather) void {
        for (&g.hcas) |*x| {
            if (x.qp) |q| _ = g.api.ibv_destroy_qp(q);
            if (x.cq) |q| _ = g.api.ibv_destroy_cq(q);
            if (x.mr) |m| _ = g.api.ibv_dereg_mr(m);
            if (x.pd) |p| _ = g.api.ibv_dealloc_pd(p);
            if (x.ctx) |c| _ = g.api.ibv_close_device(c);
            x.* = .{};
        }
    }

    fn openHca(g: *Gather, io: std.Io, gpa: std.mem.Allocator, h: usize, name: []const u8) !void {
        const a = &g.api;
        var num: c_int = 0;
        const list = a.ibv_get_device_list(&num) orelse return error.NoHca;
        defer a.ibv_free_device_list(list);
        var dev: ?*Device = null;
        for (0..@intCast(num)) |i| {
            const n = a.ibv_get_device_name(list[i].?) orelse continue;
            if (std.mem.eql(u8, std.mem.span(n), name)) dev = list[i];
        }
        const hca = &g.hcas[h];
        hca.ctx = a.ibv_open_device(dev orelse {
            g.fail("RDMA device {s} not found", .{name});
            return error.NoHca;
        }) orelse return error.HcaFailed;
        var port: PortAttr = std.mem.zeroes(PortAttr);
        if (a.ibv_query_port(hca.ctx.?, port_num, &port) != 0) return error.HcaFailed;
        if (std.mem.readInt(u32, port.bytes[0..4], .little) != port_active) {
            g.fail("{s} port 1 is not active", .{name});
            return error.NoHca;
        }
        hca.mtu = std.mem.readInt(u32, port.bytes[8..12], .little);
        hca.lid = std.mem.readInt(u16, port.bytes[34..36], .little);
        hca.gid_index = try gidIndex(io, gpa, name);
        if (a.ibv_query_gid(hca.ctx.?, port_num, hca.gid_index, &hca.gid) != 0) return error.HcaFailed;
        hca.pd = a.ibv_alloc_pd(hca.ctx.?) orelse return error.HcaFailed;
        hca.mr = a.ibv_reg_mr(hca.pd.?, g.region.bytes.ptr, g.region.bytes.len, access_local_write | access_remote_write) orelse {
            g.fail("{s}: registering the pinned region failed", .{name});
            return error.HcaFailed;
        };
        hca.cq = a.ibv_create_cq(hca.ctx.?, send_depth, null, null, 0) orelse return error.HcaFailed;
        var attr: QpInitAttr = .{ .send_cq = hca.cq, .recv_cq = hca.cq, .qp_type = qpt_rc, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 } };
        hca.qp = a.ibv_create_qp(hca.pd.?, &attr) orelse return error.HcaFailed;
        var init_attr: QpAttr = .{ .qp_state = qps_init, .port_num = port_num, .qp_access_flags = access_remote_write };
        if (a.ibv_modify_qp(hca.qp.?, &init_attr, qp_state | qp_pkey_index | qp_port | qp_access_flags) != 0) return error.HcaFailed;
    }

    fn blob(g: *Gather, b: *Blob) void {
        b.ok = 1;
        b.n_hca = g.n_hca;
        b.slot_bytes = g.layout.slot_bytes;
        b.max_bytes = g.max_bytes;
        b.addr = @intFromPtr(g.region.bytes.ptr);
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            b.rkey[h] = x.mr.?.rkey;
            b.qpn[h] = x.qp.?.qp_num;
            b.mtu[h] = x.mtu;
            b.lid[h] = x.lid;
            b.gid[h] = x.gid;
        }
    }

    fn connect(g: *Gather, peer: *const Blob) !void {
        g.peer_addr = peer.addr;
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            g.peer_rkey[h] = peer.rkey[h];
            var rtr: QpAttr = .{ .qp_state = qps_rtr, .path_mtu = @min(peer.mtu[h], x.mtu), .dest_qp_num = peer.qpn[h], .max_dest_rd_atomic = 1, .min_rnr_timer = 12, .ah_attr = .{ .is_global = 1, .dlid = peer.lid[h], .port_num = port_num, .grh = .{ .dgid = peer.gid[h], .sgid_index = @intCast(x.gid_index), .hop_limit = 64 } } };
            if (g.api.ibv_modify_qp(x.qp.?, &rtr, qp_state | qp_av | qp_path_mtu | qp_dest_qpn | qp_rq_psn | qp_max_dest_rd_atomic | qp_min_rnr_timer) != 0) {
                g.fail("HCA {d}: the queue pair refused RTR", .{h});
                return error.PeerRefused;
            }
            var rts: QpAttr = .{ .qp_state = qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .max_rd_atomic = 1 };
            if (g.api.ibv_modify_qp(x.qp.?, &rts, qp_state | qp_timeout | qp_retry_cnt | qp_rnr_retry | qp_sq_psn | qp_max_qp_rd_atomic) != 0) {
                g.fail("HCA {d}: the queue pair refused RTS", .{h});
                return error.PeerRefused;
            }
        }
        g.running.store(true, .release);
        g.thread = try std.Thread.spawn(.{}, proxy, .{g});
    }

    fn ctrl(g: *Gather) [*]u32 {
        return @ptrCast(@alignCast(g.region.bytes.ptr + g.layout.ctrl));
    }

    fn drain(g: *Gather, h: usize) !void {
        const x = &g.hcas[h];
        var wc: [32]Wc = undefined;
        const n = opsCall(PollCq, x.ctx.?, ops_poll_cq)(x.cq.?, wc.len, &wc);
        if (n < 0) return error.ProxyFailed;
        for (wc[0..@intCast(n)]) |w| {
            if (w.status != 0) {
                const why = g.api.ibv_wc_status_str(w.status) orelse "?";
                g.fail("RDMA write on HCA {d} failed: {s} (vendor 0x{x}, seq {d})", .{ h, std.mem.span(why), w.vendor_err, g.last_seq });
                return error.ProxyFailed;
            }
            x.outstanding -= 1;
        }
    }

    /// One sequence's shard to the peer: per HCA a stripe, then its flag (inline, signalled) on the same queue pair.
    fn post(g: *Gather, seq: u32, nbytes: u32) !void {
        if (nbytes == 0 or nbytes % 16 != 0 or nbytes > g.layout.slot_bytes) {
            g.fail("bad shard size {d} at seq {d}", .{ nbytes, seq });
            return error.ProxyFailed;
        }
        const slot: u64 = seq & 1;
        const base = @intFromPtr(g.region.bytes.ptr);
        const packs = nbytes / 16;
        const Completion = extern struct { seq: u32, nbytes: u32 };
        var completion: Completion = .{ .seq = seq, .nbytes = nbytes };
        var offset: u64 = 0;
        for (0..g.n_hca) |h| {
            const x = &g.hcas[h];
            while (x.outstanding >= send_depth / 4) {
                try g.drain(h);
                if (!g.running.load(.monotonic)) return error.ProxyFailed;
            }
            var n = packs / g.n_hca;
            if (h < packs % g.n_hca) n += 1;
            const bytes: u32 = n * 16;
            var flag_sge: Sge = .{ .addr = @intFromPtr(&completion), .length = @sizeOf(Completion), .lkey = 0 };
            var flag: SendWr = .{ .wr_id = h, .sg_list = &flag_sge, .num_sge = 1, .opcode = wr_rdma_write, .send_flags = send_signaled | send_inline, .remote_addr = g.peer_addr + g.layout.flags + (slot * g.n_hca + h) * flag_stride, .rkey = g.peer_rkey[h] };
            var data_sge: Sge = .{ .addr = base + g.layout.send + slot * g.layout.slot_bytes + offset, .length = bytes, .lkey = x.mr.?.lkey };
            var data: SendWr = .{ .wr_id = h, .next = &flag, .sg_list = &data_sge, .num_sge = 1, .opcode = wr_rdma_write, .remote_addr = g.peer_addr + g.layout.recv + slot * g.layout.slot_bytes + offset, .rkey = g.peer_rkey[h] };
            var bad: ?*SendWr = null;
            const first: *SendWr = if (bytes != 0) &data else &flag;
            const rc = opsCall(PostSend, x.ctx.?, ops_post_send)(x.qp.?, first, &bad);
            if (rc != 0) {
                g.fail("ibv_post_send: {d}", .{rc});
                return error.ProxyFailed;
            }
            x.outstanding += 1;
            offset += bytes;
        }
        for (0..g.n_hca) |h| try g.drain(h);
    }

    /// The proxy: spins on the doorbell and posts each sequence in order (a missed one from its slot's size).
    fn proxy(g: *Gather) void {
        const c = g.ctrl();
        var idle: u64 = 0;
        const nap: std.c.timespec = .{ .sec = 0, .nsec = 20_000 };
        while (g.running.load(.monotonic)) {
            const seq = @atomicLoad(u32, &c[c_seq], .acquire);
            if (seq == g.last_seq) {
                idle += 1;
                if (idle % 64 == 0) for (0..g.n_hca) |h| g.drain(h) catch {
                    g.setAbort();
                    g.failed.store(true, .release);
                    return;
                };
                if (idle >= idle_spins) _ = std.c.nanosleep(&nap, null);
                continue;
            }
            idle = 0;
            // a rank's next gather can ring before this thread saw the last one; peers stay within a slot's distance
            if (sequenceState(g.last_seq, seq) == null) {
                const pending = seq -% g.last_seq;
                g.fail("the doorbell skipped {d} sequences (last {d}, now {d})", .{ pending, g.last_seq, seq });
                g.setAbort();
                g.failed.store(true, .release);
                return;
            }
            var state = sequenceState(g.last_seq, seq).?;
            var s = state.next;
            while (state.pending > 0) : ({
                s +%= 1;
                state.pending -= 1;
            }) {
                const nbytes = @atomicLoad(u32, &c[c_nbytes + (s & 1)], .acquire);
                g.post(s, nbytes) catch {
                    g.setAbort();
                    g.failed.store(true, .release);
                    return;
                };
                g.last_seq = s;
            }
        }
    }

    /// The cap both ranks test in ``fits``: the smaller of the two, which each rank's slot holds.
    fn agreedCap(mine: u64, theirs: u64) usize {
        return @intCast(@min(mine, theirs));
    }

    pub fn dispatchRoce(glm: bool, world: usize, dtype: u32, sum: bool, count: usize, max_bytes: usize) bool {
        if (!glm or world != 2 or !sum or dtype > 1) return false;
        const width: usize = if (dtype == 0) 4 else 2;
        const bytes = std.math.mul(usize, count, width) catch return false;
        return count != 0 and bytes <= max_bytes and bytes % 16 == 0 and !(dtype == 0 and count % 4 != 0);
    }

    /// Whether a gather of ``bytes`` a rank goes this way: the size alone decides, so both ranks pick the same path.
    pub fn fits(g: *const Gather, bytes: usize) bool {
        return bytes > 0 and bytes <= g.max_bytes and bytes % 16 == 0;
    }

    pub fn boundTo(g: *const Gather, stream: abi.Stream) bool {
        return sameStream(g.stream_handle, stream);
    }

    /// Startup alignment check for the actual exposed RoCE and CUDA allocations.
    pub fn buffersAligned16(g: *const Gather) bool {
        return aligned16(g.region_dev) and aligned16(g.counters.ptr) and aligned16(g.reduce_scratch.ptr);
    }

    /// Both ranks' ``bytes`` from ``send`` into ``recv`` in rank order (as ncclAllGather); misaligned is an error.
    pub fn gather(g: *Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, bytes: usize) !void {
        if (!g.boundTo(stream)) return error.StreamMismatch;
        if (!aligned16(send) or !aligned16(recv)) return error.Misaligned;
        const packs = bytes / 16;
        const need = @max(1, std.math.divCeil(usize, packs, 2 * threads) catch unreachable);
        const grid: u32 = @intCast(@min(std.math.ceilPowerOfTwo(usize, need) catch max_blocks, max_blocks));
        const klass: i32 = @intCast(std.math.log2_int(u32, grid));
        const base = g.region_dev;
        var a: launch.Args = .{};
        a.add(send);
        a.add(recv);
        a.add(@as(i32, @intCast(packs)));
        a.add(@as(i32, g.rank));
        a.add(base + g.layout.recv);
        a.add(base + g.layout.flags);
        a.add(base + g.layout.send);
        a.add(base + g.layout.ctrl);
        a.add(@as(i64, @intCast(g.layout.slot_bytes)));
        a.add(g.counters.ptr);
        a.add(klass);
        a.add(g.timeout_ns);
        a.add(@as(i32, @intCast(g.n_hca)));
        try launch.launch(g.f, .{ .grid = .{ .x = grid }, .block = .{ .x = threads } }, .{ .handle = stream, .d = g.d }, &a);
        g.gathers += 1;
    }

    /// Two-rank sum: gather both operands in rank order, then reduce on the same stream.
    /// `dtype` is 0 for fp32 and 1 for bf16; other types are left to NCCL by the caller.
    pub fn allReduce(g: *Gather, stream: abi.Stream, send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, dtype: u32) !void {
        if (dtype > 1) return error.UnsupportedType;
        const size: usize = if (dtype == 0) 4 else 2;
        const bytes = std.math.mul(usize, count, size) catch return error.Misaligned;
        if (!g.fits(bytes) or (dtype == 0 and count % 4 != 0)) return error.Misaligned;
        try g.gather(stream, send, g.reduce_scratch.ptr, bytes);
        var a: launch.Args = .{};
        a.add(g.reduce_scratch.ptr);
        a.add(recv);
        a.add(g.region_dev + g.layout.ctrl);
        a.add(@as(i32, @intCast(count)));
        const f = if (dtype == 0) g.reduce_f32 else g.reduce_bf16;
        const blocks = std.math.divCeil(usize, count, 256) catch unreachable;
        try launch.launch(f, .{ .grid = .{ .x = @intCast(blocks) }, .block = .{ .x = 256 } }, .{ .handle = stream, .d = g.d }, &a);
    }

    /// An error if a wait timed out on the GPU or the proxy failed (the host checks after a synchronize).
    pub fn check(g: *Gather) !void {
        const c = g.ctrl();
        const bad = @atomicLoad(u32, &c[c_error], .acquire);
        if (bad == 0 and !g.failed.load(.acquire)) return;
        const kind = switch (c[c_error_kind]) {
            1 => "peer wait timeout",
            2 => "peer byte-count mismatch",
            3 => "host proxy or group abort",
            else => "exchange failed",
        };
        std.log.err("RoCE all-gather failed ({s}, sequence {d}, HCA {d}): {s}", .{ kind, c[c_error_seq], c[c_hca], std.mem.sliceTo(&g.err, 0) });
        return error.ProxyFailed;
    }

    fn setAbort(g: *Gather) void {
        const c = g.ctrl();
        if (@atomicLoad(u32, &c[c_error], .acquire) == 0) {
            @atomicStore(u32, &c[c_error_seq], @atomicLoad(u32, &c[c_seq], .acquire), .monotonic);
            @atomicStore(u32, &c[c_error_kind], 3, .monotonic);
            @atomicStore(u32, &c[c_error], errorFlag(0), .release);
        }
        @atomicStore(u32, &c[c_abort], 1, .release);
    }

    pub fn abort(g: *Gather) void {
        g.setAbort();
    }

    /// Spark fault-injection hook: stop the host proxy without publishing abort, forcing the GPU deadline path.
    pub fn stopProxyForTest(g: *Gather) void {
        if (g.running.swap(false, .acq_rel)) if (g.thread) |thread| thread.join();
    }

    /// Poison a graph's host-visible token/status result after its model work completes.
    pub fn gateResult(g: *Gather, stream: abi.Stream, token: abi.DevicePtr, status: abi.DevicePtr) !void {
        var a: launch.Args = .{};
        a.add(g.region_dev + g.layout.ctrl);
        a.add(token);
        a.add(status);
        try launch.launch(g.status_gate, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, .{ .handle = stream, .d = g.d }, &a);
    }

    fn teardown(g: *Gather) void {
        if (g.running.swap(false, .acq_rel)) if (g.thread) |t| t.join();
        g.closeHcas();
        g.module.unload();
        g.reduce_scratch.free();
        g.counters.free();
        g.region.free();
        g.lib.close();
    }

    pub fn deinit(g: *Gather) void {
        g.teardown();
        g.gpa.destroy(g);
    }
};

/// The HCAs to stripe over, at most two: TF_ROCE_HCA, else NCCL_IB_HCA without "=^" prefixes or ":port" suffixes.
fn hcaNames(io: std.Io, gpa: std.mem.Allocator, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) !u32 {
    _ = io;
    _ = gpa;
    const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse return error.NoHca;
    var n: u32 = 0;
    var it = std.mem.tokenizeScalar(u8, std.mem.span(raw), ',');
    while (it.next()) |w| {
        if (n == max_hcas) break;
        var s = std.mem.trimStart(u8, std.mem.trim(u8, w, " "), "=^");
        if (std.mem.indexOfScalar(u8, s, ':')) |c| s = s[0..c];
        if (s.len == 0 or s.len > 63) continue;
        @memcpy(names[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
    }
    if (n == 0) return error.NoHca;
    return n;
}

/// A triangle rank's two neighbours, lower rank first (initTriangle's [0] and [1]).
pub fn triPeers(rank: u8) [2]u8 {
    const next: u8 = (rank + 1) % 3;
    const prev: u8 = (rank + 2) % 3;
    return .{ @min(next, prev), @max(next, prev) };
}

/// The port ('0' or '1', the HCA names' f0/f1 suffix) cabled from `rank` to `peer`: port 0 faces rank + 1 and
/// port 1 rank + 2 (rank 0 f0 to rank 1 f1, rank 1 f0 to rank 2 f1, rank 2 f0 to rank 0 f1), or the reverse with `f0_prev`.
pub fn triPort(rank: u8, peer: u8, f0_prev: bool) u8 {
    return if ((peer == (rank + 1) % 3) != f0_prev) '0' else '1';
}

/// The HCAs of one port from TF_ROCE_HCA, else NCCL_IB_HCA (see portNames).
fn portHcaNames(port: u8, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) u32 {
    const raw = std.c.getenv("TF_ROCE_HCA") orelse std.c.getenv("NCCL_IB_HCA") orelse return 0;
    return portNames(std.mem.span(raw), port, names, lens);
}

/// The names in an HCA list that end f`port` (without "=^" prefixes or ":port" suffixes), at most two, sorted: on a
/// Spark, rocep1s0fN and roceP2p1s0fN are port N's two PCIe functions, and each sits on its own /30 with the same
/// function of the neighbour's port, so both ends of a cable must stripe over them in the same order.
pub fn portNames(list: []const u8, port: u8, names: *[max_hcas][64]u8, lens: *[max_hcas]usize) u32 {
    var n: u32 = 0;
    var it = std.mem.tokenizeScalar(u8, list, ',');
    while (it.next()) |w| {
        if (n == max_hcas) break;
        var s = std.mem.trimStart(u8, std.mem.trim(u8, w, " "), "=^");
        if (std.mem.indexOfScalar(u8, s, ':')) |c| s = s[0..c];
        if (s.len < 2 or s.len > 63 or s[s.len - 2] != 'f' or s[s.len - 1] != port) continue;
        @memcpy(names[n][0..s.len], s);
        lens[n] = s.len;
        n += 1;
    }
    if (n == 2 and std.mem.order(u8, names[1][0..lens[1]], names[0][0..lens[0]]) == .lt) {
        std.mem.swap([64]u8, &names[0], &names[1]);
        std.mem.swap(usize, &lens[0], &lens[1]);
    }
    return n;
}

/// The HCA's RoCE v2 IPv4 GID (TF_ROCE_GID_INDEX overrides): its index differs between hosts' tables.
fn gidIndex(io: std.Io, gpa: std.mem.Allocator, name: []const u8) !c_int {
    if (std.c.getenv("TF_ROCE_GID_INDEX")) |v| return std.fmt.parseInt(c_int, std.mem.span(v), 10) catch error.NoHca;
    const cwd = std.Io.Dir.cwd();
    var i: c_int = 0;
    while (i < 16) : (i += 1) {
        var pbuf: [160]u8 = undefined;
        const tpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, i });
        const t = readSysfsText(cwd, io, gpa, tpath) catch continue;
        defer gpa.free(t);
        if (!std.mem.startsWith(u8, t, "RoCE v2")) continue;
        const gpath = try std.fmt.bufPrint(&pbuf, "/sys/class/infiniband/{s}/ports/1/gids/{d}", .{ name, i });
        const gid = readSysfsText(cwd, io, gpa, gpath) catch continue;
        defer gpa.free(gid);
        if (std.mem.startsWith(u8, gid, "0000:0000:0000:0000:0000:ffff:")) return i;
    }
    return error.NoHca;
}

/// Read a sysfs attribute to EOF; sysfs files report a zero stat size, so readFileAlloc can return no contents.
fn readSysfsText(dir: std.Io.Dir, io: std.Io, gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try dir.openFile(io, path, .{});
    defer file.close(io);
    var scratch: [128]u8 = undefined;
    var reader = file.readerStreaming(io, &scratch);
    return reader.interface.allocRemaining(gpa, .limited(64));
}

test "sysfs text helper reads to EOF without using stat size" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const expected = "RoCE v2\n";
    try tmp.dir.writeFile(io, .{ .sub_path = "gid_type", .data = expected });
    const got = try readSysfsText(tmp.dir, io, gpa, "gid_type");
    defer gpa.free(got);
    try std.testing.expectEqualSlices(u8, expected, got);
}

test "RoCE gather: both ranks agree one cap" {
    // 511 KiB and 512 KiB round to the same slot; a 512 KiB shard must take the same path on both ranks
    try std.testing.expectEqual(Layout.of(511 << 10).slot_bytes, Layout.of(512 << 10).slot_bytes);
    const a = Gather.agreedCap(511 << 10, 512 << 10);
    const b = Gather.agreedCap(512 << 10, 511 << 10);
    try std.testing.expectEqual(a, b);
    try std.testing.expectEqual(@as(usize, 511 << 10), a);
}

test "RoCE GLM all-reduce sizes and rank-order host reference" {
    for (1..9) |rows| {
        const bytes = rows * 6144 * @sizeOf(f32);
        try std.testing.expectEqual(@as(usize, 24 * 1024 * rows), bytes);
        try std.testing.expect(bytes % 16 == 0);
    }
    const rank0 = [_]f32{ 1.0, -2.5, 0.125, 1024.0 };
    const rank1 = [_]f32{ 0.25, 3.0, -0.0625, -1.0 };
    var out0: [4]f32 = undefined;
    var out1: [4]f32 = undefined;
    var bf0: [4]u16 = undefined;
    var bf1: [4]u16 = undefined;
    for (&out0, &out1, &bf0, &bf1, rank0, rank1) |*a, *b, *ba, *bb, x, y| {
        a.* = x + y;
        b.* = x + y;
        ba.* = f32ToBf16(x + y);
        bb.* = f32ToBf16(x + y);
    }
    try std.testing.expectEqualSlices(f32, &.{ 1.25, 0.5, 0.0625, 1023.0 }, &out0);
    try std.testing.expectEqualSlices(f32, &out0, &out1);
    try std.testing.expectEqualSlices(u16, &bf0, &bf1);
    try std.testing.expectEqualSlices(u16, &.{ 0x3FA0, 0x3F00, 0x3D80, 0x4480 }, &bf0);
}

test "RoCE sequence wrap and nonzero error encoding" {
    try std.testing.expectEqual(SeqState{ .next = 0xffff_ffff, .pending = 1 }, sequenceState(0xffff_fffe, 0xffff_ffff).?);
    try std.testing.expectEqual(SeqState{ .next = 0xffff_ffff, .pending = 2 }, sequenceState(0xffff_fffe, 0).?);
    try std.testing.expectEqual(SeqState{ .next = 0, .pending = 1 }, sequenceState(0xffff_ffff, 0).?);
    try std.testing.expect(sequenceState(9, 9) == null);
    try std.testing.expect(sequenceState(9, 12) == null);
    try std.testing.expectEqual(@as(u32, 1), errorFlag(0));
    try std.testing.expectEqual(@as(u32, 1), errorFlag(0xffff_ffff));
    try std.testing.expect(aligned16(0x1000));
    try std.testing.expect(!aligned16(0x1008));
}

test "RoCE all-reduce dispatch is symmetric and respects dtype, op, size, and opt-in" {
    const max_bytes = 512 * 1024;
    try std.testing.expect(!Gather.dispatchRoce(false, 2, 0, true, 6144, max_bytes));
    try std.testing.expect(!Gather.dispatchRoce(true, 2, 2, true, 6144, max_bytes));
    try std.testing.expect(!Gather.dispatchRoce(true, 2, 0, false, 6144, max_bytes));
    try std.testing.expect(!Gather.dispatchRoce(true, 2, 0, true, max_bytes, max_bytes));
    try std.testing.expect(!Gather.dispatchRoce(true, 2, 1, true, 3, max_bytes));
    for (1..9) |rows| try std.testing.expect(Gather.dispatchRoce(true, 2, 0, true, rows * 6144, max_bytes));
}

test "RoCE triangle: pairs, ports and HCA order agree at both ends of every cable" {
    for (0..3) |r| {
        const rank: u8 = @intCast(r);
        const peers = triPeers(rank);
        try std.testing.expect(peers[0] < peers[1] and peers[0] != rank and peers[1] != rank);
        for (peers) |peer| for ([_]bool{ false, true }) |f0_prev| {
            // the cable's two ends use opposite ports, so each port faces exactly one neighbour
            try std.testing.expect(triPort(rank, peer, f0_prev) != triPort(peer, rank, f0_prev));
        };
    }
    // the cluster's cabling: rank 0 f0 to rank 1, rank 1 f0 to rank 2, rank 2 f0 to rank 0
    try std.testing.expectEqual(@as(u8, '0'), triPort(0, 1, false));
    try std.testing.expectEqual(@as(u8, '1'), triPort(1, 0, false));
    try std.testing.expectEqual(@as(u8, '0'), triPort(2, 0, false));
    var names: [max_hcas][64]u8 = undefined;
    var lens: [max_hcas]usize = undefined;
    const list = "rocep1s0f0,roceP2p1s0f0, =rocep1s0f1:1,roceP2p1s0f1";
    try std.testing.expectEqual(@as(u32, 2), portNames(list, '0', &names, &lens));
    try std.testing.expectEqualStrings("roceP2p1s0f0", names[0][0..lens[0]]);
    try std.testing.expectEqualStrings("rocep1s0f0", names[1][0..lens[1]]);
    try std.testing.expectEqual(@as(u32, 2), portNames(list, '1', &names, &lens));
    try std.testing.expectEqualStrings("roceP2p1s0f1", names[0][0..lens[0]]);
    try std.testing.expectEqualStrings("rocep1s0f1", names[1][0..lens[1]]);
    try std.testing.expectEqual(@as(u32, 0), portNames("mlx5_0,mlx5_1", '0', &names, &lens));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(Gather.Pair));
    try std.testing.expectEqual(@as(usize, 48), @offsetOf(Gather.Pair, "peer_slot"));
}

test "RoCE stream identity is explicit" {
    const a: abi.Stream = @ptrFromInt(0x1000);
    const b: abi.Stream = @ptrFromInt(0x2000);
    try std.testing.expect(!sameStream(null, a));
    try std.testing.expect(sameStream(a, a));
    try std.testing.expect(!sameStream(a, b));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Gather);
}
