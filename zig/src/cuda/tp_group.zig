// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig),
//! and three ranks cabled in a triangle add them with TF_ROCE_TRIANGLE=1 (one RoCE pair a neighbour).
const std = @import("std");
const nccl = @import("nccl.zig");
const driver = @import("driver.zig");
const stream_ = @import("stream.zig");
const memory = @import("memory.zig");
const roce = @import("roce.zig");
const abi = @import("abi.zig");

const Stream = stream_.Stream;

pub const GlmCollective = enum { nccl, roce };
pub const Route = enum { roce, nccl_size, nccl_stream };

pub fn gatherRoute(enabled: bool, has_fast: bool, fits: bool, on_bound_stream: bool) Route {
    if (enabled and has_fast and fits and on_bound_stream) return .roce;
    if (enabled and has_fast and fits and !on_bound_stream) return .nccl_stream;
    return .nccl_size;
}
pub const RoceSettings = struct { glm: bool, gather: bool, aligned: bool = true, timeout_ms: u32 = 0 };

pub fn agreesRoceSettings(rank0: RoceSettings, rank1: RoceSettings) bool {
    return rank0.glm == rank1.glm and rank0.gather == rank1.gather and rank0.aligned == rank1.aligned and rank0.timeout_ms == rank1.timeout_ms;
}

pub fn roceSettingsAllowed(settings: RoceSettings) bool {
    return (!settings.glm or settings.gather) and (settings.aligned or (!settings.glm and !settings.gather));
}

pub fn gateEnabled(has_roce: bool) bool {
    return has_roce;
}

pub fn buffersAlignedAcrossRanks(rank0: bool, rank1: bool) bool {
    return rank0 and rank1;
}

pub fn glmCollective(value: ?[]const u8) !GlmCollective {
    const v = value orelse return .nccl;
    if (std.mem.eql(u8, v, "roce")) return .roce;
    if (std.mem.eql(u8, v, "nccl")) return .nccl;
    return error.InvalidCollective;
}

pub const Group = struct {
    lib: nccl.Library,
    comm: nccl.Comm,
    rank: usize,
    world: usize,
    aborted: std.atomic.Value(bool) = .init(false),
    fast: ?*roce.Gather = null, // two ranks' small all-gathers by RoCE writes (openRoce), when both ranks brought it up
    glm_roce: bool = false, // TF_GLM53_COLLECTIVE agreed by both ranks during openRoce
    gather_roce: bool = false, // TF_ROCE agreed by both ranks; independent of GLM reductions
    routes: [3]u64 = @splat(0),
    tri: [2]?*roce.Gather = .{ null, null }, // three ranks: one pair a neighbour, lower rank first (openRoce)

    /// Whether gathers may take RoCE (two ranks' pair, or a triangle's two): RoCE poison checks and gates apply.
    pub fn hasRoce(g: *const Group) bool {
        return g.fast != null or g.tri[0] != null;
    }

    /// Rank `rank` of `world` (NCCL opened at run time; NCCL_COMM_ID names rank 0's address, every rank the same).
    pub fn open(rank: usize, world: usize) !Group {
        if (std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
        var lib = try nccl.Library.open();
        errdefer lib.close();
        var id: nccl.UniqueId = undefined;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        var comm: nccl.Comm = null;
        try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(world), id, @intCast(rank)), "ncclCommInitRank");
        return .{ .lib = lib, .comm = comm, .rank = rank, .world = world };
    }

    pub fn close(g: *Group) void {
        if (g.fast) |f| f.deinit();
        for (g.tri) |t| if (t) |f| f.deinit();
        g.destroyComm();
        g.lib.close();
        g.* = undefined;
    }

    fn destroyComm(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommDestroy(g.comm);
    }

    pub fn abort(g: *Group) void {
        if (g.fast) |f| f.abort();
        for (g.tri) |t| if (t) |f| f.abort();
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommAbort(g.comm);
    }

    /// Two ranks: gathers up to `max_bytes` a rank go by RoCE writes if both ranks bring it up; else NCCL keeps them.
    pub fn openRoce(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize, force_gather: bool) !void {
        if (g.hasRoce()) return error.AlreadyOpen;
        const local_glm = try wantedGlmCollective();
        const local_gather = roce.wanted() or force_gather;
        if (g.world == 3) return g.openTriangle(gpa, io, d, s, max_bytes, local_glm, local_gather);
        if (g.world != 2) {
            if (local_glm) return error.RoceRequiresTwoRanks;
            return;
        }
        const timeout_ms = roce.timeoutMs(std.c.getenv("TF_ROCE_TIMEOUT_MS"), local_glm) catch |err| {
            std.log.err("Invalid RoCE timeout setting on rank {d}: TF_ROCE_TIMEOUT_MS", .{g.rank});
            return err;
        };
        const Config = extern struct { glm: u8, gather: u8, timeout_ms: u32 };
        const mine = Config{ .glm = @intFromBool(local_glm), .gather = @intFromBool(local_gather), .timeout_ms = timeout_ms };
        var peers: [2]Config = undefined;
        try g.gather(Config, d, s, mine, &peers);
        const rank0 = RoceSettings{ .glm = peers[0].glm != 0, .gather = peers[0].gather != 0, .timeout_ms = peers[0].timeout_ms };
        const rank1 = RoceSettings{ .glm = peers[1].glm != 0, .gather = peers[1].gather != 0, .timeout_ms = peers[1].timeout_ms };
        if (!agreesRoceSettings(rank0, rank1)) {
            std.log.err("RoCE settings disagree: rank0 GLM={d} TF_ROCE={d} timeout_ms={d}; rank1 GLM={d} TF_ROCE={d} timeout_ms={d}", .{ @intFromBool(rank0.glm), @intFromBool(rank0.gather), rank0.timeout_ms, @intFromBool(rank1.glm), @intFromBool(rank1.gather), rank1.timeout_ms });
            return error.PeerRefused;
        }
        if (!roceSettingsAllowed(rank0)) {
            std.log.err("RoCE settings disabled: rank0 GLM={d} TF_ROCE={d}; rank1 GLM={d} TF_ROCE={d}", .{ @intFromBool(rank0.glm), @intFromBool(rank0.gather), @intFromBool(rank1.glm), @intFromBool(rank1.gather) });
            return error.RoceDisabled;
        }
        g.glm_roce = rank0.glm;
        g.gather_roce = rank0.gather;
        g.fast = try roce.Gather.init(gpa, io, d, g, s, g.gather_roce or g.glm_roce, max_bytes, timeout_ms);
        if (g.glm_roce and g.fast == null) return error.RoceUnavailable;
        if (g.fast) |f| {
            const Config2 = extern struct { aligned: u8, padding: [7]u8 = @splat(0) };
            const local2 = Config2{ .aligned = @intFromBool(f.buffersAligned16()) };
            var peers2: [2]Config2 = undefined;
            try g.gather(Config2, d, s, local2, &peers2);
            if (!buffersAlignedAcrossRanks(peers2[0].aligned != 0, peers2[1].aligned != 0)) {
                std.log.err("RoCE device buffer alignment check failed: rank0 aligned={d}; rank1 aligned={d}", .{ peers2[0].aligned, peers2[1].aligned });
                return error.RoceAlignmentViolation;
            }
        }
        if (g.glm_roce) try g.probeRoce(s, d);
    }

    /// TF_ROCE_TRIANGLE=1 (TF_ROCE not 0) on all three ranks: setup handshakes agree the switch and wait, then
    /// roce.Gather.initTriangle brings up a pair a neighbour, and a probe against ncclAllGather must match on every
    /// rank. Any rank failing setup, alignment or the probe keeps all three ranks' gathers on NCCL.
    fn openTriangle(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize, local_glm: bool, local_gather: bool) !void {
        if (local_glm) return error.RoceRequiresTwoRanks;
        const timeout_ms = roce.timeoutMs(std.c.getenv("TF_ROCE_TIMEOUT_MS"), false) catch |err| {
            std.log.err("Invalid RoCE timeout setting on rank {d}: TF_ROCE_TIMEOUT_MS", .{g.rank});
            return err;
        };
        const Config = extern struct { tri: u32, timeout_ms: u32 };
        var all: [3]Config = undefined;
        try g.gather(Config, d, s, .{ .tri = @intFromBool(local_gather and wantedTriangle()), .timeout_ms = timeout_ms }, &all);
        for (all) |c| if (c.tri != all[0].tri or c.timeout_ms != all[0].timeout_ms) {
            std.log.err("RoCE triangle settings disagree (TF_ROCE, TF_ROCE_TRIANGLE, TF_ROCE_TIMEOUT_MS): rank0 {d}/{d}, rank1 {d}/{d}, rank2 {d}/{d}", .{ all[0].tri, all[0].timeout_ms, all[1].tri, all[1].timeout_ms, all[2].tri, all[2].timeout_ms });
            return error.PeerRefused;
        };
        if (all[0].tri == 0) return;
        const pairs = try roce.Gather.initTriangle(gpa, io, d, g, s, max_bytes, timeout_ms);
        const a = pairs[0] orelse return; // initTriangle keeps both pairs on every rank or none anywhere
        const b = pairs[1].?;
        var keep = false;
        defer if (!keep) {
            a.deinit();
            b.deinit();
        };
        // g.tri stays empty until every rank agrees, so these handshakes' poison checks never see the probe's pairs
        var flags: [3]u32 = undefined;
        try g.gather(u32, d, s, @intFromBool(a.buffersAligned16() and b.buffersAligned16()), &flags);
        if (flags[0] == 0 or flags[1] == 0 or flags[2] == 0) {
            std.log.warn("RoCE triangle: device buffers not 16-byte aligned on every rank ({d}/{d}/{d}); gathers stay on NCCL", .{ flags[0], flags[1], flags[2] });
            return;
        }
        // a hard error (allocation, launch, NCCL) ends startup, as it would anywhere else in setup; a mismatch or a
        // timed-out pair is a soft failure every rank reports below
        const probed = try probeTriangle(g, gpa, a, b, s, d, timeout_ms);
        try g.gather(u32, d, s, @intFromBool(probed), &flags);
        if (flags[0] == 0 or flags[1] == 0 or flags[2] == 0) {
            std.log.warn("RoCE triangle probe against NCCL failed (rank0 {d}, rank1 {d}, rank2 {d}); gathers stay on NCCL", .{ flags[0], flags[1], flags[2] });
            return;
        }
        keep = true;
        g.tri = .{ a, b };
        g.gather_roce = true;
        std.log.info("RoCE triangle gathers: rank {d} pairs with ranks {d} and {d}, shards up to {d} KiB, probe matched NCCL", .{ g.rank, a.peer_slot, b.peer_slot, a.max_bytes >> 10 });
    }

    /// Triangle gathers of several sizes (every grid class, both slots) against ncclAllGather of the same bytes. The
    /// GPU wait is cut to 10 s meanwhile, so a pair that never connects fails here instead of hanging a request.
    fn probeTriangle(g: *Group, gpa: std.mem.Allocator, a: *roce.Gather, b: *roce.Gather, s: Stream, d: *const driver.Driver, timeout_ms: u32) !bool {
        a.setTimeoutMs(10_000);
        b.setTimeoutMs(10_000);
        defer {
            a.setTimeoutMs(timeout_ms);
            b.setTimeoutMs(timeout_ms);
        }
        const sizes = [_]usize{ 16, 4096, 24 << 10, 64 << 10, 192 << 10, 1 << 20 };
        const most = @min(sizes[sizes.len - 1], a.max_bytes);
        var buf = try memory.DeviceBuffer.alloc(d, 7 * most);
        defer buf.free();
        const mine = try gpa.alloc(u8, most);
        defer gpa.free(mine);
        const got = try gpa.alloc(u8, 6 * most);
        defer gpa.free(got);
        // Every rank runs every round whatever it saw, so the three keep issuing the same NCCL calls; a poisoned pair
        // returns at once from later launches and its neighbours stop waiting at their own deadline.
        var ok = true;
        for (0..3) |round| for (sizes) |bytes| {
            if (bytes > most) continue;
            for (mine[0..bytes], 0..) |*x, i| x.* = @intCast((i * 7 + g.rank * 13 + round * 5 + bytes) % 251);
            try buf.uploadAsync(0, mine[0..bytes], s.handle);
            try roce.Gather.triGather(a, b, s.handle, buf.ptr, buf.ptr + most, bytes);
            try g.lib.check(g.lib.api.ncclAllGather(buf.ptr, buf.ptr + 4 * most, bytes, .u8, g.comm, s.handle), "ncclAllGather(RoCE triangle probe)");
            try s.synchronize();
            if (!ok or a.failedQuietly() or b.failedQuietly()) {
                ok = false;
                continue;
            }
            try buf.download(most, got[0 .. 6 * most]);
            if (!std.mem.eql(u8, got[0 .. 3 * bytes], got[3 * most ..][0 .. 3 * bytes])) ok = false;
        };
        return ok;
    }

    /// TF_ROCE_TRIANGLE=1 asks three cabled ranks for RoCE gathers (every rank must ask; TF_ROCE=0 still wins).
    pub fn wantedTriangle() bool {
        const v = std.c.getenv("TF_ROCE_TRIANGLE") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    fn probeRoce(g: *Group, s: Stream, d: *const driver.Driver) !void {
        const n = 4096;
        var buf = try memory.DeviceBuffer.alloc(d, 7 * n);
        defer buf.free();
        var mine: [n]u8 = undefined;
        for (0..n / 2) |i| {
            const x = (@as(f32, @floatFromInt(@as(i32, @intCast((i + g.rank * 29) % 251)) - 125)) + 0.375) / 128.0;
            std.mem.writeInt(u16, @ptrCast(mine[0..].ptr + i * 2), roce.f32ToBf16(x), .little);
        }
        try buf.uploadAsync(0, &mine, s.handle);
        const f = g.fast orelse return error.RoceUnavailable;
        try f.gather(s.handle, buf.ptr, buf.ptr + n, n);
        try g.lib.check(g.lib.api.ncclAllGather(buf.ptr, buf.ptr + 3 * n, n, .u8, g.comm, s.handle), "ncclAllGather(RoCE probe)");
        try f.allReduce(s.handle, buf.ptr, buf.ptr + 5 * n, n / 2, 1);
        try g.lib.check(g.lib.api.ncclAllReduce(buf.ptr, buf.ptr + 6 * n, n / 2, .bf16, .sum, g.comm, s.handle), "ncclAllReduce(RoCE probe)");
        try g.syncChecked(s);
        var got: [6 * n]u8 = undefined;
        try buf.download(n, &got);
        if (!std.mem.eql(u8, got[0 .. 2 * n], got[2 * n .. 4 * n])) return error.RoceProbeMismatch;
        if (!std.mem.eql(u8, got[4 * n .. 5 * n], got[5 * n .. 6 * n])) return error.RoceProbeMismatch;
    }

    pub fn agreesGlmCollective(rank0: bool, rank1: bool) bool {
        return rank0 == rank1;
    }

    /// GLM's RoCE opt-in additionally enables RoCE all-reduce; TF_ROCE controls small all-gathers.
    pub fn wantedGlmCollective() !bool {
        const v = std.c.getenv("TF_GLM53_COLLECTIVE") orelse return false;
        return (try glmCollective(std.mem.span(v))) == .roce;
    }

    /// The default remains NCCL. RoCE handles only two-rank sum of fp32/bf16 inside its agreed cap.
    pub fn allReduce(g: *const Group, s: Stream, input: u64, output: u64, count: usize, dtype: nccl.DataType, op: nccl.RedOp) !void {
        const dt: u32 = switch (dtype) {
            .f32 => 0,
            .bf16 => 1,
            else => 2,
        };
        if (g.fast) |f| {
            if (roce.Gather.dispatchRoce(g.glm_roce, g.world, dt, op == .sum, count, f.max_bytes) and f.boundTo(s.handle)) {
                if (!roce.aligned16(input) or !roce.aligned16(output)) return error.RoceAlignmentViolation;
                return f.allReduce(s.handle, input, output, count, dt);
            }
        }
        try g.lib.check(g.lib.api.ncclAllReduce(input, output, count, dtype, op, g.comm, s.handle), "ncclAllReduce");
    }

    /// A failed RoCE gather's error (its rows hold nothing valid): call it after a synchronize that reads a gather.
    pub fn check(g: *const Group) !void {
        if (g.fast) |f| try f.check();
        for (g.tri) |t| if (t) |f| try f.check();
    }

    /// A step result is valid only after this synchronized poison check.
    pub fn syncChecked(g: *const Group, s: Stream) !void {
        try s.synchronize();
        g.check() catch |err| {
            @constCast(g).abort();
            return err;
        };
    }

    /// Enqueue at the end of a step graph so poisoned RoCE gathers or reductions cannot return a token id.
    pub fn gateResult(g: *const Group, s: Stream, token: u64, status: u64) !void {
        if (g.tri[0]) |a| return roce.Gather.triGateResult(a, g.tri[1].?, s.handle, token, status);
        if (gateEnabled(g.fast != null)) {
            const f = g.fast orelse return error.RoceUnavailable;
            try f.gateResult(s.handle, token, status);
        }
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        if (g.tri[0]) |a| {
            const b = g.tri[1].?;
            const route = gatherRoute(g.gather_roce, true, a.fits(bytes) and b.fits(bytes), a.boundTo(s.handle) and b.boundTo(s.handle));
            @constCast(g).routes[@intFromEnum(route)] += 1;
            if (route == .roce) {
                if (!roce.aligned16(mine) or !roce.aligned16(all)) return error.RoceAlignmentViolation;
                return roce.Gather.triGather(a, b, s.handle, mine, all, bytes);
            }
            return g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
        }
        const f = g.fast;
        const route = gatherRoute(g.gather_roce, f != null, if (f) |fast| fast.fits(bytes) else false, if (f) |fast| fast.boundTo(s.handle) else false);
        @constCast(g).routes[@intFromEnum(route)] += 1;
        if (route == .roce) if (f) |fast| {
            if (!roce.aligned16(mine) or !roce.aligned16(all)) return error.RoceAlignmentViolation;
            return fast.gather(s.handle, mine, all, bytes);
        };
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
    }

    /// Setup handshakes always cross NCCL (no deadline), so a rank still loading weights cannot trip the RoCE peer deadline (2 s under TF_GLM53_COLLECTIVE=roce).
    fn allGatherNccl(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        @constCast(g).routes[@intFromEnum(Route.nccl_size)] += 1;
        try g.lib.check(g.lib.api.ncclAllGather(mine, all, bytes, .u8, g.comm, s.handle), "ncclAllGather");
    }

    /// Host-issued route counts; captured graph replays do not pass through this function again.
    pub fn routeCounts(g: *const Group) [3]u64 {
        return g.routes;
    }

    /// Force the one-shot RoCE gather for an explicitly selected GLM fused exchange. Unlike allGather, this never
    /// falls back to NCCL: callers must establish at setup that the payload fits the negotiated cap.
    pub fn oneShotGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
        const f = g.fast orelse return error.RoceUnavailable;
        if (!g.gather_roce or !f.fits(bytes)) return error.RocePayloadTooLarge;
        if (!f.boundTo(s.handle)) return error.RoceWrongStream;
        if (!roce.aligned16(mine) or !roce.aligned16(all)) return error.RoceAlignmentViolation;
        try f.gather(s.handle, mine, all, bytes);
    }

    /// `bytes` at `ptr` to rank `peer`, on `s` (its matching recv takes the same count).
    pub fn send(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclSend(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclSend");
    }

    /// `bytes` from rank `peer` into `ptr`, on `s`.
    pub fn recv(g: *const Group, s: Stream, ptr: u64, bytes: usize, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclRecv(ptr, bytes, .u8, @intCast(peer), g.comm, s.handle), "ncclRecv");
    }

    /// Rank 0 and rank `peer` connect both ways with one small exchange (`ptr`: 128 bytes, 64 out and 64 in).
    pub fn connect(g: *const Group, s: Stream, ptr: u64, peer: usize) !void {
        try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
        const posted: anyerror!void = post: {
            g.send(s, ptr, 64, peer) catch |err| break :post err;
            g.recv(s, ptr + 64, 64, peer) catch |err| break :post err;
            break :post {};
        };
        try g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd"); // the group closes on every path
        try posted;
        try s.synchronize();
    }

    /// setup handshakes only; always NCCL.
    pub fn gather(g: *const Group, comptime T: type, d: *const driver.Driver, s: Stream, mine: T, out: []T) !void {
        if (out.len != g.world) return error.BadWorld;
        const size = @sizeOf(T);
        var dev = try memory.DeviceBuffer.alloc(d, size * g.world);
        defer dev.free();
        try dev.uploadAsync(g.rank * size, std.mem.asBytes(&mine), s.handle);
        try g.allGatherNccl(s, dev.ptr + g.rank * size, dev.ptr, size);
        try g.syncChecked(s);
        try dev.download(0, std.mem.sliceAsBytes(out));
    }
};

test "an aborted two-rank communicator is not destroyed a second time" {
    const Fake = struct {
        var aborts: usize = 0;
        var destroys: usize = 0;
        fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
            aborts += 1;
            return 0;
        }
        fn destroy(_: nccl.Comm) callconv(.c) nccl.Result {
            destroys += 1;
            return 0;
        }
    };
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    group.lib.api.ncclCommAbort = Fake.abort;
    group.lib.api.ncclCommDestroy = Fake.destroy;
    group.abort();
    group.abort();
    group.destroyComm();
    try std.testing.expectEqual(@as(usize, 1), Fake.aborts);
    try std.testing.expectEqual(@as(usize, 0), Fake.destroys);
}

test "setup handshakes take NCCL even when RoCE would fit" {
    const Fake = struct {
        var calls: usize = 0;
        var count: usize = 0;
        fn allGather(_: u64, _: u64, n: usize, _: nccl.DataType, _: nccl.Comm, _: @import("abi.zig").Stream) callconv(.c) nccl.Result {
            calls += 1;
            count = n;
            return 0;
        }
    };
    var gat: roce.Gather = undefined;
    gat.max_bytes = 4 << 20;
    gat.stream_handle = @as(abi.Stream, @ptrFromInt(0x1000));
    gat.gathers = 0;
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2 };
    group.fast = &gat;
    group.gather_roce = true;
    group.glm_roce = true;
    group.lib.api.ncclAllGather = Fake.allGather;
    const setup_stream: Stream = .{ .d = undefined, .handle = @ptrFromInt(0x1000) };
    Fake.calls = 0;
    Fake.count = 0;
    try group.allGatherNccl(setup_stream, 0x10000, 0x20000, 16);
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    try std.testing.expectEqual(@as(usize, 16), Fake.count);
    try std.testing.expectEqual([_]u64{ 0, 1, 0 }, group.routeCounts());
    try std.testing.expectEqual(@as(u64, 0), gat.gathers);
    group.fast = null;
}

test "model gathers keep their route" {
    try std.testing.expectEqual(Route.roce, gatherRoute(true, true, true, true));
    try std.testing.expectEqual(Route.nccl_stream, gatherRoute(true, true, true, false));
    try std.testing.expectEqual(Route.nccl_size, gatherRoute(true, true, false, true));
    try std.testing.expectEqual(Route.nccl_size, gatherRoute(false, true, true, true));
}

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Group);
}

test "GLM collective stays on NCCL unless explicitly selected" {
    try std.testing.expectEqual(GlmCollective.nccl, glmCollective(null));
    try std.testing.expectEqual(GlmCollective.nccl, glmCollective("nccl"));
    try std.testing.expectError(error.InvalidCollective, glmCollective("ROCE"));
    try std.testing.expectEqual(GlmCollective.roce, glmCollective("roce"));
    try std.testing.expect(Group.agreesGlmCollective(false, false));
    try std.testing.expect(Group.agreesGlmCollective(true, true));
    try std.testing.expect(!Group.agreesGlmCollective(false, true));
    try std.testing.expect(gateEnabled(true));
    try std.testing.expect(!gateEnabled(false));
    try std.testing.expect(buffersAlignedAcrossRanks(true, true));
    try std.testing.expect(!buffersAlignedAcrossRanks(false, true));
    try std.testing.expect(!buffersAlignedAcrossRanks(true, false));
    try std.testing.expect(!agreesRoceSettings(.{ .glm = false, .gather = true, .timeout_ms = 120_000 }, .{ .glm = false, .gather = true, .timeout_ms = 2_000 }));
}

test "gather route uses switch, availability, size and stream role" {
    for (0..16) |bits| {
        const enabled = bits & 1 != 0;
        const has_fast = bits & 2 != 0;
        const fits = bits & 4 != 0;
        const bound = bits & 8 != 0;
        const expected: Route = if (enabled and has_fast and fits and bound) .roce else if (enabled and has_fast and fits) .nccl_stream else .nccl_size;
        try std.testing.expectEqual(expected, gatherRoute(enabled, has_fast, fits, bound));
    }
}

test "allGather routes fitting off-stream payloads through NCCL and counts the route" {
    const Fake = struct {
        var calls: usize = 0;
        fn allGather(_: u64, _: u64, _: usize, _: nccl.DataType, _: nccl.Comm, _: @import("abi.zig").Stream) callconv(.c) nccl.Result {
            calls += 1;
            return 0;
        }
    };
    var gat: roce.Gather = undefined;
    gat.max_bytes = 4 << 20;
    gat.stream_handle = @as(abi.Stream, @ptrFromInt(0x1000));
    var group: Group = .{ .lib = undefined, .comm = null, .rank = 0, .world = 2, .fast = &gat, .gather_roce = true };
    group.lib.api.ncclAllGather = Fake.allGather;
    const other: Stream = .{ .d = undefined, .handle = @ptrFromInt(0x2000) };
    Fake.calls = 0;
    try group.allGather(other, 0x4000, 0x8000, 4096);
    try std.testing.expectEqual(@as(usize, 1), Fake.calls);
    try std.testing.expectEqual(@as(u64, 1), group.routeCounts()[@intFromEnum(Route.nccl_stream)]);
    group.gather_roce = false;
    try group.allGather(other, 0x4000, 0x8000, 4096);
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expectEqual(@as(u64, 1), group.routeCounts()[@intFromEnum(Route.nccl_size)]);
}
