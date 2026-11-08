// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
//! A tensor-parallel group over NCCL (NCCL_COMM_ID; rows in rank order); two ranks add RoCE small gathers (roce.zig).
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
        g.destroyComm();
        g.lib.close();
        g.* = undefined;
    }

    fn destroyComm(g: *Group) void {
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommDestroy(g.comm);
    }

    pub fn abort(g: *Group) void {
        if (g.fast) |f| f.abort();
        if (!g.aborted.swap(true, .acq_rel)) _ = g.lib.api.ncclCommAbort(g.comm);
    }

    /// Two ranks: gathers up to `max_bytes` a rank go by RoCE writes if both ranks bring it up; else NCCL keeps them.
    pub fn openRoce(g: *Group, gpa: std.mem.Allocator, io: std.Io, d: *const driver.Driver, s: Stream, max_bytes: usize, force_gather: bool) !void {
        if (g.fast != null) return error.AlreadyOpen;
        const local_glm = try wantedGlmCollective();
        const local_gather = roce.wanted() or force_gather;
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
        if (gateEnabled(g.fast != null)) {
            const f = g.fast orelse return error.RoceUnavailable;
            try f.gateResult(s.handle, token, status);
        }
    }

    /// `bytes` at `mine` from every rank into `all` in rank order (world * bytes there), on `s`.
    pub fn allGather(g: *const Group, s: Stream, mine: u64, all: u64, bytes: usize) !void {
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
