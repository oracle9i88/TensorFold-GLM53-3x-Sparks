// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31: zig/src/families/glm/cuda_native.zig (Apache-2.0). Changes: adds the glm_moe_dsa native contract and WP9 seam.
//! Native CUDA registration surface for the full GLM-5.3 DSA family.

const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const engine_mod = @import("engine");
const forward = @import("forward");
const moe = forward.moe;
const weights = @import("weights");
const state = @import("state");
const uma = @import("uma");
const glm_config = @import("glm_config");
const backend_adapter = @import("backend");
const manifest_mod = weights.manifest;
const follow = @import("glm_follow");
const prewarm = @import("prewarm");
const zngram = lanes.zngram;
const posix = @import("sys").posix;
const ablate = @import("ablate");
const drafts = @import("drafts");
const glm = @import("glm");
const prefix_cache = @import("prefix_cache");
const prefix_snap = @import("prefix_snap");

const libc_env = struct {
    extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
};

var follower_stop_signal: std.atomic.Value(u32) = .init(0);
var follower_prev_int: posix.Sigaction = undefined;
var follower_prev_term: posix.Sigaction = undefined;

fn followerStopSignal(sig: posix.SIG) callconv(.c) void {
    follower_stop_signal.store(@intCast(@backingInt(sig)), .release);
    const prev = if (sig == .INT) follower_prev_int else follower_prev_term;
    const handler = prev.handler.handler orelse return;
    if (handler != posix.SIG.DFL and handler != posix.SIG.IGN) handler(sig);
}

fn installFollowerStopSignals() void {
    follower_stop_signal.store(0, .release);
    const action: posix.Sigaction = .{ .handler = .{ .handler = followerStopSignal }, .mask = posix.sigemptyset(), .flags = 0 };
    posix.sigaction(.INT, &action, &follower_prev_int);
    posix.sigaction(.TERM, &action, &follower_prev_term);
}

fn restoreFollowerStopSignals() void {
    posix.sigaction(.INT, &follower_prev_int, null);
    posix.sigaction(.TERM, &follower_prev_term, null);
}

fn followerStopRequested() bool {
    return follower_stop_signal.load(.acquire) != 0;
}

fn followerStopName() []const u8 {
    return if (follower_stop_signal.load(.acquire) == @as(u32, @intCast(@backingInt(posix.SIG.TERM)))) "SIGTERM" else "SIGINT";
}

fn logFollowerStop() void {
    std.log.info("rank 1: {s}: stopping", .{followerStopName()});
}

fn serveGateEnabled() !bool {
    const raw = std.c.getenv("TF_GLM53_SERVE_GATE") orelse return true;
    const value = std.mem.span(raw);
    if (std.mem.eql(u8, value, "1")) return true;
    if (std.mem.eql(u8, value, "0")) return false;
    return error.BadServeGate;
}

fn graphCacheEnabled() bool {
    if (std.c.getenv("TF_GRAPH_CACHE")) |cache| {
        const value = std.mem.span(cache);
        if (value.len != 0 and value[0] == '0') return false;
    }
    if (std.c.getenv("TF_GLM53_NO_GRAPHS")) |graphs| {
        const value = std.mem.span(graphs);
        if (value.len != 0 and value[0] == '1') return false;
    }
    return true;
}

/// Every rank contributes a refusal bit to one NCCL max reduction, so a low-memory rank refuses the whole TP pair.
fn postLoadServeGate(own: anytype, io: std.Io, gpa: std.mem.Allocator, rank: usize, dir: []const u8, context: usize, expected_graph_count: ?usize) !void {
    const enabled = try serveGateEnabled();
    if (!enabled and expected_graph_count == null) return;
    try own.stream.synchronize();
    var info = try uma.readMeminfo(gpa, io);
    const min_avail = try state.serveMinAvailableFromEnv(info.total);
    const min_free = try state.serveMinFreeFromEnv(info.total);
    if (enabled and info.free < min_free) {
        // The weights loader advises shard pages already. Retry after optional prewarm to reclaim clean cached pages.
        weights.dropLoadedShardPages(gpa, io, dir);
        info = try uma.readMeminfo(gpa, io);
    }
    const local = if (enabled) state.serveGate(info.available, info.free, min_avail, min_free) else .ok;
    const local_byte: u8 = @intFromBool(enabled and local != .ok);
    var flag = try cuda.DeviceBuffer.alloc(own.stream.d, 1);
    defer flag.free();
    try flag.uploadAsync(0, std.mem.asBytes(&local_byte), own.stream.handle);
    try own.group.lib.check(own.group.lib.api.ncclAllReduce(flag.ptr, flag.ptr, 1, .u8, .max, own.group.comm, own.stream.handle), "ncclAllReduce(serve gate verdict)");
    try own.stream.synchronize();
    var global_byte: u8 = 0;
    try flag.download(0, std.mem.asBytes(&global_byte));
    const Metrics = extern struct { available: u64, free: u64, graphs: u64 };
    var metrics_buf: [3]Metrics = undefined;
    const pair_metrics = metrics_buf[0..own.group.world];
    const graph_count: u64 = if (own.engine.graphs) |cache| @intCast(cache.count()) else 0;
    try own.group.gather(Metrics, own.stream.d, own.stream, .{ .available = info.available, .free = info.free, .graphs = graph_count }, pair_metrics);
    const verdict: []const u8 = if (global_byte == 0) "pass" else "refuse";
    for (pair_metrics, 0..) |m, peer_rank| {
        const peer_verdict = state.serveGate(m.available, m.free, min_avail, min_free);
        std.log.info("GLM-5.3 serve gate rank {d}: MemAvailable {d:.2} GiB (need {d:.2}), MemFree {d:.2} GiB (need {d:.2}); local={s}, all-rank={s}{s}", .{ peer_rank, state.gib(m.available), state.gib(min_avail), state.gib(m.free), state.gib(min_free), if (peer_rank == rank) @tagName(local) else @tagName(peer_verdict), verdict, if (!enabled) " (gate disabled locally)" else "" });
        if (global_byte != 0 and peer_verdict != .ok) std.log.err("glm53 serve gate: rank {d} MemAvailable {d:.2} GiB (need {d:.2}), MemFree {d:.2} GiB (need {d:.2}); refusing to serve at context {d}", .{ peer_rank, state.gib(m.available), state.gib(min_avail), state.gib(m.free), state.gib(min_free), context });
    }
    if (global_byte != 0) {
        return error.ServeMemoryGateRefused;
    }
    if (expected_graph_count) |expected| {
        for (pair_metrics, 0..) |metrics, peer_rank| {
            if (metrics.graphs != expected) {
                std.log.err("glm53 serve gate: rank {d} prewarmed {d} decode graphs, need {d}; refusing to serve at context {d}", .{ peer_rank, metrics.graphs, expected, context });
                return error.DecodeGraphPrewarmIncomplete;
            }
        }
    }
}

pub const model_type = "glm_moe_dsa";
pub const formats: []const []const u8 = &.{ "fp8-block128", "exl3-mixed" };
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = 4096;
pub const snap_grid: u32 = prompt_rows;
/// Warm turns wait briefly for other work before their first prefill chunk.
pub const warm_quiet: u64 = 150 * std.time.ns_per_ms;
pub const explains_open = true;
pub const takes_drafter = true;
pub const openRefusal = "GLM-5.3 DSA CUDA requires a valid TP2 rank configuration and checkpoint";

pub const Options = struct {
    context: usize,
    drafts: bool = false,
    segments: usize = 1,
    slide: bool = false,
    room: ?Room = null,
    kv_format: pages_format = .bf16,
    page_cache_bytes: u64 = 0,
    manifest: ?[]const u8 = null,
    prewarm: ?[]const u8 = null,
    ablate_file: ?[]const u8 = null,
    drafter: ?[]const u8 = null,
    drafter_bits: u8 = 4,
};
pub const pages_format = enum { bf16, fp8 };
pub const Room = struct { ctx: *anyopaque, read: *const fn (*anyopaque) ?u64 };

pub const Place = struct { rank: usize, world: usize };

pub fn copyLeader(raw: ?[]const u8, out: *[64]u8) ![]const u8 {
    const value = raw orelse return error.NoCommId;
    if (value.len == 0 or value.len > out.len or std.mem.indexOfScalar(u8, value, ':') == null) return error.NoCommId;
    @memcpy(out[0..value.len], value);
    return out[0..value.len];
}

pub fn sideCommsWanted(attn_pipe: bool, prefetch: bool) bool {
    return attn_pipe or prefetch;
}

/// Removes NCCL_COMM_ID from the environment NCCL reads, without touching the array the process started with.
/// glibc unsetenv shifts later entries down inside the live array; at start-up that array is the block Zig's start
/// code captured with a fixed length, so its last slot became null and the first stderr write panicked in
/// Io.Threaded.Environ.scan (`opt_entry.?`, "attempt to use null value"; v12 ATTN_PIPE=1, 2026-11-03).
/// The new array lives for the process (libc points at it); it is never freed.
fn releaseCommId() !void {
    const old = std.c.environ;
    var n: usize = 0;
    while (old[n] != null) : (n += 1) {}
    const fresh = try std.heap.c_allocator.allocSentinel(?[*:0]u8, n, null);
    var k: usize = 0;
    for (old[0..n]) |entry| {
        if (std.mem.startsWith(u8, std.mem.span(entry.?), "NCCL_COMM_ID=")) continue;
        fresh[k] = entry;
        k += 1;
    }
    @memset(fresh[k..n], null);
    std.c.environ = fresh.ptr;
    if (std.c.getenv("NCCL_COMM_ID") != null) return error.CommIdUnsetFailed;
}

pub fn parseExpertRows(raw: ?[]const u8) !usize {
    const rows = if (raw) |value| std.fmt.parseInt(usize, value, 10) catch return error.BadExpertRows else 128;
    if ((rows >= 16 and rows <= 512 and rows % 16 == 0) or rows == 1024 or rows == 1536 or rows == 2048 or rows == 4096 or rows == 8192) return rows;
    return error.BadExpertRows;
}

pub fn expertRows() !usize {
    const raw = std.c.getenv("TF_GLM53_EXPERT_ROWS");
    return parseExpertRows(if (raw) |v| std.mem.span(v) else null);
}

pub fn parseExpertRowsAdapt(raw: ?[]const u8) !usize {
    const value = raw orelse return 0;
    if (std.mem.eql(u8, value, "0")) return 0;
    if (std.mem.eql(u8, value, "2048")) return 2048;
    if (std.mem.eql(u8, value, "4096")) return 4096;
    return error.BadExpertRowsAdapt;
}

pub fn parseAdaptMaxPos(raw: ?[]const u8) !usize {
    const value = raw orelse return 0;
    const n = std.fmt.parseInt(usize, value, 10) catch return error.BadExpertRowsAdaptMaxPos;
    if (n == 0) return error.BadExpertRowsAdaptMaxPos;
    return n;
}

pub fn validateExpertRowsAdapt(g: state.Geometry) !void {
    if (g.expert_rows_adapt == 0) return;
    if (g.expert_rows_adapt <= g.expert_rows) return error.ExpertRowsAdaptNotLarger;
    if (g.expert_rows_adapt > g.chunk_rows) return error.ExpertRowsAdaptExceedsChunk;
    if (!state.validExpertRows(.{ .expert_rows = g.expert_rows_adapt, .chunk_rows = g.chunk_rows, .piece_rows = g.piece_rows }) or !state.groupedExperts(g.expert_rows_adapt, g.piece_rows)) return error.ExpertRowsAdaptNotGrouped;
}

fn prefixCachePlanOptions(expert_rows: usize) !prefix_cache.PlanOptions {
    const a = try prefix_cache.kvsave.cutAlignment(expert_rows);
    const explicit_step = std.c.getenv("TF_GLM53_PREFIX_CACHE_STEP");
    const step = if (explicit_step != null)
        (try prefixCacheRowsEnv("TF_GLM53_PREFIX_CACHE_STEP", 0))
    else
        (2048 / a + @intFromBool(2048 % a != 0)) * a;
    const span = try prefixCacheRowsEnv("TF_GLM53_PREFIX_CACHE_SPAN", 0);
    if ((explicit_step != null and step % a != 0) or (span != 0 and span % a != 0)) return error.PrefixCacheBadCutGeometry;
    return .{ .expert_rows = expert_rows, .step = step, .span = span };
}

fn prefixCacheRowsEnv(name: [*:0]const u8, default: usize) !usize {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(raw), 10) catch error.PrefixCacheBadCutGeometry;
}

fn prefixCacheBudgetBytes() !u64 {
    return try prefixCacheGiBEnv("TF_GLM53_PREFIX_CACHE_GIB", 16);
}

fn prefixCacheFloorBytes() !u64 {
    const raw = std.c.getenv("TF_GLM53_PREFIX_CACHE_FREE_GIB") orelse return 50_000_000_000;
    const gib = std.fmt.parseInt(u64, std.mem.span(raw), 10) catch return error.PrefixCacheBadBudget;
    return std.math.mul(u64, gib, 1 << 30) catch error.PrefixCacheBadBudget;
}

fn prefixCacheGiBEnv(name: [*:0]const u8, default: u64) !u64 {
    const raw = std.c.getenv(name) orelse return std.math.mul(u64, default, 1 << 30) catch error.PrefixCacheBadBudget;
    const gib = std.fmt.parseInt(u64, std.mem.span(raw), 10) catch return error.PrefixCacheBadBudget;
    return std.math.mul(u64, gib, 1 << 30) catch error.PrefixCacheBadBudget;
}

fn prefixCacheFatbinIdentity() prefix_cache.kvsave.Digest {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    inline for (comptime std.meta.declarations(cuda.kernels)) |kernel_name| {
        if (comptime @TypeOf(@field(cuda.kernels, kernel_name)) == []const u8) {
            const image: []const u8 = @field(cuda.kernels, kernel_name);
            h.update(kernel_name);
            h.update(&.{0});
            h.update(image);
        }
    }
    var out: prefix_cache.kvsave.Digest = undefined;
    h.final(&out);
    return out;
}

fn prefixCacheExecutableIdentity() !prefix_cache.kvsave.Digest {
    const fd = std.c.open("/proc/self/exe", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return error.PrefixCacheExecutableOpen;
    defer _ = std.c.close(fd);
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [1 << 20]u8 = undefined;
    while (true) {
        const n = std.c.read(fd, buf[0..].ptr, buf.len);
        if (n < 0) return error.PrefixCacheExecutableRead;
        if (n == 0) break;
        h.update(buf[0..@intCast(n)]);
    }
    var out: prefix_cache.kvsave.Digest = undefined;
    h.final(&out);
    return out;
}

pub fn parseChunkRows(raw: ?[]const u8) !usize {
    const rows = if (raw) |value| std.fmt.parseInt(usize, value, 10) catch return error.BadChunkRows else state.default_chunk_rows;
    // 2048, 4096 and 8192 only: 1024-row chunks were not byte-equal to 4096 in the GPU parity (2026-10-29).
    if (rows == 2048 or rows == 4096 or rows == 8192) return rows;
    return error.BadChunkRows;
}

fn chunkRows() !usize {
    const raw = std.c.getenv("TF_GLM53_CHUNK_ROWS");
    return parseChunkRows(if (raw) |v| std.mem.span(v) else null);
}

pub fn parseL2Depth(get: anytype) !i32 {
    const raw = get("TF_EXL3X_L2D") orelse return 4;
    const depth = std.fmt.parseInt(i32, raw, 10) catch return error.BadL2Depth;
    if (depth < 0 or depth > 8) return error.BadL2Depth;
    return depth;
}

pub fn l2Depth() !i32 {
    return parseL2Depth(struct {
        fn get(name: []const u8) ?[]const u8 {
            if (!std.mem.eql(u8, name, "TF_EXL3X_L2D")) return null;
            const raw = std.c.getenv("TF_EXL3X_L2D") orelse return null;
            return std.mem.span(raw);
        }
    }.get);
}

/// Admit the requested expert batch rows or refuse before the load. No fallback: above 512 rows the routed-expert
/// collectives run in a different order than at 512/128 (all a group's attention collectives, then its MoE reduces),
/// so two ranks that fell back differently would desynchronise TP2, and a near-equal MemAvailable makes that likely
/// at the edge (review of 4cb3465c). The refusal names the largest size that fits so the operator can rerun.
pub fn admissionChoice(base: state.Geometry, requested: usize, context: usize, weight_classes: state.WeightClasses, available: u64, reserve: state.Reserve, admit: anytype) !struct { rows: usize, admission: state.Admission } {
    var geometry = base;
    geometry.expert_rows = requested;
    if (requested > geometry.chunk_rows) {
        if (!builtin.is_test) std.log.err("expert_rows {d} exceeds chunk_rows {d}; set TF_GLM53_CHUNK_ROWS >= expert_rows on both ranks", .{ requested, geometry.chunk_rows });
        return error.ExpertRowsExceedChunkRows;
    }
    const result = admit(geometry, context, weight_classes, available, reserve) catch |err| {
        if (err != error.MemoryBudget or requested == 128) return err;
        for ([_]usize{ 8192, 4096, 2048, 1536, 1024, 512, 128 }) |rows| {
            if (rows >= requested) continue;
            geometry.expert_rows = rows;
            _ = admit(geometry, context, weight_classes, available, reserve) catch continue;
            if (!builtin.is_test) std.log.err("expert_rows {d} does not fit this rank's memory; {d} fits: set TF_GLM53_EXPERT_ROWS={d} on both ranks", .{ requested, rows, rows });
            return error.ExpertRowsDoNotFit;
        }
        return err;
    };
    return .{ .rows = requested, .admission = result };
}

/// The generic CUDA host's pre-load check (native/cuda_family_memory.weights): under TP2 each rank loads about half of
/// the checkpoint (routed experts and backbone split by TP, embed and lm_head by vocabulary; measured 95.9 of 189.3 GiB
/// per rank), so 51% is a conservative share. The exact per-rank plan is admitted in open() (planRank + admissionChoice).
pub fn weightShare(_: std.mem.Allocator, _: std.Io, _: []const u8, whole: u64) u64 {
    if (std.c.getenv("TF_GLM_WORLD")) |w| {
        if (std.mem.eql(u8, std.mem.span(w), "3")) return whole / 100 * 36;
    }
    return whole / 100 * 51;
}

test "TP2 weight share is just over half the checkpoint" {
    try std.testing.expectEqual(@as(u64, 51), weightShare(std.testing.allocator, undefined, "", 100));
}

pub fn place() !Place {
    const rank_text = std.c.getenv("TF_GLM_RANK") orelse return error.NoRank;
    const world_text = std.c.getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p = Place{
        .rank = std.fmt.parseInt(usize, std.mem.span(rank_text), 10) catch return error.NoRank,
        .world = std.fmt.parseInt(usize, std.mem.span(world_text), 10) catch return error.NoRank,
    };
    if ((p.world != 2 and p.world != 3) or p.rank >= p.world) return error.UnsupportedWorld;
    return p;
}

/// NCCL's opaque id crosses the configured leader host. Port + 1 belongs to tp_watch; this
/// bootstrap uses port + 2 so all ranks initialize the same communicator before model loading.
fn ncclId(io: std.Io, lib: *const cuda.nccl.Library, p: Place, leader: []const u8) !cuda.nccl.UniqueId {
    const text = leader;
    const colon = std.mem.lastIndexOfScalar(u8, text, ':') orelse return error.NoCommId;
    const port = (std.fmt.parseInt(u16, text[colon + 1 ..], 10) catch return error.NoCommId) +% 2;
    const ip = std.Io.net.IpAddress.parse(text[0..colon], port) catch return error.NoCommId;
    if (ip != .ip4) return error.NoCommId;
    const addr: posix.sockaddr.in = .{ .port = std.mem.nativeToBig(u16, port), .addr = std.mem.readInt(u32, &ip.ip4.bytes, .little) };
    const makeSocket = struct {
        fn run() !posix.socket_t {
            const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM | posix.SOCK.CLOEXEC, 0);
            if (posix.errno(rc) != .SUCCESS) return error.SocketFailed;
            return @intCast(rc);
        }
    }.run;
    var id: cuda.nccl.UniqueId = undefined;
    if (p.rank == 0) {
        const server = try makeSocket();
        defer _ = posix.system.close(server);
        const one: c_int = 1;
        try posix.setsockopt(server, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one));
        if (posix.errno(posix.system.bind(server, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) != .SUCCESS or
            posix.errno(posix.system.listen(server, @intCast(p.world - 1))) != .SUCCESS) return error.NcclBootstrapFailed;
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        for (1..p.world) |_| {
            var fds = [_]posix.pollfd{.{ .fd = server, .events = posix.POLL.IN, .revents = 0 }};
            if ((posix.poll(&fds, 300_000) catch 0) == 0) return error.RankNeverCame;
            const rc = posix.system.accept(server, null, null);
            if (posix.errno(rc) != .SUCCESS) return error.NcclBootstrapFailed;
            const fd: posix.socket_t = @intCast(rc);
            defer _ = posix.system.close(fd);
            try writeAll(fd, std.mem.asBytes(&id));
        }
    } else {
        var tries: usize = 0;
        var fd: posix.socket_t = undefined;
        while (true) : (tries += 1) {
            fd = try makeSocket();
            if (posix.errno(posix.system.connect(fd, @ptrCast(&addr), @sizeOf(posix.sockaddr.in))) == .SUCCESS) break;
            _ = posix.system.close(fd);
            if (tries >= 3000) return error.RankNeverCame;
            try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        }
        defer _ = posix.system.close(fd);
        try readAll(fd, std.mem.asBytes(&id));
    }
    return id;
}

fn writeAll(fd: posix.socket_t, bytes: []const u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.write(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => at += @intCast(rc),
            .INTR => {},
            else => return error.NcclBootstrapFailed,
        }
    }
}

fn readAll(fd: posix.socket_t, bytes: []u8) !void {
    var at: usize = 0;
    while (at < bytes.len) {
        const rc = posix.system.read(fd, bytes[at..].ptr, bytes.len - at);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return error.NcclBootstrapFailed;
                at += @intCast(rc);
            },
            .INTR => {},
            else => return error.NcclBootstrapFailed,
        }
    }
}

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    rows: u32 = prompt_rows,
    stream_bytes: usize,
    free_streams: u32 = 1,
    facts: lanes.Model,
    target: *anyopaque,
    proposer_ctx: ?*anyopaque = null,
    proposer_make: ?*const fn (*anyopaque, std.mem.Allocator, []const u32, []const u32) anyerror!lanes.proposer.Proposer = null,
    proposer_free: ?*const fn (*anyopaque, lanes.proposer.Proposer) void = null,
    ablate: bool = false,
};
pub const ZngramEnv = struct { copy_rows: u32, min_match: u32, cap: u32, max_draft: u32, log: bool };
pub fn verifyMaxFromEnv(get: anytype) !usize {
    const value = try envInt(get, "TF_GLM53_VERIFY_MAX", 8, 8, 16);
    if (value != 8 and value != 16) return error.BadVerifyMax;
    return @intCast(value);
}
pub fn parseZngramEnv(get: anytype, verify_max: usize) !ZngramEnv {
    const max_rows: u32 = @intCast(verify_max);
    if (max_rows != 8 and max_rows != 16) return error.BadVerifyMax;
    const cap = try envInt(get, "TF_GLM53_VERIFY_CAP", 12, 2, 64);
    if (verify_max == 16 and cap > 8 and cap != 12 and cap < 16) {
        if (!builtin.is_test) std.log.err("TF_GLM53_VERIFY_CAP must be <=8, 12, or >=16 when TF_GLM53_VERIFY_MAX=16", .{});
        return error.BadZngramSetting;
    }
    return .{ .copy_rows = try envInt(get, "TF_GLM53_COPY_ROWS", 1, 1, max_rows), .min_match = try envInt(get, "TF_GLM53_ZNGRAM_MIN_MATCH", 8, 1, 64), .cap = cap, .max_draft = try envInt(get, "TF_GLM53_ZNGRAM_MAX_DRAFT", 7, 1, max_rows - 1), .log = if (get("TF_GLM53_ZNGRAM_LOG")) |v| std.mem.eql(u8, v, "1") else false };
}
pub fn parseCopyd(value: ?[]const u8, copy_rows: u32, fast_rows: usize) !bool {
    const requested = value orelse return false;
    if (!std.mem.eql(u8, requested, "1")) return false;
    if (copy_rows < 2) {
        if (!builtin.is_test) std.log.err("TF_GLM53_COPYD=1 needs TF_GLM53_COPY_ROWS of 2 or more", .{});
        return error.CopydNeedsCopyRows;
    }
    if (@as(usize, copy_rows) > fast_rows) {
        if (!builtin.is_test) std.log.err("TF_GLM53_COPYD=1 allows copy rows up to {d} (the fast exact decode kernels); got {d}", .{ fast_rows, copy_rows });
        return error.CopydRowsPastFastKernels;
    }
    return true;
}
/// Copy rows may reach 16 when TF_GLM53_DECODE_UNION16=1 with TF_GLM53_VERIFY_MAX=16 (A3 attends up to 16 rows).
pub fn copydSwitch(copy_rows: u32, verify_max: usize) !bool {
    const value = if (std.c.getenv("TF_GLM53_COPYD")) |raw| std.mem.span(raw) else null;
    const wide = forward.decodeUnion16Switch() and verify_max == forward.max_decode_rows;
    return parseCopyd(value, copy_rows, if (wide) forward.max_decode_rows else forward.decode_mla_rows);
}
fn copydRequested() bool {
    const value = std.c.getenv("TF_GLM53_COPYD") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}
fn envInt(get: anytype, name: []const u8, default: u32, min: u32, max: u32) !u32 {
    const raw = get(name) orelse return default;
    const value = std.fmt.parseInt(u32, raw, 10) catch {
        if (!builtin.is_test) std.log.err("invalid {s}; expected an integer from {d} through {d}", .{ name, min, max });
        return error.BadZngramSetting;
    };
    if (value < min or value > max) {
        if (!builtin.is_test) std.log.err("invalid {s}; expected {d} through {d}", .{ name, min, max });
        return error.BadZngramSetting;
    }
    return value;
}
pub fn zngramEnv(verify_max: usize) !ZngramEnv { return parseZngramEnv(struct { fn get(name: []const u8) ?[]const u8 { inline for (.{"TF_GLM53_COPY_ROWS", "TF_GLM53_ZNGRAM_MIN_MATCH", "TF_GLM53_VERIFY_CAP", "TF_GLM53_ZNGRAM_MAX_DRAFT", "TF_GLM53_ZNGRAM_LOG"}) |key| if (std.mem.eql(u8, name, key)) { const p = std.c.getenv(key); return if (p) |v| std.mem.span(v) else null; }; return null; } }.get, verify_max); }

const Engine = engine_mod.Engine;
const Adapter = backend_adapter.Adapter(Engine);

const Owned = struct {
    gpa: std.mem.Allocator,
    graph_keys: []engine_mod.GraphKey = &.{},
    group: cuda.tp_group.Group,
    stream: cuda.Stream,
    wire: follow.Wire,
    lane: cuda.qmmf.Lane,
    experts: cuda.exl3_experts.Experts,
    modules: forward.Kernels.Modules,
    kernels: forward.Kernels,
    model: weights.ModelWeights,
    st: state.State,
    engine: Engine,
    ablation: ablate.Ablation,
    adapter: Adapter,
    side_group: ?forward.SideGroup = null,
    pipe_group: ?forward.SideGroup = null,
    side_stream: ?cuda.Stream = null,
    side_events: [5]cuda.Event = undefined,
    side_event_count: usize = 0,
    pf: forward.Prefetch = undefined,
    host: drafts.Host = undefined,
    /// TF_GLM53_PREFILL_W8A8=1 only: the glm53_w8a8 image (prefill o_proj).
    w8a8: cuda.w8a8.Kernels = undefined,
    zfactory: zngram.Factory,
    /// Saved-prefix agreement input/output words, allocated during setup when disk caching is enabled.
    agree_flag: ?cuda.DeviceBuffer = null,
    prefix_runtime: ?*prefix_cache.Runtime = null,
    prefix_pool: ?prefix_cache.Pool = null,
    prefix_device: ?*prefix_snap.CudaDevice = null,
    prefix_snap: ?*prefix_snap.Snap(engine_mod.Engine) = null,
    prefix_cache_budget_bytes: u64 = 0,
    prefix_cache_floor_bytes: u64 = 0,
    alive_model: bool = false,
    alive_host: bool = false,
    alive_ablation: bool = false,
    alive_state: bool = false,
    alive_group: bool = false,
    alive_wire: bool = false,
    alive_engine: bool = false,
    alive_experts: bool = false,
    alive_lane: bool = false,
    alive_stream: bool = false,
    alive_side_group: bool = false,
    alive_pipe_group: bool = false,
    alive_side_stream: bool = false,
    alive_modules: u8 = 0,
    alive_w8a8: bool = false,

    fn deinit(own: *Owned) void {
        if (own.prefix_snap) |snap| { snap.deinit(); own.gpa.destroy(snap); }
        if (own.prefix_device) |device| { device.deinit(); own.gpa.destroy(device); }
        if (own.prefix_pool) |*pool| pool.deinit();
        if (own.prefix_runtime) |runtime| { runtime.deinit(); own.gpa.destroy(runtime); }
        if (own.alive_stream) own.stream.synchronize() catch {};
        if (own.alive_side_stream) if (own.side_stream) |s| s.synchronize() catch {};
        if (own.alive_host) own.host.deinit();
        if (own.agree_flag) |*flag| flag.free();
        if (own.alive_state) own.st.deinit();
        if (own.alive_ablation) own.ablation.deinit();
        if (own.alive_model) own.model.deinit();
        if (own.alive_engine) own.engine.deinit();
        if (own.alive_experts) own.experts.unload();
        if (own.alive_lane) own.lane.unload();
        if (own.alive_w8a8) own.w8a8.unload();
        inline for (.{ "glm_glue", "glm53_glue", "glm53_mla", "glm53_index", "glm53_head" }, 0..) |name, i| {
            if ((own.alive_modules & (@as(u8, 1) << i)) != 0) @field(own.modules, name).unload();
        }
        if (own.alive_wire) own.wire.deinit();
        if (own.alive_side_group) if (own.side_group) |*side| side.close();
        if (own.alive_pipe_group) if (own.pipe_group) |*pipe| pipe.close();
        for (own.side_events[0..own.side_event_count]) |*event| event.deinit();
        if (own.alive_side_stream) if (own.side_stream) |*s| s.deinit();
        if (own.alive_stream) own.stream.deinit();
        if (own.alive_group) {
            if (own.group.hasRoce()) {
                const counts = own.group.routeCounts();
                std.log.info("RoCE routes: roce {d}, nccl (size) {d}, nccl (other stream) {d}", .{ counts[0], counts[1], counts[2] });
            }
            own.group.close();
        }
        if (own.graph_keys.len != 0) own.gpa.free(own.graph_keys);
        own.gpa.destroy(own);
    }
};

fn agreeOwned(ctx: *anyopaque, words: []u32) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(ctx));
    const d = own.st.workspace.d;
    if (words.len > 128) return error.AgreementTooLarge;
    if (own.agree_flag == null) return error.AgreementBufferUnavailable;
    const flag = own.agree_flag.?.ptr;
    const bytes = std.mem.sliceAsBytes(words);
    try d.check(d.api.cuMemcpyHtoDAsync_v2(flag, bytes.ptr, bytes.len, own.stream.handle), "KV save agree input");
    try own.group.allReduce(own.stream, flag, flag + bytes.len, words.len, .u32, .min);
    try d.check(d.api.cuMemcpyDtoHAsync_v2(bytes.ptr, flag + bytes.len, bytes.len, own.stream.handle), "KV save agree result");
    try own.stream.synchronize();
}

fn prefixLookup(ctx: *anyopaque, prompt: []const u32, after: usize, _: *lanes.Stream) anyerror!?backend_adapter.DiskHit {
    const runtime: *prefix_cache.Runtime = @ptrCast(@alignCast(ctx));
    const hit = runtime.lookup(prompt, after) orelse return null;
    return .{ .at = hit.at, .key = hit.key };
}

fn prefixCuts(ctx: *anyopaque, prompt: []const u32, start: usize, s: *lanes.Stream) anyerror![]const prefix_cache.Cut {
    const runtime: *prefix_cache.Runtime = @ptrCast(@alignCast(ctx));
    return runtime.plan(prompt, start, s.shared_prefixes, s.history_len);
}

fn prefixSaveHeader(ctx: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header {
    const runtime: *prefix_cache.Runtime = @ptrCast(@alignCast(ctx));
    const h = try runtime.saveHeader(at);
    var out: follow.Header = .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at };
    std.mem.writeInt(u64, out.prefix_hash[0..8], h.key, .little);
    std.mem.writeInt(u64, out.prefix_hash[8..16], h.parent_key, .little);
    std.mem.writeInt(u64, out.prefix_hash[16..24], h.base_at, .little);
    std.mem.writeInt(u64, out.prefix_hash[24..32], h.seq, .little);
    return out;
}

fn prefixPending(ctx: *anyopaque) usize { return (@as(*prefix_cache.Runtime, @ptrCast(@alignCast(ctx)))).pending(); }
fn prefixPendingMax(ctx: *anyopaque) usize { return (@as(*prefix_cache.Runtime, @ptrCast(@alignCast(ctx)))).pendingMax(); }
fn prefixWriterIdle(ctx: *anyopaque) bool { return (@as(*prefix_cache.Runtime, @ptrCast(@alignCast(ctx)))).isWriterIdle(); }
fn prefixRingView(ctx: *anyopaque) anyerror!prefix_snap.RingView {
    const view = try drafts.Host.ringView(ctx);
    return .{ .ptr = view.ptr, .len = view.len, .from = view.from };
}

/// Production path follows WP9's bring-up ordering: header plan, fresh UMA admission, weight load,
/// then a second MemAvailable reading immediately before State allocation.
/// MemAvailable after each load step, logged as the drop since the previous step (finds load bytes the admission
/// plan does not count). A failed read only skips that line.
const MemSnap = struct {
    mem_available: i64 = 0,
    mem_free: i64 = 0,
    anon_pages: i64 = 0,
    cached: i64 = 0,
    shmem: i64 = 0,
    mlocked: i64 = 0,
    unevictable: i64 = 0,
    page_tables: i64 = 0,
    sec_page_tables: i64 = 0,
    kernel_stack: i64 = 0,
    vmalloc_used: i64 = 0,
    sunreclaim: i64 = 0,
    has_sec_page_tables: bool = false,
};

fn parseMemSnap(text: []const u8) MemSnap {
    var out: MemSnap = .{};
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse continue;
        const name = line[0..colon];
        var values = std.mem.tokenizeAny(u8, line[colon + 1 ..], " \t");
        const value = std.fmt.parseInt(i64, values.next() orelse continue, 10) catch continue;
        if (std.mem.eql(u8, name, "MemAvailable")) out.mem_available = value else if (std.mem.eql(u8, name, "MemFree")) out.mem_free = value else if (std.mem.eql(u8, name, "AnonPages")) out.anon_pages = value else if (std.mem.eql(u8, name, "Cached")) out.cached = value else if (std.mem.eql(u8, name, "Shmem")) out.shmem = value else if (std.mem.eql(u8, name, "Mlocked")) out.mlocked = value else if (std.mem.eql(u8, name, "Unevictable")) out.unevictable = value else if (std.mem.eql(u8, name, "PageTables")) out.page_tables = value else if (std.mem.eql(u8, name, "SecPageTables")) {
            out.sec_page_tables = value;
            out.has_sec_page_tables = true;
        } else if (std.mem.eql(u8, name, "KernelStack")) out.kernel_stack = value else if (std.mem.eql(u8, name, "VmallocUsed")) out.vmalloc_used = value else if (std.mem.eql(u8, name, "SUnreclaim")) out.sunreclaim = value;
    }
    return out;
}

fn readMemSnap(gpa: std.mem.Allocator, io: std.Io) !MemSnap {
    var file = try std.Io.Dir.cwd().openFile(io, "/proc/meminfo", .{});
    defer file.close(io);
    const text = try uma.readStreamingAlloc(&file, gpa, io, 1 << 20);
    defer gpa.free(text);
    return parseMemSnap(text);
}

test "parseMemSnap" {
    const sample = "MemAvailable: 1001 kB\nMemFree: 1002 kB\nAnonPages: 1003 kB\nCached: 1004 kB\nShmem: 1005 kB\nMlocked: 1006 kB\nUnevictable: 1007 kB\nPageTables: 1008 kB\nSecPageTables:    123456 kB\nKernelStack: 1010 kB\n";
    const got = parseMemSnap(sample);
    try std.testing.expectEqual(@as(i64, 1001), got.mem_available);
    try std.testing.expectEqual(@as(i64, 1002), got.mem_free);
    try std.testing.expectEqual(@as(i64, 1003), got.anon_pages);
    try std.testing.expectEqual(@as(i64, 1004), got.cached);
    try std.testing.expectEqual(@as(i64, 1005), got.shmem);
    try std.testing.expectEqual(@as(i64, 1006), got.mlocked);
    try std.testing.expectEqual(@as(i64, 1007), got.unevictable);
    try std.testing.expectEqual(@as(i64, 1008), got.page_tables);
    try std.testing.expectEqual(@as(i64, 123456), got.sec_page_tables);
    try std.testing.expectEqual(@as(i64, 1010), got.kernel_stack);
    try std.testing.expect(got.has_sec_page_tables);
    const without_sec = parseMemSnap("MemAvailable: 1001 kB\nMemFree: 1002 kB\nAnonPages: 1003 kB\nCached: 1004 kB\nShmem: 1005 kB\nMlocked: 1006 kB\nUnevictable: 1007 kB\nPageTables: 1008 kB\nKernelStack: 1010 kB\nPageTablesX: 5 kB\n");
    try std.testing.expectEqual(@as(i64, 0), without_sec.sec_page_tables);
    try std.testing.expect(!without_sec.has_sec_page_tables);
    try std.testing.expectEqual(@as(i64, 1008), without_sec.page_tables);
    try std.testing.expectEqual(MemSnap{}, parseMemSnap(""));
}

test "parseMemSnap reads VmallocUsed and SUnreclaim" {
    const got = parseMemSnap("MemAvailable: 1 kB\nSUnreclaim:   2048 kB\nSReclaimable: 7 kB\nVmallocUsed:  4096 kB\n");
    try std.testing.expectEqual(@as(i64, 1), got.mem_available);
    try std.testing.expectEqual(@as(i64, 2048), got.sunreclaim);
    try std.testing.expectEqual(@as(i64, 4096), got.vmalloc_used);
    const without = parseMemSnap("MemAvailable: 1 kB\n");
    try std.testing.expectEqual(@as(i64, 0), without.vmalloc_used);
    try std.testing.expectEqual(@as(i64, 0), without.sunreclaim);
}

const TraceCategory = enum { vmm, module, pinned, blas, nccl, drafter, other };

const Tracer = struct {
    enabled: bool = false,
    unavailable_logged: bool = false,
    prev: MemSnap = .{},
    prev_cuda_free: usize = 0,
    driver: *const cuda.Driver,
    gpa: std.mem.Allocator,
    io: std.Io,
    rank: usize,
    sums: [7]i64 = @splat(0),
    module_counts: [5]usize = @splat(0),

    fn init(driver: *const cuda.Driver, gpa: std.mem.Allocator, io: std.Io, rank: usize) Tracer {
        var self: Tracer = .{ .driver = driver, .gpa = gpa, .io = io, .rank = rank };
        const raw = std.c.getenv("TF_GLM53_LOAD_TRACE") orelse return self;
        self.enabled = std.mem.eql(u8, std.mem.span(raw), "1");
        if (!self.enabled) return self;
        self.prev = readMemSnap(gpa, io) catch {
            self.disableUnavailable();
            return self;
        };
        var total: usize = 0;
        if (driver.api.cuMemGetInfo_v2(&self.prev_cuda_free, &total) != cuda.abi.success) self.disableUnavailable();
        return self;
    }

    fn disableUnavailable(self: *Tracer) void {
        if (!self.unavailable_logged) std.log.info("glm53 load trace r{d}: trace unavailable", .{self.rank});
        self.unavailable_logged = true;
        self.enabled = false;
    }

    fn step(self: *Tracer, label: []const u8, category: TraceCategory, requested_bytes: u64) void {
        if (!self.enabled) return;
        const current = readMemSnap(self.gpa, self.io) catch {
            self.disableUnavailable();
            return;
        };
        var cuda_free: usize = 0;
        var total: usize = 0;
        if (self.driver.api.cuMemGetInfo_v2(&cuda_free, &total) != cuda.abi.success) {
            self.disableUnavailable();
            return;
        }
        const d_avail = self.prev.mem_available - current.mem_available;
        const d_free = self.prev.mem_free - current.mem_free;
        const d_anon = self.prev.anon_pages - current.anon_pages;
        const d_cached = self.prev.cached - current.cached;
        const d_shmem = self.prev.shmem - current.shmem;
        const d_mlocked = self.prev.mlocked - current.mlocked;
        const d_unevict = self.prev.unevictable - current.unevictable;
        const d_pt = self.prev.page_tables - current.page_tables;
        const d_secpt = self.prev.sec_page_tables - current.sec_page_tables;
        const d_stack = self.prev.kernel_stack - current.kernel_stack;
        const d_vmalloc = self.prev.vmalloc_used - current.vmalloc_used;
        const d_sunreclaim = self.prev.sunreclaim - current.sunreclaim;
        const d_cuda = @as(i64, @intCast(self.prev_cuda_free)) - @as(i64, @intCast(cuda_free));
        const m = 1024.0;
        self.sums[@intFromEnum(category)] += d_avail;
        if (category == .module) {
            const labels = [_][]const u8{ "module glm_glue", "module glm53_glue", "module glm53_mla", "module glm53_index", "module glm53_head" };
            for (labels, 0..) |module_label, index| if (std.mem.eql(u8, label, module_label)) {
                self.module_counts[index] += 1;
                std.log.info("glm53 module load r{d}: {s} count {d}", .{ self.rank, label, self.module_counts[index] });
            };
        }
        if (current.has_sec_page_tables and self.prev.has_sec_page_tables) {
            std.log.info("glm53 load trace r{d}: {s}: req {d:.2} | dAvail {d:.2} dFree {d:.2} dAnon {d:.2} dCached {d:.2} dShmem {d:.2} dMlocked {d:.2} dUnevict {d:.2} dPageTables {d:.2} dSecPageTables {d:.2} dKStack {d:.2} dVmalloc {d:.2} dSUnreclaim {d:.2} dCudaFree {d:.2} MiB", .{ self.rank, label, @as(f64, @floatFromInt(requested_bytes)) / (m * m), @as(f64, @floatFromInt(d_avail)) / m, @as(f64, @floatFromInt(d_free)) / m, @as(f64, @floatFromInt(d_anon)) / m, @as(f64, @floatFromInt(d_cached)) / m, @as(f64, @floatFromInt(d_shmem)) / m, @as(f64, @floatFromInt(d_mlocked)) / m, @as(f64, @floatFromInt(d_unevict)) / m, @as(f64, @floatFromInt(d_pt)) / m, @as(f64, @floatFromInt(d_secpt)) / m, @as(f64, @floatFromInt(d_stack)) / m, @as(f64, @floatFromInt(d_vmalloc)) / m, @as(f64, @floatFromInt(d_sunreclaim)) / m, @as(f64, @floatFromInt(d_cuda)) / (m * m) });
        } else {
            std.log.info("glm53 load trace r{d}: {s}: req {d:.2} | dAvail {d:.2} dFree {d:.2} dAnon {d:.2} dCached {d:.2} dShmem {d:.2} dMlocked {d:.2} dUnevict {d:.2} dPageTables {d:.2} dSecPageTables n/a dKStack {d:.2} dVmalloc {d:.2} dSUnreclaim {d:.2} dCudaFree {d:.2} MiB", .{ self.rank, label, @as(f64, @floatFromInt(requested_bytes)) / (m * m), @as(f64, @floatFromInt(d_avail)) / m, @as(f64, @floatFromInt(d_free)) / m, @as(f64, @floatFromInt(d_anon)) / m, @as(f64, @floatFromInt(d_cached)) / m, @as(f64, @floatFromInt(d_shmem)) / m, @as(f64, @floatFromInt(d_mlocked)) / m, @as(f64, @floatFromInt(d_unevict)) / m, @as(f64, @floatFromInt(d_pt)) / m, @as(f64, @floatFromInt(d_stack)) / m, @as(f64, @floatFromInt(d_vmalloc)) / m, @as(f64, @floatFromInt(d_sunreclaim)) / m, @as(f64, @floatFromInt(d_cuda)) / (m * m) });
        }
        self.prev = current;
        self.prev_cuda_free = cuda_free;
    }

    fn finish(self: *Tracer, measured_gap_bytes: i64) void {
        if (!self.enabled) return;
        const names = [_][]const u8{ "vmm", "module", "pinned", "blas", "nccl", "drafter", "other" };
        for (names, 0..) |name, i| std.log.info("glm53 load trace r{d}: sum {s} -dAvail {d:.2} MiB", .{ self.rank, name, @as(f64, @floatFromInt(self.sums[i])) / 1024.0 });
        const counted = self.sums[@intFromEnum(TraceCategory.module)] + self.sums[@intFromEnum(TraceCategory.pinned)] + self.sums[@intFromEnum(TraceCategory.blas)] + self.sums[@intFromEnum(TraceCategory.nccl)] + self.sums[@intFromEnum(TraceCategory.drafter)];
        const residual = @divTrunc(measured_gap_bytes, 1024) - counted;
        std.log.info("glm53 load trace r{d}: residual measured gap minus non-weight categories {d:.2} MiB", .{ self.rank, @as(f64, @floatFromInt(residual)) / 1024.0 });
    }

    fn graphMemory(self: *Tracer, device: cuda.abi.Device, budget_bytes: u64, widths: usize) void {
        if (!self.enabled) return;
        const query = self.driver.graph_mem_attribute orelse {
            self.disableUnavailable();
            return;
        };
        var values: [4]usize = undefined;
        for (&values, 0..) |*value, attr| {
            if (query(device, @intCast(attr), value) != cuda.abi.success) {
                self.disableUnavailable();
                return;
            }
        }
        const m = 1024.0 * 1024.0;
        std.log.info("glm53 graph mem r{d}: used {d:.2}/{d:.2} MiB reserved {d:.2}/{d:.2} MiB, budget {d:.2} MiB, widths {d}", .{ self.rank, @as(f64, @floatFromInt(values[0])) / m, @as(f64, @floatFromInt(values[1])) / m, @as(f64, @floatFromInt(values[2])) / m, @as(f64, @floatFromInt(values[3])) / m, @as(f64, @floatFromInt(budget_bytes)) / m, widths });
    }
};

fn traceWeightLoad(ctx: *anyopaque, label: []const u8, category_name: []const u8, requested_bytes: u64) void {
    const tracer: *Tracer = @ptrCast(@alignCast(ctx));
    const category: TraceCategory = if (std.mem.eql(u8, category_name, "vmm")) .vmm else if (std.mem.eql(u8, category_name, "pinned")) .pinned else .other;
    tracer.step(label, category, requested_bytes);
}

const LoadSteps = struct {
    last: u64,
    first: u64,
    last_detail: ?uma.MemDetail = null,
    reads: u64 = 0,
    fn mark(self: *LoadSteps, gpa: std.mem.Allocator, io: std.Io, what: []const u8) void {
        const m = uma.readMeminfo(gpa, io) catch return;
        self.reads += 1;
        const g = 1024.0 * 1024.0 * 1024.0;
        const drop: f64 = (@as(f64, @floatFromInt(self.last)) - @as(f64, @floatFromInt(m.available))) / g;
        const total: f64 = (@as(f64, @floatFromInt(self.first)) - @as(f64, @floatFromInt(m.available))) / g;
        const r = selfRss(gpa, io);
        std.log.info("glm53 load step: {s}: MemAvailable {d:.2} GiB (step {d:.2}, since start {d:.2}); process RssAnon {d:.2} RssFile {d:.2} RssShmem {d:.2} GiB", .{ what, @as(f64, @floatFromInt(m.available)) / g, drop, total, @as(f64, @floatFromInt(r[0])) / g, @as(f64, @floatFromInt(r[1])) / g, @as(f64, @floatFromInt(r[2])) / g });
        self.last = m.available;
        if (uma.readMeminfoDetail(gpa, io)) |detail| {
            if (self.last_detail) |previous| logMeminfoDelta(what, previous, detail);
            self.last_detail = detail;
        } else |_| {}
    }

    fn logMeminfoDelta(what: []const u8, previous: uma.MemDetail, current: uma.MemDetail) void {
        const gib_f = 1024.0 * 1024.0 * 1024.0;
        const indices = [_]usize{ 0, 3, 4, 5, 6, 10, 9, 12, 13, 14, 15 };
        const labels = [_][]const u8{ "dMemFree", "dCached", "dAnon", "dShmem", "dMlocked", "dSUnreclaim", "dSReclaimable", "dPageTables", "dSecPageTables", "dVmallocUsed", "dPercpu" };
        var buf: [1024]u8 = undefined;
        var stream: std.Io.Writer = .fixed(&buf);
        stream.print("glm53 load meminfo: {s}:", .{what}) catch return;
        for (indices, labels) |index, label| {
            const delta = @as(f64, @floatFromInt(previous.fields[index])) - @as(f64, @floatFromInt(current.fields[index]));
            const gib_delta = delta / gib_f;
            stream.print(" {s} {s}{d:.3}", .{ label, if (gib_delta < 0) "" else "+", gib_delta }) catch return;
        }
        stream.print(" GiB", .{}) catch return;
        std.log.info("{s}", .{stream.buffered()});
    }

    /// Compare an admission-table estimate with the MemAvailable drop measured over this load step.
    fn markExpected(self: *LoadSteps, gpa: std.mem.Allocator, io: std.Io, what: []const u8, label: []const u8, expected: u64) ?f64 {
        const before = self.last;
        const reads_before = self.reads;
        self.mark(gpa, io, what);
        if (self.reads == reads_before) return null;
        const gib_bytes = 1024.0 * 1024.0 * 1024.0;
        const measured = (@as(f64, @floatFromInt(before)) - @as(f64, @floatFromInt(self.last))) / gib_bytes;
        const planned = @as(f64, @floatFromInt(expected)) / gib_bytes;
        std.log.info("glm53 load gap: {s} table {d:.2} GiB, measured {d:.2} GiB, gap {d:.2} GiB", .{ label, planned, measured, measured - planned });
        return measured;
    }

    fn afterDigest(self: *LoadSteps, gpa: std.mem.Allocator, io: std.Io) void {
        self.mark(gpa, io, "after digest");
        const mem = uma.readMeminfo(gpa, io) catch return;
        const gib_bytes = 1024.0 * 1024.0 * 1024.0;
        std.log.info("glm53 load step: after digest: MemAvailable {d:.2} GiB MemFree {d:.2} GiB", .{ @as(f64, @floatFromInt(mem.available)) / gib_bytes, @as(f64, @floatFromInt(mem.free)) / gib_bytes });
        self.last = mem.available;
    }

    /// RssAnon, RssFile and RssShmem of this process in bytes from /proc/self/status (zeros when unreadable): host
    /// memory the process still holds, which MemAvailable loses on the unified-memory GB10 like device memory.
    fn selfRss(gpa: std.mem.Allocator, io: std.Io) [3]u64 {
        var out: [3]u64 = @splat(0);
        var file = std.Io.Dir.cwd().openFile(io, "/proc/self/status", .{}) catch return out;
        defer file.close(io);
        const text = uma.readStreamingAlloc(&file, gpa, io, 1 << 16) catch return out;
        defer gpa.free(text);
        const keys = [_][]const u8{ "RssAnon:", "RssFile:", "RssShmem:" };
        var lines = std.mem.splitScalar(u8, text, '\n');
        while (lines.next()) |line| for (keys, 0..) |key, i| {
            if (!std.mem.startsWith(u8, line, key)) continue;
            var it = std.mem.tokenizeAny(u8, line[key.len..], " \tkB");
            const kb = std.fmt.parseInt(u64, it.next() orelse continue, 10) catch continue;
            out[i] = kb * 1024;
        };
        return out;
    }
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, unused: ?[]const u8, o: Options) !Loaded {
    return openInner(gpa, io, ctx, dir, unused, o) catch |err| {
        std.log.err("GLM-5.3 native open failed: {s}", .{@errorName(err)});
        return err;
    };
}

fn openInner(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, _: ?[]const u8, o: Options) !Loaded {
    try validateOptions(o);
    const reuse_grain = try fineReuseGrain();
    if (reuse_grain != 0 and !residentEnabled()) return error.FineReuseNeedsResident;
    const prefix_cache_enabled = prefixCacheSwitch();
    const prefix_cache_dir = if (std.c.getenv("TF_GLM53_PREFIX_CACHE_DIR")) |v| std.mem.span(v) else null;
    if (prefix_cache_enabled) {
        if (!residentEnabled()) return error.PrefixCacheNeedsResident;
        if (o.kv_format != .bf16) return error.PrefixCacheNeedsBf16Kv;
        if (prefix_cache_dir == null or prefix_cache_dir.?.len == 0) return error.PrefixCacheNeedsDir;
        if (!probePrefixDirect(prefix_cache_dir.?)) return error.PrefixCacheNeedsODirect;
        _ = try prefixCachePlanOptions(try expertRows());
        _ = try prefixCacheBudgetBytes();
        _ = try prefixCacheFloorBytes();
    }

    const lowrank_decode = moe.decodeSwitch();
    const row_prune = envSwitch("TF_GLM53_ROW_PRUNE");
    const defer_draft_fill = envSwitch("TF_GLM53_DEFER_DRAFT_FILL");
    const decode_mla_mode = forward.decodeMlaModeSwitch();
    if (forward.megakernelSwitch() and decode_mla_mode != .off) {
        std.log.err("TF_GLM53_MK=1 and TF_GLM53_DECODE_MLA are incompatible", .{});
        return error.IncompatibleGlm53Modes;
    }

    cuda.exl3_experts.l2_depth = try l2Depth();
    const p = try place();
    if (p.world != 2 and p.world != 3) return error.UnsupportedWorld;
    var leader_buf: [64]u8 = undefined;
    const leader = try copyLeader(if (std.c.getenv("NCCL_COMM_ID")) |v| std.mem.span(v) else null, &leader_buf);
    const side_comms = sideCommsWanted(forward.attnPipeSwitch(), forward.prefetchSwitch());
    try releaseCommId();
    std.log.info("glm53 side comms: rank {d} attn_pipe={d} prefetch={d} wanted={d} leader {s} NCCL_COMM_ID released", .{ p.rank, @intFromBool(forward.attnPipeSwitch()), @intFromBool(forward.prefetchSwitch()), @intFromBool(side_comms), leader });
    if (comptime builtin.os.tag == .linux) try cuda.tp_watch.startAt(io, p.rank, p.world, leader);
    errdefer cuda.tp_watch.fail("native open failed");
    var cfg = try weights.readConfig(gpa, io, dir);
    const selected = try manifestName(o);
    var mf = if (selected) |n| try manifest_mod.Manifest.load(gpa, io, dir, n) else null;
    defer if (mf) |*m| m.deinit();
    if (mf) |*m| try checkContext(o.context, m);
    if (cfg.mtp != 0 or cfg.configured_mtp != 0) return error.MtpUnsupported;
    cfg.dcp = if (p.world == 3) 1 else 2;
    cfg.kv = if (o.kv_format == .bf16) .bf16 else .fp8;
    cfg.rank = @intCast(p.rank);
    cfg.ranks = @intCast(p.world);
    const full_layers = cfg.full.count();
    const verify_max = try verifyMaxFromEnv(struct { fn get(name: []const u8) ?[]const u8 {
        if (!std.mem.eql(u8, name, "TF_GLM53_VERIFY_MAX")) return null;
        const raw = std.c.getenv("TF_GLM53_VERIFY_MAX");
        return if (raw) |v| std.mem.span(v) else null;
    } }.get);
    const rows = try expertRows();
    const adapt_rows = try parseExpertRowsAdapt(if (std.c.getenv("TF_GLM53_EXPERT_ROWS_ADAPT")) |v| std.mem.span(v) else null);
    const adapt_maxpos = try parseAdaptMaxPos(if (std.c.getenv("TF_GLM53_EXPERT_ROWS_ADAPT_MAXPOS")) |v| std.mem.span(v) else null);
    const chunk_rows = try chunkRows();
    const index_mma_raw = std.c.getenv("TF_GLM53_INDEX_ROWS");
    const index_mma_rows = try forward.index.parseScoreRows(if (index_mma_raw) |v| std.mem.span(v) else null);

    const prompt_mla = forward.promptMlaSwitch();
    const coll_fuse_requested = collFuseSwitch();
    const coll_fuse = p.world == 2 and coll_fuse_requested and collFusePayloadFits(verify_max, 4 << 20);
    if (coll_fuse_requested and !coll_fuse) std.log.info("TF_GLM53_COLL_FUSE disabled for this decode shape: packed query or attention partial exceeds the 4 MiB RoCE cap", .{});
    const smla_p = forward.smlaPSwitch();
    const shard_qkv_a = forward.shardQkvASwitch();
    // the ROW_PRUNE cache-only last layer projects kv_a without the shard gather (half-width kv rows): refuse the pair
    if (shard_qkv_a and row_prune) return error.ShardQkvAWithRowPruneUnsupported;
    const smla_fp8 = forward.smlaFp8Switch();
    if (smla_fp8 and !smla_p) return error.SmlaFp8WithoutSmlaP;
    if (smla_fp8) std.log.info("TF_GLM53_SMLA_FP8=1: prefill attend tf_glm53_smla_p_fp8 (e4m3 score GEMM, lossy, KLD-gated; KV cache stays bf16)", .{});
    const smla_p3 = forward.smlaP3Switch();
    if (smla_p3 and !smla_p) return error.SmlaP3WithoutSmlaP;
    if (smla_p3 and smla_fp8) return error.SmlaP3WithSmlaFp8;
    if (smla_p3) std.log.info("TF_GLM53_SMLA_P3=1: prefill attend tf_glm53_smla_union + tf_glm53_smla_p3 (bit-identical to SMLA-P; union table in the attention phase)", .{});
    const decode_union = forward.decodeUnionSwitch();
    const decode_union_rows = try forward.decodeUnionRows(decode_union, forward.decodeUnion16Switch(), verify_max);
    if (decode_union and forward.megakernelSwitch()) return error.DecodeUnionWithMegakernel;
    if (decode_union) std.log.info("TF_GLM53_DECODE_UNION=1: decode attend (rows 1..{d}) tf_glm53_attend_verify + tf_glm53_attend_split_merge (byte-identical to attend_mma)", .{decode_union_rows});
    const prefill_w8a8 = forward.prefillW8a8Switch();
    const pre_fuse = forward.preFuseSwitch();
    const prep_fuse = forward.prepFuseSwitch();
    if ((pre_fuse != .off or prep_fuse != .off) and shard_qkv_a) return error.PreFuseWithShardQkvA;
    if (pre_fuse != .off and forward.megakernelSwitch()) return error.PreFuseWithMegakernel;
    std.log.info("TF_GLM53_PRE_FUSE={s} TF_GLM53_PREP_FUSE={s}: grouped q_a | kv_a | indexer wk projections (bit-identical; 0 = off)", .{ preFuseDigestValue(pre_fuse), prepFuseDigestValue(prep_fuse) });
    if (weights.backboneFp4Qa() and (pre_fuse != .off or prep_fuse != .off)) {
        std.log.err("TF_GLM53_BACKBONE_FP4_QA=1 needs TF_GLM53_PRE_FUSE=0 and TF_GLM53_PREP_FUSE=0: the grouped q_a | kv_a | wk launch takes FP8 weights only", .{});
        return error.PreFuseNeedsFp8QA;
    }
    if (weights.backboneFp4Qa() and !weights.backboneFp4()) {
        std.log.err("TF_GLM53_BACKBONE_FP4_QA=1 extends TF_GLM53_BACKBONE_FP4=1: set both", .{});
        return error.BackboneFp4QaNeedsFp4;
    }
    const prefetch_enabled = forward.prefetchSwitch();
    const tbo_enabled = forward.tboSwitch();
    const profile_prefill = forward.profilePrefillSwitch();
    try forward.validateProfilePrefillTbo(profile_prefill, tbo_enabled);
    const piece_pipe = forward.piecePipeSwitch();
    const moe_pipe = forward.moePipeSwitch();
    const ws_trim = forward.wsTrimSwitch();
    const memslim = try state.memslimFromEnv();
    const moe_post = forward.moePostSwitch();
    const comm_prio = forward.commPrioSwitch();
    const index_merge = forward.indexMergeSwitch();
    const index_pipe = forward.indexPipeSwitch();
    const attn_pipe = forward.attnPipeSwitch();
    if (attn_pipe and !piece_pipe and !moe_pipe and !index_pipe) return error.AttnPipeWithoutCommStream;
    if (attn_pipe and tbo_enabled) return error.AttnPipeWithTbo;
    std.log.info("TF_GLM53_ATTN_PIPE={d}; comm-stream gathers on their own NCCL communicator", .{@intFromBool(attn_pipe)});
    if (comm_prio and !(prefetch_enabled or piece_pipe or moe_pipe or tbo_enabled or index_pipe)) return error.CommPrioWithoutCommStream;
    if (moe_pipe and !state.groupedExperts(rows, 512)) {
        std.log.err("TF_GLM53_MOE_PIPE=1 requires grouped prefill experts (expert_rows must exceed piece_rows=512)", .{});
        return error.MoePipeRequiresGroupedExperts;
    }
    if (moe_pipe and tbo_enabled) {
        std.log.err("TF_GLM53_MOE_PIPE=1 cannot be combined with TF_GLM53_TBO=1", .{});
        return error.MoePipeWithTbo;
    }
    if (moe_pipe and row_prune) {
        std.log.err("TF_GLM53_MOE_PIPE=1 cannot be combined with TF_GLM53_ROW_PRUNE=1", .{});
        return error.MoePipeWithRowPrune;
    }
    std.log.info("TF_GLM53_MOE_PIPE={d}; grouped prefill MoE only (incompatible with TBO and ROW_PRUNE)", .{@intFromBool(moe_pipe)});
    const decode_side = forward.decodeSideSwitch();
    const prompt_experts = forward.promptExpertsMode();
    const scratch_overlay = if (std.c.getenv("TF_GLM53_SCRATCH_OVERLAY")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
    const slim_generic = memslim and scratch_overlay and prompt_experts != .off and !ws_trim;
    if (ws_trim and scratch_overlay) return error.WsTrimWithScratchOverlay;
    if (ws_trim and prompt_experts == .off) return error.WsTrimNeedsPromptExperts;
    if (moe_post and (!moe_pipe or prompt_experts == .off or tbo_enabled)) return error.MoePostNeedsPromptMoePipe;
    if (memslim and !slim_generic and p.rank == 0) std.log.info("TF_GLM53_MEMSLIM=1: generic expert scratch kept (needs TF_GLM53_SCRATCH_OVERLAY=1 and prompt experts); graph policy still applies", .{});
    const prefill_bf16_requested = try state.prefillGemmFromEnv() == .bf16;
    if (prefill_bf16_requested) std.log.info("TF_GLM53_PREFILL_GEMM=bf16 is inactive in the native server (no prefill GEMMs)", .{});
    const prefill_bf16 = false;
    const long_from = try state.longTierFrom(envSlice("TF_GLM53_LONG_CTX_ROWS"), o.context);
    const graph_budget = try state.graphBudget(envSlice("TF_GLM53_GRAPH_BUDGET_MIB"));
    var graph_rows_buf: [16]u32 = undefined;
    const graph_rows_n = try state.parseGraphRows(envSlice("TF_GLM53_GRAPH_ROWS"), verify_max, &graph_rows_buf);
    const all_graph_shapes = graph_rows_n * state.bucketCount(o.context);
    var graph_shape_buf: [512]state.GraphShape = undefined;
    const budget_graph_n = if (graph_budget) |budget| try state.graphPlan(verify_max, o.context, budget, try state.graphMiBFromEnv(), graph_rows_buf[0..graph_rows_n], &graph_shape_buf) else 0;
    const live_rows = if (memslim) try state.graphLiveRows(graph_rows_n, graph_budget, try state.graphMiBFromEnv()) else 0;
    const live_graph_n = if (memslim) try state.graphLivePlan(o.context, live_rows, graph_rows_buf[0..graph_rows_n], &graph_shape_buf) else 0;
    const graph_reserve_bytes = try state.graphsReserveFromEnv(verify_max, o.context);
    const drafter_share = drafts.shareScratchEnabled();
    const pe_flags = forward.promptExpertsFlags(prompt_experts);
    var geometry: state.Geometry = .{ .world = p.world, .full_index_layers = full_layers, .cache_format = if (o.kv_format == .bf16) .bf16 else .fp8, .vocab_local = if (p.world == 3) (if (p.rank == 0) 51712 else 51584) else cfg.vocab / 2, .expert_rows = rows, .index_mma_rows = index_mma_rows, .chunk_rows = chunk_rows, .decode_rows = verify_max, .head_rows = verify_max, .candidate_k = try state.sampleKFromEnv(), .graphs_bytes = graph_reserve_bytes, .drafter_share_bytes = 0, .prefetch = prefetch_enabled, .tbo = tbo_enabled, .piece_pipe = piece_pipe, .moe_pipe = moe_pipe, .ws_trim = ws_trim, .moe_post = moe_post, .comm_prio = comm_prio, .index_merge = index_merge, .index_pipe = index_pipe, .prompt_mla = prompt_mla, .smla_p = smla_p, .smla_fp8 = smla_fp8, .smla_p3 = smla_p3, .shard_qkv_a = shard_qkv_a, .prefill_bf16 = prefill_bf16, .decode_side = decode_side, .decode_mla = decode_mla_mode != .off, .decode_mla_v2 = decode_mla_mode == .v2 or decode_mla_mode == .v3, .decode_mla_v3 = decode_mla_mode == .v3, .prompt_experts = pe_flags.on, .prompt_experts_wide = pe_flags.wide, .prompt_experts_pmx = pe_flags.pmx, .scratch_overlay = scratch_overlay, .slim_generic = slim_generic, .coll_fuse = coll_fuse, .expert_rows_adapt = adapt_rows, .expert_rows_adapt_maxpos = adapt_maxpos, .prompt_experts_p5a = pe_flags.p5a };
    std.log.info("glm53 head: sampled candidate capacity k={d} per rank (TF_GLM53_SAMPLE_K; default 256)", .{geometry.candidate_k});
    const side_rows = try state.sideRowsFromEnv(geometry.piece_rows, o.context);
    if (side_rows != 0 and !residentEnabled()) return error.SideRowsNeedResident;
    if (side_rows != 0 and geometry.cache_format != .bf16) return error.SideRowsNeedBf16Cache;
    if (p.world == 3) {
        // TP=3 runs DCP=1: every rank keeps every position, so one rank's indexer scores the whole window, and the
        // park, prefix-cache and ablation paths (which address slot p / 2 or sum two partials) do not apply.
        const cap = state.dcp1MaxContext(geometry);
        if (o.context > cap) {
            std.log.err("TP=3 (DCP=1): the indexer scores at most {d} keys per rank, so --context must be <= {d} (got {d})", .{ cap, cap, o.context });
            return error.Tp3ContextPastIndexLimit;
        }
        if (side_rows != 0) return error.Tp3SideRowsNeedDcp2;
        if (prefixCacheSwitch()) return error.Tp3PrefixCacheNeedsDcp2;
    }
    geometry.park_rows = side_rows;
    if (long_from != 0 and side_rows != 0) geometry.park_tail_slot = long_from / 2;
    if (p.rank == 0 and side_rows != 0) std.log.info("glm53 side rows: {d} (park {d:.2} GiB per rank)", .{ side_rows, state.gib(state.parkBytes(geometry.layers, geometry.full_index_layers, side_rows)) });
    try state.validateMoePipeGeometry(geometry);
    std.log.info("TF_GLM53_PIECE_PIPE={d}; applies to grouped prefill only (non-grouped prefill and ROW_PRUNE keep the old schedule)", .{@intFromBool(piece_pipe)});
    geometry.prefill_w8a8 = prefill_w8a8;
    geometry.pre_fuse = pre_fuse;
    geometry.prep_fuse = prep_fuse;
    var geometry_without_overlay = geometry;
    geometry_without_overlay.scratch_overlay = false;
    std.log.info("TF_GLM53_SCRATCH_OVERLAY={s}: workspace total {d} bytes (without overlay {d} bytes)", .{ if (scratch_overlay) "on" else "off", state.workspaceLayout(geometry, o.context).bytes, state.workspaceLayout(geometry_without_overlay, o.context).bytes });

    const moe_balance = try weights.moeBalanceFromEnv();
    if (moe_balance != .off) {
        if (p.world != 3) {
            std.log.err("TF_GLM53_MOE_BALANCE={s} splits the routed experts over three ranks; this run has {d}", .{ @tagName(moe_balance), p.world });
            return error.MoeBalanceNeedsThreeRanks;
        }
        if (forward.megakernelSwitch()) {
            std.log.err("TF_GLM53_MOE_BALANCE={s}: the megakernel's expert tasks assume one span a layer; set TF_GLM53_MK=0", .{@tagName(moe_balance)});
            return error.MoeBalanceWithMegakernel;
        }
        std.log.info("TF_GLM53_MOE_BALANCE={s}: routed experts load with a span table ({s})", .{ @tagName(moe_balance), if (moe_balance == .expert) "six-block share rotated by expert and layer" else "one span a layer, per-expert kernels" });
    }
    if (weights.headFp8()) {
        if (forward.megakernelSwitch()) {
            std.log.err("TF_GLM53_HEAD_FP8=1: the megakernel reads the bf16 head; set TF_GLM53_MK=0", .{});
            return error.HeadFp8WithMegakernel;
        }
        std.log.info("TF_GLM53_HEAD_FP8=1: the decode head reads lm_head as block-FP8 (64x64 tile scales, quantized at load; lossy); the bf16 rows stay for the drafter's Q4 head", .{});
    }
    if (absorbRowsWanted()) std.log.info("TF_GLM53_ABSORB_ROWS=1: decode absorb v3 feeds each decoded kv_b weight to all of a thread's rows (bit-identical)", .{});
    if (glm.dflash_load.split3Switch()) {
        if (p.world != 3) {
            std.log.err("TF_GLM53_DRAFT_SPLIT3=1 splits the drafter's MLP over three ranks; this run has {d} (two ranks always split it)", .{p.world});
            return error.DraftSplit3NeedsThreeRanks;
        }
        std.log.info("TF_GLM53_DRAFT_SPLIT3=1: the drafter's MLP splits over three ranks (a third of gate/up rows and of down's K each; fp32 partials gathered, summed by df_sum3)", .{});
    }
    const selection: weights.Select = .{ .manifest = if (mf) |*m| m else null, .shard_qkv_a = shard_qkv_a, .moe_balance = moe_balance };
    var plan = try weights.planRank(gpa, io, dir, cfg, p.rank, p.world, selection);
    defer plan.deinit(gpa);
    var drafter_geometry: ?drafts.Geometry = null;
    if (o.drafter) |dd| {
        const g = try drafts.checkConfig(gpa, io, dd);
        const dm = try drafts.memoryPlan(g, 4, p.world, 1);
        if (drafter_share) {
            plan.weights.drafter = dm.deviceBytesWithSharedScratch() + dm.hostBytes();
            geometry.drafter_share_bytes = dm.shared_scratch;
        } else {
            plan.weights.drafter = dm.deviceBytes() + dm.hostBytes();
        }
        drafter_geometry = g;
    }
    const before = try uma.readMeminfo(gpa, io);
    var step: LoadSteps = .{ .last = before.available, .first = before.available, .last_detail = uma.readMeminfoDetail(gpa, io) catch null };
    var tracer = Tracer.init(ctx.d, gpa, io, p.rank);
    const reserve = try state.reserveFromEnv(before.total);
    const graph_room_bytes = try std.math.mul(u64, try state.graphMiBFromEnv(), 1 << 20);
    try uma.configureGraphRoom(reserve.guard_floor, reserve.serve_margin / 2, graph_room_bytes);
    const choice = try admissionChoice(geometry, rows, o.context, plan.weights, before.available, reserve, state.admitRank);
    geometry.expert_rows = choice.rows;
    try validateExpertRowsAdapt(geometry);
    const admission = choice.admission;
    if (moe_pipe and !state.groupedExperts(geometry.expert_rows, geometry.piece_rows)) {
        std.log.err("TF_GLM53_MOE_PIPE=1 requires grouped prefill experts after admission (expert_rows must exceed piece_rows)", .{});
        return error.MoePipeRequiresGroupedExperts;
    }
    try state.validateMoePipeGeometry(geometry);

    var own = try gpa.create(Owned);
    own.gpa = gpa;
    own.graph_keys = &.{};
    own.alive_modules = 0;
    own.alive_model = false;
    own.alive_ablation = false;
    own.alive_state = false;
    own.alive_group = false;
    own.alive_wire = false;
    own.alive_engine = false;
    own.alive_experts = false;
    own.alive_lane = false;
    own.alive_stream = false;
    own.alive_side_group = false;
    own.alive_pipe_group = false;
    own.alive_side_stream = false;
    own.alive_host = false;
    own.alive_w8a8 = false;
    own.side_event_count = 0;
    own.side_group = null;
    own.pipe_group = null;
    own.side_stream = null;
    own.agree_flag = null;
    own.prefix_runtime = null;
    own.prefix_pool = null;
    own.prefix_device = null;
    own.prefix_snap = null;
    own.prefix_cache_budget_bytes = 0;
    own.prefix_cache_floor_bytes = 0;
    errdefer own.deinit();

    if (memslim) {
        own.graph_keys = try gpa.alloc(engine_mod.GraphKey, live_graph_n);
        for (graph_shape_buf[0..live_graph_n], 0..) |shape, index_value| own.graph_keys[index_value] = .{ .rows = shape.rows, .chunk_bucket = shape.bucket };
        std.log.info("glm53 graph budget (TF_GLM53_MEMSLIM=1): {d} row widths x 1 live bucket ({d} MiB reserved); rows {any}; a bucket change swaps that width's graph", .{ live_rows, @as(u64, @intCast(live_rows)) * (try state.graphMiBFromEnv()), graph_rows_buf[0..live_rows] });
    } else if (graph_budget != null) {
        own.graph_keys = try gpa.alloc(engine_mod.GraphKey, budget_graph_n);
        for (graph_shape_buf[0..budget_graph_n], 0..) |shape, index_value| own.graph_keys[index_value] = .{ .rows = shape.rows, .chunk_bucket = shape.bucket };
        std.log.info("glm53 graph budget: {d} of {d} decode shapes graphed ({d} MiB of {d}); rows {any}; other shapes eager", .{ budget_graph_n, all_graph_shapes, budget_graph_n * (try state.graphMiBFromEnv()), all_graph_shapes * (try state.graphMiBFromEnv()), graph_rows_buf[0..graph_rows_n] });
    }

    var lib = try cuda.nccl.Library.open();
    var lib_open = true;
    errdefer if (lib_open) lib.close();
    const id = try ncclId(io, &lib, p, leader);
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    tracer.step("NCCL communicator", .nccl, 0);
    own.group = .{ .lib = lib, .comm = comm, .rank = p.rank, .world = p.world };
    own.alive_group = true;
    lib_open = false;
    step.mark(gpa, io, "NCCL communicator");
    own.stream = try cuda.Stream.init(ctx.d, false);
    own.alive_stream = true;
    try own.group.openRoce(gpa, io, ctx.d, own.stream, 4 << 20, geometry.coll_fuse);
    if (own.group.hasRoce()) std.log.info("RoCE gathers bound to the main stream; other streams and setup handshakes use NCCL", .{});
    step.mark(gpa, io, "stream + RoCE shards");
    tracer.step("RoCE shards", .nccl, 4 << 20);
    const local_switches = PrefillCommSwitches{ .prefetch = @intFromBool(geometry.prefetch), .tbo = @intFromBool(geometry.tbo), .coll_fuse = @intFromBool(geometry.coll_fuse), .index_pipe = @intFromBool(geometry.index_pipe), .attn_pipe = @intFromBool(attn_pipe) };
    var peer_switches: [8]PrefillCommSwitches = undefined;
    try own.group.gather(PrefillCommSwitches, ctx.d, own.stream, local_switches, peer_switches[0..p.world]);
    for (peer_switches[1..p.world]) |peer| {
        if (!agreesPrefillCommSwitches(peer_switches[0], peer)) return error.PrefillCommunicationSettingsDisagree;
    }
    if (geometry.index_pipe and !geometry.piece_pipe and !geometry.moe_pipe) return error.IndexPipeRequiresCommStream;
    if (geometry.prefetch) {
        // Both ranks make and exchange the side NCCL unique id in this same startup position, before weight loading.
        own.side_group = try forward.SideGroup.open(&own.group, ctx.d, own.stream);
        own.alive_side_group = true;
        own.side_stream = if (geometry.comm_prio) try cuda.Stream.initPriorityBlocking(ctx.d, .highest, false) else try cuda.Stream.init(ctx.d, false);
        own.alive_side_stream = true;
        for (&own.side_events) |*event| {
            event.* = try cuda.Event.init(ctx.d, false);
            own.side_event_count += 1;
        }
        own.pf = .{ .side = own.side_stream.?, .coll = own.side_group.?.collective(), .ready = own.side_events[0], .fetched = .{ own.side_events[1], own.side_events[2] }, .attended = .{ own.side_events[3], own.side_events[4] } };
        step.mark(gpa, io, "prefetch communicator + stream + events");
        tracer.step("prefetch communicator", .nccl, 0);
    }
    if (attn_pipe) {
        own.pipe_group = try forward.SideGroup.open(&own.group, ctx.d, own.stream);
        own.alive_pipe_group = true;
        std.log.info("glm53 attention pipe communicator: open, distinct from the main communicator", .{});
        step.mark(gpa, io, "attention pipe communicator");
        tracer.step("attention pipe communicator", .nccl, 0);
    }
    own.wire = try follow.Wire.init(gpa, ctx.d, &own.group.lib, own.group.comm, own.stream, @max(o.context, follow.max_decode_rows * ((cfg.vocab + 31) / 32)));
    own.alive_wire = true;
    step.mark(gpa, io, "follower wire");
    own.lane = try cuda.qmmf.Lane.load(ctx.d, (try ctx.capability()) / 10);
    if (geometry.pre_fuse != .off or geometry.prep_fuse != .off) try own.lane.loadGroup();
    own.alive_lane = true;
    own.experts = try cuda.exl3_experts.Experts.load(ctx.d);
    own.alive_experts = true;
    if (prompt_experts == .pmx or prompt_experts == .p5a) own.experts.pmx = try cuda.exl3_experts.pmx.Pmx.load(ctx.d);
    if (prompt_experts == .p5a) own.experts.p5a = try cuda.exl3_experts.p5a.P5a.load(ctx.d);
    if (prompt_experts != .off) std.log.info("TF_GLM53_PROMPT_EXPERTS dispatch: mode={s} wide={} pmx={} p5a={}", .{ @tagName(prompt_experts), geometry.prompt_experts_wide, geometry.prompt_experts_pmx, geometry.prompt_experts_p5a });
    if ((prompt_experts == .wide or prompt_experts == .pmx or prompt_experts == .p5a) and own.experts.prompt.wide == null) return error.PromptExpertsWideKernelsMissing;
    step.mark(gpa, io, "qmmf lanes + EXL3 expert kernels");
    own.modules = .{ .glm_glue = undefined, .glm53_glue = undefined, .glm53_mla = undefined, .glm53_index = undefined, .glm53_head = undefined };
    const module_labels = [_][]const u8{ "module glm_glue", "module glm53_glue", "module glm53_mla", "module glm53_index", "module glm53_head" };
    inline for (.{ "glm_glue", "glm53_glue", "glm53_mla", "glm53_index", "glm53_head" }, 0..) |name, i| {
        @field(own.modules, name) = try cuda.Module.load(ctx.d, @field(cuda.kernels, name));
        own.alive_modules |= @as(u8, 1) << i;
        tracer.step(module_labels[i], .module, 0);
    }
    own.kernels = try forward.Kernels.load(own.modules, &own.lane, &own.experts, geometry.index_mma_rows);
    if (geometry.smla_p3) try own.kernels.mla.loadSmlaP3(own.modules.glm53_mla);
    if (decode_union) try own.kernels.mla.loadAttendVerify(own.modules.glm53_mla);
    if (geometry.prefill_w8a8) {
        own.w8a8 = try cuda.w8a8.Kernels.load(ctx.d, cuda.kernels.glm53_w8a8);
        own.alive_w8a8 = true;
        own.kernels.w8a8 = own.w8a8.fns;
        std.log.info("TF_GLM53_PREFILL_W8A8=1: prefill o_proj runs W8A8 (e4m3 rows, fp32 scale per row and 128-K block); lossy, KLD-gated", .{});
    }
    step.mark(gpa, io, "GLM modules + kernels");
    const ablate_path = o.ablate_file orelse if (std.c.getenv("TF_GLM53_ABLATE_FILE")) |v| std.mem.span(v) else null;
    if (ablate_path) |path| {
        if (p.world == 3) return error.Tp3AblationNeedsDcp2;
        own.ablation = try ablate.Ablation.load(gpa, io, ctx.d, path, cfg.layers);
        own.alive_ablation = true;
        own.kernels.ablate = try own.modules.glm53_glue.function("tf_glm53_ablate_6144");
        step.mark(gpa, io, "ablation weights");
    }
    const weights_vmm = weights.weightsVmmFromLookup(weights.processEnvLookup);
    own.model = try weights.ModelWeights.loadConfig(dir, cfg, p.rank, p.world, .{ .allocator = gpa, .io = io, .driver = ctx.d, .device_ordinal = ctx.device, .admission = &admission, .weights_vmm = weights_vmm, .trace = .{ .ctx = &tracer, .call = traceWeightLoad } }, selection);
    own.alive_model = true;
    tracer.step("weight model load aggregate", .vmm, plan.weights.total() - (plan.weights.drafter orelse 0));
    if (prompt_experts == .wide) {
        var has_wide_shape_fallback = false;
        for (own.model.layers) |layer| {
            if (layer.ffn == .moe) {
                const routed = layer.ffn.moe.routed.layer();
                if (cuda.exl3_experts.prompt.eligibleTiles(routed) and routed.width > routed.dims) has_wide_shape_fallback = true;
            }
        }
        if (has_wide_shape_fallback) std.log.warn("TF_GLM53_PROMPT_EXPERTS=2: layers with width > dims use the =1 prompt kernels", .{});
    }
    const weight_table = plan.weights.total() - (plan.weights.drafter orelse 0);
    const measured_weights = step.markExpected(gpa, io, "model weights", "weights", weight_table);
    const gib_bytes = 1024.0 * 1024.0 * 1024.0;
    if (own.model.vmmStats()) |stats| {
        std.log.info("glm53 weights: VMM arenas on, {d} arenas, {d:.2} GiB mapped, granularity {d}", .{ own.model.allocationCount(), @as(f64, @floatFromInt(stats.mapped_bytes)) / gib_bytes, stats.granularity });
    } else {
        std.log.info("glm53 weights: VMM arenas off", .{});
    }
    const arena_bytes = own.model.arenaAllocationBytes();
    const table_gib = @as(f64, @floatFromInt(weight_table)) / gib_bytes;
    const arena_gib = @as(f64, @floatFromInt(arena_bytes)) / gib_bytes;
    if (measured_weights) |measured_gib| {
        std.log.info("glm53 load gap breakdown: table {d:.2} GiB, arena bytes {d:.2} GiB, rounding {d:.3} GiB, unattributed {d:.2} GiB, classes {d}", .{ table_gib, arena_gib, arena_gib - table_gib, measured_gib - arena_gib, own.model.allocationCount() });
        tracer.finish(@intFromFloat((measured_gib - arena_gib) * gib_bytes));
    } else {
        std.log.info("glm53 load gap breakdown: table {d:.2} GiB, arena bytes {d:.2} GiB, rounding {d:.3} GiB, unattributed n/a, classes {d}", .{ table_gib, arena_gib, arena_gib - table_gib, own.model.allocationCount() });
        tracer.finish(0);
    }
    if (drafter_geometry != null) {
        try own.host.init(gpa, ctx, own.stream, &own.group, own.modules.glm_glue, own.modules.glm53_glue, own.model.embed, own.model.head, .{ .lo = own.model.vocab.lo, .hi = own.model.vocab.hi }, p.rank);
        step.mark(gpa, io, "drafter init (modules, draft head, scratch)");
        own.alive_host = true;
        try own.host.load(io, o.drafter.?, 4, drafter_share);
        tracer.step("drafter load aggregate", .drafter, plan.weights.drafter orelse 0);
    }
    if (own.alive_host) _ = step.markExpected(gpa, io, "drafter load (weights, scratch, ring, codebooks)", "drafter", plan.weights.drafter orelse 0);
    // Admission sized the phase union from the plan's drafter scratch: refuse here if the loaded drafter needs more
    // (they drifted once, 'act'), instead of failing later in bindSharedWorkspace.
    if (own.alive_host and drafter_share) {
        const real = own.host.dr.?.shared_scratch_bytes;
        if (real > geometry.drafter_share_bytes) {
            std.log.err("TF_GLM53_DRAFTER_SHARE=1: the drafter's shared scratch is {d} B, the admission plan {d} B", .{ real, geometry.drafter_share_bytes });
            return error.DrafterSharePlanDrift;
        }
    }
    var prefix_reader_workers: usize = 4;
    if (prefix_cache_enabled) {
        const requested_workers = try prefixCacheRowsEnv("TF_GLM53_PREFIX_CACHE_THREADS", 4);
        prefix_reader_workers = prefix_cache.clampReaderWorkers(requested_workers);
        if (requested_workers > 4) std.log.warn("TF_GLM53_PREFIX_CACHE_THREADS={d} clamped to 4 (10 pinned slots reserve two for writing)", .{requested_workers});
        own.prefix_pool = try prefix_cache.Pool.init(ctx.d, ctx, prefix_reader_workers);
        step.mark(gpa, io, "prefix cache staging");
        own.prefix_cache_budget_bytes = try prefixCacheBudgetBytes();
        own.prefix_cache_floor_bytes = try prefixCacheFloorBytes();
    }
    const after = try uma.readMeminfo(gpa, io);
    if (adapt_rows != 0 and (prompt_experts == .off or geometry.tbo)) return error.ExpertRowsAdaptNeedsPromptExperts;
    var state_vmm: ?cuda.arena.Vmm = null;
    var adapt_limit_position: usize = 0;
    if (adapt_rows != 0) {
        var probe = try cuda.arena.Vmm.open(ctx.d, ctx.device, 2 << 20);
        const gran: u64 = @intCast(probe.granularity);
        probe.close();
        const ap = try state.planAdapt(gpa, geometry, o.context, gran);
        defer gpa.free(ap.tails);
        adapt_limit_position = ap.limit_position;
        state_vmm = try cuda.arena.Vmm.open(ctx.d, ctx.device, @intCast(ap.kv_bytes + ap.pool));
        std.log.info("expert rows adapt: {d} below position {d} (KV tails {d:.2} GiB)", .{ adapt_rows, ap.limit_position, state.gib(ap.pool) });
    } else {
        std.log.info("glm53 expert rows adapt: off", .{});
    }
    const adapt_local = ExpertRowsAdaptAgreement{ .rows = @intCast(adapt_rows), .pad = 0, .limit = adapt_limit_position };
    var adapt_peer: [3]ExpertRowsAdaptAgreement = undefined;
    try own.group.gather(ExpertRowsAdaptAgreement, ctx.d, own.stream, adapt_local, adapt_peer[0..p.world]);
    for (adapt_peer[1..p.world]) |peer| {
        if (!agreesExpertRowsAdapt(adapt_peer[0], peer)) return error.ExpertRowsAdaptDisagree;
    }
    var vmm_owned = state_vmm != null;
    defer if (vmm_owned) state_vmm.?.close();
    vmm_owned = false; // initWith takes ownership on both success and failure
    own.st = try state.State.initWith(ctx.d, ctx.device, state_vmm, geometry, o.context, p.rank, p.world, &admission, after.available);
    own.alive_state = true;
    if (own.group.hasRoce()) if (own.st.adapt) |ad| {
        if (!state.adaptRoceAligned(ad.ws)) return error.ExpertRowsAdaptRoceMisaligned;
        std.log.info("glm53 expert rows adapt with RoCE: adapt workspace 16-byte aligned (RoCE registers only its pinned region)", .{});
    };
    if (own.alive_host and drafter_share) try own.host.dr.?.bindSharedWorkspace(own.st.ws.phase, own.st.ws.phase_bytes);
    step.mark(gpa, io, "State (KV, indexer keys, workspace)");
    own.engine = .{ .gpa = gpa, .io = io, .world = p.world, .f = .{ .gpa = gpa, .k = &own.kernels, .w = &own.model, .st = &own.st, .stream = own.stream, .coll = forward.Collective.group(&own.group), .prompt_experts_p5a = geometry.prompt_experts_p5a, .decode_union = decode_union, .decode_union_rows = decode_union_rows, .pre_fuse = geometry.pre_fuse, .prep_fuse = geometry.prep_fuse, .ws_trim = geometry.ws_trim, .moe_post = geometry.moe_post, .comm_prio = geometry.comm_prio, .pipe_coll = if (own.alive_pipe_group) own.pipe_group.?.collective() else null, .rank = p.rank, .chunk_rows = geometry.chunk_rows, .decode_side = geometry.decode_side, .prompt_experts = geometry.prompt_experts, .prompt_experts_wide = geometry.prompt_experts_wide, .prompt_experts_pmx = geometry.prompt_experts_pmx, .prompt_mla = geometry.prompt_mla, .smla_p = geometry.smla_p, .smla_fp8 = geometry.smla_fp8, .smla_p3 = geometry.smla_p3, .shard_qkv_a = geometry.shard_qkv_a, .prefill_w8a8 = geometry.prefill_w8a8, .prefetch = if (geometry.prefetch) &own.pf else null, .prefetch_enabled = geometry.prefetch, .tbo_enabled = geometry.tbo, .piece_pipe = geometry.piece_pipe, .moe_pipe = geometry.moe_pipe, .index_merge = geometry.index_merge, .index_pipe = geometry.index_pipe, .lowrank_decode = lowrank_decode, .coll_fuse = geometry.coll_fuse, .index_mma_rows = geometry.index_mma_rows, .row_prune = row_prune }, .defer_draft_fill = defer_draft_fill, .mem = try uma.MemTrace.fromEnv(p.rank) };
    if (memslim) {
        own.engine.graph_allow = own.graph_keys;
        own.engine.decode_graph_max = @intCast(live_rows);
        own.engine.graph_live = true;
        if (live_rows == 0) {
            own.engine.auto_graphs = false;
            std.log.info("glm53 graph budget: 0 live rows, every decode step eager", .{});
        }
    } else if (graph_budget != null) {
        own.engine.graph_allow = own.graph_keys;
        own.engine.decode_graph_max = @intCast(budget_graph_n);
        if (budget_graph_n == 0) {
            own.engine.auto_graphs = false;
            std.log.info("glm53 graph budget: 0 MiB, every decode step eager", .{});
        }
    }
    if (own.group.hasRoce()) own.engine.download.group = &own.group;
    own.engine.f.prof_prefill = profile_prefill;
    own.engine.f.prof_decode_sub = profile_prefill and forward.profileDecodeSubSwitch();
    if (profile_prefill) std.log.info("TF_GLM53_PROFILE_PREFILL=1: rank 0 logs each prefill's stage times (eager, synchronized: slower)", .{});
    try own.engine.f.validateSetup();
    own.engine.mem.floor = try state.floorFromEnv();
    if (prefix_cache_enabled) {
        own.agree_flag = try cuda.DeviceBuffer.alloc(ctx.d, 1024);
        own.engine.agreement = .{ .ctx = own, .call_fn = agreeOwned };
        try own.engine.history.ensureTotalCapacity(gpa, o.context);
    }
    if (own.alive_host) {
        const dr = own.host.dr.?;
        const g = drafter_geometry.?;
        const tap = dr.k.tap; // df_tap_glm lives in the drafter's glm_dflash image (Flash: Drafter.taps)
        own.engine.drafter = .{
            .ctx = &own.host,
            .window = g.window,
            .taps_prefill = .{ .f = tap, .out = dr.taps_prefill, .hidden = 6144, .layers = dr.geometry.taps, .cap_rows = g.window },
            .taps_step = .{ .f = tap, .out = dr.taps_step, .hidden = 6144, .layers = dr.geometry.taps, .cap_rows = glm.dflash.STEP_ROWS },
            .restart_fn = drafts.Host.restartHook,
            .prompt_fn = drafts.Host.promptHook,
            .verify_fn = drafts.Host.verifyHook,
            .draft_fn = drafts.Host.draftHook,
            .state_fn = drafts.Host.stateHook,
            .adopt_fn = drafts.Host.adoptHook,
            .ring_adopt_fn = drafts.Host.ringAdopt,
        };
    }
    const resident = residentEnabled();
    own.engine.resident = resident;
    own.engine.reuse_grain = reuse_grain;
    if (side_rows != 0) own.engine.park = .{ .rows = side_rows, .adapt_limit = adapt_limit_position, .long_from = long_from };
    if (own.alive_ablation) own.engine.abl = &own.ablation;
    own.alive_engine = true;
    step.mark(gpa, io, "engine (incl. graph prewarm/capture if any)");
    tracer.step("graphs capture+instantiate", .other, graph_reserve_bytes);
    tracer.graphMemory(ctx.device, graph_reserve_bytes, if (memslim) live_rows else graph_rows_n);
    try own.engine.mem.point(gpa, io, own.stream, "state", .{});
    try own.engine.reserveSteadyState();
    try own.engine.mem.point(gpa, io, own.stream, "steady", .{});
    own.adapter = Adapter.init(gpa, &own.engine);
    own.adapter.resident = resident;
    if (reuse_grain == 0) {
        std.log.info("glm53 fine reuse: off", .{});
    } else {
        std.log.info("glm53 fine reuse: grain {d}", .{reuse_grain});
    }
    own.adapter.drafting = own.alive_host;
    own.adapter.draft_chances = own.alive_host and draftChancesSwitch();
    std.log.info("glm53 draft chances: {s}", .{if (own.adapter.draft_chances) "on" else "off"});
    const lane_chances = own.adapter.draft_chances and laneChancesSwitch();
    own.adapter.lane_chances = lane_chances;
    std.log.info("glm53 lane chances: {s}", .{if (own.adapter.lane_chances) "on" else "off"});
    own.adapter.live_costs = own.adapter.draft_chances and liveCostsSwitch();
    std.log.info("glm53 live costs: {s}", .{if (own.adapter.live_costs) "on" else "off"});
    if (liveCostsSwitch() and !own.adapter.draft_chances) std.log.warn("TF_GLM53_LIVE_COSTS=1 needs TF_GLM53_DRAFT_CHANCES=1; live costs stay off", .{});
    own.adapter.link_ctx = &own.wire;
    own.adapter.send_fn = sendFrame;
    const ze = try zngramEnv(verify_max);
    own.adapter.copy_rows = ze.copy_rows;
    own.adapter.copy_match = ze.min_match;
    own.adapter.batch_cap = ze.cap;
    own.adapter.verify_max = verify_max;
    const copyd = try copydSwitch(ze.copy_rows, verify_max);
    own.adapter.copyd = copyd;
    std.log.info("glm53 copyd: {s} (copy rows {d})", .{ if (copyd) "on" else "off", ze.copy_rows });
    // C1 recalibration (backend recal.zig): TF_GLM53_DRAFT_RECAL (file), TF_GLM53_DRAFT_SWEEP (measurement), TF_GLM53_ROUND_TRACE
    // (rank 0 JSONL). All default off. A refusal fails the load on every rank (tp_watch errdefer above).
    const recal_mod = backend_adapter.recal;
    const draft_dspark = if (drafter_geometry) |g| g.kind == .dspark else false;
    var recal_sha: ?[32]u8 = null;
    if (std.c.getenv("TF_GLM53_DRAFT_RECAL")) |recal_raw| {
        const recal_path = std.mem.span(recal_raw);
        if (recal_path.len != 0) {
            if (!own.adapter.draft_chances) return error.DraftRecalNeedsDraftChances;
            const recal_bytes = try std.Io.Dir.cwd().readFileAlloc(io, recal_path, gpa, .limited(64 << 10));
            defer gpa.free(recal_bytes);
            const recal_file = try recal_mod.parse(gpa, recal_bytes);
            if (recal_file.conf != null) {
                if (!draft_dspark) return error.DraftRecalConfNeedsDspark;
                if ((try drafts.confTempSwitch()) != 1.0) return error.DraftRecalWithConfTemp;
            }
            own.adapter.recal_costs = recal_file.costs;
            own.adapter.recal_conf = recal_file.conf;
            recal_sha = recal_file.sha;
        }
    }
    if (p.rank == 0) std.log.info("glm53 draft recal: {s}", .{if (recal_sha != null) "on" else "off"});
    const sweep_on = try recal_mod.parseSweep(if (std.c.getenv("TF_GLM53_DRAFT_SWEEP")) |v| std.mem.span(v) else null);
    if (sweep_on and (!own.adapter.draft_chances or own.adapter.live_costs or own.adapter.lane_chances)) return error.DraftSweepNeedsChancesOnly;
    own.adapter.sweep = sweep_on;
    if (sweep_on and p.rank == 0) std.log.warn("glm53 draft sweep: on (measurement only; rounds verify 0..7 held drafts in turn)", .{});
    if (std.c.getenv("TF_GLM53_ROUND_TRACE")) |trace_raw| {
        if (trace_raw[0] != 0) {
            if (!own.adapter.draft_chances) return error.RoundTraceNeedsDraftChances;
            if (p.rank == 0) {
                if (!recal_mod.writable(trace_raw)) return error.RoundTraceUnwritable;
                own.adapter.trace_path = trace_raw;
                own.adapter.trace_info = .{ .draft_chances = own.adapter.draft_chances, .live_costs = own.adapter.live_costs, .lane_chances = own.adapter.lane_chances, .copyd = copyd, .copy_rows = ze.copy_rows, .verify_max = verify_max, .conf_temp = try drafts.confTempSwitch(), .dspark = draft_dspark, .recal = recal_sha, .sweep = sweep_on };
                std.log.info("glm53 round trace: {s}", .{std.mem.span(trace_raw)});
            }
        }
    }
    own.zfactory = .{ .params = .{ .min_match = ze.min_match, .max_draft = if (copyd) ze.copy_rows - 1 else ze.max_draft, .adaptive = verify_max == 16 and !copyd }, .log = ze.log };
    const prewarm_path = o.prewarm orelse if (std.c.getenv("TF_GLM53_PREWARM")) |v| std.mem.span(v) else null;
    if (prewarm_path != null and !resident) return error.PrewarmNeedsResident;

    const digest = recal_mod.digest(try settingsDigest(gpa, io, dir, o, cfg, plan, choice.rows, lowrank_decode, if (own.alive_ablation) own.ablation.digest else null, geometry.prompt_mla, geometry.smla_p, geometry.smla_fp8, geometry.smla_p3, geometry.shard_qkv_a, geometry.prefill_w8a8, prompt_experts, prefetch_enabled, tbo_enabled, geometry.coll_fuse, piece_pipe, moe_pipe, index_merge, index_pipe, drafter_share, graph_reserve_bytes, row_prune, defer_draft_fill, decode_mla_mode, geometry.index_mma_rows, verify_max, attn_pipe, ws_trim, moe_post, comm_prio, geometry.pre_fuse, geometry.prep_fuse, decode_union, lane_chances, adapt_rows, adapt_limit_position, reuse_grain, side_rows, memslim), recal_sha, sweep_on);
    try cuda.tp_watch.ready(io, digest);
    if (prefix_cache_enabled) {
        const a = try prefix_cache.kvsave.cutAlignment(try expertRows());
        const identity = prefix_cache.kvsave.identityOf(digest, prefixCacheFatbinIdentity(), try prefixCacheExecutableIdentity(), @intCast(p.rank), @intCast(p.world), @intCast(a));
        const hex = std.fmt.bytesToHex(identity, .lower);
        const cache_path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}/rank{d}", .{ prefix_cache_dir.?, &hex, p.rank }, 0);
        defer gpa.free(cache_path);
        try std.Io.Dir.cwd().createDirPath(io, cache_path);
        var tmp_removed: usize = 0;
        if (std.Io.Dir.cwd().openDir(io, cache_path, .{ .iterate = true })) |opened_rank_dir| {
            var rank_dir = opened_rank_dir;
            defer rank_dir.close(io);
            var entries = rank_dir.iterate();
            while (entries.next(io) catch null) |entry| {
                if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".tmp")) {
                    rank_dir.deleteFile(io, entry.name) catch continue;
                    tmp_removed += 1;
                }
            }
        } else |_| {}
        std.log.info("prefix cache: removed {d} temporary files", .{tmp_removed});
        var identities_removed: usize = 0;
        if (std.Io.Dir.cwd().openDir(io, prefix_cache_dir.?, .{ .iterate = true })) |opened_root_dir| {
            var root_dir = opened_root_dir;
            defer root_dir.close(io);
            var entries = root_dir.iterate();
            while (entries.next(io) catch null) |entry| {
                if (entry.kind != .directory or entry.name.len != 64 or std.mem.eql(u8, entry.name, &hex)) continue;
                var valid = true;
                for (entry.name) |ch| if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) { valid = false; break; };
                if (!valid) continue;
                const old = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ prefix_cache_dir.?, entry.name });
                defer gpa.free(old);
                std.Io.Dir.cwd().deleteTree(io, old) catch continue;
                identities_removed += 1;
            }
        } else |_| {}
        std.log.info("prefix cache: removed {d} old identity directories", .{identities_removed});
        const runtime = try gpa.create(prefix_cache.Runtime);
        errdefer gpa.destroy(runtime);
        runtime.* = try prefix_cache.Runtime.init(gpa, prefix_cache_dir.?, identity, @intCast(p.rank), try prefixCachePlanOptions(try expertRows()), own.prefix_cache_budget_bytes, own.prefix_cache_floor_bytes, own.prefix_pool);
        own.prefix_pool = null;
        own.prefix_runtime = runtime;
        try runtime.index.reconcile(io, cache_path, identity, @intCast(p.rank), try expertRows(), null);
        const local = runtime.index.startupDigest();
        var words: [8]u32 = undefined;
        @memcpy(std.mem.sliceAsBytes(&words), &local);
        try agreeOwned(own, &words);
        var agreed: prefix_cache.kvsave.Digest = undefined;
        @memcpy(&agreed, std.mem.sliceAsBytes(&words));
        var peer_mismatch = !std.mem.eql(u8, &local, &agreed);
        for (&words) |*word| word.* = ~word.*;
        try agreeOwned(own, &words);
        for (&words) |*word| word.* = ~word.*;
        @memcpy(&agreed, std.mem.sliceAsBytes(&words));
        peer_mismatch = peer_mismatch or !std.mem.eql(u8, &local, &agreed);
        if (peer_mismatch) {
            try std.Io.Dir.cwd().deleteTree(io, cache_path);
            try std.Io.Dir.cwd().createDirPath(io, cache_path);
            runtime.index.clear();
            runtime.used_bytes = 0;
            std.log.warn("prefix cache: rank indexes differ at startup; both ranks wiped {s}", .{cache_path});
        }
        const budget_mib_raw = runtime.budget_bytes >> 20;
        const floor_mib_raw = runtime.floor_bytes >> 20;
        const free_mib_raw = (prefix_cache.kvsave.freeBytes(cache_path) orelse 0) >> 20;
        const max_mib = @as(u64, std.math.maxInt(u32));
        var budget_words = [3]u32{
            @intCast(@min(budget_mib_raw, max_mib)),
            ~@as(u32, @intCast(@min(floor_mib_raw, max_mib))),
            @intCast(@min(free_mib_raw, max_mib)),
        };
        try agreeOwned(own, &budget_words);
        runtime.budget_bytes = @as(u64, budget_words[0]) << 20;
        runtime.floor_bytes = @as(u64, ~budget_words[1]) << 20;
        const agreed_free = @as(u64, budget_words[2]) << 20;
        runtime.recount();
        _ = runtime.room(0, agreed_free);
        std.log.info("prefix cache on: dir={s} A={d} nodes={d} used={d} MiB budget={d} MiB floor={d} MiB threads={d} staging=40 MiB", .{ cache_path, a, runtime.index.nodes.items.len, runtime.used_bytes >> 20, runtime.budget_bytes >> 20, runtime.floor_bytes >> 20, prefix_reader_workers });
        const dev = try gpa.create(prefix_snap.CudaDevice);
        errdefer gpa.destroy(dev);
        dev.* = try prefix_snap.CudaDevice.init(ctx.d, ctx, own.stream);
        own.prefix_device = dev;
        const snap = try gpa.create(prefix_snap.Snap(engine_mod.Engine));
        errdefer gpa.destroy(snap);
        const ring: ?prefix_snap.RingHooks = if (own.alive_host) .{ .ctx = &own.host, .view_fn = prefixRingView } else null;
        var slots: [10][]align(prefix_cache.kvsave.alignment) u8 = undefined;
        for (&slots, 0..) |*slot, idx| slot.* = runtime.pool.?.bytes(idx);
        try snap.init(gpa, io, runtime, &own.engine, dev.device(), ring, try prefix_snap.layoutFromState(&own.st, state.latent_row_bytes, state.rope_row_bytes, state.index_row_bytes), slots, o.context, prefix_reader_workers);
        own.prefix_snap = snap;
        own.engine.snap_ops = snap.ops();
        if (p.rank == 0) own.adapter.disk = .{
            .ctx = runtime,
            .lookup_fn = prefixLookup,
            .cuts_fn = prefixCuts,
            .save_header_fn = prefixSaveHeader,
            .pending_fn = prefixPending,
            .pending_max_fn = prefixPendingMax,
            .writer_idle_fn = prefixWriterIdle,
            .max_pending = prefix_snap.max_jobs - 18,
        };
    }
    if (p.rank != 0) {
        follow.Link.run(&own.engine, &own.wire) catch |err| {
            cuda.tp_watch.fail(@errorName(err));
            std.c._exit(1);
        };
        cuda.tp_watch.ended();
        own.deinit();
        std.process.exit(0);
    }
    // Rank 0 pre-warms after the rank barrier: rank 1 must already be in its follower loop to receive the frames.
    if (prewarm_path) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
        defer gpa.free(bytes);
        const prompts = try prewarm.parse(gpa, bytes);
        defer prewarm.deinit(gpa, prompts);
        _ = try prewarm.run(&own.adapter, prompts, {}, cfg.vocab);
    }
    const zon = ze.copy_rows > 1;
    return .{ .backend = own.adapter.backend(), .ctx = own, .deinit = close, .stream_bytes = @intCast(admission.state_bytes + (if (drafter_geometry) |g| drafts.ringBytes(g) else 0)), .facts = own.adapter.facts(), .target = &own.engine, .proposer_ctx = if (zon) &own.zfactory else null, .proposer_make = if (zon) zngram.Factory.make else null, .proposer_free = if (zon) zngram.Factory.free else null, .ablate = own.alive_ablation };
}

fn residentEnabled() bool {
    const value = std.c.getenv("TF_GLM53_RESIDENT") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

pub fn parseFineReuse(fine: ?[]const u8, grain: ?[]const u8) !usize {
    const fine_value = fine orelse return 0;
    if (std.mem.eql(u8, fine_value, "0")) return 0;
    if (!std.mem.eql(u8, fine_value, "1")) return error.BadFineReuse;
    const selected = if (grain) |value| std.fmt.parseInt(usize, value, 10) catch return error.BadReuseGrain else 1;
    if (selected == 1 or selected == 2 or selected == 4 or selected == 8 or selected == 16 or selected == 32 or selected == 64 or selected == 128 or selected == 256) return selected;
    return error.BadReuseGrain;
}
/// Cache is opt-in. Reading the environment happens once in openInner; the off path adds no digest bytes.
fn parseSwitch(value: ?[*:0]const u8) bool {
    const raw = value orelse return false;
    return raw[0] == '1' and raw[1] == 0;
}

/// TF_GLM53_PREFIX_CACHE=1 (off by default; needs TF_GLM53_RESIDENT=1 and bf16 KV) | TF_GLM53_PREFIX_CACHE_DIR (per-node NVMe dir, required) | TF_GLM53_PREFIX_CACHE_GIB (16; min across ranks) | TF_GLM53_PREFIX_CACHE_FREE_GIB (47 = 50 GB; max across ranks) | TF_GLM53_PREFIX_CACHE_STEP (multiple of A) | TF_GLM53_PREFIX_CACHE_SPAN (multiple of A; 0 = 16384 rounded up to A) | TF_GLM53_PREFIX_CACHE_THREADS (4; 1..4)
pub fn prefixCacheSwitch() bool {
    return parseSwitch(std.c.getenv("TF_GLM53_PREFIX_CACHE"));
}

fn probePrefixDirect(dir: []const u8) bool {
    if (comptime builtin.os.tag != .linux) return false;
    const path = std.fmt.allocPrintSentinel(std.heap.page_allocator, "{s}/.probe", .{dir}, 0) catch return false;
    defer std.heap.page_allocator.free(path);
    var flags: std.c.O = .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true };
    if (comptime @hasField(std.c.O, "DIRECT")) flags.DIRECT = true else return false;
    const fd = std.c.open(path.ptr, flags, @as(std.c.mode_t, 0o600));
    if (fd < 0) return false;
    defer {
        _ = std.c.close(fd);
        _ = std.c.unlink(path.ptr);
    }
    const aligned = std.heap.page_allocator.alignedAlloc(u8, .fromByteUnits(4096), 4096) catch return false;
    defer std.heap.page_allocator.free(aligned);
    @memset(aligned, 0x5a);
    const wrote = std.c.write(fd, aligned.ptr, aligned.len);
    if (wrote < 0 or @as(usize, @intCast(wrote)) != aligned.len) return false;
    if (std.c.lseek(fd, 0, std.c.SEEK.SET) < 0) return false;
    @memset(aligned, 0);
    const read = std.c.read(fd, aligned.ptr, aligned.len);
    if (read < 0 or @as(usize, @intCast(read)) != aligned.len) return false;
    for (aligned) |v| if (v != 0x5a) return false;
    return true;
}


pub fn fineReuseGrain() !usize {
    const fine_raw = std.c.getenv("TF_GLM53_FINE_REUSE");
    const grain_raw = std.c.getenv("TF_GLM53_REUSE_GRAIN");
    return parseFineReuse(if (fine_raw) |value| std.mem.span(value) else null, if (grain_raw) |value| std.mem.span(value) else null);
}

fn hashFineReuseSwitch(hash: *std.crypto.hash.sha2.Sha256, grain: usize) void {
    if (grain == 0) return;
    var text: [32]u8 = undefined;
    const value = std.fmt.bufPrint(&text, "TF_GLM53_FINE_REUSE={d}", .{grain}) catch unreachable;
    hash.update(value);
}

test "fine reuse parser defaults off and accepts only supported divisors" {
    try std.testing.expectEqual(@as(usize, 0), try parseFineReuse(null, null));
    try std.testing.expectEqual(@as(usize, 0), try parseFineReuse("0", "3"));
    try std.testing.expectEqual(@as(usize, 1), try parseFineReuse("1", null));
    try std.testing.expectEqual(@as(usize, 64), try parseFineReuse("1", "64"));
    try std.testing.expectError(error.BadReuseGrain, parseFineReuse("1", "512"));
    try std.testing.expectError(error.BadReuseGrain, parseFineReuse("1", "3"));
    try std.testing.expectError(error.BadReuseGrain, parseFineReuse("1", "0"));
    const digest = struct {
        fn make(grain: usize) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashFineReuseSwitch(&hash, grain);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var empty = std.crypto.hash.sha2.Sha256.init(.{});
    var empty_digest: [32]u8 = undefined;
    empty.final(&empty_digest);
    try std.testing.expectEqualSlices(u8, &empty_digest, &digest(0));
}

/// TF_GLM53_DRAFT_CHANCES=1 passes selector probabilities into allocation-based draft window sizing.
pub fn draftChancesSwitch() bool {
    const value = std.c.getenv("TF_GLM53_DRAFT_CHANCES") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

/// TF_GLM53_LANE_CHANCES=1 enables grant-following holds; it is gated by draft chances.
pub fn parseLaneChancesSwitch(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}
pub fn laneChancesSwitch() bool {
    const value = std.c.getenv("TF_GLM53_LANE_CHANCES") orelse return false;
    return parseLaneChancesSwitch(std.mem.span(value));
}
pub fn parseLiveCostsSwitch(value: ?[]const u8) bool {
    return if (value) |v| std.mem.eql(u8, v, "1") else false;
}
pub fn liveCostsSwitch() bool {
    const value = std.c.getenv("TF_GLM53_LIVE_COSTS") orelse return false;
    return parseLiveCostsSwitch(std.mem.span(value));
}


const PrefillCommSwitches = extern struct { prefetch: u8, tbo: u8, coll_fuse: u8, index_pipe: u8, attn_pipe: u8 };
const ExpertRowsAdaptAgreement = extern struct { rows: u32, pad: u32, limit: u64 };

fn agreesExpertRowsAdapt(a: ExpertRowsAdaptAgreement, b: ExpertRowsAdaptAgreement) bool {
    return a.rows == b.rows and a.limit == b.limit;
}

fn agreesPrefillCommSwitches(rank_zero: PrefillCommSwitches, rank_one: PrefillCommSwitches) bool {
    return rank_zero.prefetch == rank_one.prefetch and rank_zero.tbo == rank_one.tbo and rank_zero.coll_fuse == rank_one.coll_fuse and rank_zero.index_pipe == rank_one.index_pipe and rank_zero.attn_pipe == rank_one.attn_pipe;
}

fn hashTransportSwitches(hash: *std.crypto.hash.sha2.Sha256, prefetch_enabled: bool, tbo_enabled: bool, coll_fuse: bool) void {
    hash.update("TF_GLM53_PREFETCH");
    hash.update(std.mem.asBytes(&prefetch_enabled));
    hash.update("TF_GLM53_TBO");
    hash.update(std.mem.asBytes(&tbo_enabled));
    hash.update("TF_GLM53_COLL_FUSE");
    hash.update(std.mem.asBytes(&coll_fuse));
}

fn hashIndexSwitches(hash: *std.crypto.hash.sha2.Sha256, index_merge: u8, index_pipe: bool) void {
    if (index_merge != 0) hash.update("TF_GLM53_INDEX_MERGE=2");
    if (index_pipe) hash.update("TF_GLM53_INDEX_PIPE=1");
}

fn collFuseSwitch() bool {
    const value = std.c.getenv("TF_GLM53_COLL_FUSE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

fn collFusePayloadFits(decode_rows: usize, cap: usize) bool {
    const query_bytes = decode_rows * state.local_heads * state.kv_row_bytes;
    const partial_bytes = decode_rows * state.decode_partial_row_bytes;
    return query_bytes <= cap and partial_bytes <= cap;
}

fn hashSmlaPSwitch(hash: *std.crypto.hash.sha2.Sha256, smla_p: bool) void {
    if (smla_p) hash.update("TF_GLM53_SMLA_P=1");
}

fn hashShardQkvASwitch(hash: *std.crypto.hash.sha2.Sha256, shard_qkv_a: bool) void {
    if (shard_qkv_a) hash.update("TF_GLM53_SHARD_QKV_A=1");
}

fn hashMoePipeSwitch(hash: *std.crypto.hash.sha2.Sha256, moe_pipe: bool) void {
    if (moe_pipe) hash.update("TF_GLM53_MOE_PIPE=1");
}

fn hashAttnPipeSwitch(hash: *std.crypto.hash.sha2.Sha256, on: bool) void {
    if (on) hash.update("TF_GLM53_ATTN_PIPE=1");
}

fn hashWorkspaceTrimSwitch(hash: *std.crypto.hash.sha2.Sha256, ws_trim: bool) void {
    if (ws_trim) hash.update("TF_GLM53_WS_TRIM=1");
}

fn hashMemslimSwitch(hash: *std.crypto.hash.sha2.Sha256, on: bool) void {
    if (on) hash.update("TF_GLM53_MEMSLIM=1");
}

fn hashMoePostSwitch(hash: *std.crypto.hash.sha2.Sha256, moe_post: bool) void {
    if (moe_post) hash.update("TF_GLM53_MOE_POST=1");
}

fn hashCommPrioSwitch(hash: *std.crypto.hash.sha2.Sha256, comm_prio: bool) void {
    if (comm_prio) hash.update("TF_GLM53_COMM_PRIO=1");
}

/// Off hashes nothing, so the default digest equals the one before SMLA-FP8 existed.
/// TF_GLM53_ABSORB_ROWS=1 (read again here for the log and the settings digest; mla.zig picks the kernel).
fn absorbRowsWanted() bool {
    const v = std.c.getenv("TF_GLM53_ABSORB_ROWS") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn hashSmlaFp8Switch(hash: *std.crypto.hash.sha2.Sha256, smla_fp8: bool) void {
    if (smla_fp8) hash.update("TF_GLM53_SMLA_FP8=1");
}

/// Off hashes nothing, so the default digest equals the one before SMLA-P3 existed. P3 is bit-identical to SMLA-P; it
/// is hashed anyway so a prefix cache records which attend kernel wrote it.
fn hashSmlaP3Switch(hash: *std.crypto.hash.sha2.Sha256, smla_p3: bool) void {
    if (smla_p3) hash.update("TF_GLM53_SMLA_P3=1");
}

/// Off hashes nothing, so default digests stay what they were before the switch existed; on adds its value.
fn hashPrefillW8a8Switch(hash: *std.crypto.hash.sha2.Sha256, prefill_w8a8: bool) void {
    if (prefill_w8a8) hash.update("TF_GLM53_PREFILL_W8A8=1");
}

/// Off hashes nothing, so default digests stay what they were. Both switches are bit-identical; they are hashed so a
/// saved prefix records which launch path wrote it.
fn hashPreFuseSwitches(hash: *std.crypto.hash.sha2.Sha256, pre_fuse: forward.PreFuse, prep_fuse: forward.PrepFuse) void {
    if (pre_fuse != .off) { hash.update("TF_GLM53_PRE_FUSE="); hash.update(preFuseDigestValue(pre_fuse)); }
    if (prep_fuse != .off) { hash.update("TF_GLM53_PREP_FUSE="); hash.update(prepFuseDigestValue(prep_fuse)); }
}
fn preFuseDigestValue(v: forward.PreFuse) []const u8 { return switch (v) { .off => "0", .on => "1", .prefetch => "2" }; }
fn prepFuseDigestValue(v: forward.PrepFuse) []const u8 { return switch (v) { .off => "0", .on => "1", .split => "2" }; }
/// Off hashes nothing, preserving the default digest before A3 existed.
fn hashDecodeUnionSwitch(hash: *std.crypto.hash.sha2.Sha256, decode_union: bool) void {
    if (decode_union) hash.update("TF_GLM53_DECODE_UNION=1");
}

fn hashDecodeUnion16Switch(hash: *std.crypto.hash.sha2.Sha256, union16: bool) void {
    if (union16) hash.update("TF_GLM53_DECODE_UNION16=1");
}

fn hashPromptExperts(hash: *std.crypto.hash.sha2.Sha256, mode: forward.PromptExperts) void {
    switch (mode) {
        .off => {},
        .tiles => hash.update("TF_GLM53_PROMPT_EXPERTS=1"),
        .wide => hash.update("TF_GLM53_PROMPT_EXPERTS=2"),
        .pmx => hash.update("TF_GLM53_PROMPT_EXPERTS=3"),
        .p5a => hash.update("TF_GLM53_PROMPT_EXPERTS=10"),
    }
}

fn envSwitch(name: [*:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    return value[0] == '1' and value[1] == 0;
}

fn decodeMlaDigestValue(mode: forward.DecodeMlaMode) []const u8 {
    return switch (mode) { .off => "0", .v1 => "1", .v2 => "2", .v3 => "3" };
}

test "decode MLA digest tags preserve modes 0 through 2 and distinguish v3" {
    try std.testing.expectEqualStrings("0", decodeMlaDigestValue(.off));
    try std.testing.expectEqualStrings("1", decodeMlaDigestValue(.v1));
    try std.testing.expectEqualStrings("2", decodeMlaDigestValue(.v2));
    try std.testing.expectEqualStrings("3", decodeMlaDigestValue(.v3));
    try std.testing.expect(!std.mem.eql(u8, decodeMlaDigestValue(.v2), decodeMlaDigestValue(.v3)));
}

test "pre-fuse digest is unchanged off and distinguishes the four enabled values" {
    const digest = struct {
        fn make(pre: forward.PreFuse, prep: forward.PrepFuse) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashPreFuseSwitches(&hash, pre, prep);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var empty = std.crypto.hash.sha2.Sha256.init(.{});
    var empty_digest: [32]u8 = undefined;
    empty.final(&empty_digest);
    try std.testing.expectEqualSlices(u8, &empty_digest, &digest(.off, .off));
    const values = [_][32]u8{ digest(.on, .off), digest(.prefetch, .off), digest(.off, .on), digest(.off, .split) };
    for (values, 0..) |a, i| for (values[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, &a, &b));
}

fn settingsDigest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, o: Options, cfg: weights.Config, plan: weights.Plan, effective_expert_rows: usize, lowrank_decode: bool, ablate_digest: ?[32]u8, prompt_mla: forward.PromptMlaMode, smla_p: bool, smla_fp8: bool, smla_p3: bool, shard_qkv_a: bool, prefill_w8a8: bool, prompt_experts: forward.PromptExperts, prefetch_enabled: bool, tbo_enabled: bool, coll_fuse: bool, piece_pipe: bool, moe_pipe: bool, index_merge: u8, index_pipe: bool, drafter_share: bool, graph_reserve_bytes: u64, row_prune: bool, defer_draft_fill: bool, decode_mla: forward.DecodeMlaMode, index_mma_rows: usize, verify_max: usize, attn_pipe: bool, ws_trim: bool, moe_post: bool, comm_prio: bool, pre_fuse: forward.PreFuse, prep_fuse: forward.PrepFuse, decode_union: bool, lane_chances: bool, adapt_rows: usize, adapt_limit: usize, fine_reuse_grain: usize, side_rows: usize, memslim: bool) ![32]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 24));
    defer gpa.free(raw);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(raw);
    if (try manifestName(o)) |name| {
        var m = try manifest_mod.Manifest.load(gpa, io, dir, name);
        defer m.deinit();
        hash.update(&m.digest);
        hash.update(std.mem.asBytes(&m.max_context));
    }
    const ze = try zngramEnv(verify_max);
    if (ze.copy_rows > 1) {
        hash.update(std.mem.asBytes(&ze.copy_rows));
        hash.update(std.mem.asBytes(&ze.min_match));
        hash.update(std.mem.asBytes(&ze.cap));
        hash.update(std.mem.asBytes(&ze.max_draft));
        hash.update(std.mem.asBytes(&ze.log));
    }
    hash.update(std.mem.asBytes(&o.context));
    hash.update("TF_GLM53_INDEX_ROWS=");
    hash.update(std.mem.asBytes(&index_mma_rows));
    // Off (8) hashes nothing, so the default digest is unchanged; 16 adds its value.
    if (verify_max != 8) {
        hash.update("TF_GLM53_VERIFY_MAX=");
        hash.update(std.mem.asBytes(&verify_max));
    }
    hash.update(if (lowrank_decode) "TF_GLM53_LOWRANK_DECODE=1" else "TF_GLM53_LOWRANK_DECODE=0");
    hashTransportSwitches(&hash, prefetch_enabled, tbo_enabled, coll_fuse);
    if (piece_pipe) hash.update("TF_GLM53_PIECE_PIPE=1");
    hashMoePipeSwitch(&hash, moe_pipe);
    hashWorkspaceTrimSwitch(&hash, ws_trim);
    hashMemslimSwitch(&hash, memslim);
    hashMoePostSwitch(&hash, moe_post);
    hashCommPrioSwitch(&hash, comm_prio);
    hashIndexSwitches(&hash, index_merge, index_pipe);
    hashAttnPipeSwitch(&hash, attn_pipe);
    const rows = try expertRows();
    if (rows != 128) hash.update(std.mem.asBytes(&rows));
    if (effective_expert_rows != 128) hash.update(std.mem.asBytes(&effective_expert_rows));
    if (adapt_rows != 0) {
        hash.update("TF_GLM53_EXPERT_ROWS_ADAPT=");
        hash.update(std.mem.asBytes(&adapt_rows));
        hash.update(std.mem.asBytes(&adapt_limit));
    }
    const chunk_rows = try chunkRows();
    if (chunk_rows != state.default_chunk_rows) hash.update(std.mem.asBytes(&chunk_rows));
    hash.update("TF_GLM53_DECODE_MLA=");
    hash.update(decodeMlaDigestValue(decode_mla));
    if (residentEnabled()) hash.update("TF_GLM53_RESIDENT=1");
    hashFineReuseSwitch(&hash, fine_reuse_grain);
    if (side_rows != 0) {
        hash.update("TF_GLM53_SIDE_ROWS=");
        hash.update(std.mem.asBytes(&side_rows));
    }
    // the split changes every rank's routed work, so timings recorded under another split do not carry over
    const moe_balance = try weights.moeBalanceFromEnv();
    if (moe_balance != .off) {
        hash.update("TF_GLM53_MOE_BALANCE=");
        hash.update(@tagName(moe_balance));
    }
    // the head's bytes per step change with its format, so draft timings recorded with the bf16 head do not carry over
    if (weights.headFp8()) hash.update("TF_GLM53_HEAD_FP8=1");
    if (glm.dflash_load.split3Switch()) hash.update("TF_GLM53_DRAFT_SPLIT3=1");
    if (absorbRowsWanted()) hash.update("TF_GLM53_ABSORB_ROWS=1");
    // Every rank must cool the chances alike or their draft windows would part; 1 (off) hashes nothing.
    const chance_temp = try drafts.chanceTempSwitch();
    if (chance_temp != 1.0) {
        hash.update("TF_GLM53_DRAFT_CHANCE_TEMP=");
        hash.update(std.mem.asBytes(&chance_temp));
    }
    try hashPrefixCacheSwitch(&hash, prefixCacheSwitch(), try expertRows());
    hashPromptExperts(&hash, prompt_experts);
    if (prompt_mla != .off) {
        hash.update("TF_GLM53_PROMPT_MLA=");
        hash.update(if (prompt_mla == .tile16) "1" else "2");
    }
    hashSmlaPSwitch(&hash, smla_p);
    hashSmlaFp8Switch(&hash, smla_fp8);
    hashSmlaP3Switch(&hash, smla_p3);
    hashShardQkvASwitch(&hash, shard_qkv_a);
    hashPrefillW8a8Switch(&hash, prefill_w8a8);
    hashPreFuseSwitches(&hash, pre_fuse, prep_fuse);
    hashDecodeUnionSwitch(&hash, decode_union);
    hashDecodeUnion16Switch(&hash, forward.decodeUnion16Switch());
    hashGlmLaneChances(&hash, draftChancesSwitch(), lane_chances);
    hashGlmLiveCosts(&hash, draftChancesSwitch(), liveCostsSwitch());
    if (copydRequested()) hash.update("TF_GLM53_COPYD=1");
    if (std.c.getenv("TF_GLM53_ATTEND")) |value| {
        const attend = std.mem.span(value);
        if (!std.mem.eql(u8, attend, "mma")) {
            hash.update("TF_GLM53_ATTEND=");
            hash.update(attend);
        }
    }
    if (std.c.getenv("TF_GLM53_LOOP_GUARD")) |value| {
        if (std.mem.eql(u8, std.mem.span(value), "long")) hash.update("TF_GLM53_LOOP_GUARD=long");
    }
    hashAdmissionSwitches(&hash, drafter_share, graph_reserve_bytes);
    hashAccountingOverrides(&hash, envSlice("TF_GLM53_RUNTIME_RESERVE_MIB"), envSlice("TF_GLM53_GRAPH_MIB"), envSlice("TF_GLM53_SERVE_MARGIN_MIB"));
    if (envSlice("TF_GLM53_GRAPH_BUDGET_MIB")) |budget_value| {
        hash.update("TF_GLM53_GRAPH_BUDGET_MIB=");
        hash.update(budget_value);
        var graph_rows: [16]u32 = undefined;
        const graph_rows_len = try state.parseGraphRows(envSlice("TF_GLM53_GRAPH_ROWS"), verify_max, &graph_rows);
        hash.update("TF_GLM53_GRAPH_ROWS=");
        hash.update(std.mem.sliceAsBytes(graph_rows[0..graph_rows_len]));
    }
    const long_rows = try state.longTierFrom(envSlice("TF_GLM53_LONG_CTX_ROWS"), o.context);
    if (long_rows != 0 and side_rows != 0) {
        hash.update("TF_GLM53_LONG_CTX_ROWS=");
        hash.update(envSlice("TF_GLM53_LONG_CTX_ROWS") orelse "200000");
    }
    if (row_prune) hash.update("TF_GLM53_ROW_PRUNE=1");
    if (defer_draft_fill) hash.update("TF_GLM53_DEFER_DRAFT_FILL=1");
    hash.update(std.mem.asBytes(&o.kv_format));
    hash.update(std.mem.asBytes(&cfg.routed_codebook));
    hash.update(std.mem.asBytes(&cfg.dcp));
    const checkpoint = try core.checkpoint_identity.fullIdentity(gpa, io, dir);
    hash.update(&checkpoint);
    if (o.drafter) |dd| {
        const drafter_path = try std.fs.path.join(gpa, &.{ dd, "config.json" });
        defer gpa.free(drafter_path);
        const drafter_config = try std.Io.Dir.cwd().readFileAlloc(io, drafter_path, gpa, .limited(1 << 20));
        defer gpa.free(drafter_config);
        hash.update("drafter");
        hash.update(drafter_config);
        hash.update(std.mem.asBytes(&o.drafter_bits));
    }
    if (cfg.ranks == 2) {
        const w = [_]u64{ plan.weights.routed, plan.weights.backbone_fp8, plan.weights.embed_head, plan.weights.router_norms, plan.weights.lowrank, plan.weights.drafter orelse 0 };
        const logical = [_]u64{ plan.logical.routed, plan.logical.backbone_fp8, plan.logical.embed_head, plan.logical.router_norms, plan.logical.lowrank, plan.logical.drafter orelse 0 };
        hash.update(std.mem.sliceAsBytes(&w));
        hash.update(std.mem.sliceAsBytes(&logical));
    } else {
        hash.update(std.mem.asBytes(&cfg.ranks));
    }
    if (ablate_digest) |digest| hash.update(&digest);
    for (plan.unused) |name| hash.update(name);
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}

fn envSlice(name: [:0]const u8) ?[]const u8 {
    const raw = std.c.getenv(name) orelse return null;
    return std.mem.span(raw);
}

fn hashAccountingOverrides(hash: *std.crypto.hash.sha2.Sha256, runtime_mib: ?[]const u8, graph_mib: ?[]const u8, serve_margin_gib: ?[]const u8) void {
    if (runtime_mib) |v| { hash.update("TF_GLM53_RUNTIME_RESERVE_MIB="); hash.update(v); }
    if (graph_mib) |v| { hash.update("TF_GLM53_GRAPH_MIB="); hash.update(v); }
    if (serve_margin_gib) |v| { hash.update("TF_GLM53_SERVE_MARGIN_MIB="); hash.update(v); }
}

test "settings digest includes only explicitly supplied accounting overrides" {
    const digest = struct {
        fn run(runtime: ?[]const u8, graph: ?[]const u8, margin: ?[]const u8) [32]u8 {
            var h = std.crypto.hash.sha2.Sha256.init(.{});
            hashAccountingOverrides(&h, runtime, graph, margin);
            var out: [32]u8 = undefined;
            h.final(&out);
            return out;
        }
    }.run;
    const defaults = digest(null, null, null);
    try std.testing.expect(std.mem.eql(u8, &defaults, &digest(null, null, null)));
    try std.testing.expect(!std.mem.eql(u8, &defaults, &digest("2560", null, null)));
    try std.testing.expect(!std.mem.eql(u8, &defaults, &digest(null, "64", null)));
    try std.testing.expect(!std.mem.eql(u8, &defaults, &digest(null, null, "4.0")));
}

fn hashGlmLaneChances(hash: *std.crypto.hash.sha2.Sha256, draft_chances: bool, lane_chances: bool) void {
    if (draft_chances and lane_chances) hash.update("TF_GLM53_LANE_CHANCES=1");
}

fn hashGlmLiveCosts(hash: *std.crypto.hash.sha2.Sha256, draft_chances: bool, live_costs: bool) void {
    if (draft_chances and live_costs) hash.update("TF_GLM53_LIVE_COSTS=1");
}

test "lane chances switch parses exactly 1 and requires draft chances" {
    try std.testing.expect(!parseLaneChancesSwitch(null));
    try std.testing.expect(!parseLaneChancesSwitch("0"));
    try std.testing.expect(!parseLaneChancesSwitch("true"));
    try std.testing.expect(parseLaneChancesSwitch("1"));
    var b: backend_adapter.Adapter(Engine) = undefined;
    b.drafting = true;
    b.draft_chances = false;
    b.lane_chances = true;
    try std.testing.expect(!b.facts().granted_depth);
    b.draft_chances = true;
    try std.testing.expect(b.facts().granted_depth and b.facts().copy_first);
    const digest = struct {
        fn run(draft: bool, lane: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashGlmLaneChances(&hash, draft, lane);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.run;
    try std.testing.expect(std.mem.eql(u8, &digest(false, true), &digest(false, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(true, true), &digest(true, false)));
}

test "live costs switch parses exactly 1, requires draft chances, digest unchanged off" {
    try std.testing.expect(!parseLiveCostsSwitch(null));
    try std.testing.expect(!parseLiveCostsSwitch("0"));
    try std.testing.expect(!parseLiveCostsSwitch("true"));
    try std.testing.expect(parseLiveCostsSwitch("1"));
    var b: backend_adapter.Adapter(Engine) = undefined;
    b.drafting = true;
    b.lane_chances = false;
    b.copy_rows = 1;
    b.copy_match = 8;
    b.batch_cap = 12;
    b.draft_chances = false;
    b.live_costs = true;
    try std.testing.expect(!b.facts().plain_guard and b.facts().lone_cost_tile == 0);
    b.draft_chances = true;
    b.live_costs = false;
    try std.testing.expect(!b.facts().plain_guard and b.facts().lone_cost_tile == 0);
    b.live_costs = true;
    try std.testing.expect(b.facts().plain_guard and b.facts().lone_cost_tile == 2);
    const digest = struct {
        fn run(draft: bool, live: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashGlmLiveCosts(&hash, draft, live);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.run;
    try std.testing.expect(std.mem.eql(u8, &digest(false, true), &digest(false, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(true, true), &digest(true, false)));
}

fn hashPrefixCacheSwitch(hash: *std.crypto.hash.sha2.Sha256, enabled: bool, expert_rows: usize) !void {
    if (!enabled) return;
    hash.update("TF_GLM53_PREFIX_CACHE=1");
    const a = try prefix_cache.kvsave.cutAlignment(expert_rows);
    hash.update("TF_GLM53_PREFIX_CACHE_ALIGNMENT=");
    hash.update(std.mem.asBytes(&a));
}

test "prefix cache switch leaves the off settings digest input unchanged" {
    var off = std.crypto.hash.sha2.Sha256.init(.{});
    var control = std.crypto.hash.sha2.Sha256.init(.{});
    try hashPrefixCacheSwitch(&off, false, 1024);
    var a: [32]u8 = undefined;
    var b: [32]u8 = undefined;
    off.final(&a);
    control.final(&b);
    try std.testing.expectEqualSlices(u8, &a, &b);
    try std.testing.expectError(error.BadExpertRows, hashPrefixCacheSwitch(&off, true, 0));
}

test "prefix cache off adds no settings digest bytes and no startup work" {
    var off = std.crypto.hash.sha2.Sha256.init(.{});
    var control = std.crypto.hash.sha2.Sha256.init(.{});
    try hashPrefixCacheSwitch(&off, false, 1024);
    var actual: [32]u8 = undefined;
    var expected: [32]u8 = undefined;
    off.final(&actual);
    control.final(&expected);
    try std.testing.expectEqualSlices(u8, &actual, &expected);
    try std.testing.expect(!parseSwitch(null));
    try std.testing.expect(parseSwitch("1"));
    try std.testing.expect(!parseSwitch("0"));
    try std.testing.expect(!parseSwitch("10"));
    try std.testing.expect(!parseSwitch("1 "));
}

fn hashAdmissionSwitches(hash: *std.crypto.hash.sha2.Sha256, share: bool, graph_bytes: u64) void {
    hash.update("TF_GLM53_DRAFTER_SHARE=");
    hash.update(if (share) "1" else "0");
    hash.update("TF_GLM53_GRAPH_RESERVE_MIB=");
    hash.update(std.mem.asBytes(&graph_bytes));
}

test "settings digest input changes with either admission switch" {
    const digest = struct {
        fn run(share: bool, graph_bytes: u64) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashAdmissionSwitches(&hash, share, graph_bytes);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.run;
    const base = digest(false, state.graphs_runtime_placeholder);
    const shared = digest(true, state.graphs_runtime_placeholder);
    const smaller_graphs = digest(false, 512 << 20);
    try std.testing.expect(!std.mem.eql(u8, &base, &shared));
    try std.testing.expect(!std.mem.eql(u8, &base, &smaller_graphs));
}

fn sendFrame(ctx: *anyopaque, header: follow.Header, ids: []const u32) anyerror!void {
    const wire: *follow.Wire = @ptrCast(@alignCast(ctx));
    try wire.send(header, ids);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    if (own.group.rank == 0) {
        own.adapter.commitPending() catch {};
        own.wire.send(.{ .kind = @backingInt(follow.Kind.end) }, &.{}) catch {};
    }
    own.stream.synchronize() catch {};
    cuda.tp_watch.ended();
    own.deinit();
}

pub fn validateOptions(o: Options) !void {
    if (o.context == 0 or o.context > default_context) return error.BadContext;
    if (o.drafts and o.drafter == null) return error.MtpUnsupported;
    if (o.drafter != null and !o.drafts) return error.DrafterNeedsDrafts;
    if (o.drafter != null and o.drafter_bits != 4) return error.BadDrafterBits;
    if (o.segments != 1) return error.UnsupportedSegments;
    if (o.slide) return error.UnsupportedSlide;
}

test "native drafter options require drafts and four-bit weights" {
    try std.testing.expectError(error.MtpUnsupported, validateOptions(.{ .context = 128, .drafts = true }));
    try std.testing.expectError(error.DrafterNeedsDrafts, validateOptions(.{ .context = 128, .drafter = "draft" }));
    try std.testing.expectError(error.BadDrafterBits, validateOptions(.{ .context = 128, .drafts = true, .drafter = "draft", .drafter_bits = 8 }));
    try validateOptions(.{ .context = 128, .drafts = true, .drafter = "draft" });
    try validateOptions(.{ .context = 128, .drafts = false });
}

test "expert row environment accepts only supported batching sizes" {
    try std.testing.expectEqual(@as(usize, 128), try parseExpertRows(null));
    inline for (.{ "16", "128", "512", "1024", "1536", "2048", "4096", "8192" }) |value| {
        try std.testing.expectEqual(try std.fmt.parseInt(usize, value, 10), try parseExpertRows(value));
    }
    inline for (.{ "0", "15", "17", "513", "768", "16384", "bad" }) |value| {
        try std.testing.expectError(error.BadExpertRows, parseExpertRows(value));
    }
}

test "chunk row environment accepts only supported batching sizes" {
    try std.testing.expectEqual(state.default_chunk_rows, try parseChunkRows(null));
    inline for ([_][]const u8{ "2048", "4096", "8192" }) |value| {
        try std.testing.expectEqual(try std.fmt.parseInt(usize, value, 10), try parseChunkRows(value));
    }
    inline for ([_][]const u8{ "0", "256", "512", "1000", "1024", "1536", "3072", "4608", "x" }) |value| {
        try std.testing.expectError(error.BadChunkRows, parseChunkRows(value));
    }
}

test "expert row admission refuses an oversized request before the load instead of falling back" {
    const fake = struct {
        fn admit(g: state.Geometry, context: usize, w: state.WeightClasses, available: u64, reserve: state.Reserve) !state.Admission {
            if (g.expert_rows > 512) return error.MemoryBudget;
            return state.admitRank(g, context, w, available, reserve);
        }
    }.admit;
    const g: state.Geometry = .{ .layers = 1, .full_index_layers = 0, .graphs_bytes = 0 };
    const r = state.reserveFor(1 << 40, false, 0);
    try std.testing.expectError(error.ExpertRowsDoNotFit, admissionChoice(g, 2048, 8, .{}, 1 << 40, r, fake));
    try std.testing.expectEqual(@as(usize, 512), (try admissionChoice(g, 512, 8, .{}, 1 << 40, r, fake)).rows);
    try std.testing.expectEqual(@as(usize, 128), (try admissionChoice(g, 128, 8, .{}, 1 << 40, r, fake)).rows);
    const none = struct {
        fn admit(_: state.Geometry, _: usize, _: state.WeightClasses, _: u64, _: state.Reserve) !state.Admission {
            return error.MemoryBudget;
        }
    }.admit;
    try std.testing.expectError(error.MemoryBudget, admissionChoice(g, 2048, 8, .{}, 1 << 40, r, none));
    try std.testing.expectError(error.MemoryBudget, admissionChoice(g, 128, 8, .{}, 1 << 40, r, none));
}

test "expert rows above chunk rows are refused before memory admission" {
    const never = struct {
        fn admit(_: state.Geometry, _: usize, _: state.WeightClasses, _: u64, _: state.Reserve) !state.Admission {
            return error.MemoryBudget;
        }
    }.admit;
    const g: state.Geometry = .{ .chunk_rows = 4096 };
    const r = state.reserveFor(1 << 40, false, 0);
    try std.testing.expectError(error.ExpertRowsExceedChunkRows, admissionChoice(g, 8192, 8, .{}, 1 << 40, r, never));
}

pub fn contextCap(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: ?[]const u8) !?i64 {
    const n = name orelse (if (std.c.getenv("TF_GLM53_MANIFEST")) |v| std.mem.span(v) else return null);
    var m = try manifest_mod.Manifest.load(a, io, dir, n);
    defer m.deinit();
    return m.max_context;
}

pub fn checkContext(o_context: usize, m: *const manifest_mod.Manifest) !void {
    if (o_context > m.max_context) return error.ContextPastManifest;
}

test "manifest context cap refuses a larger requested window" {
    var m = try manifest_mod.Manifest.parse(std.testing.allocator, "limited", "{\"max_context\":131072}");
    defer m.deinit();
    try checkContext(131072, &m);
    try std.testing.expectError(error.ContextPastManifest, checkContext(262144, &m));
}

fn manifestName(o: Options) !?[]const u8 {
    const name = o.manifest orelse (if (std.c.getenv("TF_GLM53_MANIFEST")) |v| std.mem.span(v) else return null);
    if (!manifest_mod.validName(name)) return error.ManifestBadName;
    return name;
}

pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadL2Depth => "TF_EXL3X_L2D must be an integer from 0 through 8",
        error.BadChunkRows => "TF_GLM53_CHUNK_ROWS must be 2048, 4096, or 8192",
        error.ExpertRowsExceedChunkRows => "TF_GLM53_EXPERT_ROWS exceeds TF_GLM53_CHUNK_ROWS; set TF_GLM53_CHUNK_ROWS >= expert_rows on both ranks",
        error.ExpertRowsDoNotFit => "TF_GLM53_EXPERT_ROWS does not fit this rank's memory (the log names the size that fits); set the same smaller value on both ranks",
        error.ContextPastManifest => "requested context exceeds the selected manifest's max_context",
        error.ManifestMatrixMissing => "the selected manifest requests a routed matrix missing from its pack store",
        error.PackWidthMismatch => "a routed pack matrix shape disagrees with the selected manifest width",
        error.ManifestIncomplete => "the selected manifest does not specify every routed matrix",
        error.DrafterNeedsDrafts => "--drafter requires --drafts",
        error.BadDrafterBits => "GLM-5.3 DFlash2 currently requires --drafter-bits 4",
        error.DraftHeadShape => "DFlash2 target draft head does not match this rank's vocabulary and hidden dimensions",
        else => blk: {
            std.log.err("GLM-5.3: {s}", .{@errorName(err)});
            break :blk openRefusal;
        },
    };
}

test "TF_EXL3X_L2D accepts the supported depths and refuses invalid values" {
    const Env = struct {
        var value: ?[]const u8 = null;
        fn read(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_EXL3X_L2D")) return value;
            return null;
        }
    };
    try std.testing.expectEqual(@as(i32, 4), try parseL2Depth(Env.read));
    Env.value = "0";
    try std.testing.expectEqual(@as(i32, 0), try parseL2Depth(Env.read));
    Env.value = "4";
    try std.testing.expectEqual(@as(i32, 4), try parseL2Depth(Env.read));
    Env.value = "8";
    try std.testing.expectEqual(@as(i32, 8), try parseL2Depth(Env.read));
    Env.value = "9";
    try std.testing.expectError(error.BadL2Depth, parseL2Depth(Env.read));
    Env.value = "x";
    try std.testing.expectError(error.BadL2Depth, parseL2Depth(Env.read));
    try std.testing.expectEqualStrings("TF_EXL3X_L2D must be an integer from 0 through 8", explain(null, error.BadL2Depth).?);
}

pub fn snap_grid_fn(_: *anyopaque) u32 {
    return snap_grid;
}

/// Run a prompt followed by resumed turns. This is the lane server's continuity path: every
/// later turn goes through the WP9 Engine.nextTurn frontier hook.
pub fn serveTurns(engine: anytype, turns: []const []const u32, out: []u32) !void {
    if (turns.len == 0 or out.len < turns.len) return error.BadServeTurns;
    if (turns[0].len == 0) return error.EmptyPrompt;
    out[0] = try engine.prefill(turns[0]);
    for (turns[1..], 1..) |turn, i| out[i] = try engine.nextTurn(turn);
}

test "native contract names the DSA checkpoint and fixed TP2 context" {
    try std.testing.expectEqualStrings("glm_moe_dsa", model_type);
    try std.testing.expectEqual(@as(i64, 262144), default_context);
    try std.testing.expectEqual(@as(u32, 4096), snap_grid);
}

test "native open option checks reject MTP and unsupported serving geometry" {
    try std.testing.expectError(error.MtpUnsupported, validateOptions(.{ .context = 4096, .drafts = true }));
    try std.testing.expectError(error.UnsupportedSegments, validateOptions(.{ .context = 4096, .segments = 2 }));
    try std.testing.expectError(error.UnsupportedSlide, validateOptions(.{ .context = 4096, .slide = true }));
    try std.testing.expectError(error.BadContext, validateOptions(.{ .context = 0 }));
}

test "native settings digest material changes for either prefill communication switch" {
    const digest = struct {
        fn make(prefetch_enabled: bool, tbo_enabled: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashTransportSwitches(&hash, prefetch_enabled, tbo_enabled, false);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expect(!std.mem.eql(u8, &digest(false, false), &digest(true, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(false, false), &digest(false, true)));
    const fuse_off = blk: {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hashTransportSwitches(&hash, false, false, false);
        var out: [32]u8 = undefined;
        hash.final(&out);
        break :blk out;
    };
    const fuse_on = blk: {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hashTransportSwitches(&hash, false, false, true);
        var out: [32]u8 = undefined;
        hash.final(&out);
        break :blk out;
    };
    try std.testing.expect(!std.mem.eql(u8, &fuse_off, &fuse_on));
}

test "collective fusion stays off when its maximum decode payload exceeds the cap" {
    try std.testing.expect(collFusePayloadFits(8, 4 << 20));
    try std.testing.expect(!collFusePayloadFits(128, 4 << 20));
}

test "native settings digest material changes with SMLA-P" {
    const digest = struct {
        fn make(smla_p: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashSmlaPSwitch(&hash, smla_p);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes with q_a / kv_a sharding" {
    const digest = struct {
        fn make(shard_qkv_a: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashShardQkvASwitch(&hash, shard_qkv_a);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material for TF_GLM53_PROMPT_EXPERTS keeps off..3 and distinguishes 10" {
    const digest = struct {
        fn make(mode: forward.PromptExperts) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base"); hashPromptExperts(&hash, mode);
            var out: [32]u8 = undefined; hash.final(&out); return out;
        }
    }.make;
    var base_hash = std.crypto.hash.sha2.Sha256.init(.{}); base_hash.update("base");
    var base: [32]u8 = undefined; base_hash.final(&base);
    try std.testing.expectEqualSlices(u8, &base, &digest(.off));
    for ([_]struct { mode: forward.PromptExperts, suffix: []const u8 }{ .{ .mode = .tiles, .suffix = "TF_GLM53_PROMPT_EXPERTS=1" }, .{ .mode = .wide, .suffix = "TF_GLM53_PROMPT_EXPERTS=2" }, .{ .mode = .pmx, .suffix = "TF_GLM53_PROMPT_EXPERTS=3" } }) |case| {
        var expected_hash = std.crypto.hash.sha2.Sha256.init(.{}); expected_hash.update("base"); expected_hash.update(case.suffix);
        var expected: [32]u8 = undefined; expected_hash.final(&expected); try std.testing.expectEqualSlices(u8, &expected, &digest(case.mode));
    }
    for ([_]forward.PromptExperts{ .off, .tiles, .wide, .pmx }) |mode| try std.testing.expect(!std.mem.eql(u8, &digest(.p5a), &digest(mode)));
}

test "native settings digest material changes with TF_GLM53_PREFILL_W8A8 and is unchanged when off" {
    const digest = struct {
        fn make(on: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashPrefillW8a8Switch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain = std.crypto.hash.sha2.Sha256.init(.{});
    plain.update("base");
    var before: [32]u8 = undefined;
    plain.final(&before);
    try std.testing.expectEqualSlices(u8, &before, &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes with TF_GLM53_MOE_PIPE and is unchanged when off" {
    const digest = struct {
        fn make(on: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashMoePipeSwitch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain = std.crypto.hash.sha2.Sha256.init(.{});
    plain.update("base");
    var before: [32]u8 = undefined;
    plain.final(&before);
    try std.testing.expectEqualSlices(u8, &before, &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes with TF_GLM53_ATTN_PIPE and is unchanged when off" {
    const digest = struct {
        fn make(on: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashAttnPipeSwitch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain = std.crypto.hash.sha2.Sha256.init(.{});
    plain.update("base");
    var before: [32]u8 = undefined;
    plain.final(&before);
    try std.testing.expectEqualSlices(u8, &before, &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest distinguishes WS_TRIM, MOE_POST and COMM_PRIO only when enabled" {
    const digest = struct {
        fn make(ws_trim: bool, moe_post: bool, comm_prio: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashWorkspaceTrimSwitch(&hash, ws_trim);
            hashMoePostSwitch(&hash, moe_post);
            hashCommPrioSwitch(&hash, comm_prio);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain_hash = std.crypto.hash.sha2.Sha256.init(.{});
    plain_hash.update("base");
    var plain: [32]u8 = undefined;
    plain_hash.final(&plain);
    try std.testing.expectEqualSlices(u8, &plain, &digest(false, false, false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false, false, false), &digest(true, false, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(false, false, false), &digest(false, true, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(false, false, false), &digest(false, false, true)));
}

test "native settings digest distinguishes MEMSLIM only when enabled" {
    const digest = struct {
        fn make(on: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashMemslimSwitch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain_hash = std.crypto.hash.sha2.Sha256.init(.{});
    plain_hash.update("base");
    var plain: [32]u8 = undefined;
    plain_hash.final(&plain);
    try std.testing.expectEqualSlices(u8, &plain, &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes for index merge and pipe while off stays unchanged" {
    const digest = struct {
        fn make(merge: u8, pipe: bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hash.update("base");
            hashIndexSwitches(&hash, merge, pipe);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    var plain = std.crypto.hash.sha2.Sha256.init(.{});
    plain.update("base");
    var before: [32]u8 = undefined;
    plain.final(&before);
    try std.testing.expectEqualSlices(u8, &before, &digest(0, false));
    try std.testing.expect(!std.mem.eql(u8, &digest(0, false), &digest(2, false)));
    try std.testing.expect(!std.mem.eql(u8, &digest(0, false), &digest(0, true)));
}

test "native settings digest material changes with SMLA-FP8 and is unchanged when it is off" {
    const digest = struct {
        fn make(smla_fp8: ?bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashSmlaPSwitch(&hash, true);
            if (smla_fp8) |on| hashSmlaFp8Switch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expectEqualSlices(u8, &digest(null), &digest(false)); // off: the digest before SMLA-FP8 existed
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes with SMLA-P3 and is unchanged when it is off" {
    const digest = struct {
        fn make(smla_p3: ?bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashSmlaPSwitch(&hash, true);
            hashSmlaFp8Switch(&hash, false);
            if (smla_p3) |on| hashSmlaP3Switch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expectEqualSlices(u8, &digest(null), &digest(false)); // off: the digest before SMLA-P3 existed
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native settings digest material changes with decode union and is unchanged when it is off" {
    const digest = struct {
        fn make(decode_union: ?bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            hashSmlaP3Switch(&hash, false);
            if (decode_union) |on| hashDecodeUnionSwitch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expectEqualSlices(u8, &digest(null), &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "decode union 16 rows needs decode union and verify max 16; digest unchanged off" {
    try std.testing.expectEqual(@as(usize, 8), try forward.decodeUnionRows(false, false, 8));
    try std.testing.expectEqual(@as(usize, 8), try forward.decodeUnionRows(true, false, 16));
    try std.testing.expectEqual(@as(usize, 16), try forward.decodeUnionRows(true, true, 16));
    try std.testing.expectError(error.DecodeUnion16WithoutDecodeUnion, forward.decodeUnionRows(false, true, 16));
    try std.testing.expectError(error.DecodeUnion16NeedsVerifyMax16, forward.decodeUnionRows(true, true, 8));
    try std.testing.expect(!forward.decodeUnion16Selected(null));
    try std.testing.expect(!forward.decodeUnion16Selected("0"));
    try std.testing.expect(!forward.decodeUnion16Selected("16"));
    try std.testing.expect(forward.decodeUnion16Selected("1"));
    const digest = struct {
        fn make(union16: ?bool) [32]u8 {
            var hash = std.crypto.hash.sha2.Sha256.init(.{});
            if (union16) |on| hashDecodeUnion16Switch(&hash, on);
            var out: [32]u8 = undefined;
            hash.final(&out);
            return out;
        }
    }.make;
    try std.testing.expectEqualSlices(u8, &digest(null), &digest(false));
    try std.testing.expect(!std.mem.eql(u8, &digest(false), &digest(true)));
}

test "native prefill communicator settings require rank agreement before side-group creation" {
    try std.testing.expect(agreesPrefillCommSwitches(.{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }));
    try std.testing.expect(!agreesPrefillCommSwitches(.{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 1, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }));
    try std.testing.expect(!agreesPrefillCommSwitches(.{ .prefetch = 1, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 1, .tbo = 1, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }));
    try std.testing.expect(!agreesPrefillCommSwitches(.{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 0, .tbo = 0, .coll_fuse = 1, .index_pipe = 0, .attn_pipe = 0 }));
    try std.testing.expect(!agreesPrefillCommSwitches(.{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 1, .attn_pipe = 0 }));
    try std.testing.expect(!agreesPrefillCommSwitches(.{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 0 }, .{ .prefetch = 0, .tbo = 0, .coll_fuse = 0, .index_pipe = 0, .attn_pipe = 1 }));
}

test "copyLeader keeps the rank address" {
    var out: [64]u8 = undefined;
    const copied = try copyLeader("192.0.2.1:29600", &out);
    try std.testing.expectEqualSlices(u8, "192.0.2.1:29600", copied);
    try std.testing.expectError(error.NoCommId, copyLeader(null, &out));
    try std.testing.expectError(error.NoCommId, copyLeader("", &out));
    try std.testing.expectError(error.NoCommId, copyLeader("192.0.2.1", &out));
    const too_long: [65]u8 = @splat('x');
    try std.testing.expectError(error.NoCommId, copyLeader(&too_long, &out));
}

test "side communicators are wanted only with ATTN_PIPE or PREFETCH" {
    const cases = [_]struct { attn_pipe: bool, prefetch: bool, wanted: bool }{
        .{ .attn_pipe = false, .prefetch = false, .wanted = false },
        .{ .attn_pipe = false, .prefetch = true, .wanted = true },
        .{ .attn_pipe = true, .prefetch = false, .wanted = true },
        .{ .attn_pipe = true, .prefetch = true, .wanted = true },
    };
    for (cases) |c| try std.testing.expectEqual(c.wanted, sideCommsWanted(c.attn_pipe, c.prefetch));
}

test "releaseCommId removes NCCL_COMM_ID from this process" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const saved = std.c.environ;
    var saved_value: [4097]u8 = @splat(0);
    const saved_ptr = std.c.getenv("NCCL_COMM_ID");
    const had_saved = saved_ptr != null;
    const saved_len = if (saved_ptr) |v| std.mem.span(v).len else 0;
    try std.testing.expect(saved_len <= 4096);
    if (saved_ptr) |v| @memcpy(saved_value[0..saved_len], std.mem.span(v));
    saved_value[saved_len] = 0;
    defer {
        std.c.environ = saved;
        if (had_saved) {
            _ = libc_env.setenv("NCCL_COMM_ID", saved_value[0..saved_len :0], 1);
        } else {
            _ = libc_env.unsetenv("NCCL_COMM_ID");
        }
    }

    try std.testing.expectEqual(@as(c_int, 0), libc_env.setenv("NCCL_COMM_ID", "192.0.2.1:29600", 1));
    var out: [64]u8 = undefined;
    const copied = try copyLeader(std.mem.span(std.c.getenv("NCCL_COMM_ID") orelse return error.NoCommId), &out);
    try std.testing.expectEqualSlices(u8, "192.0.2.1:29600", copied);
    try releaseCommId();
    try std.testing.expect(std.c.getenv("NCCL_COMM_ID") == null);
    try std.testing.expectEqualSlices(u8, "192.0.2.1:29600", copied);
}

test "releaseCommId keeps the start-up environment block free of nulls (v12 ATTN_PIPE null panic)" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const saved = std.c.environ;
    var saved_value: [4097]u8 = @splat(0);
    const saved_ptr = std.c.getenv("NCCL_COMM_ID");
    const had_saved = saved_ptr != null;
    const saved_len = if (saved_ptr) |v| std.mem.span(v).len else 0;
    try std.testing.expect(saved_len <= 4096);
    if (saved_ptr) |v| @memcpy(saved_value[0..saved_len], std.mem.span(v));
    saved_value[saved_len] = 0;
    defer {
        std.c.environ = saved;
        if (had_saved) {
            _ = libc_env.setenv("NCCL_COMM_ID", saved_value[0..saved_len :0], 1);
        } else {
            _ = libc_env.unsetenv("NCCL_COMM_ID");
        }
    }

    try std.testing.expectEqual(@as(c_int, 0), libc_env.setenv("NCCL_COMM_ID", "192.0.2.1:29600", 1));
    const live = std.c.environ;
    var n: usize = 0;
    while (live[n] != null) : (n += 1) {}
    const block = live[0..n :null];
    try releaseCommId();
    try std.testing.expect(std.c.getenv("NCCL_COMM_ID") == null);
    var found_nccl = false;
    for (block) |entry| {
        // With the old unsetenv body, this check reproduced the null slot that panicked during startup environment scanning.
        try std.testing.expect(entry != null);
        if (std.mem.startsWith(u8, std.mem.span(entry.?), "NCCL_COMM_ID=")) found_nccl = true;
    }
    try std.testing.expect(found_nccl);
    var environ: std.Io.Threaded.Environ = .{ .process_environ = .{ .block = .{ .slice = block } } };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    environ.scan(arena.allocator());
}

test "releaseCommId keeps the other variables" {
    if (builtin.os.tag != .linux) return error.SkipZigTest;
    const saved = std.c.environ;
    var saved_nccl: [4097]u8 = @splat(0);
    const saved_nccl_ptr = std.c.getenv("NCCL_COMM_ID");
    const had_nccl = saved_nccl_ptr != null;
    const saved_nccl_len = if (saved_nccl_ptr) |v| std.mem.span(v).len else 0;
    try std.testing.expect(saved_nccl_len <= 4096);
    if (saved_nccl_ptr) |v| @memcpy(saved_nccl[0..saved_nccl_len], std.mem.span(v));
    saved_nccl[saved_nccl_len] = 0;
    var saved_probe: [4097]u8 = @splat(0);
    const saved_probe_ptr = std.c.getenv("TF_P3FIX_PROBE");
    const had_probe = saved_probe_ptr != null;
    const saved_probe_len = if (saved_probe_ptr) |v| std.mem.span(v).len else 0;
    try std.testing.expect(saved_probe_len <= 4096);
    if (saved_probe_ptr) |v| @memcpy(saved_probe[0..saved_probe_len], std.mem.span(v));
    saved_probe[saved_probe_len] = 0;
    defer {
        std.c.environ = saved;
        if (had_nccl) {
            _ = libc_env.setenv("NCCL_COMM_ID", saved_nccl[0..saved_nccl_len :0], 1);
        } else {
            _ = libc_env.unsetenv("NCCL_COMM_ID");
        }
        if (had_probe) {
            _ = libc_env.setenv("TF_P3FIX_PROBE", saved_probe[0..saved_probe_len :0], 1);
        } else {
            _ = libc_env.unsetenv("TF_P3FIX_PROBE");
        }
    }

    try std.testing.expectEqual(@as(c_int, 0), libc_env.setenv("NCCL_COMM_ID", "192.0.2.1:29600", 1));
    try std.testing.expectEqual(@as(c_int, 0), libc_env.setenv("TF_P3FIX_PROBE", "1", 1));
    try releaseCommId();
    try std.testing.expect(std.c.getenv("NCCL_COMM_ID") == null);
    try std.testing.expectEqualStrings("1", std.mem.span(std.c.getenv("TF_P3FIX_PROBE") orelse return error.MissingProbe));
}

test "GLM zngram settings parse defaults, overrides and reject out of range" {
    const Empty = struct { fn get(_: []const u8) ?[]const u8 { return null; } };
    const defaults = try parseZngramEnv(Empty.get, 8);
    try std.testing.expectEqual(@as(u32, 1), defaults.copy_rows);
    try std.testing.expectEqual(@as(u32, 8), defaults.min_match);
    try std.testing.expectEqual(@as(u32, 12), defaults.cap);
    const Values = struct { fn get(name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_GLM53_COPY_ROWS")) return "8";
        if (std.mem.eql(u8, name, "TF_GLM53_ZNGRAM_MIN_MATCH")) return "6";
        if (std.mem.eql(u8, name, "TF_GLM53_VERIFY_CAP")) return "20";
        if (std.mem.eql(u8, name, "TF_GLM53_ZNGRAM_MAX_DRAFT")) return "5";
        if (std.mem.eql(u8, name, "TF_GLM53_ZNGRAM_LOG")) return "1";
        return null;
    } };
    const parsed = try parseZngramEnv(Values.get, 8);
    try std.testing.expectEqual(@as(u32, 8), parsed.copy_rows);
    try std.testing.expectEqual(@as(u32, 6), parsed.min_match);
    try std.testing.expectEqual(@as(u32, 20), parsed.cap);
    try std.testing.expectEqual(@as(u32, 5), parsed.max_draft);
    try std.testing.expect(parsed.log);
    const Bad = struct { fn get(name: []const u8) ?[]const u8 { return if (std.mem.eql(u8, name, "TF_GLM53_COPY_ROWS")) "0" else null; } };
    try std.testing.expectError(error.BadZngramSetting, parseZngramEnv(Bad.get, 8));
}

test "copyd switch is opt in and stays within fast decode rows" {
    try std.testing.expect(!(try parseCopyd(null, 8, 8)));
    try std.testing.expect(!(try parseCopyd("0", 8, 8)));
    try std.testing.expect(try parseCopyd("1", 8, 8));
    try std.testing.expectError(error.CopydNeedsCopyRows, parseCopyd("1", 1, 8));
    try std.testing.expectError(error.CopydRowsPastFastKernels, parseCopyd("1", 16, 8));
    try std.testing.expect(try parseCopyd("1", 16, 16));
}

test "verify max controls suffix and copy environment ranges" {
    const Max16 = struct { fn get(name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_GLM53_COPY_ROWS")) return "16";
        if (std.mem.eql(u8, name, "TF_GLM53_ZNGRAM_MAX_DRAFT")) return "15";
        return null;
    } };
    const parsed = try parseZngramEnv(Max16.get, 16);
    try std.testing.expectEqual(@as(u32, 16), parsed.copy_rows);
    try std.testing.expectEqual(@as(u32, 15), parsed.max_draft);
    try std.testing.expectError(error.BadZngramSetting, parseZngramEnv(Max16.get, 8));
    const Verify = struct { fn get(_: []const u8) ?[]const u8 { return "16"; } };
    try std.testing.expectEqual(@as(usize, 16), try verifyMaxFromEnv(Verify.get));
    const BadVerify = struct { fn get(_: []const u8) ?[]const u8 { return "12"; } };
    try std.testing.expectError(error.BadVerifyMax, verifyMaxFromEnv(BadVerify.get));
}

test "native serve loop uses prefill once then Engine.nextTurn for resumed requests" {
    const FakeEngine = struct {
        calls: [4]u8 = .{ 0, 0, 0, 0 },
        len: usize = 0,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            e.calls[e.len] = 1;
            e.len += 1;
            return ids[0];
        }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 {
            e.calls[e.len] = 2;
            e.len += 1;
            return ids[0];
        }
    };
    var engine: FakeEngine = .{};
    var out: [2]u32 = undefined;
    try serveTurns(&engine, &.{ &.{ 11, 12 }, &.{ 21, 22 } }, &out);
    try std.testing.expectEqualSlices(u32, &.{ 11, 21 }, &out);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2 }, engine.calls[0..2]);
}

test {
    _ = prefix_cache;
    _ = prefix_snap;
}

test "native declarations and public nested contracts are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Options);
    std.testing.refAllDecls(Room);
    std.testing.refAllDecls(Loaded);
    std.testing.refAllDecls(Place);
}

test "expert rows adapt switch parsing and refusal geometry" {
    try std.testing.expectEqual(@as(usize, 0), try parseExpertRowsAdapt(null));
    try std.testing.expectEqual(@as(usize, 0), try parseExpertRowsAdapt("0"));
    try std.testing.expectEqual(@as(usize, 2048), try parseExpertRowsAdapt("2048"));
    try std.testing.expectEqual(@as(usize, 4096), try parseExpertRowsAdapt("4096"));
    try std.testing.expectError(error.BadExpertRowsAdapt, parseExpertRowsAdapt("1024"));
    try std.testing.expectEqual(@as(usize, 0), try parseAdaptMaxPos(null));
    try std.testing.expectEqual(@as(usize, 1024), try parseAdaptMaxPos("1024"));
    try std.testing.expectError(error.BadExpertRowsAdaptMaxPos, parseAdaptMaxPos("0"));
    try std.testing.expectError(error.ExpertRowsAdaptNotLarger, validateExpertRowsAdapt(.{ .expert_rows = 2048, .chunk_rows = 4096, .expert_rows_adapt = 2048 }));
    try std.testing.expectError(error.ExpertRowsAdaptExceedsChunk, validateExpertRowsAdapt(.{ .expert_rows = 1024, .chunk_rows = 2048, .expert_rows_adapt = 4096 }));
    try std.testing.expectError(error.ExpertRowsAdaptNotGrouped, validateExpertRowsAdapt(.{ .expert_rows = 128, .piece_rows = 512, .chunk_rows = 4096, .expert_rows_adapt = 256 }));
}

test "TP2 adaptive expert agreement includes rows and selected limit" {
    const a = ExpertRowsAdaptAgreement{ .rows = 2048, .pad = 0, .limit = 237000 };
    try std.testing.expect(agreesExpertRowsAdapt(a, a));
    try std.testing.expect(!agreesExpertRowsAdapt(a, .{ .rows = 4096, .pad = 0, .limit = 237000 }));
    try std.testing.expect(!agreesExpertRowsAdapt(a, .{ .rows = 2048, .pad = 0, .limit = 236000 }));
}
