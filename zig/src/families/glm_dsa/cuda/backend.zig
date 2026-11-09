// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Lane backend for the WP9 GLM-5.3 Engine. Grammar masks enable strict auto tools, required/named tools and response_format; lane constrain remains authoritative.
const std = @import("std");
const lanes = @import("lanes");

/// TF_GLM53_DRAFT_CHANCES (C1) prices verify rows with these: full-round ms by window rows on two GB10s (4,096-token context:
/// 1 88.0, 2 104.1, 3 120.2, 4 130.3, 6 159.8, 8 191.8, 12 274.3, 16 332.2; other widths interpolated).
/// Without a table the lane allocator treats rows as free and verifies the whole held block every round.
const gb10_verify_costs = [_]lanes.config.Cost{
    .{ .width = 1, .ms = 88.0 },  .{ .width = 2, .ms = 104.1 },  .{ .width = 3, .ms = 120.2 },  .{ .width = 4, .ms = 130.3 },
    .{ .width = 5, .ms = 145.1 }, .{ .width = 6, .ms = 159.8 },  .{ .width = 7, .ms = 175.8 },  .{ .width = 8, .ms = 191.8 },
    .{ .width = 9, .ms = 212.4 }, .{ .width = 10, .ms = 233.1 }, .{ .width = 11, .ms = 253.7 }, .{ .width = 12, .ms = 274.3 },
    .{ .width = 13, .ms = 288.8 }, .{ .width = 14, .ms = 303.2 }, .{ .width = 15, .ms = 317.7 }, .{ .width = 16, .ms = 332.2 },
};
// Probes widths 1, 2, 4, 6, 8 measured in gb10_verify_costs; probe starts at 1, then max(2, tile), then adds tile.
const live_cost_tile: u32 = 2;
const sampler = @import("glm_sampler");
const be = lanes.backend;
const Stream = lanes.Stream;
const follow = @import("glm_follow");
const prefix_cache = @import("prefix_cache");
pub const recal = @import("recal.zig");
test { _ = recal; }

pub const DiskHit = struct { at: usize, key: u64, files: usize = 1 };
pub const DiskHooks = struct {
    ctx: *anyopaque,
    lookup_fn: *const fn (ctx: *anyopaque, prompt: []const u32, after: usize, s: *Stream) anyerror!?DiskHit,
    cuts_fn: *const fn (ctx: *anyopaque, prompt: []const u32, start: usize, s: *Stream) anyerror![]const prefix_cache.Cut,
    save_header_fn: *const fn (ctx: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header,
    pending_fn: *const fn (ctx: *anyopaque) usize,
    pending_max_fn: *const fn (ctx: *anyopaque) usize,
    writer_idle_fn: *const fn (ctx: *anyopaque) bool,
    max_pending: usize,
};

const SaveBase = struct { parent_key: u64, parent_at: u64, seq: u64 };

fn saveBaseOf(h: follow.Header) SaveBase {
    return .{
        .parent_key = std.mem.readInt(u64, h.prefix_hash[8..16], .little),
        .parent_at = std.mem.readInt(u64, h.prefix_hash[16..24], .little),
        .seq = std.mem.readInt(u64, h.prefix_hash[24..32], .little),
    };
}

fn nextBase(current: ?SaveBase, cut: follow.Header, skip: bool) SaveBase {
    if (skip) return current orelse saveBaseOf(cut);
    return .{
        .parent_key = std.mem.readInt(u64, cut.prefix_hash[0..8], .little),
        .parent_at = cut.position,
        .seq = std.mem.readInt(u64, cut.prefix_hash[24..32], .little) +| 1,
    };
}

fn applySaveBase(h: *follow.Header, base: SaveBase) void {
    std.mem.writeInt(u64, h.prefix_hash[8..16], base.parent_key, .little);
    std.mem.writeInt(u64, h.prefix_hash[16..24], base.parent_at, .little);
    std.mem.writeInt(u64, h.prefix_hash[24..32], base.seq, .little);
}

fn saveFrame(position: u64, key: u64, parent_key: u64, parent_at: u64, seq: u64) follow.Header {
    var h: follow.Header = .{ .kind = @backingInt(follow.Kind.snap_save), .position = position };
    std.mem.writeInt(u64, h.prefix_hash[0..8], key, .little);
    std.mem.writeInt(u64, h.prefix_hash[8..16], parent_key, .little);
    std.mem.writeInt(u64, h.prefix_hash[16..24], parent_at, .little);
    std.mem.writeInt(u64, h.prefix_hash[24..32], seq, .little);
    return h;
}

test "skipped middle save keeps the last saved cut as the next base" {
    const first = saveFrame(1024, 0x111, 0, 0, 1);
    var base: ?SaveBase = null;
    base = nextBase(base, first, false);
    const middle = saveFrame(2048, 0x222, 0x111, 1024, 2);
    base = nextBase(base, middle, true);
    var third = saveFrame(3072, 0x333, 0x222, 2048, 3);
    applySaveBase(&third, base.?);
    try std.testing.expectEqual(@as(u64, 0x111), std.mem.readInt(u64, third.prefix_hash[8..16], .little));
    try std.testing.expectEqual(@as(u64, 1024), std.mem.readInt(u64, third.prefix_hash[16..24], .little));
    try std.testing.expectEqual(@as(u64, 2), std.mem.readInt(u64, third.prefix_hash[24..32], .little));
}

test "first skipped save keeps its planned committed parent or base zero" {
    const first = saveFrame(1024, 0x111, 0xabc, 512, 7);
    const base = nextBase(null, first, true);
    try std.testing.expectEqual(@as(u64, 0xabc), base.parent_key);
    try std.testing.expectEqual(@as(u64, 512), base.parent_at);
    const full = saveFrame(1024, 0x111, 0, 0, 1);
    const zero_base = nextBase(null, full, true);
    try std.testing.expectEqual(@as(u64, 0), zero_base.parent_key);
    try std.testing.expectEqual(@as(u64, 0), zero_base.parent_at);
}

pub fn noDraftSwitch(raw: ?[]const u8) bool {
    return if (raw) |value| std.mem.eql(u8, value, "1") else false;
}

pub fn draftProposalCount(disabled: bool, requested: usize) usize {
    return if (disabled) 0 else requested;
}

test "GLM-5.3 no-draft switch is opt-in and exact" {
    try std.testing.expect(!noDraftSwitch(null));
    try std.testing.expect(!noDraftSwitch("0"));
    try std.testing.expect(!noDraftSwitch("true"));
    try std.testing.expect(noDraftSwitch("1"));
    try std.testing.expectEqual(@as(usize, 0), draftProposalCount(true, 7));
    try std.testing.expectEqual(@as(usize, 3), draftProposalCount(false, 3));
}

fn noDraftEnabled() bool {
    const raw = std.c.getenv("TF_GLM53_NO_DRAFT") orelse return false;
    return noDraftSwitch(std.mem.span(raw));
}

const VerifyTiming = struct {
    offered: usize,
    accepted: usize,

    fn record(enabled: bool, offered: usize, accepted: usize) ?@This() {
        if (!enabled) return null;
        return .{ .offered = offered, .accepted = accepted };
    }
};

fn verifyTimingEnabled() bool {
    return std.c.getenv("TF_TIMING") != null;
}

const SaveMemory = struct { free_kib: u64, available_kib: u64 };

fn meminfoValue(contents: []const u8, name: []const u8) ?u64 {
    var lines = std.mem.splitScalar(u8, contents, '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, name)) continue;
        var at = name.len;
        while (at < line.len and std.ascii.isWhitespace(line[at])) : (at += 1) {}
        const start = at;
        while (at < line.len and std.ascii.isDigit(line[at])) : (at += 1) {}
        if (start == at) return null;
        return std.fmt.parseInt(u64, line[start..at], 10) catch null;
    }
    return null;
}

fn readSaveMemory() ?SaveMemory {
    if (comptime @import("builtin").os.tag != .linux) return null;
    const fd = std.c.open("/proc/meminfo", .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buffer: [4096]u8 = undefined;
    const count = std.c.read(fd, &buffer, buffer.len);
    if (count <= 0) return null;
    const contents = buffer[0..@intCast(count)];
    return .{
        .free_kib = meminfoValue(contents, "MemFree:") orelse return null,
        .available_kib = meminfoValue(contents, "MemAvailable:") orelse return null,
    };
}

fn memoryThresholdGiB(name: [:0]const u8, default: u64) u64 {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(raw), 10) catch default;
}

fn prefixCacheEnabled() bool {
    const raw = std.c.getenv("TF_GLM53_PREFIX_CACHE") orelse return false;
    return raw[0] == '1' and raw[1] == 0;
}

fn saveMemoryBelowThreshold(memory: SaveMemory) bool {
    const kib_per_gib: u64 = 1 << 20;
    return memory.free_kib < memoryThresholdGiB("TF_GLM53_PREFIX_CACHE_MIN_FREE_GIB", 6) *| kib_per_gib or
        memory.available_kib < memoryThresholdGiB("TF_GLM53_PREFIX_CACHE_MIN_AVAIL_GIB", 12) *| kib_per_gib;
}

/// TF_GLM53_SLOTS: an adapter slot's per-stream fields while another slot is bound (Adapter.bindLane swaps them).
const Lane = struct {
    owner: ?*Stream = null,
    window_start: ?usize = null,
    window_rows: u32 = 0,
    held: [7]u32 = undefined,
    held_n: usize = 0,
    first_mask: []u32 = &.{},
    first_tok: ?u32 = null,
    chances: [7]f64 = undefined,
    conf_buf: [32]f64 = @splat(0),
};

pub fn Adapter(comptime Engine: type) type {
    return struct {
        gpa: std.mem.Allocator,
        e: *Engine,
        owner: ?*Stream = null,
        link_ctx: ?*anyopaque = null,
        send_fn: ?*const fn (*anyopaque, follow.Header, []const u32) anyerror!void = null,
        ring: [32]u32 = undefined,
        puts: u64 = 0,
        resident: bool = false,
        copy_rows: u32 = 1,
        copy_match: u32 = 8,
        batch_cap: u32 = 12,
        verify_max: usize = 8,
        window_start: ?usize = null,
        window_rows: u32 = 0,
        drafting: bool = false,
        disk: ?DiskHooks = null,
        last_snap_header: follow.Header = .{ .kind = @backingInt(follow.Kind.snap_load) },
        held: [7]u32 = undefined,
        held_n: usize = 0,
        first_mask: []u32 = &.{},
        chances: [7]f64 = undefined,
        draft_chances: bool = false,
        lane_chances: bool = false,
        live_costs: bool = false,
        copyd: bool = false,
        verified_rows: u64 = 0,
        matched_drafts: u64 = 0,
        recal_costs: ?[16]lanes.config.Cost = null, // TF_GLM53_DRAFT_RECAL "costs": full-round ms by rows 1..16 in place of gb10_verify_costs
        recal_conf: ?recal.ConfSets = null, // TF_GLM53_DRAFT_RECAL "conf": per-slot logistic on the DSpark features; chances only
        conf_buf: [32]f64 = @splat(0), // DSpark chances [0,8), z [8,16), margin [16,24), logp [24,32) when the trace or conf recal is on
        trace_path: ?[*:0]const u8 = null, // TF_GLM53_ROUND_TRACE (rank 0): one JSON line per draft, verify and release
        trace_info: recal.Open = .{},
        trace_opened: bool = false,
        trace_req: u64 = 0,
        sweep: bool = false, // TF_GLM53_DRAFT_SWEEP=1: held drafts verified 0,1,..,7,0,.. in turn (cost measurement; tokens unchanged)
        sweep_count: u64 = 0,
        sweep_k: usize = 0,
        /// TF_GLM53_SLOTS: request slots (one stream each; 1: the single-stream adapter as before), each slot's
        /// lane fields while another is bound, the bound slot, and the bound slot's first token kept from its prefill
        /// (another stream's forward may overwrite the head rows before `first` reads them).
        slots: usize = 1,
        lanes: [follow.max_lanes]Lane = @splat(.{}),
        lane: usize = 0,
        first_tok: ?u32 = null,
        /// TF_GLM53_SLOTS with the drafter (TF_GLM53_SLOT_DRAFTS unset or "solo"): only a stream alone drafts; a
        /// shared round's streams take their kept rows into their drafter contexts (depth 0), ready to draft alone.
        solo_drafts: bool = false,

        pub fn init(gpa: std.mem.Allocator, e: *Engine) @This() {
            return .{ .gpa = gpa, .e = e };
        }

        pub fn backend(b: *@This()) be.Backend {
            return .{ .ptr = b, .vtable = if (b.drafting and !noDraftEnabled()) &drafting_vt else &plain_vt };
        }

        const plain_vt: be.Backend.VTable = .{ .prefill = prefill, .first = first, .first_masked = firstMasked, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release };
        const drafting_vt: be.Backend.VTable = vt: {
            var v = plain_vt;
            v.held = heldFn;
            v.probabilities = chancesFn;
            break :vt v;
        };

        pub fn facts(b: *const @This()) lanes.Model {
            if (noDraftEnabled()) return .{ .exact_width = 1, .first_copy_rows = 1, .copy_drafts = false, .copy_match = 0, .verify_widths = true, .mtp = false, .speculate = false, .gpu_tokens = false, .max_streams = 1, .batch_rows = b.batch_cap };
            if (!b.drafting) return .{ .exact_width = b.copy_rows, .first_copy_rows = b.copy_rows, .copy_drafts = b.copy_rows > 1, .copy_match = @intCast(b.copy_match), .verify_widths = true, .mtp = false, .speculate = false, .gpu_tokens = false, .max_streams = @intCast(b.slots), .hidden_rows = b.slots > 1, .shared_costs = if (b.slots > 1) &gb10_verify_costs else &.{}, .batch_rows = if (b.slots > 1) @min(b.batch_cap, follow.max_decode_rows) else b.batch_cap, .copy_arbitrate = b.copyd, .copy_costs = if (b.copyd) &gb10_verify_costs else &.{} };
            return .{ .exact_width = @max(b.copy_rows, 8), .first_copy_rows = b.copy_rows, .copy_drafts = b.copy_rows > 1, .copy_match = @intCast(b.copy_match), .verify_widths = true, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 7, .draft_streams = true, .gpu_tokens = false, .max_streams = @intCast(b.slots), .hidden_rows = b.slots > 1, .shared_costs = if (b.slots > 1) &gb10_verify_costs else &.{}, .solo_drafts = b.solo_drafts, .batch_rows = if (b.slots > 1) @min(b.batch_cap, @max(b.copy_rows, 8)) else b.batch_cap, .draft_probabilities = b.draft_chances, .window_costs = if (b.draft_chances) b.costTable() else &.{}, .granted_depth = b.lane_chances and b.draft_chances, .copy_first = b.lane_chances and b.draft_chances, .plain_guard = b.live_costs and b.draft_chances, .lone_cost_tile = if (b.live_costs and b.draft_chances) live_cost_tile else 0, .copy_arbitrate = b.copyd, .copy_costs = if (b.copyd) b.costTable() else &.{} };
        }

        pub fn appendBegin(b: *@This(), ids: []const u32) !void {
            if (comptime @hasField(Engine, "park")) if (b.e.park != null and b.e.park.?.held) return error.ParkHeldAppend;
            const r = if (ids.len == 0) b.e.pos / b.e.f.st.ws.piece_rows * b.e.f.st.ws.piece_rows else b.e.reusable(ids);
            try b.commitBeforeRollback(r);
            if (r != b.e.pos) try emit(b, .rollback, b.e.pos, &.{}, @intCast(r));
            try b.e.appendBegin(ids);
        }

        pub fn appendTokens(b: *@This(), ids: []const u32) !void {
            const piece = b.e.f.st.ws.piece_rows;
            try b.e.append_pending.appendSlice(b.e.gpa, ids);
            while (b.e.append_pending.items.len > piece) {
                const chunk = b.e.append_pending.items[0..piece];
                try emit(b, .prefill_rows, b.e.pos, chunk, @intCast(chunk.len));
                try b.e.prefillRows(chunk);
                const remain = b.e.append_pending.items.len - piece;
                std.mem.copyForwards(u32, b.e.append_pending.items[0..remain], b.e.append_pending.items[piece..]);
                b.e.append_pending.items.len = remain;
            }
        }

        pub fn appendFinish(b: *@This()) !u32 {
            const pending = b.e.append_pending.items;
            try emit(b, .prefill, b.e.pos, pending, @intCast(pending.len));
            return b.e.appendFinish();
        }

        pub fn warm(b: *@This(), ids: []const u32) !void {
            if (comptime @hasField(Engine, "park")) if (b.e.park != null and b.e.park.?.held) return error.ParkHeldDuringWarm;
            try b.commitBeforeRollback(0);
            try emit(b, .rollback, b.e.pos, &.{}, 0);
            try b.e.rollback(0);
            const count = ids.len / b.e.f.st.ws.piece_rows * b.e.f.st.ws.piece_rows;
            if (count != 0) {
                try emit(b, .prefill_rows, 0, ids[0..count], @intCast(count));
                try b.e.prefillRows(ids[0..count]);
            }
            try b.e.settle();
        }

        fn self(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }

        /// The verify-cost table C1 prices rows with: the recal file's when given, else the measured GB10 table.
        fn costTable(b: *const @This()) []const lanes.config.Cost {
            if (b.recal_costs) |*table| return table;
            return &gb10_verify_costs;
        }

        /// The request's sampling temperature (0: greedy); engines without a sampling rule are greedy.
        fn temperature(b: *const @This()) f64 {
            if (comptime @hasField(Engine, "rule")) return b.e.rule.temperature;
            return 0;
        }

        /// TF_GLM53_ROUND_TRACE's path, writing the open line first (once).
        fn traceFile(b: *@This()) ?[*:0]const u8 {
            const path = b.trace_path orelse return null;
            if (!b.trace_opened) {
                b.trace_opened = true;
                recal.openLine(path, b.trace_info);
            }
            return path;
        }

        fn put(b: *@This(), value: u32) u64 {
            const handle = b.puts;
            b.ring[handle % b.ring.len] = value;
            b.puts += 1;
            return handle;
        }

        fn token(b: *@This(), feed: be.Feed) !u32 {
            return switch (feed) {
                .value => |v| v,
                .handle => |h| if (h < b.puts and b.puts - h <= b.ring.len) b.ring[h % b.ring.len] else error.StaleHandle,
            };
        }

        fn arm(b: *@This(), s: *Stream) !void {
            if (s.sampling) |sampling| {
                const rule: @TypeOf(b.e.rule) = .{ .temperature = sampling.temperature, .top_k = sampling.top_k, .top_p = sampling.top_p, .min_log = sampling.minLog() };
                b.e.seed = sampling.seed;
                if (comptime @hasDecl(Engine, "setSamplingRule")) try b.e.setSamplingRule(rule) else b.e.rule = rule;
                const k = if (comptime @hasDecl(Engine, "candidateHeadK")) b.e.candidateHeadK() else 64;
                std.log.info("glm53 head: candidates k={d} (temperature {d}, top_p {d})", .{ k, rule.temperature, rule.top_p });
            } else {
                const rule: @TypeOf(b.e.rule) = .{ .temperature = 0 };
                if (comptime @hasDecl(Engine, "setSamplingRule")) try b.e.setSamplingRule(rule) else b.e.rule = rule;
                std.log.info("glm53 head: greedy (temperature 0, top_p 1)", .{});
            }
        }

        fn parkStep(b: *@This(), prompt: []const u32, ablate: bool) !void {
            if (comptime @hasField(Engine, "park")) {
                if (b.e.park == null) return;
                const pk = &b.e.park.?;
                const ablated = if (comptime @hasField(Engine, "ablate_on")) b.e.ablate_on else false;
                if (ablate or ablated or !b.resident) {
                    if (pk.held) {
                        const want = pk.effective();
                        const ids = [_]u32{@intCast(want)};
                        try emit(b, .unpark, b.e.pos, &ids, 1);
                        if (try b.e.parkRestore() != want) return error.ParkMismatch;
                    }
                    return;
                }
                const d = pk.plan(b.e.history.items, b.e.frontier, prompt, b.e.f.st.ws.piece_rows);
                if (d.unpark) {
                    const want = pk.effective();
                    const ids = [_]u32{@intCast(want)};
                    try emit(b, .unpark, b.e.pos, &ids, 1);
                    if (try b.e.parkRestore() != want) return error.ParkMismatch;
                }
                if (d.park) |r| {
                    const ids = [_]u32{ @intCast(r.lo), @intCast(r.hi) };
                    try emit(b, .park, b.e.pos, &ids, 2);
                    try b.e.parkSave(r.lo, r.hi);
                }
            }
        }

        fn emit(b: *@This(), kind: follow.Kind, position: usize, ids: []const u32, count: u32) !void {
            try emitSlot(b, kind, position, ids, count, 0);
        }
        fn emitSlot(b: *@This(), kind: follow.Kind, position: usize, ids: []const u32, count: u32, slot: u32) !void {
            if (b.send_fn) |send| try send(b.link_ctx.?, follow.withLane(.{ .kind = @backingInt(kind), .n = count, .slot = slot, .position = position }, b.lane), ids);
        }

        fn armMasks(b: *@This(), bits: []const u32, rows: usize) !void {
            if (comptime @hasDecl(Engine, "setMasks")) {
                try b.e.setMasks(bits, rows);
                try emitSlot(b, .masks, b.e.pos, bits, @intCast(bits.len), @intCast(rows));
            } else return error.StructuresUnsupported;
        }

        fn armFirstMask(b: *@This(), s: *Stream) !void {
            if (s.grammar) |constraint| {
                if (try constraint.next(b.gpa)) |bits| {
                    b.first_mask = bits;
                    try armMasks(b, bits, 1);
                }
            }
        }

        /// Disk frames and prefill_rows start on a 512-row boundary, but fine reuse may stop anywhere: step back to the boundary.
        fn alignReuse(b: *@This(), rows: usize) !usize {
            const aligned = rows / 512 * 512;
            if (aligned == rows) return rows;
            try b.commitBeforeRollback(aligned);
            try emit(b, .rollback, b.e.pos, &.{}, @intCast(aligned));
            if (comptime @hasDecl(Engine, "rollback")) try b.e.rollback(aligned);
            return aligned;
        }

        /// TF_GLM53_SLOTS: bind slot `i`: its lane fields onto the adapter, its sequence onto the engine.
        fn bindLane(b: *@This(), i: usize) !void {
            if (i == b.lane) return;
            if (i >= b.slots) return error.UnknownSlot;
            b.lanes[b.lane] = .{ .owner = b.owner, .window_start = b.window_start, .window_rows = b.window_rows, .held = b.held, .held_n = b.held_n, .first_mask = b.first_mask, .first_tok = b.first_tok, .chances = b.chances, .conf_buf = b.conf_buf };
            const l = b.lanes[i];
            b.owner = l.owner;
            b.window_start = l.window_start;
            b.window_rows = l.window_rows;
            b.held = l.held;
            b.held_n = l.held_n;
            b.first_mask = l.first_mask;
            b.first_tok = l.first_tok;
            b.chances = l.chances;
            b.conf_buf = l.conf_buf;
            if (comptime @hasDecl(Engine, "bind")) try b.e.bind(i);
            b.lane = i;
        }

        fn slotOwner(b: *const @This(), i: usize) ?*Stream {
            return if (i == b.lane) b.owner else b.lanes[i].owner;
        }

        /// The slot serving `s`, bound. One slot: nothing to bind (the owner checks below stand as before).
        fn bindStream(b: *@This(), s: *const Stream) !void {
            if (b.slots == 1) return;
            for (0..b.slots) |i| if (b.slotOwner(i) == s) return b.bindLane(i);
            return error.UnknownStream;
        }

        /// A free slot for a new stream, bound.
        fn bindFree(b: *@This()) !void {
            if (b.slots == 1) return;
            for (0..b.slots) |i| if (b.slotOwner(i) == null) return b.bindLane(i);
            return error.SingleStreamOnly;
        }

        /// A free slot for a prompt, bound. With resident reuse, the free slot whose kept prefill shares the longest
        /// reusable prefix with it; on a tie, the one that drops the fewest kept rows, so a long document stays for
        /// its follow-ups while unrelated requests take the other slot. Without reuse: the first free slot.
        fn bindFreeFor(b: *@This(), prompt: []const u32) !void {
            if (b.slots == 1) return;
            if (comptime @hasDecl(Engine, "reusableOn")) if (b.resident) {
                var best: ?usize = null;
                var best_reuse: usize = 0;
                var best_drop: usize = 0;
                for (0..b.slots) |i| {
                    if (b.slotOwner(i) != null) continue;
                    const reuse = b.e.reusableOn(i, prompt);
                    const drop = b.e.frontierOn(i) -| reuse;
                    if (best == null or reuse > best_reuse or (reuse == best_reuse and drop < best_drop)) {
                        best = i;
                        best_reuse = reuse;
                        best_drop = drop;
                    }
                }
                return b.bindLane(best orelse return error.SingleStreamOnly);
            };
            return b.bindFree();
        }

        fn prefill(ptr: *anyopaque, s: *Stream) anyerror!void {
            const b = self(ptr);
            try b.bindFreeFor(s.prompt());
            b.first_tok = null;
            b.gpa.free(b.first_mask);
            b.first_mask = &.{};
            if (comptime @hasDecl(Engine, "clearMasks")) b.e.clearMasks();
            if (b.owner != null) return error.SingleStreamOnly;
            // TF_GLM53_SLOTS: a server started with --drafter marks every stream drafted; slots serve them undrafted
            // (the facts say no head, so the lane core asks no drafts).
            if ((s.drafts and !b.drafting and b.slots == 1) or s.images.len > 0 or s.reuse.saved != null) return error.UnsupportedStreamFeatures;
            // TF_GLM53_SLOTS: shared windows run unablated (Engine.verifyShared), so slots take no ablated stream.
            if (b.slots > 1 and s.ablate) return error.UnsupportedStreamFeatures;
            b.held_n = 0;
            try arm(b, s);
            try parkStep(b, s.prompt(), s.ablate);
            if (comptime @hasDecl(Engine, "setAblate")) {
                if (comptime @hasField(Engine, "abl")) if (s.ablate and b.e.abl == null) return error.NoAblation;
                // The ablation flag is part of the reuse key: resident rows were computed under the engine's current
                // flag, so a request with the other flag drops them (on both ranks, before `begin` flips the flag) and
                // prefills from row 0. The rollback also empties the history, so `reusable` finds nothing below.
                if (s.ablate != b.e.ablate_on and b.e.pos != 0) {
                    try b.commitBeforeRollback(0);
                    try emit(b, .rollback, b.e.pos, &.{}, 0);
                    try b.e.rollback(0);
                }
                try b.e.setAblate(s.ablate);
            }
            var sampling: [9]u32 = @splat(0);
            const rule = b.e.rule;
            const temp: u64 = @bitCast(rule.temperature);
            const top_p: u64 = @bitCast(rule.top_p);
            const min_log: u64 = @bitCast(rule.min_log);
            sampling = .{ @truncate(temp), @truncate(temp >> 32), @intCast(rule.top_k), @truncate(top_p), @truncate(top_p >> 32), @truncate(min_log), @truncate(min_log >> 32), @truncate(b.e.seed), @truncate(b.e.seed >> 32) };
            try emitSlot(b, .begin, 0, &sampling, @intCast(sampling.len), @as(u32, @intFromBool(s.ablate)));
            const prompt = s.prompt();
            const reused: usize = if (b.resident) (if (comptime @hasDecl(Engine, "reusablePrompt")) b.e.reusablePrompt(prompt) else if (comptime @hasDecl(Engine, "reusable")) b.e.reusable(prompt) else 0) else 0;
            var disk_hit: ?DiskHit = null;
            if (b.disk) |d| {
                if (!s.ablate) {
                    disk_hit = try d.lookup_fn(d.ctx, prompt, reused, s);
                    const pending = d.pending_fn(d.ctx);
                    if (pending != 0 and (pending > d.max_pending or reused < d.pending_max_fn(d.ctx) or (disk_hit != null and disk_hit.?.at > reused) or d.writer_idle_fn(d.ctx))) {
                        try commitSnapshots(b, pending);
                        // The commit can drop a pending node that the first lookup found: look again.
                        disk_hit = try d.lookup_fn(d.ctx, prompt, reused, s);
                    }
                }
            }
            if (b.resident and reused != b.e.pos) {
                try emit(b, .rollback, b.e.pos, &.{}, @intCast(reused));
                if (comptime @hasDecl(Engine, "rollback")) try b.e.rollback(reused);
            }
            var start = reused;
            if (disk_hit) |hit| if (hit.at > reused) {
                const base = try alignReuse(b, reused);
                try emitSnapshotLoad(b, base, prompt[0..hit.at], hit.key);
                if (comptime @hasDecl(Engine, "snapLoad")) start = try b.e.snapLoad(lastHeader(b), prompt[0..hit.at]) else return error.UnsupportedOperation;
            };
            if (b.disk) |d| {
                if (!s.ablate) {
                    const save_cuts = try d.cuts_fn(d.ctx, prompt, start, s);
                    for (save_cuts) |cut| {
                        if (cut.at > start and cut.at <= prompt.len) {
                            start = try alignReuse(b, start);
                            break;
                        }
                    }
                    var from = start;
                    var save_base: ?SaveBase = null;
                    var skipped = false;
                    for (save_cuts) |cut| {
                        if (cut.at <= from or cut.at > prompt.len) continue;
                        const piece = prompt[from..cut.at];
                        try emit(b, .prefill_rows, from, piece, @intCast(piece.len));
                        if (comptime @hasDecl(Engine, "prefillRows")) try b.e.prefillRows(piece) else return error.UnsupportedOperation;
                        if (prefixCacheEnabled()) {
                            if (readSaveMemory()) |memory| {
                                if (saveMemoryBelowThreshold(memory)) {
                                    std.log.warn("prefix cache: save skipped mem free {d} MiB avail {d} MiB", .{ memory.free_kib / 1024, memory.available_kib / 1024 });
                                    // save_header_fn is side-effect free: native.zig:505-513 and prefix_cache.zig:92-117 only read state.
                                    const skipped_header = try d.save_header_fn(d.ctx, cut.at, cut.kind);
                                    save_base = nextBase(save_base, skipped_header, true);
                                    skipped = true;
                                    from = cut.at;
                                    continue;
                                }
                            }
                        }
                        var h = try d.save_header_fn(d.ctx, cut.at, cut.kind);
                        if (skipped) applySaveBase(&h, save_base.?);
                        h.kind = @backingInt(follow.Kind.snap_save);
                        h.n = 0;
                        h.slot = @intFromEnum(cut.kind);
                        h.position = cut.at;
                        try emitHeader(b, h);
                        if (comptime @hasDecl(Engine, "snapSave")) try b.e.snapSave(h) else return error.UnsupportedOperation;
                        save_base = nextBase(save_base, h, false);
                        from = cut.at;
                    }
                    try armFirstMask(b, s);
                    try emit(b, .prefill, from, prompt[from..], @intCast(prompt.len - from));
                    s.cached = @intCast(start);
                    start = from;
                } else {
                    try armFirstMask(b, s);
                    try emit(b, .prefill, b.e.pos, prompt[reused..], @intCast(prompt.len - reused));
                    s.cached = @intCast(reused);
                }
            } else {
                try armFirstMask(b, s);
                try emit(b, .prefill, b.e.pos, prompt[reused..], @intCast(prompt.len - reused));
                s.cached = @intCast(reused);
            }
            b.owner = s;
            if (b.resident) {
                if (comptime @hasField(Engine, "last_reused")) b.e.last_reused = reused;
            }
            // A warm pass takes this same exact prefill. It has no chunk gate here, so it cannot step aside mid-pass;
            // it only prefills rows the next turn needs, and lane_host's quiet wait keeps it from starting beside work.
            if (start < prompt.len) {
                const t = try b.e.prefill(prompt[start..]);
                if (b.slots > 1) b.first_tok = t;
            }
        }

        fn lastHeader(b: *@This()) follow.Header {
            return b.last_snap_header;
        }

        fn emitHeader(b: *@This(), h: follow.Header) !void {
            b.last_snap_header = h;
            if (b.send_fn) |send| try send(b.link_ctx.?, follow.withLane(h, b.lane), &.{});
        }

        fn emitSnapshotLoad(b: *@This(), r: usize, ids: []const u32, key: u64) !void {
            var h: follow.Header = .{ .kind = @backingInt(follow.Kind.snap_load), .n = @intCast(ids.len), .position = r };
            var key_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &key_bytes, key, .little);
            @memcpy(h.prefix_hash[0..8], &key_bytes);
            b.last_snap_header = h;
            if (b.send_fn) |send| try send(b.link_ctx.?, h, ids);
        }

        fn commitSnapshots(b: *@This(), n: usize) !void {
            try emit(b, .snap_commit, 0, &.{}, @intCast(n));
            if (comptime @hasDecl(Engine, "snapCommit")) try b.e.snapCommit(n) else return error.UnsupportedOperation;
        }

        fn commitBeforeRollback(b: *@This(), target: usize) !void {
            if (b.disk) |d| {
                const pending = d.pending_fn(d.ctx);
                if (pending != 0 and target < d.pending_max_fn(d.ctx)) try commitSnapshots(b, pending);
            }
        }

        pub fn commitPending(b: *@This()) !void {
            if (b.disk) |d| {
                const pending = d.pending_fn(d.ctx);
                if (pending != 0) try commitSnapshots(b, pending);
            }
        }

        fn first(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
            const b = self(ptr);
            try b.bindStream(s);
            if (b.owner != s or position != b.e.pos) return error.PositionMismatch;
            if (b.first_tok) |t| {
                b.first_tok = null;
                return put(b, t);
            }
            var out: [1]u32 = undefined;
            try b.e.draw(b.e.head_rows - 1, 1, position - 1, &out);
            return put(b, out[0]);
        }

        fn firstMasked(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
            const b = self(ptr);
            try b.bindStream(s);
            if (b.owner != s or position != b.e.pos) return error.PositionMismatch;
            if (b.first_mask.len == 0) return error.FirstMaskNotArmed;
            defer {
                b.gpa.free(b.first_mask);
                b.first_mask = &.{};
            }
            if (!std.mem.eql(u32, mask, b.first_mask)) return error.FirstMaskChanged;
            if (b.first_tok) |t| {
                b.first_tok = null;
                return put(b, t);
            }
            var out: [1]u32 = undefined;
            try b.e.draw(b.e.head_rows - 1, 1, position - 1, &out);
            return put(b, out[0]);
        }

        fn queue(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
            const b = self(ptr);
            try b.bindStream(s);
            if (b.owner != s or position != b.e.pos + 1) return error.PositionMismatch;
            const id = try token(b, feed);
            var out: [1]u32 = undefined;
            try emit(b, .verify, b.e.pos, &.{id}, 1);
            try b.e.verify(&.{id}, &out);
            try emit(b, .keep, b.e.pos, &.{}, 1);
            try b.e.keep(1);
            return put(b, out[0]);
        }

        fn read(ptr: *anyopaque, handle: u64) anyerror!u32 {
            const b = self(ptr);
            if (handle >= b.puts or b.puts - handle > b.ring.len) return error.StaleHandle;
            return b.ring[handle % b.ring.len];
        }

        fn heldFn(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
            const b = self(ptr);
            try b.bindStream(s);
            if (b.owner != s) return error.UnknownStream;
            if (out.len > b.held_n) return error.NoHeldDrafts;
            @memcpy(out, b.held[0..out.len]);
        }

        /// TF_GLM53_DRAFT_CHANCES=1 enables conditional draft chances for allocation-based window sizing.
        /// TF_GLM53_DRAFT_RECAL conf: per-slot logistic on (z, margin, logp, 1), greedy or sampled set by the request temperature.
        fn chancesFn(ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool {
            const b = self(ptr);
            try b.bindStream(s);
            if (!b.draft_chances or b.owner != s or b.held_n == 0) return false;
            const n: usize = @min(out.len, b.held_n);
            if (b.sweep) {
                // TF_GLM53_DRAFT_SWEEP: the first sweep_k drafts look certain and the rest worthless, so the allocator
                // verifies exactly sweep_k of them (every width 0..7 in turn, same text and context).
                for (out[0..n], 0..) |*chance, i| chance.* = if (i < b.sweep_k) 1.0 else 0.0;
                for (out[n..]) |*chance| chance.* = 0;
                return true;
            }
            const mode: usize = @intFromBool(b.temperature() > 0);
            var chain: f64 = 1;
            for (out[0..n], b.chances[0..n], 0..) |*chance, conditional, i| {
                const c = if (b.recal_conf) |*sets| recal.calibrate(sets[mode][i], b.conf_buf[8 + i], b.conf_buf[16 + i], b.conf_buf[24 + i]) else conditional;
                chain *= c;
                chance.* = chain;
            }
            for (out[n..]) |*chance| chance.* = 0;
            return true;
        }

        fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
            const b = self(ptr);
            if (windows.len > 1 and out.len >= windows.len) return b.verifyMany(windows, out);
            if (windows.len != 1 or out.len != 1) return error.SingleStreamOnly;
            try b.bindStream(windows[0].stream);
            return b.verifyOne(windows[0], &out[0]);
        }

        /// One stream's window in a forward of its own (the single-stream verify).
        fn verifyOne(b: *@This(), w: be.Window, o: *be.Verified) !void {
            const t0: u64 = if (b.trace_path != null) recal.nowNs() else 0;
            if (b.owner != w.stream or w.parents != null or w.early) return error.UnsupportedWindow;
            if (w.rows() > follow.max_decode_rows or w.rows() > b.verify_max or o.sampled.len < w.rows()) return error.WindowTooWide;
            if (w.held != 0 and !b.drafting) return error.UnsupportedWindow;
            if (w.held > b.held_n) return error.NoHeldDrafts;
            if (w.positions.len == w.rows() and w.positions[0] != b.e.pos + 1) return error.PositionMismatch;
            var ids = try b.gpa.alloc(u32, w.rows());
            defer b.gpa.free(ids);
            ids[0] = w.pending;
            const drafts = if (w.held > 0) b.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            if (w.masks.len != 0) {
                if (w.masks.len % w.rows() != 0) return error.InvalidGrammarMasks;
                try armMasks(b, w.masks, w.rows());
            }
            const start = b.e.pos;
            try emit(b, .verify, start, ids, @intCast(ids.len));
            var sampled: [follow.max_decode_rows]u32 = undefined;
            if (comptime @hasField(Engine, "capture_held")) b.e.capture_held = @intCast(w.held);
            try b.e.verify(ids, sampled[0..ids.len]);
            if (b.drafting) b.verified_rows += ids.len;
            if (b.traceFile()) |path| recal.verifyLine(path, b.trace_req, start, ids.len, w.held, if (w.held > 0) 0 else w.tokens.len, t0, recal.nowNs() - t0, ids, sampled[0..ids.len]);
            try emit(b, .keep, start, &.{}, @intCast(ids.len));
            try b.e.keep(ids.len);
            b.window_start = start;
            b.window_rows = @intCast(ids.len);
            @memcpy(o.sampled[0..ids.len], sampled[0..ids.len]);
            if (w.held > 0) {
                if (o.drafts.len < w.held) return error.WindowTooWide;
                @memcpy(o.drafts[0..w.held], drafts);
            } else {
                o.drafts = &.{};
            }
        }

        fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
            const b = self(ptr);
            if (windows.len == 0 or windows.len != paths.len or (windows.len > 1 and b.slots == 1)) return error.SingleStreamOnly;
            for (windows, paths) |w, path| {
                try b.bindStream(w.stream);
                try b.keepOne(w, path);
            }
        }

        fn keepOne(b: *@This(), w: be.Window, path: []const u32) !void {
            if (b.owner != w.stream) return error.SingleStreamOnly;
            if (w.parents != null or (w.held != 0 and !b.drafting)) return error.UnsupportedWindow;
            const kept = path.len;
            const start = b.window_start orelse return error.NoVerifiedWindow;
            if (kept > b.window_rows) return error.BadKeep;
            if (kept < b.window_rows) {
                const to = start + kept;
                try emit(b, .rollback, b.e.pos, &.{}, @intCast(to));
                if (comptime @hasDecl(Engine, "rollback")) try b.e.rollback(to) else return error.RollbackUnsupported;
            }
            // Ported from ashhart/TensorFold 2e0869e3 (Apache-2.0)
            if (VerifyTiming.record(verifyTimingEnabled(), b.window_rows, kept)) |timing| {
                std.debug.print("[timing] verify_frame offered={d} accepted={d}\n", .{ timing.offered, timing.accepted });
            }
            b.window_start = null;
            b.window_rows = 0;
        }

        /// TF_GLM53_SLOTS: a round's windows, one per stream. One shared forward when every window can share it
        /// (no held drafts, no grammar masks, one head mode), else a forward a window in turn.
        fn verifyMany(b: *@This(), windows: []const be.Window, out: []be.Verified) !void {
            if (comptime @hasDecl(Engine, "verifyShared")) {
                if (b.slots == 1 or windows.len > b.slots or windows.len > follow.max_lanes) return error.SingleStreamOnly;
                var share = true;
                var head: @TypeOf(b.e.f.head) = .greedy;
                for (windows, 0..) |w, k| {
                    try b.bindStream(w.stream);
                    if (w.masks.len != 0 or w.parents != null or w.early) share = false;
                    if (k == 0) head = b.e.f.head else if (!std.meta.eql(head, b.e.f.head)) share = false;
                }
                if (!share) {
                    for (windows, out[0..windows.len]) |w, *o| {
                        try b.bindStream(w.stream);
                        try b.verifyOne(w, o);
                    }
                    return;
                }
                const P = Engine.SharedPart;
                var parts: [follow.max_lanes]P = undefined;
                var ids: [follow.max_lanes][follow.max_decode_rows]u32 = undefined;
                var sampled: [follow.max_lanes][follow.max_decode_rows]u32 = undefined;
                var frame: [1 + follow.max_lanes * (2 + follow.max_decode_rows)]u32 = undefined;
                frame[0] = @intCast(windows.len);
                var at: usize = 1;
                var total: usize = 0;
                for (windows, out[0..windows.len], 0..) |w, o, k| {
                    try b.bindStream(w.stream);
                    if (b.owner != w.stream) return error.UnsupportedWindow;
                    const rows = w.rows();
                    if (rows > follow.max_decode_rows or rows > b.verify_max or o.sampled.len < rows or total + rows > follow.max_decode_rows) return error.WindowTooWide;
                    if (w.positions.len == rows and w.positions[0] != b.e.pos + 1) return error.PositionMismatch;
                    if (w.held != 0 and !b.drafting) return error.UnsupportedWindow;
                    if (w.held > b.held_n) return error.NoHeldDrafts;
                    ids[k][0] = w.pending;
                    // the slot's held drafts (a stream that drafted alone, or its first drafts) or copy proposals
                    const drafts = if (w.held > 0) b.held[0..w.held] else w.tokens;
                    @memcpy(ids[k][1..][0..drafts.len], drafts);
                    parts[k] = .{ .slot = b.lane, .ids = ids[k][0..rows], .out = sampled[k][0..rows] };
                    frame[at] = @intCast(b.lane);
                    frame[at + 1] = @intCast(rows);
                    @memcpy(frame[at + 2 ..][0..rows], ids[k][0..rows]);
                    at += 2 + rows;
                    total += rows;
                }
                try emit(b, .verify_shared, 0, frame[0..at], @intCast(at));
                try b.e.verifyShared(parts[0..windows.len]);
                for (windows, out[0..windows.len], 0..) |w, *o, k| {
                    try b.bindStream(w.stream);
                    const rows = parts[k].ids.len;
                    const start = b.e.pos;
                    try emit(b, .keep, start, &.{}, @intCast(rows));
                    try b.e.keep(rows);
                    b.window_start = start;
                    b.window_rows = @intCast(rows);
                    if (b.drafting) b.verified_rows += rows;
                    @memcpy(o.sampled[0..rows], sampled[k][0..rows]);
                    if (w.held > 0) {
                        if (o.drafts.len < w.held) return error.WindowTooWide;
                        @memcpy(o.drafts[0..w.held], ids[k][1..][0..w.held]);
                    } else {
                        o.drafts = &.{};
                    }
                }
            } else return error.SingleStreamOnly;
        }

        fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
            const b = self(ptr);
            const disabled = noDraftEnabled();
            if (disabled) {
                b.held_n = 0;
                b.window_start = null;
                b.window_rows = 0;
                if (comptime @hasField(Engine, "draft_fill_pending")) b.e.draft_fill_pending.clearRetainingCapacity();
                return;
            }
            if (!b.drafting) {
                if (requests.len != 0) return error.MtpUnsupported;
                return;
            }
            // TF_GLM53_SLOTS: a shared round asks every stream's head in one call; each drafts on its own slot.
            if (requests.len != 1 and b.slots == 1) return error.SingleStreamOnly;
            for (requests) |r| {
                try b.bindStream(r.stream);
                try b.draftOne(r);
            }
        }

        fn draftOne(b: *@This(), r: be.DraftRequest) !void {
            if (b.owner != r.stream or r.lanes != null or r.early) return error.TreesNotBuilt;
            if (r.position == 0) return error.PositionMismatch;
            const pos: usize = @intCast(r.position - 1);
            if (r.rows) |rows| {
                if (rows.len == 0 or rows.len != r.follow.len) return error.TreesNotBuilt;
                if (pos != r.start + rows.len) return error.PositionMismatch;
            }
            var n: usize = 0;
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                n = rows.len;
                b.matched_drafts += n - 1;
            }
            // TF_GLM53_SLOTS: a round that holds no drafts (a shared round's) takes its kept rows into the slot's
            // drafter context and runs no block pass, so the stream drafts from its first round alone.
            if (b.slots > 1 and r.depth == 0) {
                if (comptime @hasDecl(Engine, "absorbBlock")) {
                    const one = [_]u32{@intCast(n)};
                    try emit(b, .absorb, pos, &one, 1);
                    try b.e.absorbBlock(pos, n);
                } else return error.UnsupportedOperation;
                b.held_n = 0;
                return;
            }
            const anchor = if (r.rows != null) r.follow[n - 1] else try token(b, r.first orelse return error.NoFirstToken);
            const ids = [_]u32{ @intCast(n), anchor };
            try emit(b, .draft_block, pos, &ids, 2);
            // The trace and the conf recal need the selector's features: a 32-slot buffer asks for them (dflash_select.zig).
            const feats = b.draft_chances and (b.trace_path != null or b.recal_conf != null);
            if (feats) @memset(&b.conf_buf, 0);
            const probs: ?[]f64 = if (!b.draft_chances) null else if (feats) b.conf_buf[0..] else b.chances[0..];
            if (comptime @hasDecl(Engine, "draftBlock")) try b.e.draftBlock(pos, n, anchor, b.held[0..], probs) else return error.UnsupportedOperation;
            if (feats) @memcpy(&b.chances, b.conf_buf[0..7]);
            const depth: usize = @min(@as(usize, @intCast(r.depth)), 7);
            b.held_n = depth;
            if (b.sweep) {
                b.sweep_k = @intCast(b.sweep_count % 8);
                b.sweep_count += 1;
            }
            if (b.traceFile()) |path| recal.draftLine(path, b.trace_req, pos, anchor, n, depth, b.temperature(), b.held[0..], b.chances[0..], b.conf_buf[0..]);
        }

        fn release(ptr: *anyopaque, s: *Stream) void {
            const b = self(ptr);
            b.bindStream(s) catch return;
            if (b.owner == s) {
                // TF_GLM53_CAPTURE_DIR: the transcript goes out before any rollback below drops it.
                if (comptime @hasDecl(Engine, "captureEnd")) b.e.captureEnd();
                b.gpa.free(b.first_mask);
                b.first_mask = &.{};
                if (comptime @hasDecl(Engine, "clearMasks")) b.e.clearMasks();
                if (b.traceFile()) |path| recal.releaseLine(path, b.trace_req, b.e.pos, recal.nowNs());
                b.trace_req += 1;
                b.window_start = null;
                b.window_rows = 0;
                if (b.drafting) {
                    b.held_n = 0;
                    std.log.info("glm53 dflash2: matched_draft_tokens {d} verified_rows {d}", .{ b.matched_drafts, b.verified_rows });
                }
                if (b.resident) {
                    if (comptime @hasDecl(Engine, "settle")) {
                        b.emit(.rollback, b.e.pos, &.{}, @intCast(b.e.frontier)) catch {};
                        b.e.settle() catch {};
                    }
                }
                // Publish this request's saves now. A save that is only sealed stays `.tmp` and is deleted at the
                // next startup, so a restart right after a reply would lose it. Decode rows are at or above the cuts,
                // so the join only waits for the background writer.
                b.commitPending() catch |err| std.log.warn("prefix cache: end-of-request commit failed: {s}", .{@errorName(err)});
                b.emit(.release, b.e.pos, &.{}, 0) catch {};
                b.owner = null;
                // The Engine owns position-addressed cache rows. A later lane stream starts a fresh
                // transcript, so drop the prior frontier before the next prompt is admitted.
                // Resident mode keeps the settled prefill rows, and with them the ablation flag they were computed under.
                if (!b.resident) {
                    if (comptime @hasDecl(Engine, "rollback")) b.e.rollback(0) catch {};
                    if (comptime @hasField(Engine, "ablate_on")) b.e.ablate_on = false;
                }
                if (comptime @hasDecl(Engine, "requestEnd")) b.e.requestEnd() catch {};
            }
        }
    };
}

test "engine adapter serves one greedy stream and rejects a second owner" {
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        calls: [8]u8 = @splat(0),
        n: usize = 0,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            e.calls[e.n] = 1;
            e.n += 1;
            e.pos += ids.len;
            return ids[0] + 1;
        }
        pub fn draw(e: *@This(), _: usize, _: usize, _: u64, out: []u32) !void {
            e.calls[e.n] = 2;
            e.n += 1;
            out[0] = 9;
        }
        pub fn verify(e: *@This(), ids: []const u32, out: []u32) !void {
            e.calls[e.n] = 3;
            e.n += 1;
            out[0] = ids[0] + 1;
        }
        pub fn keep(e: *@This(), rows: usize) !void {
            e.pos += rows;
        }
    };
    var fake: FakeEngine = .{};
    const B = Adapter(FakeEngine);
    var b = B.init(std.testing.allocator, &fake);
    const facts_before = B.facts(&b);
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    try stream.context.appendSlice(std.testing.allocator, &.{ 5, 6 });
    stream.prompt_len = 2;
    try B.prefill(&b, &stream);
    b.first_mask = try std.testing.allocator.dupe(u32, &.{ 1, 2 });
    const masked_handle = try B.firstMasked(&b, &stream, 2, &.{ 1, 2 });
    try std.testing.expectEqual(@as(u32, 9), try B.read(&b, masked_handle));
    b.first_mask = try std.testing.allocator.dupe(u32, &.{ 1, 2 });
    try std.testing.expectError(error.FirstMaskChanged, B.firstMasked(&b, &stream, 2, &.{ 2, 1 }));
    try std.testing.expectError(error.FirstMaskNotArmed, B.firstMasked(&b, &stream, 2, &.{ 1, 2 }));
    const handle = try B.first(&b, &stream, 2);
    try std.testing.expectEqual(@as(u32, 9), try B.read(&b, handle));
    const next = try B.queue(&b, &stream, .{ .value = 10 }, 3);
    try std.testing.expectEqual(@as(u32, 11), try B.read(&b, next));
    var other = std.mem.zeroes(Stream);
    try std.testing.expectError(error.SingleStreamOnly, B.prefill(&b, &other));
    try std.testing.expectError(error.StaleHandle, B.read(&b, 99));
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 2, 3 }, fake.calls[0..4]);
    try std.testing.expectEqual(facts_before, B.facts(&b));
    try std.testing.expectEqual(@as(u32, 1), B.facts(&b).exact_width);
    std.testing.refAllDecls(@This());
}

test "verify frame timing: accepted widths exist only when timing is enabled" {
    try std.testing.expect(VerifyTiming.record(false, 8, 5) == null);
    const timing = VerifyTiming.record(true, 8, 5) orelse return error.MissingTiming;
    try std.testing.expectEqual(@as(usize, 8), timing.offered);
    try std.testing.expectEqual(@as(usize, 5), timing.accepted);
}

test "glm masks: verify sends row masks immediately before verify and keeps masked windows" {
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        set_calls: usize = 0,
        clear_calls: usize = 0,
        rows: usize = 0,
        ids_n: usize = 0,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 1; }
        pub fn draw(_: *@This(), _: usize, _: usize, _: u64, out: []u32) !void { @memset(out, 4); }
        pub fn setMasks(e: *@This(), ids: []const u32, rows: usize) !void { e.set_calls += 1; e.rows = rows; e.ids_n = ids.len; }
        pub fn clearMasks(e: *@This()) void { e.clear_calls += 1; }
        pub fn verify(_: *@This(), _: []const u32, out: []u32) !void { @memset(out, 9); }
        pub fn keep(e: *@This(), rows: usize) !void { e.pos += rows; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    const Frames = struct {
        kinds: [8]u32 = @splat(0),
        n: usize = 0,
        fn send(ptr: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const self_ptr: *@This() = @ptrCast(@alignCast(ptr));
            self_ptr.kinds[self_ptr.n] = h.kind;
            self_ptr.n += 1;
        }
    };
    var fake: FakeEngine = .{};
    var frames: Frames = .{};
    const B = Adapter(FakeEngine);
    var b = B.init(std.testing.allocator, &fake);
    b.drafting = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    b.send_fn = Frames.send;
    b.link_ctx = &frames;
    const positions = [_]u64{ 1, 2 };
    const bits = [_]u32{ 3, 5, 7, 9 };
    const w: be.Window = .{ .stream = &stream, .pending = 1, .held = 0, .tokens = &.{2}, .parents = null, .positions = &positions, .masks = &bits };
    var sampled: [2]u32 = undefined;
    var drafted: [0]u32 = .{};
    var verified: [1]be.Verified = .{.{ .sampled = &sampled, .drafts = &drafted }};
    try B.verify(&b, &.{w}, &verified);
    try std.testing.expectEqual(@as(usize, 1), fake.set_calls);
    try std.testing.expectEqual(@as(usize, 2), fake.rows);
    try std.testing.expectEqual(@as(usize, 4), fake.ids_n);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.masks)), frames.kinds[0]);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.verify)), frames.kinds[1]);
    const accepted = [_]u32{1};
    const paths = [_][]const u32{&accepted};
    try B.keep(&b, &.{w}, &paths);
    try std.testing.expectEqual(@as(usize, 1), fake.pos);

    var plain_fake: FakeEngine = .{};
    var plain_frames: Frames = .{};
    var plain = B.init(std.testing.allocator, &plain_fake);
    plain.drafting = true;
    plain.owner = &stream;
    plain.send_fn = Frames.send;
    plain.link_ctx = &plain_frames;
    var plain_verified: [1]be.Verified = .{.{ .sampled = &sampled, .drafts = &drafted }};
    try B.verify(&plain, &.{.{ .stream = &stream, .pending = 1, .held = 0, .tokens = &.{2}, .parents = null, .positions = &positions }}, &plain_verified);
    try std.testing.expectEqual(@as(usize, 0), plain_fake.set_calls);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.verify)), plain_frames.kinds[0]);
}

test "glm masks: an active prefill grammar arms its one head row before prefill" {
    const Rule = struct {
        fn accept(_: *anyopaque, _: u32) anyerror!bool { return true; }
        fn rollback(_: *anyopaque, _: usize) anyerror!void {}
        fn terminated(_: *anyopaque) bool { return false; }
        fn fill(_: *anyopaque, bits: []u32) anyerror!void { @memset(bits, 0xffffffff); }
        fn drafts(_: *anyopaque, _: []u32) anyerror!usize { return 0; }
        fn free(_: *anyopaque) void {}
    };
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        mask_rows: usize = 0,
        mask_words: usize = 0,
        prefill_mask_rows: usize = 0,
        clears: usize = 0,
        pub fn clearMasks(e: *@This()) void { e.mask_rows = 0; e.clears += 1; }
        pub fn setMasks(e: *@This(), bits: []const u32, rows: usize) !void { e.mask_rows = rows; e.mask_words = bits.len / rows; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.prefill_mask_rows = e.mask_rows; e.mask_rows = 0; e.pos += ids.len; return 1; }
        pub fn draw(_: *@This(), _: usize, _: usize, _: u64, out: []u32) !void { @memset(out, 3); }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), rows: usize) !void { e.pos += rows; }
    };
    const Frames = struct {
        kinds: [8]u32 = @splat(0),
        n: usize = 0,
        fn send(ptr: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const capture: *@This() = @ptrCast(@alignCast(ptr));
            capture.kinds[capture.n] = h.kind;
            capture.n += 1;
        }
    };
    var rule_state: u8 = 0;
    const constraint = try std.testing.allocator.create(lanes.grammar.Constraint);
    constraint.* = .init(.{ .ptr = &rule_state, .vtable = &.{ .accept = Rule.accept, .rollback = Rule.rollback, .terminated = Rule.terminated, .fill = Rule.fill, .drafts = Rule.drafts, .free = Rule.free } }, 4, null);
    defer constraint.deinit(std.testing.allocator);
    var fake: FakeEngine = .{};
    var frames: Frames = .{};
    const B = Adapter(FakeEngine);
    var b = B.init(std.testing.allocator, &fake);
    b.send_fn = Frames.send;
    b.link_ctx = &frames;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    try stream.context.append(std.testing.allocator, 5);
    stream.prompt_len = 1;
    stream.grammar = constraint;
    try B.prefill(&b, &stream);
    try std.testing.expectEqual(@as(usize, 1), fake.prefill_mask_rows);
    try std.testing.expectEqual(@as(usize, 4), fake.mask_words);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.masks)), frames.kinds[1]);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.prefill)), frames.kinds[2]);
    try std.testing.expectEqual(@as(usize, 0), fake.mask_rows);
    B.release(&b, &stream);
    try std.testing.expect(fake.clears > 0);
}

test "resident adapter emits reuse, append, warm, and release frames" {
    const FakeEngine = struct {
        const Ws = struct { piece_rows: usize = 512 };
        const St = struct { ws: Ws = .{} };
        const F = struct { st: St = .{} };
        gpa: std.mem.Allocator,
        f: F = .{},
        pos: usize = 0,
        frontier: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        history: std.ArrayList(u32) = .empty,
        append_pending: std.ArrayList(u32) = .empty,
        last_reused: usize = 0,
        reuse_grain: usize = 0,
        park: ?@import("engine").side_park.Park = null,
        pub fn reusable(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.history.items.len and n < ids.len and e.history.items[n] == ids[n]) : (n += 1) {}
            if (n >= ids.len) n = ids.len -| 1;
            return n / 512 * 512;
        }
        pub fn reusablePrompt(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.history.items.len and n < ids.len and e.history.items[n] == ids[n]) : (n += 1) {}
            if (n >= ids.len) n = ids.len -| 1;
            const grain = if (e.reuse_grain == 0) 512 else e.reuse_grain;
            return n / grain * grain;
        }
        pub fn rollback(e: *@This(), r: usize) !void {
            e.pos = r;
            e.frontier = r;
            e.history.items.len = @min(e.history.items.len, r);
        }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            if (e.park) |*pk| pk.notePrefill(e.pos, e.pos + ids.len);
            try e.history.appendSlice(e.gpa, ids);
            e.pos += ids.len;
            e.frontier = e.pos;
            return if (ids.len == 0) 0 else ids[ids.len - 1];
        }
        pub fn prefillRows(e: *@This(), ids: []const u32) !void {
            if (e.park) |*pk| pk.notePrefill(e.pos, e.pos + ids.len);
            try e.history.appendSlice(e.gpa, ids);
            e.pos += ids.len;
            e.frontier = e.pos;
        }
        pub fn appendBegin(e: *@This(), ids: []const u32) !void {
            e.append_pending.clearRetainingCapacity();
            try e.append_pending.appendSlice(e.gpa, ids[e.pos..]);
        }
        pub fn appendFinish(e: *@This()) !u32 {
            const p = e.append_pending.items;
            const first = p[0];
            try e.history.appendSlice(e.gpa, p);
            e.pos += p.len;
            e.frontier = e.pos;
            e.append_pending.clearRetainingCapacity();
            return first;
        }
        pub fn settle(e: *@This()) !void { e.pos = e.frontier; }
        pub fn parkSave(e: *@This(), lo: usize, hi: usize) !void {
            const pk = &e.park.?;
            _ = try pk.save(e.gpa, e.history.items, e.pos, e.frontier, .{ .lo = lo, .hi = hi }, 512);
        }
        pub fn parkRestore(e: *@This()) !usize {
            const pk = &e.park.?;
            const r = try pk.restore(&e.history);
            e.pos = r.frontier;
            e.frontier = r.frontier;
            return r.frontier;
        }
        pub fn draw(_: *@This(), _: usize, _: usize, _: u64, _: []u32) !void {}
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(_: *@This(), _: usize) !void {}
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32, ids: [4096]u32 = @splat(0), len: usize = 0 };
        frames: [32]Frame = undefined,
        count: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, ids: []const u32) anyerror!void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            var f: Frame = .{ .kind = @enumFromInt(h.kind), .position = h.position, .n = h.n, .len = ids.len };
            @memcpy(f.ids[0..ids.len], ids);
            r.frames[r.count] = f;
            r.count += 1;
        }
    };
    const E = FakeEngine;
    var e = E{ .gpa = std.testing.allocator };
    defer e.history.deinit(std.testing.allocator);
    defer e.append_pending.deinit(std.testing.allocator);
    const B = Adapter(E);
    var b = B.init(std.testing.allocator, &e);
    b.resident = true;
    var rec: Recorder = .{};
    b.link_ctx = &rec;
    b.send_fn = Recorder.send;

    var seed_ids: [1024]u32 = undefined;
    for (&seed_ids, 0..) |*v, i| v.* = @intCast(i + 1);
    _ = try e.prefill(&seed_ids);
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    try stream.context.appendSlice(std.testing.allocator, &seed_ids);
    stream.prompt_len = seed_ids.len;
    try B.prefill(&b, &stream);
    try std.testing.expectEqual(@as(u32, 512), stream.cached);
    try std.testing.expectEqual(follow.Kind.begin, rec.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 512), rec.frames[1].n);
    try std.testing.expectEqual(follow.Kind.prefill, rec.frames[2].kind);
    try std.testing.expectEqual(@as(u32, 512), rec.frames[2].n);
    B.release(&b, &stream);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[3].kind);
    try std.testing.expectEqual(follow.Kind.release, rec.frames[4].kind);

    // Fine reuse emits the exact rollback target and suffix; zero grain retains the 512-row behavior.
    rec.count = 0;
    b.owner = null;
    try e.rollback(0);
    _ = try e.prefill(&seed_ids);
    var fine_prompt: [900]u32 = undefined;
    @memcpy(fine_prompt[0..700], seed_ids[0..700]);
    @memset(fine_prompt[700..], 99);
    e.reuse_grain = 1;
    var fine_stream = std.mem.zeroes(Stream);
    defer fine_stream.context.deinit(std.testing.allocator);
    try fine_stream.context.appendSlice(std.testing.allocator, &fine_prompt);
    fine_stream.prompt_len = fine_prompt.len;
    try B.prefill(&b, &fine_stream);
    try std.testing.expectEqual(@as(u32, 700), fine_stream.cached);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 700), rec.frames[1].n);
    try std.testing.expectEqualSlices(u32, fine_prompt[700..], rec.frames[2].ids[0..rec.frames[2].len]);
    B.release(&b, &fine_stream);

    rec.count = 0;
    b.owner = null;
    try e.rollback(0);
    _ = try e.prefill(&seed_ids);
    e.reuse_grain = 0;
    var coarse_stream = std.mem.zeroes(Stream);
    defer coarse_stream.context.deinit(std.testing.allocator);
    try coarse_stream.context.appendSlice(std.testing.allocator, &fine_prompt);
    coarse_stream.prompt_len = fine_prompt.len;
    try B.prefill(&b, &coarse_stream);
    try std.testing.expectEqual(@as(u32, 512), coarse_stream.cached);
    B.release(&b, &coarse_stream);

    // Append sends completed pieces without a head, then the final prefill with a head.
    rec.count = 0;
    b.owner = null;
    try e.rollback(0);
    e.history.clearRetainingCapacity();
    try B.appendBegin(&b, seed_ids[0..1]);
    try B.appendTokens(&b, seed_ids[0..512]);
    _ = try B.appendFinish(&b);
    try std.testing.expectEqual(follow.Kind.prefill_rows, rec.frames[0].kind);
    try std.testing.expectEqual(@as(u32, 512), rec.frames[0].n);
    try std.testing.expectEqual(follow.Kind.prefill, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 1), rec.frames[1].n);

    rec.count = 0;
    try B.warm(&b, &seed_ids);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.prefill_rows, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 1024), rec.frames[1].n);

    var parked = E{ .gpa = std.testing.allocator, .park = .{ .rows = 4096 } };
    defer parked.history.deinit(std.testing.allocator);
    defer parked.append_pending.deinit(std.testing.allocator);
    defer parked.park.?.deinit(std.testing.allocator);
    var pb = B.init(std.testing.allocator, &parked);
    pb.resident = true;
    pb.link_ctx = &rec;
    pb.send_fn = Recorder.send;
    rec.count = 0;
    var main_ids: [2048]u32 = undefined;
    for (&main_ids, 0..) |*v, i| v.* = @intCast(i + 1);
    const side_ids: [1024]u32 = @splat(10000);
    var main_next: [2648]u32 = undefined;
    @memcpy(main_next[0..main_ids.len], &main_ids);
    for (main_next[main_ids.len..], main_ids.len..) |*v, i| v.* = @intCast(i + 1);
    var main_stream = std.mem.zeroes(Stream);
    defer main_stream.context.deinit(std.testing.allocator);
    try main_stream.context.appendSlice(std.testing.allocator, &main_ids);
    main_stream.prompt_len = main_ids.len;
    try B.prefill(&pb, &main_stream);
    B.release(&pb, &main_stream);
    var side_stream = std.mem.zeroes(Stream);
    defer side_stream.context.deinit(std.testing.allocator);
    try side_stream.context.appendSlice(std.testing.allocator, &side_ids);
    side_stream.prompt_len = side_ids.len;
    try B.prefill(&pb, &side_stream);
    B.release(&pb, &side_stream);
    var main2_stream = std.mem.zeroes(Stream);
    defer main2_stream.context.deinit(std.testing.allocator);
    try main2_stream.context.appendSlice(std.testing.allocator, &main_next);
    main2_stream.prompt_len = main_next.len;
    try B.prefill(&pb, &main2_stream);
    try std.testing.expectEqual(@as(u32, 2048), main2_stream.cached);
    try std.testing.expectEqualSlices(u32, &main_next, parked.history.items);
    try std.testing.expect(!parked.park.?.held);
    var saw_park = false;
    var saw_unpark = false;
    for (rec.frames[0..rec.count]) |frame| {
        if (frame.kind == .park) {
            saw_park = true;
            try std.testing.expectEqualSlices(u32, &.{ 0, 2048 }, frame.ids[0..2]);
        }
        if (frame.kind == .unpark) {
            saw_unpark = true;
            try std.testing.expectEqual(@as(u32, 2048), frame.ids[0]);
        }
    }
    try std.testing.expect(saw_park and saw_unpark);
    B.release(&pb, &main2_stream);
    std.testing.refAllDecls(B);
}

test "disk-on adapter emits prefill_rows and snap_save at each planned cut, then prefill" {
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        saves: [2]usize = .{ 0, 0 },
        save_count: usize = 0,
        pub fn prefillRows(e: *@This(), ids: []const u32) !void { e.pos += ids.len; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
        pub fn snapSave(e: *@This(), h: follow.Header) !void { e.saves[e.save_count] = @intCast(h.position); e.save_count += 1; }
    };
    const Hooks = struct {
        cuts: [2]prefix_cache.Cut = .{ .{ .at = 1024, .kind = .grid }, .{ .at = 2048, .kind = .grid } },
        fn lookup(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit { return null; }
        fn cutsFn(ctx: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror![]const prefix_cache.Cut { return (@as(*@This(), @ptrCast(@alignCast(ctx)))).cuts[0..]; }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(_: *anyopaque) usize { return 0; }
        fn pendingMax(_: *anyopaque) usize { return 0; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const Recorder = struct {
        frames: [8]Frame = undefined,
        len: usize = 0,
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    var hooks: Hooks = .{};
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [3000]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 2), fake.save_count);
    try std.testing.expectEqualSlices(usize, &.{ 1024, 2048 }, &fake.saves);
    try std.testing.expectEqual(@as(usize, 6), recorder.len);
    try std.testing.expectEqual(follow.Kind.begin, recorder.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.prefill_rows, recorder.frames[1].kind);
    try std.testing.expectEqual(follow.Kind.snap_save, recorder.frames[2].kind);
    try std.testing.expectEqual(follow.Kind.prefill_rows, recorder.frames[3].kind);
    try std.testing.expectEqual(follow.Kind.snap_save, recorder.frames[4].kind);
    try std.testing.expectEqual(follow.Kind.prefill, recorder.frames[5].kind);
    try std.testing.expectEqual(@as(usize, 2048), recorder.frames[5].position);
}

test "fine reuse aligns a disk snapshot load to the 512-row boundary" {
    const FakeEngine = struct {
        pos: usize = 10,
        history: [20000]u32 = @splat(0),
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn reusablePrompt(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.pos and n < ids.len and e.history[n] == ids[n]) : (n += 1) {}
            return n;
        }
        pub fn rollback(e: *@This(), at: usize) !void { e.pos = at; }
        pub fn snapLoad(e: *@This(), h: follow.Header, _: []const u32) !usize { e.pos = @intCast(h.position + 512); return e.pos; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
    };
    const Hooks = struct {
        fn lookup(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit { return .{ .at = 512, .key = 0xabc }; }
        fn cutsFn(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror![]const prefix_cache.Cut { return &.{}; }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(_: *anyopaque) usize { return 0; }
        fn pendingMax(_: *anyopaque) usize { return 0; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const Recorder = struct {
        frames: [8]Frame = undefined,
        len: usize = 0,
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    for (&fake.history, 0..) |*id, i| id.* = @intCast(i + 1);
    var hooks: Hooks = .{};
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.resident = true;
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [1024]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    var rollback_index: ?usize = null;
    var load_index: ?usize = null;
    for (recorder.frames[0..recorder.len], 0..) |frame, i| {
        if (frame.kind == .rollback and frame.n == 0) rollback_index = i;
        if (frame.kind == .snap_load) {
            load_index = i;
            try std.testing.expectEqual(@as(usize, 0), frame.position);
        }
    }
    try std.testing.expect(rollback_index != null and load_index != null);
    try std.testing.expect(rollback_index.? < load_index.?);
}

test "fine reuse aligns a save cut before prefill_rows" {
    const FakeEngine = struct {
        pos: usize = 10,
        history: [20000]u32 = @splat(0),
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn reusablePrompt(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.pos and n < ids.len and e.history[n] == ids[n]) : (n += 1) {}
            return n;
        }
        pub fn rollback(e: *@This(), at: usize) !void { e.pos = at; }
        pub fn prefillRows(e: *@This(), ids: []const u32) !void { e.pos += ids.len; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
        pub fn snapSave(_: *@This(), _: follow.Header) !void {}
    };
    const Hooks = struct {
        const cuts = [_]prefix_cache.Cut{.{ .at = 16384, .kind = .grid }};
        fn lookup(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit { return null; }
        fn cutsFn(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror![]const prefix_cache.Cut { return &cuts; }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(_: *anyopaque) usize { return 0; }
        fn pendingMax(_: *anyopaque) usize { return 0; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        frames: [8]Frame = undefined,
        len: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    for (&fake.history, 0..) |*id, i| id.* = @intCast(i + 1);
    var hooks: Hooks = .{};
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.resident = true;
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [20000]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    var rollback_index: ?usize = null;
    var rows_index: ?usize = null;
    for (recorder.frames[0..recorder.len], 0..) |frame, i| {
        if (frame.kind == .rollback and frame.n == 0) rollback_index = i;
        if (frame.kind == .prefill_rows) {
            rows_index = i;
            try std.testing.expectEqual(@as(usize, 0), frame.position);
        }
    }
    try std.testing.expect(rollback_index != null and rows_index != null);
    try std.testing.expect(rollback_index.? < rows_index.?);
}

test "fine reuse keeps ten resident rows when there is no save cut" {
    const FakeEngine = struct {
        pos: usize = 10,
        history: [20]u32 = @splat(0),
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn reusablePrompt(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.pos and n < ids.len and e.history[n] == ids[n]) : (n += 1) {}
            return n;
        }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
    };
    const Recorder = struct {
        frames: [4]Frame = undefined,
        len: usize = 0,
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    for (&fake.history, 0..) |*id, i| id.* = @intCast(i + 1);
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.resident = true;
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    try std.testing.expectEqual(@as(u32, 10), stream.cached);
    for (recorder.frames[0..recorder.len]) |frame| try std.testing.expect(frame.kind != .rollback);
}

test "disk-on adapter commits a lookup hit and plans cuts from the restored row" {
    const FakeEngine = struct {
        pos: usize = 1024,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        commits: usize = 0,
        loads: usize = 0,
        saves: usize = 0,
        pub fn reusable(_: *@This(), _: []const u32) usize { return 1024; }
        pub fn rollback(e: *@This(), at: usize) !void { e.pos = at; }
        pub fn snapCommit(e: *@This(), _: usize) !void { e.commits += 1; }
        pub fn snapLoad(e: *@This(), _: follow.Header, _: []const u32) !usize { e.loads += 1; e.pos = 2048; return 2048; }
        pub fn prefillRows(e: *@This(), ids: []const u32) !void { e.pos += ids.len; }
        pub fn snapSave(e: *@This(), _: follow.Header) !void { e.saves += 1; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
    };
    const Hooks = struct {
        cuts: [1]prefix_cache.Cut = .{.{ .at = 3072, .kind = .history }},
        fn lookup(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit { return .{ .at = 2048, .key = 0xabc }; }
        fn cutsFn(ctx: *anyopaque, _: []const u32, start: usize, _: *Stream) anyerror![]const prefix_cache.Cut {
            const hooks: *@This() = @ptrCast(@alignCast(ctx));
            if (start != 2048) return error.WrongStart;
            return &hooks.cuts;
        }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(_: *anyopaque) usize { return 1; }
        fn pendingMax(_: *anyopaque) usize { return 3072; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        frames: [8]Frame = undefined,
        len: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    var hooks: Hooks = .{};
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.resident = true;
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [4096]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 1), fake.commits);
    try std.testing.expectEqual(@as(usize, 1), fake.loads);
    try std.testing.expectEqual(@as(usize, 1), fake.saves);
    try std.testing.expectEqual(@as(u32, 2048), stream.cached);
    try std.testing.expectEqual(@as(usize, 6), recorder.len);
    try std.testing.expectEqual(follow.Kind.begin, recorder.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.snap_commit, recorder.frames[1].kind);
    try std.testing.expectEqual(follow.Kind.snap_load, recorder.frames[2].kind);
    try std.testing.expectEqual(@as(usize, 1024), recorder.frames[2].position);
    try std.testing.expectEqual(follow.Kind.prefill_rows, recorder.frames[3].kind);
    try std.testing.expectEqual(@as(usize, 2048), recorder.frames[3].position);
    try std.testing.expectEqual(follow.Kind.snap_save, recorder.frames[4].kind);
    try std.testing.expectEqual(follow.Kind.prefill, recorder.frames[5].kind);
}

test "disk-on adapter commits pending saves at request end, before the release frame" {
    const Hooks = struct {
        count: usize = 0,
        fn lookup(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit { return null; }
        fn cutsFn(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror![]const prefix_cache.Cut { return &.{}; }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(ctx: *anyopaque) usize { return (@as(*@This(), @ptrCast(@alignCast(ctx)))).count; }
        fn pendingMax(_: *anyopaque) usize { return 0; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        commits: usize = 0,
        hooks: *Hooks,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
        pub fn snapCommit(e: *@This(), n: usize) !void { e.commits += n; e.hooks.count = 0; }
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        frames: [8]Frame = undefined,
        len: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var hooks: Hooks = .{};
    var fake: FakeEngine = .{ .hooks = &hooks };
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [2000]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    hooks.count = 1; // one save is waiting for its writer when the reply ends
    B.release(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 1), fake.commits);
    try std.testing.expectEqual(@as(usize, 4), recorder.len);
    try std.testing.expectEqual(follow.Kind.begin, recorder.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.prefill, recorder.frames[1].kind);
    try std.testing.expectEqual(follow.Kind.snap_commit, recorder.frames[2].kind);
    try std.testing.expectEqual(@as(u32, 1), recorder.frames[2].n);
    try std.testing.expectEqual(follow.Kind.release, recorder.frames[3].kind);
    // Nothing pending: release sends no commit.
    try B.prefill(&adapter, &stream);
    B.release(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 1), fake.commits);
    try std.testing.expectEqual(@as(usize, 7), recorder.len);
    for (recorder.frames[4..7]) |frame| try std.testing.expect(frame.kind != .snap_commit);
}

test "disk-on adapter looks up again when the commit point drops the pending hit" {
    const Hooks = struct {
        alive: bool = true,
        count: usize = 1,
        lookups: usize = 0,
        fn lookup(ctx: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror!?DiskHit {
            const h: *@This() = @ptrCast(@alignCast(ctx));
            h.lookups += 1;
            if (!h.alive) return null;
            return .{ .at = 2048, .key = 0xabc };
        }
        fn cutsFn(_: *anyopaque, _: []const u32, _: usize, _: *Stream) anyerror![]const prefix_cache.Cut { return &.{}; }
        fn header(_: *anyopaque, at: usize, kind: prefix_cache.CutKind) anyerror!follow.Header { return .{ .kind = @backingInt(follow.Kind.snap_save), .slot = @intFromEnum(kind), .position = at }; }
        fn pending(ctx: *anyopaque) usize { return (@as(*@This(), @ptrCast(@alignCast(ctx)))).count; }
        fn pendingMax(_: *anyopaque) usize { return 2048; }
        fn idle(_: *anyopaque) bool { return false; }
    };
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        loads: usize = 0,
        hooks: *Hooks,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
        pub fn snapCommit(e: *@This(), _: usize) !void { e.hooks.alive = false; e.hooks.count = 0; }
        pub fn snapLoad(e: *@This(), _: follow.Header, _: []const u32) !usize { e.loads += 1; e.pos = 2048; return 2048; }
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32 };
        frames: [8]Frame = undefined,
        len: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.frames[rec.len] = .{ .kind = @enumFromInt(h.kind), .position = @intCast(h.position), .n = h.n };
            rec.len += 1;
        }
    };
    var hooks: Hooks = .{};
    var fake: FakeEngine = .{ .hooks = &hooks };
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    adapter.disk = .{ .ctx = &hooks, .lookup_fn = Hooks.lookup, .cuts_fn = Hooks.cutsFn, .save_header_fn = Hooks.header, .pending_fn = Hooks.pending, .pending_max_fn = Hooks.pendingMax, .writer_idle_fn = Hooks.idle, .max_pending = 14 };
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    var ids: [3000]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 1);
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 2), hooks.lookups);
    try std.testing.expectEqual(@as(usize, 0), fake.loads);
    try std.testing.expectEqual(@as(u32, 0), stream.cached);
    try std.testing.expectEqual(@as(usize, 3), recorder.len);
    try std.testing.expectEqual(follow.Kind.begin, recorder.frames[0].kind);
    try std.testing.expectEqual(follow.Kind.snap_commit, recorder.frames[1].kind);
    try std.testing.expectEqual(follow.Kind.prefill, recorder.frames[2].kind);
}

test "disk-off adapter emits no snapshot frames or hooks" {
    const FakeEngine = struct {
        pos: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        calls: usize = 0,
        pub fn snapSave(e: *@This(), _: follow.Header) !void { e.calls += 1; }
        pub fn snapLoad(e: *@This(), _: follow.Header, _: []const u32) !usize { e.calls += 1; return 0; }
        pub fn snapCommit(e: *@This(), _: usize) !void { e.calls += 1; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return ids[ids.len - 1]; }
    };
    const Recorder = struct {
        kinds: [4]follow.Kind = undefined,
        len: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const rec: *@This() = @ptrCast(@alignCast(ctx));
            rec.kinds[rec.len] = @enumFromInt(h.kind);
            rec.len += 1;
        }
    };
    var fake: FakeEngine = .{};
    const B = Adapter(FakeEngine);
    var adapter = B.init(std.testing.allocator, &fake);
    var recorder: Recorder = .{};
    adapter.link_ctx = &recorder;
    adapter.send_fn = Recorder.send;
    var stream = std.mem.zeroes(Stream);
    defer stream.context.deinit(std.testing.allocator);
    const ids = [_]u32{ 1, 2, 3, 4 };
    try stream.context.appendSlice(std.testing.allocator, &ids);
    stream.prompt_len = ids.len;
    try B.prefill(&adapter, &stream);
    try std.testing.expectEqual(@as(usize, 0), fake.calls);
    for (recorder.kinds[0..recorder.len]) |kind| try std.testing.expect(kind != .snap_save and kind != .snap_load and kind != .snap_commit);
}

test "adapter commits every single-row lane round and advances the engine" {
    const core = @import("lanes");
    const FakeEngine = struct {
        pos: usize = 0, max_pos: usize = 0, head_rows: usize = 1, seed: u64 = 0, rule: sampler.Sampling = .{ .temperature = 0 },
        history: std.ArrayList(u32) = .empty,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { try e.history.appendSlice(std.testing.allocator, ids); e.pos += ids.len; return next(e.history.items); }
        pub fn draw(e: *@This(), _: usize, _: usize, _: u64, out: []u32) !void { out[0] = next(e.history.items); }
        pub fn verify(e: *@This(), ids: []const u32, out: []u32) !void { for (ids, 0..) |id, i| { try e.history.append(std.testing.allocator, id); out[i] = next(e.history.items); } }
        pub fn keep(e: *@This(), rows: usize) !void { e.pos += rows; if (e.pos > e.max_pos) e.max_pos = e.pos; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; e.history.shrinkRetainingCapacity(to); }
        fn next(h: []const u32) u32 { var sum: u32 = 0; for (h) |x| sum +%= x; return sum % 89; }
    };
    const gpa = std.testing.allocator;
    var cfg = try core.Config.init(gpa, .{ .exact_width = 1, .first_copy_rows = 1, .batch_rows = 1 }, 8, 7);
    defer cfg.deinit(gpa);
    var target: FakeEngine = .{};
    defer target.history.deinit(gpa);
    const B = Adapter(FakeEngine);
    var adapter = B.init(gpa, &target);
    const DummyClock = struct {
        fn start(_: *anyopaque) void {}
        fn elapsed(_: *anyopaque, _: core.backend.Mark) f64 { return 1; }
        fn clock(ctx: *@This()) core.backend.Clock { return .{ .ptr = ctx, .vtable = &.{ .start = start, .elapsed_ms = elapsed } }; }
    };
    var clock: DummyClock = .{};
    var engine = core.Engine.init(gpa, &cfg, adapter.backend(), clock.clock());
    defer engine.deinit();
    var stream = try core.Stream.init(gpa, .{ .id = "adapter-regression", .prompt = &.{ 1, 2, 3 }, .max_new = 12, .drafts = false });
    defer stream.deinit(gpa);
    try engine.addStream(&stream);
    while (engine.activeCount() > 0) try engine.step();
    try std.testing.expectEqual(@as(usize, 14), target.max_pos);
    try std.testing.expectEqual(@as(usize, 12), stream.emitted().len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &.{ 1, 2, 3 });
    for (stream.emitted()) |token| { try std.testing.expectEqual(FakeEngine.next(history.items), token); try history.append(gpa, token); }
}

test "drafting adapter: block draft fills held drafts and the window verifies them" {
    const FakeEngine = struct {
        pos: usize = 0,
        pub fn draftBlock(_: *@This(), _: usize, _: usize, _: u32, out: ?[]u32, probs: ?[]f64) !void {
            const dst = out orelse return error.MissingDraftOutput;
            for (dst, 0..) |*id, i| id.* = @intCast(100 + i);
            if (probs) |p| @memcpy(p[0..7], &[_]f64{ 0.9, 0.8, 0.5, 0.5, 0.5, 0.5, 0.5 });
        }
        pub fn verify(_: *@This(), ids: []const u32, out: []u32) !void {
            @memcpy(out[0..ids.len], ids);
        }
        pub fn keep(e: *@This(), n: usize) !void {
            e.pos += n;
        }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    try std.testing.expect(B.facts(&b).draft_probabilities);
    // C1 prices rows with the measured GB10 table: widths 1..16 ascending in width and ms.
    const costs = B.facts(&b).window_costs;
    try std.testing.expectEqual(@as(usize, 16), costs.len);
    for (costs, 1..) |c, w| {
        try std.testing.expectEqual(@as(u32, @intCast(w)), c.width);
        if (w > 1) try std.testing.expect(c.ms > costs[w - 2].ms);
    }
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    try B.draft(&b, &.{request});
    try std.testing.expectEqual(@as(usize, 7), b.held_n);
    var chances: [9]f64 = @splat(-1);
    try std.testing.expect(try B.chancesFn(&b, &stream, &chances));
    const expected = [_]f64{ 0.9, 0.72, 0.36, 0.18, 0.09, 0.045, 0.0225, 0, 0 };
    for (expected, chances) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-12);
    const positions = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const window: be.Window = .{ .stream = &stream, .pending = 9, .held = 7, .tokens = &.{}, .parents = null, .positions = &positions };
    var sampled: [8]u32 = undefined;
    var copied: [7]u32 = undefined;
    var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &copied }};
    try B.verify(&b, &.{window}, &verified);
    try std.testing.expectEqualSlices(u32, &.{ 9, 100, 101, 102, 103, 104, 105, 106 }, &sampled);
    try std.testing.expectEqualSlices(u32, &.{ 100, 101, 102, 103, 104, 105, 106 }, &copied);
    try std.testing.expectEqual(@as(u64, 8), b.verified_rows);
}

test "drafting adapter: chances switch off passes null and declines probabilities" {
    const FakeEngine = struct {
        saw_null: bool = false,
        pub fn draftBlock(e: *@This(), _: usize, _: usize, _: u32, _: ?[]u32, probs: ?[]f64) !void {
            e.saw_null = probs == null;
        }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    try B.draft(&b, &.{request});
    try std.testing.expect(engine.saw_null);
    try std.testing.expect(!B.facts(&b).draft_probabilities);
    try std.testing.expectEqual(@as(usize, 0), B.facts(&b).window_costs.len);
    var out: [7]f64 = undefined;
    try std.testing.expect(!(try B.chancesFn(&b, &stream, &out)));

    b.draft_chances = true;
    var other = std.mem.zeroes(Stream);
    try std.testing.expect(!(try B.chancesFn(&b, &other, &out)));
    b.held_n = 0;
    try std.testing.expect(!(try B.chancesFn(&b, &stream, &out)));
}

test "drafting adapter: recal off keeps the 7-slot buffer and the GB10 table" {
    const FakeEngine = struct {
        probs_len: usize = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn draftBlock(e: *@This(), _: usize, _: usize, _: u32, out: ?[]u32, probs: ?[]f64) !void {
            const dst = out orelse return error.MissingDraftOutput;
            for (dst, 0..) |*id, i| id.* = @intCast(100 + i);
            e.probs_len = if (probs) |p| p.len else 0;
            if (probs) |p| {
                const zs = [_]f64{ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0 };
                for (0..8) |j| {
                    if (j < p.len) p[j] = 1.0 / (1.0 + @exp(-zs[j]));
                    if (p.len >= 32) {
                        p[8 + j] = zs[j];
                        p[16 + j] = 1.0;
                        p[24 + j] = -0.5;
                    }
                }
            }
        }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    try B.draft(&b, &.{request});
    try std.testing.expectEqual(@as(usize, 7), engine.probs_len);
    try std.testing.expect(B.facts(&b).window_costs.ptr == @as([*]const lanes.config.Cost, &gb10_verify_costs));
    var out: [7]f64 = undefined;
    try std.testing.expect(try B.chancesFn(&b, &stream, &out));
    const zs = [_]f64{ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0 };
    var chain: f64 = 1;
    for (0..7) |j| {
        chain *= 1.0 / (1.0 + @exp(-zs[j]));
        try std.testing.expectApproxEqAbs(chain, out[j], 1e-15);
    }
}

test "drafting adapter: recal costs replace the window and copy tables" {
    const FakeEngine = struct {};
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    b.copyd = true;
    var table: [16]lanes.config.Cost = undefined;
    for (&table, 1..) |*c, w| c.* = .{ .width = @intCast(w), .ms = 50.0 + 10.0 * @as(f64, @floatFromInt(w)) };
    b.recal_costs = table;
    const m = B.facts(&b);
    try std.testing.expectEqual(@as(usize, 16), m.window_costs.len);
    try std.testing.expectEqual(@as(f64, 60), m.window_costs[0].ms);
    try std.testing.expectEqual(@as(f64, 210), m.window_costs[15].ms);
    try std.testing.expectEqual(@as(usize, 16), m.copy_costs.len);
    try std.testing.expectEqual(@as(f64, 90), m.copy_costs[3].ms);
    b.recal_costs = null;
    try std.testing.expectEqual(@as(f64, 88.0), B.facts(&b).window_costs[0].ms);
}

test "drafting adapter: recal conf identity equals off; weights move chances, never drafts" {
    const FakeEngine = struct {
        probs_len: usize = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn draftBlock(e: *@This(), _: usize, _: usize, _: u32, out: ?[]u32, probs: ?[]f64) !void {
            const dst = out orelse return error.MissingDraftOutput;
            for (dst, 0..) |*id, i| id.* = @intCast(100 + i);
            e.probs_len = if (probs) |p| p.len else 0;
            if (probs) |p| {
                const zs = [_]f64{ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0 };
                for (0..8) |j| {
                    if (j < p.len) p[j] = 1.0 / (1.0 + @exp(-zs[j]));
                    if (p.len >= 32) {
                        p[8 + j] = zs[j];
                        p[16 + j] = 1.0;
                        p[24 + j] = -0.5;
                    }
                }
            }
        }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    try B.draft(&b, &.{request});
    var off: [7]f64 = undefined;
    try std.testing.expect(try B.chancesFn(&b, &stream, &off));
    const held_off = b.held;
    var ident: recal.ConfSets = undefined;
    for (&ident) |*mode| {
        for (mode) |*row| row.* = .{ 1, 0, 0, 0 };
    }
    b.recal_conf = ident;
    try B.draft(&b, &.{request});
    var same: [7]f64 = undefined;
    try std.testing.expect(try B.chancesFn(&b, &stream, &same));
    try std.testing.expectEqual(@as(usize, 32), engine.probs_len);
    try std.testing.expectEqualSlices(f64, &off, &same);
    try std.testing.expectEqualSlices(u32, &held_off, &b.held);
    var sets: recal.ConfSets = undefined;
    for (0..recal.slots) |slot| {
        sets[0][slot] = .{ 0, 0, 0, 0 };
        sets[1][slot] = .{ 1, 0, 0, 0 };
    }
    b.recal_conf = sets;
    engine.rule.temperature = 0;
    try B.draft(&b, &.{request});
    var greedy: [7]f64 = undefined;
    try std.testing.expect(try B.chancesFn(&b, &stream, &greedy));
    var survival: f64 = 1;
    for (&greedy) |chance| {
        survival *= 0.5;
        try std.testing.expectApproxEqAbs(survival, chance, 1e-15);
    }
    try std.testing.expectEqualSlices(u32, &held_off, &b.held);
    engine.rule.temperature = 1.0;
    try B.draft(&b, &.{request});
    var sampled: [7]f64 = undefined;
    try std.testing.expect(try B.chancesFn(&b, &stream, &sampled));
    try std.testing.expectEqualSlices(f64, &off, &sampled);
    try std.testing.expectEqualSlices(u32, &held_off, &b.held);
}

test "drafting adapter: recal sweep cycles widths through the chances" {
    const FakeEngine = struct {
        probs_len: usize = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn draftBlock(e: *@This(), _: usize, _: usize, _: u32, out: ?[]u32, probs: ?[]f64) !void {
            const dst = out orelse return error.MissingDraftOutput;
            for (dst, 0..) |*id, i| id.* = @intCast(100 + i);
            e.probs_len = if (probs) |p| p.len else 0;
            if (probs) |p| {
                const zs = [_]f64{ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0 };
                for (0..8) |j| {
                    if (j < p.len) p[j] = 1.0 / (1.0 + @exp(-zs[j]));
                    if (p.len >= 32) {
                        p[8 + j] = zs[j];
                        p[16 + j] = 1.0;
                        p[24 + j] = -0.5;
                    }
                }
            }
        }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    b.sweep = true;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    for (0..9) |round| {
        try B.draft(&b, &.{request});
        var out: [9]f64 = undefined;
        try std.testing.expect(try B.chancesFn(&b, &stream, &out));
        const leading: usize = round % 8;
        for (out, 0..) |chance, i| try std.testing.expectEqual(@as(f64, if (i < leading) 1.0 else 0.0), chance);
        try std.testing.expectEqual(@as(usize, 7), b.held_n);
    }
}

test "drafting adapter: recal trace writes without changing the round" {
    const FakeEngine = struct {
        pos: usize = 0,
        probs_len: usize = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        pub fn draftBlock(e: *@This(), _: usize, _: usize, _: u32, out: ?[]u32, probs: ?[]f64) !void {
            const dst = out orelse return error.MissingDraftOutput;
            for (dst, 0..) |*id, i| id.* = @intCast(100 + i);
            e.probs_len = if (probs) |p| p.len else 0;
            if (probs) |p| {
                const zs = [_]f64{ 2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0, -3.0 };
                for (0..8) |j| {
                    if (j < p.len) p[j] = 1.0 / (1.0 + @exp(-zs[j]));
                    if (p.len >= 32) {
                        p[8 + j] = zs[j];
                        p[16 + j] = 1.0;
                        p[24 + j] = -0.5;
                    }
                }
            }
        }
        pub fn verify(_: *@This(), ids: []const u32, out: []u32) !void { @memcpy(out[0..ids.len], ids); }
        pub fn keep(e: *@This(), n: usize) !void { e.pos += n; }
    };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    b.draft_chances = true;
    b.trace_path = "/dev/null";
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const request: be.DraftRequest = .{ .stream = &stream, .follow = &.{}, .first = .{ .value = 42 }, .rows = null, .start = 0, .position = 1, .depth = 7 };
    try B.draft(&b, &.{request});
    const positions = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const window: be.Window = .{ .stream = &stream, .pending = 9, .held = 7, .tokens = &.{}, .parents = null, .positions = &positions };
    var sampled: [8]u32 = undefined;
    var copied: [7]u32 = undefined;
    var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &copied }};
    try B.verify(&b, &.{window}, &verified);
    try std.testing.expectEqualSlices(u32, &.{ 9, 100, 101, 102, 103, 104, 105, 106 }, &sampled);
    try std.testing.expectEqualSlices(u32, &.{ 100, 101, 102, 103, 104, 105, 106 }, &copied);
    B.release(&b, &stream);
    try std.testing.expectEqual(@as(usize, 32), engine.probs_len);
    try std.testing.expect(b.trace_opened);
    try std.testing.expectEqual(@as(u64, 1), b.trace_req);
}

test "drafting adapter: live costs turn on lone pricing only with chances" {
    const FakeEngine = struct {
        pub fn draftBlock(_: *@This(), _: usize, _: usize, _: u32, _: ?[]u32, _: ?[]f64) !void {}
    };
    const FakeE = struct { alone: bool, cfg: *const lanes.Config };
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.drafting = true;
    var stream = std.mem.zeroes(Stream);
    const plan: lanes.windows.Plan = .{ .stream = &stream, .position = 1, .kind = .head, .held = 7 };

    b.draft_chances = false;
    b.live_costs = false;
    const no_chances_off = B.facts(&b);
    try std.testing.expect(!no_chances_off.plain_guard and no_chances_off.lone_cost_tile == 0);
    var cfg_no_chances_off = try lanes.Config.init(std.testing.allocator, no_chances_off, 16, 15);
    defer cfg_no_chances_off.deinit(std.testing.allocator);
    try std.testing.expect(!lanes.lone_costs.enabled(FakeE{ .alone = true, .cfg = &cfg_no_chances_off }, plan));

    b.live_costs = true;
    const no_chances_on = B.facts(&b);
    try std.testing.expect(!no_chances_on.plain_guard and no_chances_on.lone_cost_tile == 0);
    var cfg_no_chances_on = try lanes.Config.init(std.testing.allocator, no_chances_on, 16, 15);
    defer cfg_no_chances_on.deinit(std.testing.allocator);
    try std.testing.expect(!lanes.lone_costs.enabled(FakeE{ .alone = true, .cfg = &cfg_no_chances_on }, plan));

    b.draft_chances = true;
    b.live_costs = false;
    const chances_off = B.facts(&b);
    try std.testing.expect(!chances_off.plain_guard and chances_off.lone_cost_tile == 0);
    var cfg_chances_off = try lanes.Config.init(std.testing.allocator, chances_off, 16, 15);
    defer cfg_chances_off.deinit(std.testing.allocator);
    try std.testing.expect(!lanes.lone_costs.enabled(FakeE{ .alone = true, .cfg = &cfg_chances_off }, plan));

    b.live_costs = true;
    const both_on = B.facts(&b);
    try std.testing.expect(both_on.plain_guard);
    try std.testing.expectEqual(@as(u32, 2), both_on.lone_cost_tile);
    try std.testing.expect(both_on.draft_probabilities);
    try std.testing.expectEqual(@as(usize, 16), both_on.window_costs.len);
    var cfg_both_on = try lanes.Config.init(std.testing.allocator, both_on, 16, 15);
    defer cfg_both_on.deinit(std.testing.allocator);
    try std.testing.expect(cfg_both_on.timed and cfg_both_on.own_prices and cfg_both_on.plain_guard and cfg_both_on.lone_cost_tile == 2 and cfg_both_on.draft_confidence == null);
    try std.testing.expect(lanes.lone_costs.enabled(FakeE{ .alone = true, .cfg = &cfg_both_on }, plan));
    try std.testing.expect(!lanes.lone_costs.enabled(FakeE{ .alone = false, .cfg = &cfg_both_on }, plan));
    var copy_plan = plan;
    copy_plan.kind = .copy;
    try std.testing.expect(!lanes.lone_costs.enabled(FakeE{ .alone = true, .cfg = &cfg_both_on }, copy_plan));
    try std.testing.expectEqual(@as(f64, 191.8), cfg_both_on.shared_costs.get(8).?);
}

test "adapter copy arbitration prices both branches without changing window costs" {
    const FakeEngine = struct {};
    const B = Adapter(FakeEngine);
    var engine: FakeEngine = .{};
    var b = B.init(std.testing.allocator, &engine);
    b.copy_rows = 8;
    const plain_off = B.facts(&b);
    try std.testing.expect(!plain_off.copy_arbitrate);
    try std.testing.expectEqual(@as(usize, 0), plain_off.copy_costs.len);
    b.copyd = true;
    const plain_on = B.facts(&b);
    try std.testing.expect(plain_on.copy_arbitrate);
    try std.testing.expectEqual(@as(usize, 16), plain_on.copy_costs.len);
    try std.testing.expectEqual(@as(usize, 0), plain_on.window_costs.len);
    b.drafting = true;
    b.copyd = false;
    const draft_off = B.facts(&b);
    try std.testing.expect(!draft_off.copy_arbitrate);
    try std.testing.expectEqual(@as(usize, 0), draft_off.copy_costs.len);
    b.copyd = true;
    const draft_on = B.facts(&b);
    try std.testing.expect(draft_on.copy_arbitrate);
    try std.testing.expectEqual(@as(usize, 16), draft_on.copy_costs.len);
    try std.testing.expectEqual(@as(usize, 0), draft_on.window_costs.len);
    try std.testing.expect(!draft_on.draft_probabilities);
}

test "adapter commits verify windows then rolls back only the unaccepted suffix" {
    const TestEngine = struct {
        pos: usize = 10,
        pub fn verify(_: *@This(), ids: []const u32, out: []u32) !void { for (ids, 0..) |id, i| out[i] = id + 100; }
        pub fn keep(e: *@This(), rows: usize) !void { e.pos += rows; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    const Capture = struct {
        frames: [8]follow.Header = undefined,
        n: usize = 0,
        fn send(ptr: *anyopaque, h: follow.Header, ids: []const u32) anyerror!void { _ = ids; const c: *@This() = @ptrCast(@alignCast(ptr)); c.frames[c.n] = h; c.n += 1; }
    };
    var target: TestEngine = .{};
    var capture: Capture = .{};
    const B = Adapter(TestEngine);
    var b = B.init(std.testing.allocator, &target);
    b.send_fn = Capture.send;
    b.link_ctx = &capture;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    const positions = [_]u64{ 11, 12, 13, 14 };
    const tokens = [_]u32{ 2, 3, 4 };
    const w = be.Window{ .stream = &stream, .pending = 1, .held = 0, .tokens = &tokens, .parents = null, .positions = &positions };
    var sampled: [4]u32 = undefined;
    var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
    try B.verify(&b, &.{w}, &verified);
    try std.testing.expectEqual(@as(usize, 14), target.pos);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.verify)), capture.frames[0].kind);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.keep)), capture.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 4), capture.frames[1].n);
    const accepted = [_]u32{ 10, 11 };
    try B.keep(&b, &.{w}, &.{&accepted});
    try std.testing.expectEqual(@as(usize, 12), target.pos);
    try std.testing.expectEqual(@as(u32, 9), capture.frames[2].kind);
    try std.testing.expectEqual(@as(u32, 12), capture.frames[2].n);
    const one_pos = [_]u64{13};
    const one = be.Window{ .stream = &stream, .pending = 5, .held = 0, .tokens = &.{}, .parents = null, .positions = &one_pos };
    capture.n = 0;
    try B.verify(&b, &.{one}, &verified);
    const one_accepted = [_]u32{6};
    try B.keep(&b, &.{one}, &.{&one_accepted});
    try std.testing.expectEqual(@as(usize, 13), target.pos);
    try std.testing.expectEqual(@as(usize, 2), capture.n);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.verify)), capture.frames[0].kind);
    try std.testing.expectEqual(@as(u32, @backingInt(follow.Kind.keep)), capture.frames[1].kind);
    var bad = one;
    const wrong_pos = [_]u64{99};
    bad.positions = &wrong_pos;
    try std.testing.expectError(error.PositionMismatch, B.verify(&b, &.{bad}, &verified));
}

test "16-row adapter verifies 12 and 16 and refuses 17 before emitting a frame" {
    const TestEngine = struct {
        pos: usize = 0,
        pub fn verify(_: *@This(), ids: []const u32, out: []u32) !void { @memcpy(out[0..ids.len], ids); }
        pub fn keep(e: *@This(), n: usize) !void { e.pos += n; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    const Capture = struct {
        n: usize = 0,
        fn send(ptr: *anyopaque, _: follow.Header, _: []const u32) anyerror!void { const c: *@This() = @ptrCast(@alignCast(ptr)); c.n += 1; }
    };
    const B = Adapter(TestEngine);
    var target: TestEngine = .{};
    var b = B.init(std.testing.allocator, &target);
    b.verify_max = 16;
    var capture: Capture = .{};
    b.link_ctx = &capture;
    b.send_fn = Capture.send;
    var stream = std.mem.zeroes(Stream);
    b.owner = &stream;
    var positions: [17]u64 = undefined;
    var tokens: [16]u32 = undefined;
    var sampled: [follow.max_decode_rows]u32 = undefined;
    for (0..17) |rows| {
        if (rows != 12 and rows != 16) continue;
        for (0..rows) |i| positions[i] = @intCast(target.pos + i + 1);
        for (&tokens, 0..) |*token, i| token.* = @intCast(i + 10);
        const w: be.Window = .{ .stream = &stream, .pending = 1, .held = 0, .tokens = tokens[0 .. rows - 1], .parents = null, .positions = positions[0..rows] };
        var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
        try B.verify(&b, &.{w}, &verified);
        const accepted = [_]u32{1};
        try B.keep(&b, &.{w}, &.{&accepted});
    }
    // per pass: verify, keep and the rollback of the unkept rows (one row kept of 12 / 16)
    try std.testing.expectEqual(@as(usize, 6), capture.n);
    const before = capture.n;
    for (0..17) |i| positions[i] = @intCast(i + 1);
    const too_wide: be.Window = .{ .stream = &stream, .pending = 1, .held = 0, .tokens = &tokens, .parents = null, .positions = &positions };
    var verified = [_]be.Verified{.{ .sampled = &sampled, .drafts = &.{} }};
    try std.testing.expectError(error.WindowTooWide, B.verify(&b, &.{too_wide}, &verified));
    try std.testing.expectEqual(before, capture.n);
}

test "an ablated request never shares resident rows with a plain one" {
    const FakeEngine = struct {
        gpa: std.mem.Allocator,
        pos: usize = 0,
        frontier: usize = 0,
        head_rows: usize = 1,
        seed: u64 = 0,
        rule: sampler.Sampling = .{ .temperature = 0 },
        history: std.ArrayList(u32) = .empty,
        abl: ?*const u8 = null,
        ablate_on: bool = false,
        pub fn setAblate(e: *@This(), on: bool) !void {
            if (on and e.abl == null) return error.NoAblation;
            if (on != e.ablate_on and e.pos != 0) return error.AblateWithResidentRows;
            e.ablate_on = on;
        }
        pub fn reusable(e: *@This(), ids: []const u32) usize {
            var n: usize = 0;
            while (n < e.history.items.len and n < ids.len and e.history.items[n] == ids[n]) : (n += 1) {}
            if (n >= ids.len) n = ids.len -| 1;
            return n / 512 * 512;
        }
        pub fn rollback(e: *@This(), r: usize) !void {
            e.pos = r;
            e.frontier = @min(e.frontier, r);
            e.history.items.len = @min(e.history.items.len, r);
        }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            try e.history.appendSlice(e.gpa, ids);
            e.pos += ids.len;
            e.frontier = e.pos;
            return 0;
        }
        pub fn settle(e: *@This()) !void {
            try e.rollback(e.frontier);
        }
    };
    const Recorder = struct {
        const Frame = struct { kind: follow.Kind, position: usize, n: u32, slot: u32 };
        frames: [16]Frame = undefined,
        count: usize = 0,
        fn send(ctx: *anyopaque, h: follow.Header, _: []const u32) anyerror!void {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            r.frames[r.count] = .{ .kind = @enumFromInt(h.kind), .position = h.position, .n = h.n, .slot = h.slot };
            r.count += 1;
        }
    };
    var marker: u8 = 0;
    var e = FakeEngine{ .gpa = std.testing.allocator, .abl = &marker };
    defer e.history.deinit(std.testing.allocator);
    const B = Adapter(FakeEngine);
    var b = B.init(std.testing.allocator, &e);
    b.resident = true;
    var rec: Recorder = .{};
    b.link_ctx = &rec;
    b.send_fn = Recorder.send;
    var prompt: [1024]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast(i + 1);

    // plain A: cold prefill
    var a = std.mem.zeroes(Stream);
    defer a.context.deinit(std.testing.allocator);
    try a.context.appendSlice(std.testing.allocator, &prompt);
    a.prompt_len = prompt.len;
    try B.prefill(&b, &a);
    try std.testing.expectEqual(@as(u32, 0), a.cached);
    B.release(&b, &a);
    try std.testing.expectEqual(@as(usize, 1024), e.pos);
    try std.testing.expect(!e.ablate_on);

    // ablated B, same prompt: the resident plain rows are dropped before `begin`, nothing is reused
    rec.count = 0;
    var ab = std.mem.zeroes(Stream);
    defer ab.context.deinit(std.testing.allocator);
    try ab.context.appendSlice(std.testing.allocator, &prompt);
    ab.prompt_len = prompt.len;
    ab.ablate = true;
    try B.prefill(&b, &ab);
    try std.testing.expectEqual(@as(u32, 0), ab.cached);
    try std.testing.expect(e.ablate_on);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[0].kind);
    try std.testing.expectEqual(@as(u32, 0), rec.frames[0].n);
    try std.testing.expectEqual(@as(usize, 1024), rec.frames[0].position);
    try std.testing.expectEqual(follow.Kind.begin, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 1), rec.frames[1].slot);
    try std.testing.expectEqual(follow.Kind.prefill, rec.frames[2].kind);
    try std.testing.expectEqual(@as(usize, 0), rec.frames[2].position);
    try std.testing.expectEqual(@as(u32, 1024), rec.frames[2].n);
    B.release(&b, &ab);
    try std.testing.expect(e.ablate_on);

    // ablated C, same prompt: ablated rows are shared with the same flag
    rec.count = 0;
    var ac = std.mem.zeroes(Stream);
    defer ac.context.deinit(std.testing.allocator);
    try ac.context.appendSlice(std.testing.allocator, &prompt);
    ac.prompt_len = prompt.len;
    ac.ablate = true;
    try B.prefill(&b, &ac);
    try std.testing.expectEqual(@as(u32, 512), ac.cached);
    try std.testing.expectEqual(follow.Kind.begin, rec.frames[0].kind);
    try std.testing.expectEqual(@as(u32, 1), rec.frames[0].slot);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[1].kind);
    try std.testing.expectEqual(@as(u32, 512), rec.frames[1].n);
    B.release(&b, &ac);

    // plain D: flipping back drops the ablated rows too
    rec.count = 0;
    var d = std.mem.zeroes(Stream);
    defer d.context.deinit(std.testing.allocator);
    try d.context.appendSlice(std.testing.allocator, &prompt);
    d.prompt_len = prompt.len;
    try B.prefill(&b, &d);
    try std.testing.expectEqual(@as(u32, 0), d.cached);
    try std.testing.expect(!e.ablate_on);
    try std.testing.expectEqual(follow.Kind.rollback, rec.frames[0].kind);
    try std.testing.expectEqual(@as(u32, 0), rec.frames[0].n);
    try std.testing.expectEqual(@as(u32, 0), rec.frames[1].slot);
    B.release(&b, &d);

    // an engine without vectors refuses an ablated request before any frame is sent
    e.abl = null;
    rec.count = 0;
    var bad = std.mem.zeroes(Stream);
    bad.ablate = true;
    try std.testing.expectError(error.NoAblation, B.prefill(&b, &bad));
    try std.testing.expectEqual(@as(usize, 0), rec.count);
}

test {
    _ = @import("grammar_test.zig");
}
