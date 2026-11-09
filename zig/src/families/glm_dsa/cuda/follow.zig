// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Adapted from ashhart/TensorFold@d1018b31: zig/src/families/glm/cuda_follow.zig (Apache-2.0). Changes: defines the DSA rank protocol header and cache-hit messages.
//! Rank-0 protocol for GLM-5.3 CUDA. Step payloads carry prompt ids and their agreed DCP layout.

const std = @import("std");
const cuda = @import("cuda");
pub const max_decode_rows = 16;

pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draw, end, release, cache_hit, rollback, prefill_rows, draft_block, park, unpark, snap_save, snap_load, snap_commit, prewarm_graph, masks = 18, verify_shared = 19, absorb = 20 };

/// TF_GLM53_SLOTS: request slots a follower binds at most (engine.max_shared). A frame names its slot in the high
/// half of `protocol` (slot 0: protocol 1, the single-stream wire as before).
pub const max_lanes = 8;

pub fn withLane(h: Header, lane: usize) Header {
    var x = h;
    x.protocol = 1 | (@as(u32, @intCast(lane)) << 16);
    return x;
}

pub fn laneOf(h: Header) usize {
    return h.protocol >> 16;
}

pub fn hasPayload(kind: u32) bool {
    return kind == @backingInt(Kind.begin) or kind == @backingInt(Kind.prefill) or kind == @backingInt(Kind.verify) or kind == @backingInt(Kind.cache_hit) or kind == @backingInt(Kind.prefill_rows) or kind == @backingInt(Kind.draft_block) or kind == @backingInt(Kind.park) or kind == @backingInt(Kind.unpark) or kind == @backingInt(Kind.snap_load) or kind == @backingInt(Kind.prewarm_graph) or kind == @backingInt(Kind.masks) or kind == @backingInt(Kind.verify_shared) or kind == @backingInt(Kind.absorb);
}


pub const Header = extern struct {
    kind: u32,
    protocol: u32 = 1,
    n: u32 = 0,
    slot: u32 = 0,
    position: u64 = 0,
    prefix_hash: [32]u8 = @splat(0),

    comptime {
        std.debug.assert(@sizeOf(Header) == 56);
    }
};

pub const Link = struct {
    /// Host-testable wire validation. A follower must reject malformed rank-0 frames before touching its engine.
    pub fn validate(h: Header, ids: []const u32) !void {
        if (h.protocol & 0xffff != 1) return error.ProtocolMismatch;
        if (laneOf(h) >= max_lanes) return error.BadPayload;
        if (h.kind < @backingInt(Kind.begin) or h.kind > @backingInt(Kind.absorb)) return error.BadKind;
        const payload_kind = hasPayload(h.kind);

        if (h.kind == @backingInt(Kind.begin) and h.slot > 1) return error.BadPayload;
        if (payload_kind and h.n != ids.len) return error.BadPayload;
        if (!payload_kind and ids.len != 0) return error.BadPayload;
        if ((h.kind == @backingInt(Kind.prefill) or h.kind == @backingInt(Kind.cache_hit) or h.kind == @backingInt(Kind.prefill_rows)) and ids.len == 0) return error.EmptyPayload;
        if (h.kind == @backingInt(Kind.prefill_rows) and (h.n % 512 != 0 or h.position % 512 != 0)) return error.BadAlignment;
        if (h.kind == @backingInt(Kind.begin) and ids.len != 9) return error.BadPayload;
        if (h.kind == @backingInt(Kind.draft_block) and (ids.len != 2 or ids[0] > max_decode_rows)) return error.BadPayload;
        // TF_GLM53_SLOTS: [n], the kept rows a round that holds no drafts takes into the slot's drafter context.
        if (h.kind == @backingInt(Kind.absorb) and (ids.len != 1 or ids[0] > max_decode_rows)) return error.BadPayload;

        if (h.kind == @backingInt(Kind.park) and (ids.len != 2 or ids[0] % 512 != 0 or ids[0] >= ids[1] or ids[1] - ids[0] > 1 << 20)) return error.BadPayload;
        if (h.kind == @backingInt(Kind.unpark) and ids.len != 1) return error.BadPayload;
        if (h.kind == @backingInt(Kind.snap_save) and (h.n != 0 or ids.len != 0 or h.position == 0 or h.position % 512 != 0 or h.slot > 2)) return error.BadPayload;
        if (h.kind == @backingInt(Kind.snap_load) and (h.n != ids.len or h.n % 512 != 0 or h.position % 512 != 0 or h.position >= h.n)) return error.BadPayload;
        if (h.kind == @backingInt(Kind.snap_commit) and (h.n > 64 or ids.len != 0)) return error.BadPayload;
        if (h.kind == @backingInt(Kind.prewarm_graph) and (h.n == 0 or h.n > max_decode_rows or h.n != ids.len)) return error.BadPayload;

        if (h.kind == @backingInt(Kind.masks) and (h.slot == 0 or h.slot > max_decode_rows or ids.len == 0 or ids.len % @as(usize, h.slot) != 0)) return error.BadPayload;

        if (h.kind == @backingInt(Kind.keep) and h.n > max_decode_rows) return error.BadPayload;
        if (h.kind == @backingInt(Kind.rollback) and h.n > 1 << 20) return error.BadPayload;
        if ((h.kind == @backingInt(Kind.draw) or h.kind == @backingInt(Kind.end) or h.kind == @backingInt(Kind.release)) and h.n != 0) return error.BadPayload;
    }

    pub fn validateRanks(all: []const @import("ranks.zig").Fingerprint) !void {
        if (@import("ranks.zig").mismatch(all)) |m| return switch (m.field) {
            .protocol => error.ProtocolMismatch,
            else => error.RankMismatch,
        };
    }

    /// Apply a validated rank-0 frame to WP9's Engine. This is also the host protocol oracle;
    /// transport receives a complete frame before invoking it, so malformed commands fail closed.
    pub fn apply(engine: anytype, h: Header, ids: []const u32, out: []u32) !void {
        try Link.validate(h, ids);
        // TF_GLM53_SLOTS: every frame runs on the slot rank 0 sent it from (slot 0 without slots).
        if (comptime @hasDecl(@TypeOf(engine.*), "bind")) try engine.bind(laneOf(h)) else if (laneOf(h) != 0) return error.BadPayload;
        switch (@as(Kind, @fromBackingInt(@intCast(h.kind)))) {
            .verify_shared => {
                // [parts, then each part's slot, rows and input ids]; the parts' draws go to `out` in part order.
                if (comptime !@hasDecl(@TypeOf(engine.*), "verifyShared")) return error.UnsupportedOperation;
                const P = @TypeOf(engine.*).SharedPart;
                var parts: [max_lanes]P = undefined;
                if (ids.len == 0 or ids[0] == 0 or ids[0] > max_lanes) return error.BadPayload;
                const n: usize = ids[0];
                var at: usize = 1;
                var row: usize = 0;
                for (parts[0..n]) |*pt| {
                    if (at + 2 > ids.len) return error.BadPayload;
                    const slot: usize = ids[at];
                    const rows: usize = ids[at + 1];
                    at += 2;
                    if (rows == 0 or at + rows > ids.len or row + rows > out.len) return error.BadPayload;
                    pt.* = .{ .slot = slot, .ids = ids[at..][0..rows], .out = out[row..][0..rows] };
                    at += rows;
                    row += rows;
                }
                if (at != ids.len) return error.BadPayload;
                try engine.verifyShared(parts[0..n]);
            },
            .begin => {
                const temp: u64 = ids[0] | (@as(u64, ids[1]) << 32);
                const top_p: u64 = ids[3] | (@as(u64, ids[4]) << 32);
                const min_log: u64 = ids[5] | (@as(u64, ids[6]) << 32);
                engine.seed = ids[7] | (@as(u64, ids[8]) << 32);
                const rule: @TypeOf(engine.rule) = .{ .temperature = @bitCast(temp), .top_k = ids[2], .top_p = @bitCast(top_p), .min_log = @bitCast(min_log) };
                if (comptime @hasDecl(@TypeOf(engine.*), "setSamplingRule")) try engine.setSamplingRule(rule) else engine.rule = rule;
                if (comptime @hasDecl(@TypeOf(engine.*), "setAblate")) try engine.setAblate(h.slot == 1);
                if (comptime @hasDecl(@TypeOf(engine.*), "clearMasks")) engine.clearMasks();
            },
            .prefill => {
                if (engine.pos != h.position) return error.PositionMismatch;
                _ = try engine.prefill(ids);
            },
            .cache_hit => {
                if (engine.pos != h.position) return error.PositionMismatch;
                _ = try engine.nextTurn(ids);
            },
            .rollback => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (@as(usize, h.n) > engine.pos) return error.BadRollback;
                try engine.rollback(@as(usize, h.n));
            },
            .prefill_rows => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "prefillRows")) try engine.prefillRows(ids) else return error.UnsupportedOperation;
            },
            .verify => {
                if (engine.pos != h.position) return error.PositionMismatch;
                try engine.verify(ids, out);
            },
            .prewarm_graph => {
                if (engine.pos != 0 or ids.len != h.n) return error.PositionMismatch;
                engine.pos = @intCast(h.position);
                defer engine.pos = 0;
                try engine.verify(ids, out[0..ids.len]);
            },
            .draft_block => {
                if (comptime @hasDecl(@TypeOf(engine.*), "draftBlock"))
                    try engine.draftBlock(@intCast(h.position), @intCast(ids[0]), ids[1], null, null)
                else
                    return error.UnsupportedOperation;
            },
            .absorb => {
                if (comptime @hasDecl(@TypeOf(engine.*), "absorbBlock"))
                    try engine.absorbBlock(@intCast(h.position), @intCast(ids[0]))
                else
                    return error.UnsupportedOperation;
            },

            .park => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "parkSave")) try engine.parkSave(ids[0], ids[1]) else return error.UnsupportedOperation;
            },
            .unpark => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "parkRestore")) {
                    if (try engine.parkRestore() != ids[0]) return error.ParkMismatch;
                } else return error.UnsupportedOperation;
            },
            .snap_save => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "snapSave")) try engine.snapSave(h) else return error.UnsupportedOperation;
            },
            .snap_load => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "snapLoad")) {
                    _ = try engine.snapLoad(h, ids);
                } else return error.UnsupportedOperation;
            },
            .snap_commit => {
                if (comptime @hasDecl(@TypeOf(engine.*), "snapCommit")) try engine.snapCommit(h.n) else return error.UnsupportedOperation;
            },
            .masks => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "setMasks")) try engine.setMasks(ids, @intCast(h.slot)) else return error.UnsupportedOperation;

            },
            .keep => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasField(@TypeOf(engine.*), "window_rows")) if (engine.window_rows < h.n) return error.BadKeep;
                try engine.keep(@as(usize, h.n));
            },
            .release => {
                if (engine.pos != h.position) return error.PositionMismatch;
                if (comptime @hasDecl(@TypeOf(engine.*), "clearMasks")) engine.clearMasks();
                if (comptime @hasDecl(@TypeOf(engine.*), "rollback")) {
                    // Resident mode keeps the settled prefill rows (and the ablation flag they were computed under).
                    const drop = if (comptime @hasField(@TypeOf(engine.*), "resident")) !engine.resident else true;
                    if (drop) {
                        try engine.rollback(0);
                        if (comptime @hasField(@TypeOf(engine.*), "ablate_on")) engine.ablate_on = false;
                    }
                }
                if (comptime @hasDecl(@TypeOf(engine.*), "requestEnd")) engine.requestEnd() catch {}; // trace only: never end the follow loop over it (rank 0 would wait)
            },
            .draw, .end => {},
        }
    }

    /// Drive the rank-1 engine until rank 0 sends end. A transport supplies complete frames; receive errors stop
    /// the follower immediately, and validation always precedes engine mutation.
    pub fn run(engine: anytype, transport: anytype) !void {
        return runUntil(engine, transport, struct {
            fn never() bool { return false; }
        }.never);
    }

    /// Drive the follower until rank 0 sends end or a local stop request arrives. A signal stop is checked
    /// between complete frames so the two ranks never diverge halfway through an engine operation.
    pub fn runUntil(engine: anytype, transport: anytype, should_stop: anytype) !void {
        var out: [max_decode_rows]u32 = undefined;
        while (!should_stop()) {
            const frame = try transport.receive();
            try apply(engine, frame.header, frame.ids, &out);
            if (frame.header.kind == @backingInt(Kind.end)) return;
        }
    }
};

/// NCCL broadcast frame transport shared by the rank-0 adapter and rank-1 follower.
pub const Wire = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    device: cuda.DeviceBuffer,
    ids: []u32,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize) !Wire {
        const device = try cuda.DeviceBuffer.alloc(d, @sizeOf(Header) + max_ids * @sizeOf(u32));
        errdefer {
            var b = device;
            b.free();
        }
        return .{ .gpa = gpa, .d = d, .lib = lib, .comm = comm, .stream = stream, .device = device, .ids = try gpa.alloc(u32, max_ids) };
    }

    pub fn deinit(w: *Wire) void {
        w.gpa.free(w.ids);
        w.device.free();
    }

    pub fn send(w: *Wire, h: Header, ids: []const u32) !void {
        try Link.validate(h, ids);
        if (ids.len > w.ids.len) return error.TooManyIds;
        try w.device.uploadAsync(0, std.mem.asBytes(&h), w.stream.handle);
        try w.broadcast(0, @sizeOf(Header));
        if (ids.len != 0) {
            try w.device.uploadAsync(@sizeOf(Header), std.mem.sliceAsBytes(ids), w.stream.handle);
            try w.broadcast(@sizeOf(Header), ids.len * @sizeOf(u32));
        }
    }

    pub fn receive(w: *Wire) !struct { header: Header, ids: []const u32 } {
        try w.broadcast(0, @sizeOf(Header));
        try w.stream.synchronize();
        var h: Header = undefined;
        try w.device.download(0, std.mem.asBytes(&h));
        const has_payload = hasPayload(h.kind);

        if (has_payload and h.n > w.ids.len) return error.TooManyIds;
        if (has_payload and h.n != 0) {
            try w.broadcast(@sizeOf(Header), @as(usize, h.n) * @sizeOf(u32));
            try w.stream.synchronize();
            try w.device.download(@sizeOf(Header), std.mem.sliceAsBytes(w.ids[0..h.n]));
        }
        const n: usize = if (has_payload) h.n else 0;
        try Link.validate(h, w.ids[0..n]);
        return .{ .header = h, .ids = w.ids[0..n] };
    }

    fn broadcast(w: *Wire, offset: usize, bytes: usize) !void {
        try w.lib.check(w.lib.api.ncclBroadcast(w.device.ptr + offset, w.device.ptr + offset, bytes, .u8, 0, w.comm, w.stream.handle), "GLM-5.3 follower broadcast");
    }
};

test "follower protocol validates payload and protocol version" {
    const h: Header = .{ .kind = @backingInt(Kind.prefill), .n = 3, .position = 256 };
    try Link.validate(h, &.{ 1, 2, 3 });
    try std.testing.expectError(error.BadPayload, Link.validate(h, &.{ 1, 2 }));
    const begin: Header = .{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 1 };
    try Link.validate(begin, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 2 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }));
    var old = h;
    old.protocol = 2;
    try std.testing.expectError(error.ProtocolMismatch, Link.validate(old, &.{ 1, 2, 3 }));
}


test "park protocol frames validate payloads and preserve the payload kind map" {
    const park_kind = @backingInt(Kind.park);
    const unpark_kind = @backingInt(Kind.unpark);
    try Link.validate(.{ .kind = park_kind, .n = 2 }, &.{ 512, 1024 });
    try Link.validate(.{ .kind = unpark_kind, .n = 1 }, &.{ 2048 });
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = park_kind, .n = 2 }, &.{ 1, 512 }));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = park_kind, .n = 2 }, &.{ 1024, 1024 }));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = unpark_kind, .n = 1 }, &.{}));
    // 17 = prewarm_graph, 18 = masks, 19 = verify_shared and 20 = absorb; 21 is past the last kind.
    try std.testing.expectError(error.BadKind, Link.validate(.{ .kind = 21 }, &.{}));
    for (1..21) |k| try std.testing.expectEqual(k == 1 or k == 2 or k == 3 or k == 8 or k == 10 or k == 11 or k == 12 or k == 13 or k == 15 or k == 17 or k == 18 or k == 19 or k == 20, hasPayload(@intCast(k)));
}

test "park protocol apply checks position and the follower restored frontier" {
    const Fake = struct {
        pos: usize = 100,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: u32 = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        saved: bool = false,
        restored: usize = 120,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
        pub fn prefillRows(e: *@This(), ids: []const u32) !void { e.pos += ids.len; }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), rows: usize) !void { e.pos += rows; }
        pub fn parkSave(e: *@This(), lo: usize, hi: usize) !void { e.saved = lo == 0 and hi == 512; }
        pub fn parkRestore(e: *@This()) !usize { return e.restored; }
    };
    var e: Fake = .{};
    try Link.apply(&e, .{ .kind = @backingInt(Kind.park), .n = 2, .position = 100 }, &.{ 0, 512 }, &.{});
    try std.testing.expect(e.saved);
    try std.testing.expectError(error.ParkMismatch, Link.apply(&e, .{ .kind = @backingInt(Kind.unpark), .n = 1, .position = 100 }, &.{ 121 }, &.{}));
    try std.testing.expectError(error.PositionMismatch, Link.apply(&e, .{ .kind = @backingInt(Kind.unpark), .n = 1, .position = 99 }, &.{ 120 }, &.{}));
}
test "snapshot protocol validates aligned save, load and bounded commit frames" {
    try Link.validate(.{ .kind = @backingInt(Kind.snap_save), .position = 1024, .slot = 2 }, &.{});
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.snap_save), .position = 0 }, &.{}));
    const zeros: [1024]u32 = @splat(0);
    try Link.validate(.{ .kind = @backingInt(Kind.snap_load), .n = 1024, .position = 512 }, &zeros);
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.snap_load), .n = 512, .position = 1 }, zeros[0..512]));
    try Link.validate(.{ .kind = @backingInt(Kind.snap_commit), .n = 64 }, &.{});
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.snap_commit), .n = 65 }, &.{}));
}


test "follower masks frames validate, arm and clear" {
    const kind = @backingInt(Kind.masks);
    const too_many: [17]u32 = @splat(1);
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = kind, .n = 1, .slot = 0 }, &.{1}));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = kind, .n = 17, .slot = 17 }, &too_many));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = kind, .n = 3, .slot = 2 }, &.{ 1, 2, 3 }));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = kind, .n = 0, .slot = 1 }, &.{}));
    try std.testing.expectError(error.BadKind, Link.validate(.{ .kind = 21 }, &.{}));
    const Fake = struct {
        pos: usize = 0,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        rows: usize = 0,
        bits: [4]u32 = @splat(0),
        pub fn prefill(_: *@This(), _: []const u32) !u32 { return 0; }
        pub fn nextTurn(_: *@This(), _: []const u32) !u32 { return 0; }
        pub fn prefillRows(_: *@This(), _: []const u32) !void {}
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(_: *@This(), _: usize) !void {}
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
        pub fn draftBlock(_: *@This(), _: usize, _: usize, _: u32, _: ?[]u32, _: ?[]f64) !void {}
        pub fn setMasks(e: *@This(), ids: []const u32, rows: usize) !void {
            e.rows = rows;
            @memcpy(e.bits[0..ids.len], ids);
        }
        pub fn clearMasks(e: *@This()) void { e.rows = 0; }
    };
    var fake: Fake = .{};
    try Link.apply(&fake, .{ .kind = kind, .n = 4, .slot = 2 }, &.{ 4, 5, 6, 7 }, &.{});
    try std.testing.expectEqual(@as(usize, 2), fake.rows);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 6, 7 }, &fake.bits);
    try Link.apply(&fake, .{ .kind = @backingInt(Kind.release) }, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 0), fake.rows);
    try Link.apply(&fake, .{ .kind = @backingInt(Kind.masks), .n = 2, .slot = 1 }, &.{ 8, 9 }, &.{});
    try Link.apply(&fake, .{ .kind = @backingInt(Kind.begin), .n = 9 }, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 0 }, &.{});
    try std.testing.expectEqual(@as(usize, 0), fake.rows);

}

test "follower fails closed when rank settings disagree" {
    const ranks = @import("ranks.zig");
    const a = ranks.Fingerprint.of(2, 0, .{ .overrides = @splat(3) });
    const b = ranks.Fingerprint.of(2, 1, .{ .overrides = @splat(4) });
    try std.testing.expectError(error.RankMismatch, Link.validateRanks(&.{ a, b }));
    try std.testing.expectError(error.EmptyPayload, Link.validate(.{ .kind = @backingInt(Kind.prefill) }, &.{}));
}

test "follower validates and applies draft_block frames" {
    // apply() compiles every frame kind against the engine type, so the fake carries the whole surface.
    const FakeEngine = struct {
        pos: usize = 0,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        calls: usize = 0,
        draft_pos: usize = 0,
        n: usize = 0,
        anchor: u32 = 0,
        had_out: bool = true,
        pub fn prefill(_: *@This(), _: []const u32) !u32 {
            return 0;
        }
        pub fn nextTurn(_: *@This(), _: []const u32) !u32 {
            return 0;
        }
        pub fn prefillRows(_: *@This(), _: []const u32) !void {}
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(_: *@This(), _: usize) !void {}
        pub fn rollback(e: *@This(), to: usize) !void {
            e.pos = to;
        }
        pub fn draftBlock(e: *@This(), pos: usize, n: usize, anchor: u32, out: ?[]u32, _: ?[]f64) !void {
            e.calls += 1;
            e.draft_pos = pos;
            e.n = n;
            e.anchor = anchor;
            e.had_out = out != null;
        }
    };
    const frame: Header = .{ .kind = @backingInt(Kind.draft_block), .n = 2, .position = 44 };
    var fake: FakeEngine = .{};
    try Link.apply(&fake, frame, &.{ 7, 123 }, &.{});
    try std.testing.expectEqual(@as(usize, 1), fake.calls);
    try std.testing.expectEqual(@as(usize, 44), fake.draft_pos);
    try std.testing.expectEqual(@as(usize, 7), fake.n);
    try std.testing.expectEqual(@as(u32, 123), fake.anchor);
    try std.testing.expect(!fake.had_out);
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.draft_block), .n = 1 }, &.{1}));
    try std.testing.expectError(error.BadPayload, Link.validate(frame, &.{ 1, 2, 3 }));
    try std.testing.expectError(error.BadPayload, Link.validate(frame, &.{ 17, 2 }));
}

test "prewarm graph frames verify at a synthetic position and restore the empty frontier" {
    const Fake = struct {
        pos: usize = 0,
        seen: usize = 0,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        pub fn verify(e: *@This(), ids: []const u32, out: []u32) !void {
            e.seen = e.pos;
            @memcpy(out[0..ids.len], ids);
        }
        // Link.apply compiles every frame kind for its engine type; these frames are not sent in this test.
        pub fn prefill(_: *@This(), _: []const u32) !u32 { return error.Unexpected; }
        pub fn nextTurn(_: *@This(), _: []const u32) !u32 { return error.Unexpected; }
        pub fn keep(_: *@This(), _: usize) !void { return error.Unexpected; }
        pub fn rollback(_: *@This(), _: usize) !void { return error.Unexpected; }
    };
    var engine: Fake = .{};
    var out: [2]u32 = undefined;
    const ids = [_]u32{ 7, 9 };
    const frame: Header = .{ .kind = @intFromEnum(Kind.prewarm_graph), .n = 2, .position = 2046 };
    try Link.apply(&engine, frame, &ids, &out);
    try std.testing.expectEqual(@as(usize, 2046), engine.seen);
    try std.testing.expectEqual(@as(usize, 0), engine.pos);
    try std.testing.expectEqualSlices(u32, &ids, &out);
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @intFromEnum(Kind.prewarm_graph), .n = 3, .position = 1 }, &ids));
    engine.pos = 1;
    try std.testing.expectError(error.PositionMismatch, Link.apply(&engine, frame, &ids, &out));
}

test "follower applies rank-0 prefill and cache-hit frames to the Engine frontier" {
    const FakeEngine = struct {
        pos: usize = 0,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        prefill_calls: usize = 0,
        turn_calls: usize = 0,
        rows_calls: usize = 0,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            e.prefill_calls += 1;
            e.pos += ids.len;
            return 1;
        }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 {
            e.turn_calls += 1;
            e.pos += ids.len;
            return 2;
        }
        pub fn prefillRows(e: *@This(), ids: []const u32) !void {
            e.rows_calls += 1;
            e.pos += ids.len;
        }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), n: usize) !void {
            e.pos += n;
        }
        pub fn rollback(e: *@This(), to: usize) !void {
            e.pos = to;
        }
    };
    var engine: FakeEngine = .{};
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.prefill), .n = 2, .position = 0 }, &.{ 4, 5 }, &.{});
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.cache_hit), .n = 1, .position = 2 }, &.{6}, &.{});
    try std.testing.expectEqual(@as(usize, 1), engine.prefill_calls);
    try std.testing.expectEqual(@as(usize, 1), engine.turn_calls);
    try std.testing.expectError(error.PositionMismatch, Link.apply(&engine, .{ .kind = @backingInt(Kind.cache_hit), .n = 1, .position = 0 }, &.{7}, &.{}));
    const short_rows: [511]u32 = @splat(0);
    try std.testing.expectError(error.BadAlignment, Link.validate(.{ .kind = @backingInt(Kind.prefill_rows), .n = 511 }, &short_rows));
    try std.testing.expectError(error.EmptyPayload, Link.validate(.{ .kind = @backingInt(Kind.prefill_rows) }, &.{}));
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.rollback), .n = 0, .position = 3 }, &.{}, &.{});
    const rows: [512]u32 = @splat(0);
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.prefill_rows), .n = 512, .position = 0 }, &rows, &.{});
    try std.testing.expectEqual(@as(usize, 1), engine.rows_calls);
    try std.testing.expectEqual(@as(usize, 512), engine.pos);
}

test "follower applies rollback frames and rejects bad targets or payloads" {
    const Fake = struct {
        pos: usize = 8,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), n: usize) !void { e.pos += n; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    var e: Fake = .{};
    const frame: Header = .{ .kind = @backingInt(Kind.rollback), .position = 8, .n = 5 };
    try Link.apply(&e, frame, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 5), e.pos);
    e.pos = 8;
    try std.testing.expectError(error.BadRollback, Link.apply(&e, .{ .kind = @backingInt(Kind.rollback), .position = 8, .n = 9 }, &.{}, &.{}));
    try std.testing.expectError(error.BadPayload, Link.apply(&e, frame, &.{1}, &.{}));
    try std.testing.expectError(error.BadPayload, Link.validate(.{ .kind = @backingInt(Kind.rollback), .n = (1 << 20) + 1 }, &.{}));
    try std.testing.expectError(error.PositionMismatch, Link.apply(&e, .{ .kind = @backingInt(Kind.rollback), .position = 7, .n = 4 }, &.{}, &.{}));
}

test "follower applies begin ablation slot and release clears it" {
    const FakeEngine = struct {
        pos: usize = 0,
        seed: u64 = 0,
        ablate_on: bool = false,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        pub fn setAblate(e: *@This(), on: bool) !void { e.ablate_on = on; }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), n: usize) !void { e.pos += n; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    var engine: FakeEngine = .{};
    const sampling = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 1 }, &sampling, &.{});
    try std.testing.expect(engine.ablate_on);
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.release) }, &.{}, &.{});
    try std.testing.expect(!engine.ablate_on);
}

test "follower begin selects sampled candidates and preserves greedy" {
    const FakeEngine = struct {
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        candidates: bool = false,
        k: usize = 64,
        pos: usize = 0,
        pub fn setSamplingRule(e: *@This(), rule: @TypeOf(e.rule)) !void {
            e.rule = rule;
            e.candidates = rule.temperature > 0;
        }
        // Link.apply compiles every frame kind for its engine type; only .begin is sent in this test.
        pub fn prefill(_: *@This(), _: []const u32) !u32 { return error.Unexpected; }
        pub fn nextTurn(_: *@This(), _: []const u32) !u32 { return error.Unexpected; }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void { return error.Unexpected; }
        pub fn keep(_: *@This(), _: usize) !void { return error.Unexpected; }
        pub fn rollback(_: *@This(), _: usize) !void { return error.Unexpected; }
    };
    var engine: FakeEngine = .{};
    const sampled = [_]u32{ 0, 0x3ff00000, 0, 0, 0x3fee6666, 0, 0xfff00000, 0, 0 };
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9 }, &sampled, &.{});
    try std.testing.expect(engine.candidates);
    try std.testing.expectEqual(@as(f64, 1), engine.rule.temperature);
    const greedy = [_]u32{ 0, 0, 0, 0, 0x3ff00000, 0, 0xfff00000, 0, 0 };
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9 }, &greedy, &.{});
    try std.testing.expect(!engine.candidates);
}

test "resident follower keeps its ablation flag on release and takes a flag flip after the rollback frame" {
    const FakeEngine = struct {
        pos: usize = 0,
        seed: u64 = 0,
        resident: bool = true,
        ablate_on: bool = false,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        pub fn setAblate(e: *@This(), on: bool) !void {
            if (on != e.ablate_on and e.pos != 0) return error.AblateWithResidentRows;
            e.ablate_on = on;
        }
        pub fn prefill(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 { e.pos += ids.len; return 0; }
        pub fn verify(_: *@This(), _: []const u32, _: []u32) !void {}
        pub fn keep(e: *@This(), n: usize) !void { e.pos += n; }
        pub fn rollback(e: *@This(), to: usize) !void { e.pos = to; }
    };
    var engine: FakeEngine = .{};
    const sampling = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 0 }, &sampling, &.{});
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.prefill), .n = 2, .position = 0 }, &.{ 1, 2 }, &.{});
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.release), .position = 2 }, &.{}, &.{});
    try std.testing.expectEqual(@as(usize, 2), engine.pos);
    // a flip with resident rows is refused; rank 0 sends the rollback frame first
    try std.testing.expectError(error.AblateWithResidentRows, Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 1 }, &sampling, &.{}));
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.rollback), .n = 0, .position = 2 }, &.{}, &.{});
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.begin), .n = 9, .slot = 1 }, &sampling, &.{});
    try std.testing.expect(engine.ablate_on);
    try Link.apply(&engine, .{ .kind = @backingInt(Kind.release), .position = 0 }, &.{}, &.{});
    try std.testing.expect(engine.ablate_on);
}

test "follower protocol public declarations are fully analyzed" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Header);
    std.testing.refAllDecls(Link);
    std.testing.refAllDecls(Wire);
}

fn loopbackCase(resident: bool, want_pos: usize) !void {
    const Frame = struct { header: Header, ids: []const u32 };
    const FakeEngine = struct {
        pos: usize = 0,
        resident: bool,
        seed: u64 = 0,
        rule: struct { temperature: f64 = 0, top_k: usize = 0, top_p: f64 = 1, min_log: f64 = -std.math.inf(f64) } = .{},
        calls: [5]u8 = .{ 0, 0, 0, 0, 0 },
        n: usize = 0,
        pub fn prefill(e: *@This(), ids: []const u32) !u32 {
            e.calls[e.n] = 1;
            e.n += 1;
            e.pos += ids.len;
            return 0;
        }
        pub fn nextTurn(e: *@This(), ids: []const u32) !u32 {
            e.calls[e.n] = 2;
            e.n += 1;
            e.pos += ids.len;
            return 0;
        }
        pub fn verify(e: *@This(), _: []const u32, _: []u32) !void {
            e.calls[e.n] = 3;
            e.n += 1;
        }
        pub fn keep(e: *@This(), n: usize) !void {
            e.pos += n;
        }
        pub fn rollback(e: *@This(), to: usize) !void {
            e.pos = to;
        }
    };
    const Loopback = struct {
        frames: []const Frame,
        at: usize = 0,
        pub fn receive(t: *@This()) !@TypeOf(t.frames[0]) {
            if (t.at == t.frames.len) return error.LoopbackClosed;
            defer t.at += 1;
            return t.frames[t.at];
        }
    };
    const frames = [_]Frame{
        .{ .header = .{ .kind = @backingInt(Kind.begin), .n = 9 }, .ids = &.{ 0, 0, 0, 0, 0, 0, 0, 19, 0 } },
        .{ .header = .{ .kind = @backingInt(Kind.prefill), .n = 2, .position = 0 }, .ids = &.{ 1, 2 } },
        .{ .header = .{ .kind = @backingInt(Kind.cache_hit), .n = 1, .position = 2 }, .ids = &.{3} },
        .{ .header = .{ .kind = @backingInt(Kind.verify), .n = 1, .position = 3 }, .ids = &.{4} },
        .{ .header = .{ .kind = @backingInt(Kind.keep), .n = 1, .position = 3 }, .ids = &.{} },
        // Rank 0 settles decode-written rows to its prefill frontier before release.
        .{ .header = .{ .kind = @backingInt(Kind.rollback), .n = 3, .position = 4 }, .ids = &.{} },
        .{ .header = .{ .kind = @backingInt(Kind.release), .position = 3 }, .ids = &.{} },
        .{ .header = .{ .kind = @backingInt(Kind.end) }, .ids = &.{} },
    };
    var e: FakeEngine = .{ .resident = resident };
    var t = Loopback{ .frames = &frames };
    try Link.run(&e, &t);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3 }, e.calls[0..3]);
    try std.testing.expectEqual(@as(u64, 19), e.seed);
    try std.testing.expectEqual(want_pos, e.pos);
    try std.testing.expectEqual(@as(usize, 8), t.at);
}

test "follower loop in resident mode keeps the settled prefill frontier on release" {
    try loopbackCase(true, 3);
}

test "follower loop in non-resident mode rolls back to zero on release" {
    try loopbackCase(false, 0);
}
