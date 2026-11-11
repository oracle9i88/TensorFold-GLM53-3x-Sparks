// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the glm53-tp3-3x-sparks contributors
//! TF_GLM53_POOL: request slots on one shared cache, each claiming the rows its request needs instead of a fixed
//! window. The semantics (a claim of what the request needs, first-fit holes, idle kept prefixes give way, honest
//! waiting when nothing fits) follow the shared KV pool of ciprianveg/gb10-tensorfold (branch glm53-nvfp4-csf, credited
//! there to Mia's design); this is an independent host-side implementation for the TP3 slots of this family.
//!
//! The ledger lives on rank 0 only. Rank 0 plans every claim here and sends the result as `claim` frames, so the
//! followers rebuild the same views without a ledger of their own. Nothing here touches a device.
const std = @import("std");

pub const max_slots = 8;
/// The fewest positions a slot's view holds (State.slotViewRows refuses fewer).
pub const min_positions = 2048;
/// Rows past prompt + reply: a verify window's draft rows near the reply's end stay inside the claim.
pub const margin = 64;

/// Which idle kept prefix gives way first when a claim needs room.
pub const Evict = enum {
    /// the fewest kept rows (the cheapest to prefill again), the least recently used on a tie
    small,
    /// the least recently used
    old,
};

pub const Mask = [max_slots]bool;
pub const none: Mask = @splat(false);

/// What the planner needs to know about a slot: a stream on it, the prefill rows it keeps, and how many of them the
/// new prompt reuses (0 for the admission check, which sees no prompt).
pub const Slot = struct { busy: bool = false, kept: usize = 0, reuse: usize = 0 };

pub const Action = enum {
    /// the slot's interval already holds the request: bind it as it is
    keep,
    /// the slot keeps its rows and its interval grows in place to `rows`
    extend,
    /// a new interval at `start`; the slot's kept rows (if any) are dropped
    claim,
};

pub const Plan = struct {
    slot: usize,
    action: Action,
    start: usize,
    rows: usize,
    /// idle slots emptied first (their kept rows dropped, their intervals released)
    evict: Mask = none,
};

pub const Ledger = struct {
    /// The cache's rows (the positions it holds + 2).
    rows_total: usize,
    count: usize,
    start: [max_slots]usize = @splat(0),
    /// A slot's interval, sentinel rows included; 0: none.
    rows: [max_slots]usize = @splat(0),
    /// The tick of each slot's last bind (Evict.old, and Evict.small's tie-break).
    used: [max_slots]u64 = @splat(0),
    tick: u64 = 0,
    /// Claims of at least this many rows are placed from the bottom of the cache, smaller ones from the top, so a
    /// long conversation has room to grow while short requests come and go above it.
    big: usize = 32768,
    evict: Evict = .small,
    /// A kept prefix of at least this many rows waits for its interval to grow in place (another stream in the way
    /// finishes first) instead of moving, which would prefill it again.
    wait_reuse: usize = 32768,

    pub fn init(rows_total: usize, count: usize) !Ledger {
        if (count < 2 or count > max_slots) return error.BadPool;
        if (rows_total < min_positions + 2) return error.BadPool;
        return .{ .rows_total = rows_total, .count = count };
    }

    /// The rows a request of `need` positions (prompt + reply) claims.
    pub fn rowsFor(need: usize) usize {
        return @max(need + margin, min_positions) + 2;
    }

    /// Where `n` rows fit with the slots in `ignore` treated as free: the lowest hole for a big claim, the top of the
    /// highest hole for a small one. Null: no hole holds them.
    pub fn hole(l: *const Ledger, n: usize, ignore: Mask) ?usize {
        const Iv = struct { lo: usize, hi: usize };
        var live: [max_slots]Iv = undefined;
        var k: usize = 0;
        for (0..l.count) |i| {
            if (l.rows[i] == 0 or ignore[i]) continue;
            live[k] = .{ .lo = l.start[i], .hi = l.start[i] + l.rows[i] };
            k += 1;
        }
        std.mem.sort(Iv, live[0..k], {}, struct {
            fn lt(_: void, a: Iv, b: Iv) bool {
                return a.lo < b.lo;
            }
        }.lt);
        var best: ?usize = null;
        var at: usize = 0;
        for (0..k + 1) |j| {
            const hi = if (j < k) live[j].lo else l.rows_total;
            if (hi >= at and hi - at >= n) {
                if (n >= l.big) return at; // the lowest hole that holds it
                best = hi - n; // keep walking: the highest hole wins for a small claim
            }
            if (j < k) at = @max(at, live[j].hi);
        }
        return best;
    }

    /// The slots whose intervals lie in the rows slot `i` grows into when it takes `n` rows in place; null when the
    /// growth would run past the cache.
    pub fn growthBlockers(l: *const Ledger, i: usize, n: usize) ?Mask {
        const from = l.start[i] + l.rows[i];
        const to = l.start[i] + n;
        if (to > l.rows_total) return null;
        var m = none;
        for (0..l.count) |j| {
            if (j == i or l.rows[j] == 0) continue;
            if (l.start[j] >= to or l.start[j] + l.rows[j] <= from) continue;
            m[j] = true;
        }
        return m;
    }

    /// The idle slots with an interval (other than those in `skip`), in the order they give way.
    fn evictionOrder(l: *const Ledger, slots: []const Slot, skip: Mask, out: *[max_slots]usize) usize {
        var k: usize = 0;
        for (0..l.count) |i| {
            if (skip[i] or slots[i].busy or l.rows[i] == 0) continue;
            out[k] = i;
            k += 1;
        }
        const Ctx = struct { l: *const Ledger, s: []const Slot };
        std.mem.sort(usize, out[0..k], Ctx{ .l = l, .s = slots }, struct {
            fn lt(c: Ctx, a: usize, b: usize) bool {
                if (c.l.evict == .small and c.s[a].kept != c.s[b].kept) return c.s[a].kept < c.s[b].kept;
                if (c.l.used[a] != c.l.used[b]) return c.l.used[a] < c.l.used[b];
                return a < b;
            }
        }.lt);
        return k;
    }

    /// Where a request of `need` positions goes. `slots[i]` describes slot i (count of them). LaneMemory: not now,
    /// another stream frees room when it ends (the lane host queues the request again); ContextFull: never.
    pub fn plan(l: *const Ledger, slots: []const Slot, need: usize) error{ LaneMemory, ContextFull }!Plan {
        std.debug.assert(slots.len == l.count);
        const n = rowsFor(need);
        if (n > l.rows_total) return error.ContextFull;
        var any_busy = false;
        var any_free = false;
        for (slots) |s| {
            any_busy = any_busy or s.busy;
            any_free = any_free or !s.busy;
        }
        if (!any_free) return error.LaneMemory;

        // A free slot whose kept rows the prompt reuses keeps them, unless it would drop more of them than it reuses
        // (a short shared opening must not wipe another conversation's long prefix).
        var best: ?usize = null;
        for (slots, 0..) |s, i| {
            if (s.busy or l.rows[i] == 0 or s.reuse == 0) continue;
            if (s.kept -| s.reuse > s.reuse) continue;
            if (best == null or s.reuse > slots[best.?].reuse or (s.reuse == slots[best.?].reuse and l.rows[i] > l.rows[best.?])) best = i;
        }
        if (best) |i| {
            if (l.rows[i] >= n) return .{ .slot = i, .action = .keep, .start = l.start[i], .rows = l.rows[i] };
            if (l.growthBlockers(i, n)) |m| {
                var blocked = false;
                for (0..l.count) |j| blocked = blocked or (m[j] and slots[j].busy);
                if (!blocked) return .{ .slot = i, .action = .extend, .start = l.start[i], .rows = n, .evict = m };
                if (slots[i].reuse >= l.wait_reuse) return error.LaneMemory;
            }
        }

        // A new interval: on a free slot without one, else on the free slot whose kept rows give way first.
        var self: ?usize = null;
        for (slots, 0..) |s, i| if (!s.busy and l.rows[i] == 0) {
            self = i;
            break;
        };
        if (self == null) {
            var order: [max_slots]usize = undefined;
            if (l.evictionOrder(slots, none, &order) == 0) return error.LaneMemory;
            self = order[0];
        }
        const me = self.?;
        var ignore = none;
        ignore[me] = true;
        if (l.hole(n, ignore)) |at| return .{ .slot = me, .action = .claim, .start = at, .rows = n };
        var order: [max_slots]usize = undefined;
        const k = l.evictionOrder(slots, ignore, &order);
        for (order[0..k]) |j| {
            ignore[j] = true;
            if (l.hole(n, ignore)) |at| {
                var gone = ignore;
                gone[me] = false;
                return .{ .slot = me, .action = .claim, .start = at, .rows = n, .evict = gone };
            }
        }
        return if (any_busy) error.LaneMemory else error.ContextFull;
    }

    /// Record a plan rank 0 carried out.
    pub fn apply(l: *Ledger, p: Plan) void {
        for (0..l.count) |j| if (p.evict[j]) {
            l.rows[j] = 0;
        };
        l.start[p.slot] = p.start;
        l.rows[p.slot] = p.rows;
        l.tick += 1;
        l.used[p.slot] = l.tick;
    }

    pub fn release(l: *Ledger, i: usize) void {
        l.rows[i] = 0;
    }

    /// Every interval inside the cache and no two overlapping (the invariant `apply` keeps).
    pub fn consistent(l: *const Ledger) bool {
        for (0..l.count) |i| {
            if (l.rows[i] == 0) continue;
            if (l.start[i] + l.rows[i] > l.rows_total) return false;
            for (i + 1..l.count) |j| {
                if (l.rows[j] == 0) continue;
                if (l.start[i] < l.start[j] + l.rows[j] and l.start[j] < l.start[i] + l.rows[i]) return false;
            }
        }
        return true;
    }
};

fn slotsOf(comptime n: usize, busy: [n]bool, kept: [n]usize, reuse: [n]usize) [n]Slot {
    var s: [n]Slot = undefined;
    for (0..n) |i| s[i] = .{ .busy = busy[i], .kept = kept[i], .reuse = reuse[i] };
    return s;
}

test "pool: big claims go to the bottom, small ones to the top" {
    var l = try Ledger.init(139_266, 3);
    const s = slotsOf(3, .{ false, false, false }, .{ 0, 0, 0 }, .{ 0, 0, 0 });
    const big = try l.plan(&s, 100_000);
    try std.testing.expectEqual(Action.claim, big.action);
    try std.testing.expectEqual(@as(usize, 0), big.start);
    try std.testing.expectEqual(Ledger.rowsFor(100_000), big.rows);
    l.apply(big);
    var s2 = s;
    s2[big.slot].busy = true;
    const small = try l.plan(&s2, 6_000);
    try std.testing.expectEqual(l.rows_total - Ledger.rowsFor(6_000), small.start);
    l.apply(small);
    try std.testing.expect(l.consistent());
}

test "pool: a short request beside a busy long one, then waiting when the rest is too small" {
    var l = try Ledger.init(139_266, 3);
    l.apply(.{ .slot = 0, .action = .claim, .start = 0, .rows = 114_000 });
    const s = slotsOf(3, .{ true, false, false }, .{ 0, 0, 0 }, .{ 0, 0, 0 });
    const p = try l.plan(&s, 20_000);
    try std.testing.expectEqual(@as(usize, 1), p.slot);
    try std.testing.expect(p.start >= 114_000);
    try std.testing.expectError(error.LaneMemory, l.plan(&s, 30_000));
    // alone (nothing busy), a full-window request takes the cache by emptying the idle ones
    const idle = slotsOf(3, .{ false, false, false }, .{ 100_000, 0, 0 }, .{ 0, 0, 0 });
    const whole = try l.plan(&idle, 131_072);
    try std.testing.expect(whole.evict[0]);
    try std.testing.expect(whole.slot != 0);
    try std.testing.expectError(error.ContextFull, l.plan(&idle, 139_266));
}

test "pool: a follow-up keeps its rows, grows in place past an idle short one, waits for a busy one" {
    var l = try Ledger.init(139_266, 3);
    l.apply(.{ .slot = 0, .action = .claim, .start = 0, .rows = 100_066 });
    l.apply(.{ .slot = 1, .action = .claim, .start = 120_000, .rows = 10_000 });
    // fits as it is
    var s = slotsOf(3, .{ false, true, false }, .{ 90_000, 0, 0 }, .{ 90_000, 0, 0 });
    try std.testing.expectEqual(Action.keep, (try l.plan(&s, 95_000)).action);
    // grows into free rows
    const g = try l.plan(&s, 110_000);
    try std.testing.expectEqual(Action.extend, g.action);
    try std.testing.expectEqual(@as(usize, 0), g.start);
    // the growth reaches the busy short one: a long kept prefix waits
    try std.testing.expectError(error.LaneMemory, l.plan(&s, 125_000));
    // the short one idle now: it gives way
    s[1].busy = false;
    const e = try l.plan(&s, 125_000);
    try std.testing.expectEqual(Action.extend, e.action);
    try std.testing.expect(e.evict[1]);
    l.apply(e);
    try std.testing.expect(l.consistent());
}

test "pool: a short shared opening does not wipe another conversation's long prefix" {
    var l = try Ledger.init(139_266, 3);
    l.apply(.{ .slot = 0, .action = .claim, .start = 0, .rows = 100_066 });
    const s = slotsOf(3, .{ false, false, false }, .{ 90_000, 0, 0 }, .{ 2_048, 0, 0 });
    const p = try l.plan(&s, 8_000);
    try std.testing.expect(p.slot != 0);
    try std.testing.expectEqual(Action.claim, p.action);
    try std.testing.expect(!p.evict[0]);
}

test "pool: the cheapest idle prefix gives way first (small), or the oldest (old)" {
    var l = try Ledger.init(60_000, 4);
    l.apply(.{ .slot = 0, .action = .claim, .start = 0, .rows = 30_000 }); // used 1
    l.apply(.{ .slot = 1, .action = .claim, .start = 30_000, .rows = 10_000 }); // used 2
    l.apply(.{ .slot = 2, .action = .claim, .start = 40_000, .rows = 20_000 }); // used 3
    const s = slotsOf(4, .{ false, false, false, false }, .{ 29_000, 9_000, 19_000, 0 }, .{ 0, 0, 0, 0 });
    const p = try l.plan(&s, 9_000);
    try std.testing.expectEqual(@as(usize, 3), p.slot);
    try std.testing.expect(p.evict[1] and !p.evict[0] and !p.evict[2]);
    l.evict = .old;
    const q = try l.plan(&s, 9_000);
    try std.testing.expect(q.evict[0] and !q.evict[1] and !q.evict[2]);
}

test "pool: growth that would run past the cache moves when the prefix is short, waits when long and blocked" {
    var l = try Ledger.init(50_000, 2);
    l.apply(.{ .slot = 0, .action = .claim, .start = 40_000, .rows = 10_000 });
    var s = slotsOf(2, .{ false, true }, .{ 9_000, 0 }, .{ 9_000, 0 });
    const p = try l.plan(&s, 20_000);
    try std.testing.expectEqual(Action.claim, p.action);
    try std.testing.expectEqual(@as(usize, 0), p.slot);
    s[1].busy = false;
    try std.testing.expect(l.consistent());
}
