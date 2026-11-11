// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! Host-side saved-prefix index and deterministic cut policy for GLM-5.3.
//! Index/sweep design follows MiaAI-Lab patch 0132 (Apache-2.0, Copyright 2026 Mia's AI Lab);
//! file format follows ashhart/TensorFold sessions/format.zig (Apache-2.0).
const std = @import("std");
const cuda = @import("cuda");
const uma = @import("uma");
pub const kvsave = @import("kvsave");

pub const max_nodes = 4096;
pub const piece_rows = 512;
pub const Node = struct {
    key: u64,
    at: usize,
    alignment: usize = piece_rows,
    base_at: usize = 0,
    parent_key: u64 = 0,
    seq: u64 = 0,
    bytes: u64 = 0,
    depth: u8 = 0,
    last_used: u64 = 0,
    children: u16 = 0,
    pending_children: u16 = 0,
    pending: bool = false,
    pinned: bool = false,
};

pub const PlanOptions = struct {
    expert_rows: usize,
    step: usize = 0,
    span: usize = 16384,
};

/// Per-rank owner for the fixed staging pool and the identical saved-file index.
/// Call `init` during native setup; request paths reuse the pool and its fixed-size
/// cut scratch. The adapter and engine callbacks keep this object as their context.
pub const Runtime = struct {
    gpa: std.mem.Allocator,
    pool: ?Pool,
    index: Index,
    identity: kvsave.Digest,
    root: []u8,
    rank: u32,
    world: u32 = 2,
    options: PlanOptions,
    budget_bytes: u64,
    floor_bytes: u64,
    used_bytes: u64 = 0,
    pending_count: usize = 0,
    pending_max_at: usize = 0,
    writer_idle: std.atomic.Value(bool) = .init(true),
    planned: [18]Cut = undefined,
    planned_len: usize = 0,
    existing: [max_nodes]usize = undefined,
    planned_ids: []const u32 = &.{},
    chain_key: u64 = 0,
    chain_at: usize = 0,
    removed: [max_nodes]Node = undefined,

    pub fn init(gpa: std.mem.Allocator, root: []const u8, identity: kvsave.Digest, rank: u32, options: PlanOptions, budget_bytes: u64, floor_bytes: u64, pool: ?Pool) !Runtime {
        const owned_root = try gpa.dupe(u8, root);
        errdefer gpa.free(owned_root);
        var index = Index.init(gpa);
        errdefer index.deinit();
        try index.reserve(max_nodes);
        return .{ .gpa = gpa, .pool = pool, .index = index, .identity = identity, .root = owned_root, .rank = rank, .options = options, .budget_bytes = budget_bytes, .floor_bytes = floor_bytes };
    }

    pub fn deinit(r: *Runtime) void {
        if (r.pool) |*pool| pool.deinit();
        r.index.deinit();
        r.gpa.free(r.root);
        r.* = undefined;
    }

    pub fn lookup(r: *Runtime, ids: []const u32, after: usize) ?Node {
        const node = r.index.lookup(ids, after, kvsave.cutAlignment(r.options.expert_rows) catch return null) orelse return null;
        r.index.touchChain(node.key);
        return node;
    }

    pub fn plan(r: *Runtime, prompt: []const u32, start: usize, shared: []const u32, history_len: usize) ![]const Cut {
        const a = try kvsave.cutAlignment(r.options.expert_rows);
        var deepest: ?Node = null;
        const count = r.index.existingOnPrefix(prompt, a, start, r.existing[0..], &deepest);
        r.planned_len = try cutsInto(r.planned[0..], prompt.len, start, shared, history_len, r.options, r.existing[0..count]);
        r.planned_ids = prompt;
        r.chain_key = if (deepest) |node| node.key else 0;
        r.chain_at = if (deepest) |node| node.at else 0;
        return r.planned[0..r.planned_len];
    }

    pub fn saveHeader(r: *Runtime, at: usize) !kvsave.Header {
        if (at == 0 or at > r.planned_ids.len or at % (try kvsave.cutAlignment(r.options.expert_rows)) != 0) return error.BadNode;
        const key = kvsave.keyOf(r.planned_ids[0..at]);
        const existing = r.index.get(key);
        if (existing) |node| {
            return .{ .identity = r.identity, .key = key, .base_at = node.base_at, .parent_key = node.parent_key, .seq = node.seq, .rank = r.rank, .world = r.world, .at = node.at, .ids_len = node.at - node.base_at, .body_len = 0, .digest_count = 0 };
        }
        var parent_at = r.chain_at;
        var parent_key = r.chain_key;
        if (parent_key != 0) if (r.index.get(parent_key)) |node| {
            if (node.depth >= 64) {
                var cursor = node;
                var walked: usize = 0;
                while (cursor.depth > 32 and cursor.parent_key != 0 and walked < 64) : (walked += 1) {
                    cursor = r.index.get(cursor.parent_key) orelse return error.BadNode;
                }
                parent_key = cursor.key;
                parent_at = cursor.at;
            }
        };
        if (at <= parent_at) return error.BadNode;
        const seq = r.index.next_seq;
        return .{ .identity = r.identity, .key = key, .base_at = parent_at, .parent_key = parent_key, .seq = seq, .rank = r.rank, .world = r.world, .at = at, .ids_len = at - parent_at, .body_len = 0, .digest_count = 0 };
    }

    pub fn path(r: *const Runtime, key: u64) ![:0]u8 {
        return kvsave.pathFor(r.gpa, r.root, r.identity, r.rank, key);
    }

    pub fn pending(r: *const Runtime) usize { return r.pending_count; }
    pub fn pendingMax(r: *const Runtime) usize { return r.pending_max_at; }
    pub fn isWriterIdle(r: *const Runtime) bool { return r.writer_idle.load(.acquire); }

    pub fn dropSubtree(r: *Runtime, key: u64) usize {
        const count = r.index.removeSubtree(key, &r.removed);
        for (r.removed[0..count]) |node| {
            var path_buf: [512]u8 = undefined;
            var removed_file = false;
            if (kvsave.pathInto(&path_buf, r.root, r.identity, r.rank, node.key)) |file_path| {
                if (std.c.unlink(file_path.ptr) == 0) {
                    removed_file = true;
                } else {
                    std.log.warn("prefix cache rank {d}: unlink failed for {s}", .{ r.rank, file_path });
                }
            } else |err| std.log.warn("prefix cache rank {d}: path formatting failed: {s}", .{ r.rank, @errorName(err) });
            if (removed_file) r.used_bytes -|= node.bytes;
        }
        return count;
    }

    pub fn recount(r: *Runtime) void {
        r.used_bytes = 0;
        for (r.index.nodes.items) |node| r.used_bytes +|= node.bytes;
    }

    pub fn room(r: *Runtime, incoming: u64, free_bytes: u64) bool {
        var free = free_bytes;
        while (r.used_bytes +| incoming > r.budget_bytes or free < r.floor_bytes +| incoming) {
            const victim = r.index.victim() orelse return false;
            var buf: [512]u8 = undefined;
            if (kvsave.pathInto(&buf, r.root, r.identity, r.rank, victim.key)) |file_path| {
                if (std.c.unlink(file_path.ptr) != 0) {
                    std.log.warn("prefix cache rank {d}: unlink failed for {s}; save skipped", .{ r.rank, file_path });
                    return false;
                }
            } else |err| {
                std.log.warn("prefix cache rank {d}: path formatting failed: {s}; save skipped", .{ r.rank, @errorName(err) });
                return false;
            }
            r.index.remove(victim.key);
            r.used_bytes -|= victim.bytes;
            free +|= victim.bytes;
        }
        return true;
    }
};

/// Every persisted cut must preserve whole expert groups as well as the 512 row kernel piece.
pub const EffectiveBudget = struct { budget_bytes: u64, floor_bytes: u64 };

pub fn effectiveBudget(rank0_budget: u64, rank1_budget: u64, rank0_floor: u64, rank1_floor: u64) EffectiveBudget {
    return .{ .budget_bytes = @min(rank0_budget, rank1_budget), .floor_bytes = @max(rank0_floor, rank1_floor) };
}

pub fn limitFor(len: usize, a: usize) usize {
    return (len -| 1) / a * a;
}

pub const CutKind = enum(u32) { grid = 0, history = 1, system = 2 };
pub const Cut = struct { at: usize, kind: CutKind };

/// Produce at most 18 strictly increasing aligned cuts. Existing cuts suppress duplicate work.
pub fn cuts(gpa: std.mem.Allocator, prompt_len: usize, start: usize, shared: []const u32, history_len: usize, options: PlanOptions, existing: []const usize) ![]Cut {
    var storage: [18]Cut = undefined;
    const len = try cutsInto(&storage, prompt_len, start, shared, history_len, options, existing);
    return gpa.dupe(Cut, storage[0..len]);
}

/// Allocation-free cut planner used by Runtime after setup. `out` must hold 18 cuts.
pub fn cutsInto(out: []Cut, prompt_len: usize, start: usize, shared: []const u32, history_len: usize, options: PlanOptions, existing: []const usize) !usize {
    if (out.len < 18) return error.CutBufferTooSmall;
    const a = try kvsave.cutAlignment(options.expert_rows);
    const min_step: usize = @max(2048, a);
    const min_span: usize = @max(16384, a);
    const step: usize = if (options.step == 0) ceilMultiple(min_step, a) else options.step;
    const span: usize = if (options.span == 0) ceilMultiple(min_span, a) else options.span;
    if (step % a != 0 or span % a != 0) return error.BadCutGeometry;
    const limit = limitFor(prompt_len, a);
    const from = @max(start, a);
    if (from >= limit) return 0;
    var candidates: [20]Cut = undefined;
    var candidate_len: usize = 0;

    var shared_end: usize = 0;
    for (shared) |v| shared_end = @max(shared_end, v);
    const sys = shared_end / a * a;
    if (from < sys and sys <= limit and !contains(existing, sys)) {
        candidates[candidate_len] = .{ .at = sys, .kind = .system };
        candidate_len += 1;
    }

    const hist_input = @min(if (history_len == 0) limit else history_len, limit);
    const hist = @min(hist_input / a * a, limit);
    var grid = (from / span + 1) * span;
    while (grid < hist) : (grid += span) {
        if (grid > from and !contains(existing, grid)) {
            if (candidate_len < candidates.len) {
                candidates[candidate_len] = .{ .at = grid, .kind = .grid };
                candidate_len += 1;
            }
        }
    }
    if (hist > from and !contains(existing, hist) and candidate_len < candidates.len) {
        candidates[candidate_len] = .{ .at = hist, .kind = .history };
        candidate_len += 1;
    }

    std.mem.sort(Cut, candidates[0..candidate_len], {}, struct {
        fn less(_: void, x: Cut, y: Cut) bool { return x.at < y.at; }
    }.less);
    var prev = from;
    var kept: usize = 0;
    for (candidates[0..candidate_len]) |cut| {
        var previous_existing = from;
        for (existing) |at| if (at <= cut.at) { previous_existing = @max(previous_existing, at); };
        const spacing_base = @max(prev, previous_existing);
        if (cut.kind != .system and cut.at - spacing_base < step) continue;
        out[kept] = cut;
        kept += 1;
        prev = cut.at;
        if (kept == 18) break;
    }
    return kept;
}

fn ceilMultiple(value: usize, multiple: usize) usize {
    return (value / multiple + @intFromBool(value % multiple != 0)) * multiple;
}

fn contains(xs: []const usize, value: usize) bool {
    for (xs) |x| if (x == value) return true;
    return false;
}

pub const Index = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    by_key: std.AutoHashMap(u64, usize),
    clock: u64 = 0,
    next_seq: u64 = 1,

    pub fn init(gpa: std.mem.Allocator) Index {
        return .{ .gpa = gpa, .by_key = std.AutoHashMap(u64, usize).init(gpa) };
    }

    /// Reserve all index storage during runtime setup so inserting a committed save cannot grow host buffers.
    pub fn reserve(i: *Index, count: usize) !void {
        if (count > @as(usize, std.math.maxInt(u32))) return error.IndexCapacityTooLarge;
        try i.nodes.ensureTotalCapacity(i.gpa, count);
        try i.by_key.ensureTotalCapacity(@intCast(count));
    }

    pub fn deinit(i: *Index) void {
        i.nodes.deinit(i.gpa);
        i.by_key.deinit();
    }

    pub fn get(i: *const Index, key: u64) ?*const Node {
        const at = i.by_key.get(key) orelse return null;
        return &i.nodes.items[at];
    }

    pub fn insert(i: *Index, node_in: Node) !void {
        if (i.nodes.items.len >= max_nodes) return error.IndexFull;
        if (node_in.at == 0 or node_in.alignment == 0 or node_in.at % node_in.alignment != 0 or node_in.at % piece_rows != 0 or node_in.base_at >= node_in.at) return error.BadNode;
        if (i.by_key.contains(node_in.key)) return error.DuplicateNode;
        if (node_in.parent_key != 0) {
            const p = i.by_key.get(node_in.parent_key) orelse return error.MissingParent;
            if (i.nodes.items[p].at > node_in.base_at) return error.BadParent;
            i.nodes.items[p].children +|= 1;
        } else if (node_in.base_at != 0) return error.MissingParent;
        try i.nodes.append(i.gpa, node_in);
        errdefer i.nodes.items.len -= 1;
        try i.by_key.put(node_in.key, i.nodes.items.len - 1);
        i.next_seq = @max(i.next_seq, node_in.seq +| 1);
    }

    pub fn touch(i: *Index, key: u64) void {
        const at = i.by_key.get(key) orelse return;
        i.clock +|= 1;
        i.nodes.items[at].last_used = i.clock;
    }

    pub fn touchChain(i: *Index, leaf_key: u64) void {
        var key = leaf_key;
        var traversed: usize = 0;
        while (key != 0 and traversed < max_nodes) : (traversed += 1) {
            const idx = i.by_key.get(key) orelse break;
            i.clock +|= 1;
            i.nodes.items[idx].last_used = i.clock;
            key = i.nodes.items[idx].parent_key;
        }
    }

    /// Exact-token lookup at every aligned boundary; key collisions are confirmed against ids by the reader.
    pub fn lookup(i: *const Index, ids: []const u32, after: usize, a: usize) ?Node {
        if (a == 0) return null;
        const limit = @min((ids.len -| 1) / a * a, ids.len);
        var h = std.hash.Wyhash.init(0x1e47);
        var at: usize = 0;
        var best: ?Node = null;
        while (at < limit) {
            const end = @min(at + piece_rows, limit);
            h.update(std.mem.sliceAsBytes(ids[at..end]));
            at = end;
            if (at <= after or at % a != 0) continue;
            var copy = h;
            const key = copy.final();
            const idx = i.by_key.get(key) orelse continue;
            const n = i.nodes.items[idx];
            if (n.at == at and n.alignment == a and (best == null or n.at > best.?.at)) best = n;
        }
        return best;
    }

    pub fn existingOnPrefix(i: *const Index, ids: []const u32, a: usize, limit_le: usize, out_at: []usize, deepest_le: *?Node) usize {
        deepest_le.* = null;
        if (a == 0) return 0;
        const limit = @min((ids.len -| 1) / a * a, ids.len);
        var h = std.hash.Wyhash.init(0x1e47);
        var at: usize = 0;
        var count: usize = 0;
        while (at < limit) {
            const end = @min(at + piece_rows, limit);
            h.update(std.mem.sliceAsBytes(ids[at..end]));
            at = end;
            if (at % a != 0) continue;
            var copy = h;
            const idx = i.by_key.get(copy.final()) orelse continue;
            const node = i.nodes.items[idx];
            if (node.at != at or node.alignment != a) continue;
            if (count < out_at.len) out_at[count] = at;
            count += 1;
            if (at <= limit_le and (deepest_le.* == null or at > deepest_le.*.?.at)) deepest_le.* = node;
        }
        return @min(count, out_at.len);
    }

    pub fn insertPending(i: *Index, node_in: Node) !void {
        if (i.nodes.items.len == max_nodes) return error.IndexFull;
        if (node_in.at == 0 or node_in.alignment == 0 or node_in.at % node_in.alignment != 0 or node_in.at % piece_rows != 0 or node_in.base_at >= node_in.at) return error.BadNode;
        if (i.by_key.contains(node_in.key)) return error.DuplicateNode;
        var node = node_in;
        node.pending = true;
        if (node.parent_key != 0) {
            const parent_idx = i.by_key.get(node.parent_key) orelse return error.MissingParent;
            const parent = &i.nodes.items[parent_idx];
            if (parent.at != node.base_at) return error.BadParent;
            node.depth = parent.depth +| 1;
            parent.pending_children +|= 1;
        } else if (node.base_at != 0) return error.MissingParent else node.depth = 0;
        i.nodes.appendAssumeCapacity(node);
        i.by_key.putAssumeCapacity(node.key, i.nodes.items.len - 1);
        i.next_seq = @max(i.next_seq, node.seq +| 1);
    }

    pub fn commitPending(i: *Index, key: u64) void {
        const idx = i.by_key.get(key) orelse return;
        const parent_key = i.nodes.items[idx].parent_key;
        i.nodes.items[idx].pending = false;
        if (parent_key != 0) if (i.by_key.get(parent_key)) |parent_idx| {
            i.nodes.items[parent_idx].pending_children -|= 1;
            i.nodes.items[parent_idx].children +|= 1;
        };
        i.clock +|= 1;
        i.nodes.items[idx].last_used = i.clock;
    }

    pub fn dropPending(i: *Index, key: u64) void {
        const idx = i.by_key.get(key) orelse return;
        const node = i.nodes.items[idx];
        if (!node.pending or node.pending_children != 0) return;
        if (node.parent_key != 0) {
            if (i.by_key.get(node.parent_key)) |parent_idx| {
                i.nodes.items[parent_idx].pending_children -|= 1;
            }
        }
        _ = i.by_key.remove(key);
        _ = i.nodes.swapRemove(idx);
        if (idx < i.nodes.items.len) selfFixMap(i, idx);
    }

    /// Deterministic childless LRU victim. Caller applies file removal and then removes the node.
    pub fn victim(i: *const Index) ?Node {
        var best: ?Node = null;
        for (i.nodes.items) |n| {
            if (n.pending or n.children != 0 or n.pending_children != 0 or n.pinned) continue;
            if (best == null or n.last_used < best.?.last_used or (n.last_used == best.?.last_used and n.seq < best.?.seq)) best = n;
        }
        return best;
    }

    pub fn remove(i: *Index, key: u64) void {
        const idx = i.by_key.get(key) orelse return;
        const node = i.nodes.items[idx];
        if (node.children != 0 or node.pending_children != 0 or node.pending) return;
        if (node.parent_key != 0) if (i.by_key.get(node.parent_key)) |p| { i.nodes.items[p].children -|= 1; };
        _ = i.by_key.remove(key);
        _ = i.nodes.swapRemove(idx);
        if (idx < i.nodes.items.len) selfFixMap(i, idx);
    }

    /// Root-to-leaf chain of committed nodes strictly above `r`.
    pub fn chainAbove(i: *const Index, leaf_key: u64, r: usize, out: []Node) !usize {
        var reverse: [max_nodes]Node = undefined;
        var count: usize = 0;
        var key = leaf_key;
        var walked: usize = 0;
        while (key != 0 and walked < max_nodes) : (walked += 1) {
            const node_ptr = i.get(key) orelse return error.ChainBroken;
            const node = node_ptr.*;
            if (node.at <= r) break;
            if (node.pending) return error.ChainPending;
            if (count == reverse.len) return error.ChainTooLong;
            reverse[count] = node;
            count += 1;
            key = node.parent_key;
        }
        if (walked == max_nodes) return error.ChainBroken;
        if (count > out.len) return error.ChainTooLong;
        if (count == 0 or reverse[0].key != leaf_key) return error.ChainBroken;
        for (0..count) |idx| out[idx] = reverse[count - idx - 1];
        for (1..count) |idx| if (out[idx].parent_key != out[idx - 1].key) return error.ChainBroken;
        return count;
    }

    /// Remove a committed node and its committed descendants, leaf-first.
    pub fn removeSubtree(i: *Index, key: u64, out: []Node) usize {
        var found: [max_nodes]Node = undefined;
        var selected: [max_nodes]bool = @splat(false);
        var count: usize = 0;
        var changed = true;
        while (changed) {
            changed = false;
            for (i.nodes.items) |node| {
                const node_idx = i.by_key.get(node.key) orelse continue;
                if (selected[node_idx]) continue;
                var belongs = node.key == key;
                if (!belongs) {
                    for (i.nodes.items, 0..) |ancestor, idx| {
                        if (selected[idx] and node.parent_key == ancestor.key) { belongs = true; break; }
                    }
                }
                if (!belongs) continue;
                if (node.pending) {
                    std.log.warn("prefix cache: subtree removal skipped pending node {x}", .{node.key});
                    continue;
                }
                selected[node_idx] = true;
                found[count] = node;
                count += 1;
                changed = true;
            }
        }
        const output_count: usize = @min(count, out.len);
        var reverse = count;
        var written: usize = 0;
        while (reverse > 0) {
            reverse -= 1;
            if (written < output_count) out[written] = found[reverse];
            i.remove(found[reverse].key);
            written += 1;
        }
        return output_count;
    }

    fn selfFixMap(i: *Index, idx: usize) void {
        i.by_key.put(i.nodes.items[idx].key, idx) catch {};
    }

    pub fn makeRoom(i: *Index, budget: u64, floor: u64, incoming: u64, free_bytes: u64, used_bytes: *u64) bool {
        var free = free_bytes;
        while (used_bytes.* +| incoming > budget or free < floor +| incoming) {
            const n = i.victim() orelse return false;
            // The caller unlinks the file; a deletion error is handled as a skipped save at the protocol layer.
            i.remove(n.key);
            used_bytes.* -|= n.bytes;
            free +|= n.bytes;
        }
        return true;
    }

    pub fn clear(i: *Index) void {
        i.nodes.clearRetainingCapacity();
        i.by_key.clearRetainingCapacity();
        i.clock = 0;
        i.next_seq = 1;
    }

    pub fn startupDigest(i: *const Index) kvsave.Digest {
        const Tuple = struct { key: u64, at: u64, base_at: u64, parent_key: u64, seq: u64 };
        var tuples: [max_nodes]Tuple = undefined;
        for (i.nodes.items, 0..) |n, j| tuples[j] = .{ .key = n.key, .at = n.at, .base_at = n.base_at, .parent_key = n.parent_key, .seq = n.seq };
        std.mem.sort(Tuple, tuples[0..i.nodes.items.len], {}, struct {
            fn less(_: void, x: Tuple, y: Tuple) bool {
                if (x.key != y.key) return x.key < y.key;
                if (x.at != y.at) return x.at < y.at;
                return x.seq < y.seq;
            }
        }.less);
        var sha = std.crypto.hash.sha2.Sha256.init(.{});
        for (tuples[0..i.nodes.items.len]) |t| {
            for ([_]u64{ t.key, t.at, t.base_at, t.parent_key, t.seq }) |word| {
                var bytes: [8]u8 = undefined;
                std.mem.writeInt(u64, &bytes, word, .little);
                sha.update(&bytes);
            }
        }
        var out: kvsave.Digest = undefined;
        sha.final(&out);
        return out;
    }

    /// Rebuild in-memory topology from B1's header-only scan; returns false when peer state requires a directory wipe.
    pub fn reconcileRecords(i: *Index, records: []const Node, peer_digest: kvsave.Digest) !bool {
        i.clear();
        var added: usize = 0;
        var progress = true;
        while (progress and added < records.len) {
            progress = false;
            for (records) |record| {
                if (i.by_key.contains(record.key)) continue;
                if (record.parent_key != 0 and !i.by_key.contains(record.parent_key)) continue;
                var node = record;
                node.pending = false;
                node.pinned = false;
                node.last_used = node.seq;
                try i.insert(node);
                added += 1;
                progress = true;
            }
        }
        // A header with an absent/corrupt parent is intentionally omitted along with its descendants.
        i.next_seq = 1;
        for (i.nodes.items) |node| i.next_seq = @max(i.next_seq, node.seq +| 1);
        const local_digest = i.startupDigest();
        if (!std.mem.eql(u8, &local_digest, &peer_digest)) {
            i.clear();
            return false;
        }
        return true;
    }

    /// Read v2 headers from a single rank directory. Corrupt or foreign files are logged and skipped.
    pub fn reconcile(i: *Index, io: std.Io, dir_path: []const u8, identity: kvsave.Digest, rank: u32, expert_rows: usize, peer_digest: ?kvsave.Digest) !void {
        const a = try kvsave.cutAlignment(expert_rows);
        var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true });
        defer dir.close(io);
        var records: std.ArrayList(Node) = .empty;
        defer records.deinit(i.gpa);
        var it = dir.iterate();
        while (try it.next(io)) |entry| {
            if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".tfkv")) continue;
            const path = try std.fmt.allocPrintSentinel(i.gpa, "{s}/{s}", .{ dir_path, entry.name }, 0);
            defer i.gpa.free(path);
            const h = kvsave.Reader.headerOnly(i.gpa, path, identity) catch |err| {
                std.log.warn("prefix cache: skipping invalid header {s}: {s}", .{ path, @errorName(err) });
                continue;
            };
            if (h.rank != rank or (h.world != 2 and h.world != 3) or h.at > std.math.maxInt(usize) or h.base_at > std.math.maxInt(usize) or
                h.at % a != 0 or h.at % piece_rows != 0) {
                std.log.warn("prefix cache: skipping incompatible header {s}", .{path});
                continue;
            }
            const body_len = std.math.cast(usize, h.body_len) orelse continue;
            const ids_len = std.math.cast(usize, h.ids_len) orelse continue;
            try records.append(i.gpa, .{ .key = h.key, .at = @intCast(h.at), .alignment = a, .base_at = @intCast(h.base_at), .parent_key = h.parent_key, .seq = h.seq, .bytes = kvsave.fileBytes(body_len, ids_len) });
        }
        _ = try i.reconcileRecords(records.items, peer_digest orelse blk: {
            var local = Index.init(i.gpa);
            defer local.deinit();
            // Build the same valid parent graph without comparing against a peer yet.
            var progress = true;
            while (progress) {
                progress = false;
                for (records.items) |record| {
                    if (local.by_key.contains(record.key)) continue;
                    if (record.parent_key != 0 and !local.by_key.contains(record.parent_key)) continue;
                    try local.insert(record);
                    progress = true;
                }
            }
            break :blk local.startupDigest();
        });
    }
};

test "save alignment is the lcm of 512 and expert_rows" {
    try std.testing.expectEqual(@as(usize, 1024), try kvsave.cutAlignment(1024));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.cutAlignment(2048));
    try std.testing.expectEqual(@as(usize, 4096), try kvsave.cutAlignment(4096));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.roundCutDown(2559, 1024));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.roundCutDown(4095, 2048));
    try std.testing.expectEqual(@as(usize, 4096), try kvsave.roundCutDown(8191, 4096));
}

test "victim order is stable and respects child and pin state" {
    var i = Index.init(std.testing.allocator);
    defer i.deinit();
    try i.insert(.{ .key = 1, .at = 512, .seq = 2, .bytes = 100, .last_used = 7 });
    try i.insert(.{ .key = 2, .at = 1024, .base_at = 512, .parent_key = 1, .seq = 3, .bytes = 200, .last_used = 1 });
    try std.testing.expectEqual(@as(u64, 2), i.victim().?.key);
    i.nodes.items[1].pinned = true;
    try std.testing.expect(i.victim() == null);
}

test "Index hit miss and LRU eviction stay inside the configured byte budget" {
    var index = Index.init(std.testing.allocator);
    defer index.deinit();
    try index.reserve(4);
    try index.insert(.{ .key = 10, .at = 512, .seq = 1, .bytes = 80, .last_used = 1 });
    try index.insert(.{ .key = 20, .at = 1024, .seq = 2, .bytes = 80, .last_used = 2 });
    try std.testing.expect(index.get(99) == null);
    var used: u64 = 160;
    try std.testing.expect(index.makeRoom(160, 0, 80, 80, &used));
    try std.testing.expectEqual(@as(u64, 80), used);
    try std.testing.expect(index.get(10) == null);
    try std.testing.expect(index.get(20) != null);
}

test "Runtime cut planning uses fixed capacity and caps aligned cuts at eighteen" {
    var index = Index.init(std.testing.allocator);
    defer index.deinit();
    var runtime: Runtime = .{
        .gpa = std.testing.allocator,
        .pool = undefined,
        .index = index,
        .identity = @splat(0),
        .root = &.{},
        .rank = 0,
        .options = .{ .expert_rows = 1024, .step = 1024, .span = 1024 },
        .budget_bytes = 0,
        .floor_bytes = 0,
    };
    const prompt = try std.testing.allocator.alloc(u32, 65537);
    defer std.testing.allocator.free(prompt);
    @memset(prompt, 7);
    const planned_cuts = try runtime.plan(prompt, 0, &.{}, 60000);
    try std.testing.expect(planned_cuts.len <= 18);
    for (planned_cuts) |cut| try std.testing.expectEqual(@as(usize, 0), cut.at % 1024);
}

test "mock two rank snapshot agreement mismatch falls back without exposing partial state" {
    const MockRank = struct {
        kv: [128]u8 = @splat(0x31),
        indexer: [32]u8 = @splat(0x42),
        ring: [24]u8 = @splat(0x53),
        restored: bool = false,
        agree_calls: usize = 0,
        fn agree(self: *@This(), peer_ok: bool) bool {
            self.agree_calls += 1;
            return peer_ok;
        }
        fn save(self: *const @This(), slot: *@This()) void {
            slot.kv = self.kv;
            slot.indexer = self.indexer;
            slot.ring = self.ring;
        }
        fn load(self: *@This(), slot: *const @This(), peer_ok: bool) void {
            const local_ok = self.agree(peer_ok);
            if (!local_ok) {
                self.restored = false;
                return;
            }
            self.kv = slot.kv;
            self.indexer = slot.indexer;
            self.ring = slot.ring;
            self.restored = true;
        }
    };
    var cold: [2]MockRank = .{ .{}, .{} };
    var saved: [2]MockRank = .{ .{}, .{} };
    for (0..2) |rank| cold[rank].save(&saved[rank]);
    for (0..2) |rank| {
        cold[rank].kv = @splat(0xa1);
        cold[rank].indexer = @splat(0xb2);
        cold[rank].ring = @splat(0xc3);
        cold[rank].load(&saved[rank], true);
        try std.testing.expectEqualSlices(u8, &saved[rank].kv, &cold[rank].kv);
        try std.testing.expectEqualSlices(u8, &saved[rank].indexer, &cold[rank].indexer);
        try std.testing.expectEqualSlices(u8, &saved[rank].ring, &cold[rank].ring);
    }
    for (0..2) |rank| cold[rank].load(&saved[rank], false);
    try std.testing.expect(!cold[0].restored);
    try std.testing.expect(!cold[1].restored);
    try std.testing.expectEqual(cold[0].agree_calls, cold[1].agree_calls);
}

test "cut policy preserves system boundary and rounds every cut to expert groups" {
    const p = try cuts(std.testing.allocator, 32769, 0, &.{ 3001 }, 30001, .{ .expert_rows = 1024 }, &.{});
    defer std.testing.allocator.free(p);
    try std.testing.expect(p.len > 0);
    try std.testing.expectEqual(CutKind.system, p[0].kind);
    for (p) |c| try std.testing.expectEqual(@as(usize, 0), c.at % 1024);
}

test "startup digest ignores insertion order" {
    var a = Index.init(std.testing.allocator);
    defer a.deinit();
    var b = Index.init(std.testing.allocator);
    defer b.deinit();
    try a.insert(.{ .key = 11, .at = 512, .seq = 1 });
    try a.insert(.{ .key = 22, .at = 1024, .seq = 2 });
    try b.insert(.{ .key = 22, .at = 1024, .seq = 2 });
    try b.insert(.{ .key = 11, .at = 512, .seq = 1 });
    const da = a.startupDigest();
    const db = b.startupDigest();
    try std.testing.expectEqualSlices(u8, &da, &db);
}

test "budget takes the minimum and free floor takes the maximum" {
    const e = effectiveBudget(16, 12, 5, 9);
    try std.testing.expectEqual(@as(u64, 12), e.budget_bytes);
    try std.testing.expectEqual(@as(u64, 9), e.floor_bytes);
}

test "lookup returns the deepest aligned token-exact key after resident reuse" {
    var ids: [4096]u32 = undefined;
    for (&ids, 0..) |*id, n| id.* = @intCast(n * 17 + 3);
    var i = Index.init(std.testing.allocator);
    defer i.deinit();
    const key = kvsave.keyOf(ids[0..2048]);
    try i.insert(.{ .key = key, .at = 2048, .alignment = 1024, .seq = 1 });
    try std.testing.expectEqual(@as(usize, 2048), i.lookup(&ids, 1024, 1024).?.at);
    try std.testing.expect(i.lookup(&ids, 2048, 1024) == null);
}

test "chainAbove orders committed delta nodes and removeSubtree removes leaf first" {
    var index = Index.init(std.testing.allocator);
    defer index.deinit();
    try index.reserve(8);
    try index.insert(.{ .key = 11, .at = 2048, .alignment = 1024, .seq = 1 });
    try index.insert(.{ .key = 22, .at = 4096, .alignment = 1024, .base_at = 2048, .parent_key = 11, .seq = 2 });
    try index.insert(.{ .key = 33, .at = 6144, .alignment = 1024, .base_at = 4096, .parent_key = 22, .seq = 3 });
    var chain: [max_nodes]Node = undefined;
    const count = try index.chainAbove(33, 0, &chain);
    try std.testing.expectEqual(@as(usize, 3), count);
    try std.testing.expectEqual(@as(u64, 11), chain[0].key);
    try std.testing.expectEqual(@as(u64, 22), chain[1].key);
    try std.testing.expectEqual(@as(u64, 33), chain[2].key);
    var removed: [max_nodes]Node = undefined;
    const removed_count = index.removeSubtree(22, &removed);
    try std.testing.expectEqual(@as(usize, 2), removed_count);
    try std.testing.expectEqual(@as(u64, 33), removed[0].key);
    try std.testing.expectEqual(@as(u64, 22), removed[1].key);
    try std.testing.expect(index.get(11) != null);
    try std.testing.expect(index.get(22) == null);
    try std.testing.expect(index.get(33) == null);
}

test "existingOnPrefix matches lookup and the old full-hash scan" {
    var ids: [8192]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx * 17 + 13);
    var index = Index.init(std.testing.allocator);
    defer index.deinit();
    for ([_]usize{ 1024, 2048, 4096, 7168 }) |at| try index.insert(.{ .key = kvsave.keyOf(ids[0..at]), .at = at, .alignment = 1024, .seq = at });
    var actual: [max_nodes]usize = undefined;
    var deepest: ?Node = null;
    const count = index.existingOnPrefix(&ids, 1024, 4096, &actual, &deepest);
    var expected: [4]usize = undefined;
    var expected_count: usize = 0;
    for (index.nodes.items) |node| if (node.at <= ids.len and kvsave.keyOf(ids[0..node.at]) == node.key) {
        expected[expected_count] = node.at;
        expected_count += 1;
    };
    try std.testing.expectEqual(expected_count, count);
    for (actual[0..count], expected[0..expected_count]) |left, right| try std.testing.expectEqual(left, right);
    try std.testing.expectEqual(@as(usize, 4096), deepest.?.at);
    try std.testing.expectEqual(@as(usize, 7168), index.lookup(&ids, 4096, 1024).?.at);
}

test "saveHeader uses next_seq and the last accepted save as parent" {
    var index = Index.init(std.testing.allocator);
    try index.reserve(max_nodes);
    var ids: [4096]u32 = undefined;
    for (&ids, 0..) |*id, idx| id.* = @intCast(idx + 1);
    const parent_key = kvsave.keyOf(ids[0..1024]);
    try index.insert(.{ .key = parent_key, .at = 1024, .alignment = 1024, .seq = 1 });
    var runtime: Runtime = .{ .gpa = std.testing.allocator, .pool = null, .index = index, .identity = @splat(0), .root = &.{}, .rank = 0, .options = .{ .expert_rows = 1024, .step = 1024, .span = 1024 }, .budget_bytes = 0, .floor_bytes = 0 };
    defer runtime.index.deinit();
    _ = try runtime.plan(&ids, 1024, &.{}, ids.len);
    const skipped_header = try runtime.saveHeader(2048);
    try std.testing.expectEqual(parent_key, skipped_header.parent_key);
    try std.testing.expectEqual(@as(u64, 2), skipped_header.seq);
    const accepted_key = kvsave.keyOf(ids[0..2048]);
    try runtime.index.insertPending(.{ .key = accepted_key, .at = 2048, .alignment = 1024, .base_at = 1024, .parent_key = parent_key, .seq = skipped_header.seq, .bytes = 1 });
    runtime.chain_key = accepted_key;
    runtime.chain_at = 2048;
    const next_header = try runtime.saveHeader(3072);
    try std.testing.expectEqual(accepted_key, next_header.parent_key);
    try std.testing.expectEqual(@as(u64, 3), next_header.seq);
}

test "a saved cut is shared by any prompt with the same aligned prefix and the longest shared cut wins" {
    const gpa = std.testing.allocator;
    const p1 = try gpa.alloc(u32, 16602);
    defer gpa.free(p1);
    for (p1, 0..) |*id, k| id.* = @intCast(k * 7 + 5);
    var runtime: Runtime = .{ .gpa = gpa, .pool = null, .index = Index.init(gpa), .identity = @splat(0), .root = &.{}, .rank = 0, .options = .{ .expert_rows = 1024 }, .budget_bytes = 0, .floor_bytes = 0 };
    defer runtime.index.deinit();
    try runtime.index.reserve(max_nodes);
    // P1 = a 10000-token system text + a user turn: a system cut at 9216 and a history cut at the limit 16384.
    const planned = try runtime.plan(p1, 0, &.{10000}, 0);
    try std.testing.expectEqual(@as(usize, 2), planned.len);
    try std.testing.expectEqual(@as(usize, 9216), planned[0].at);
    try std.testing.expectEqual(CutKind.system, planned[0].kind);
    try std.testing.expectEqual(@as(usize, 16384), planned[1].at);
    for (planned) |cut| {
        const h = try runtime.saveHeader(cut.at);
        try runtime.index.insertPending(.{ .key = h.key, .at = cut.at, .alignment = 1024, .base_at = @intCast(h.base_at), .parent_key = h.parent_key, .seq = h.seq, .bytes = 1 });
        runtime.index.commitPending(h.key);
        runtime.chain_key = h.key;
        runtime.chain_at = cut.at;
    }
    // A new conversation that shares 16500 tokens hits the deepest saved cut.
    const p2 = try gpa.alloc(u32, 16605);
    defer gpa.free(p2);
    @memcpy(p2[0..16500], p1[0..16500]);
    for (p2[16500..], 0..) |*id, k| id.* = @intCast(900_000 + k);
    try std.testing.expectEqual(@as(usize, 16384), runtime.lookup(p2, 0).?.at);
    // Shares 12000 tokens: the 16384 cut is not shared, the system cut is.
    const p3 = try gpa.dupe(u32, p1);
    defer gpa.free(p3);
    for (p3[12000..], 0..) |*id, k| id.* = @intCast(900_000 + k);
    try std.testing.expectEqual(@as(usize, 9216), runtime.lookup(p3, 0).?.at);
    // Resident rows already cover 9216: nothing deeper is shared.
    try std.testing.expect(runtime.lookup(p3, 9216) == null);
    // One token before the system cut differs: miss.
    p3[9215] = 4_000_000_000;
    try std.testing.expect(runtime.lookup(p3, 0) == null);
    // A different first token: miss.
    p2[0] = 4_000_000_001;
    try std.testing.expect(runtime.lookup(p2, 0) == null);
}

test "reconcile indexes valid v2 headers and skips a corrupt file" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/cache/rank0", .{tmp.sub_path});
    defer gpa.free(root);
    try std.Io.Dir.cwd().createDirPath(io, root);
    const identity = kvsave.identityOf(@splat(1), @splat(2), @splat(3), 0, 2, 512);
    const ids_a = try gpa.alloc(u32, 512);
    defer gpa.free(ids_a);
    const ids_b = try gpa.alloc(u32, 512);
    defer gpa.free(ids_b);
    for (ids_a, 0..) |*id, n| id.* = @intCast(n + 11);
    for (ids_b, 0..) |*id, n| id.* = @intCast(n + 900);
    const body = try gpa.alignedAlloc(u8, .fromByteUnits(kvsave.alignment), kvsave.chunk_bytes);
    defer gpa.free(body);
    @memset(body, 0x5a);
    const names = [_][]const u8{ "a.tfkv", "b.tfkv" };
    const id_sets = [_][]const u32{ ids_a, ids_b };
    var expected = Index.init(gpa);
    defer expected.deinit();
    for (names, id_sets, 0..) |name, ids, n| {
        const path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ root, name }, 0);
        defer gpa.free(path);
        var writer = try kvsave.Writer.begin(gpa, path, identity, 0, 2, ids, 0, ids.len, 0, n + 1, kvsave.chunk_bytes, 0, 0);
        try writer.writeChunkAt(0, body, body.len);
        try writer.finish();
        try expected.insert(.{ .key = kvsave.keyOf(ids), .at = ids.len, .alignment = 512, .seq = n + 1, .bytes = kvsave.fileBytes(kvsave.chunk_bytes, ids.len) });
    }
    const corrupt = try std.fmt.allocPrint(gpa, "{s}/broken.tfkv", .{root});
    defer gpa.free(corrupt);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = corrupt, .data = "not a v2 header" });
    var actual = Index.init(gpa);
    defer actual.deinit();
    try actual.reconcile(io, root, identity, 0, 512, expected.startupDigest());
    try std.testing.expectEqual(@as(usize, 2), actual.nodes.items.len);
    try std.testing.expect(actual.get(kvsave.keyOf(ids_a)) != null);
    try std.testing.expect(actual.get(kvsave.keyOf(ids_b)) != null);
}


// ---- B1: IO resources (pinned slots, reader/writer workers). Policy and index above are B2. ----

// IO API
//   kvsave.Writer.begin(gpa,path,identity,rank,world,ids_full,base_at,at,parent_key,seq,body_len,ring_from,ring_span)
//   Writer.writeChunkAt(i, aligned_buf, n), Writer.seal(), Writer.publish(), Writer.abort()
//   kvsave.Reader.open(gpa,path,identity,rank,world,ids_full,base_at,at,body_len)
//   Reader.readChunkAt(i, aligned_buf), Reader.close()
//   cutAlignment(expert_rows), roundCutDown(cut,expert_rows), Pool.init(driver,context), Pool.run(...), Pool.deinit()
// Ring device hooks: drafts.Host.ringView(ctx) -> {ptr,len,from}; drafts.Host.ringAdopt(ctx,from).

pub const Pool = struct {
    slots: [10]uma.PinnedHost,
    ctx: *const cuda.Context,
    reader_workers: usize = 4,
    /// Slots 0..7 are reader slots; 8..9 are writer slots.
    pub fn init(d: *const cuda.Driver, ctx: *const cuda.Context, reader_workers: usize) !Pool {
        if (reader_workers == 0 or reader_workers > 4) return error.BadWorkerCount;
        var slots: [10]uma.PinnedHost = undefined;
        var made: usize = 0;
        errdefer for (slots[0..made]) |*slot| slot.free();
        for (&slots) |*slot| {
            slot.* = try uma.PinnedHost.alloc(d, kvsave.chunk_bytes);
            made += 1;
        }
        return .{ .slots = slots, .ctx = ctx, .reader_workers = reader_workers };
    }

    pub fn deinit(p: *Pool) void {
        for (&p.slots) |*slot| slot.free();
        p.* = undefined;
    }

    pub fn bytes(p: *Pool, i: usize) []align(kvsave.alignment) u8 {
        return @alignCast(p.slots[i].bytes());
    }

    pub const WorkFn = *const fn (ctx: *anyopaque, worker: usize, a: []align(kvsave.alignment) u8, b: []align(kvsave.alignment) u8) anyerror!void;
    const Work = struct { pool: *Pool, ctx: *anyopaque, worker: usize, writer: bool, fn_ptr: WorkFn, failed: bool = false };

    fn workerMain(w: *Work) void {
        w.pool.ctx.makeCurrent() catch {
            w.failed = true;
            return;
        };
        const slot0: usize = if (w.writer) 8 else w.worker * 2;
        w.fn_ptr(w.ctx, w.worker, w.pool.bytes(slot0), w.pool.bytes(slot0 + 1)) catch {
            w.failed = true;
        };
    }

    /// Run 1..4 reader workers (two slots each) or one writer worker (the reserved slots 8 and 9).
    pub fn run(p: *Pool, ctx: *anyopaque, count: usize, writer: bool, fn_ptr: WorkFn) !void {
        if (count == 0 or (writer and count != 1)) return error.BadWorkerCount;
        const effective_count = if (writer) count else @min(count, p.reader_workers);
        var work: [4]Work = undefined;
        var threads: [4]std.Thread = undefined;
        var made: usize = 0;
        errdefer for (threads[0..made]) |thread| thread.join();
        for (0..effective_count) |i| {
            work[i] = .{ .pool = p, .ctx = ctx, .worker = i, .writer = writer, .fn_ptr = fn_ptr };
            threads[i] = try std.Thread.spawn(.{}, workerMain, .{&work[i]});
            made += 1;
        }
        for (0..effective_count) |i| threads[i].join();
        made = 0;
        for (work[0..effective_count]) |w| if (w.failed) return error.IoWorkerFailed;
    }
};

pub fn clampReaderWorkers(requested: usize) usize {
    const bounded: usize = @min(@max(requested, 1), 4);
    return bounded;
}

test "reader worker requests clamp to the four pinned reader pairs" {
    try std.testing.expectEqual(@as(usize, 1), clampReaderWorkers(0));
    try std.testing.expectEqual(@as(usize, 4), clampReaderWorkers(4));
    try std.testing.expectEqual(@as(usize, 4), clampReaderWorkers(9));
}

test "prefix cache cut alignment includes expert row geometry" {
    try std.testing.expectEqual(@as(usize, 1024), try kvsave.cutAlignment(1024));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.cutAlignment(2048));
    try std.testing.expectEqual(@as(usize, 4096), try kvsave.cutAlignment(4096));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.roundCutDown(2560, 2048));
    try std.testing.expectEqual(@as(usize, 2048), try kvsave.roundCutDown(3071, 1024));
}
