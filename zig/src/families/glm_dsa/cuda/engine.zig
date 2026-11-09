// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! GLM-5.3 CUDA engine for one TP2 rank (WP9): prefill, verify, keep and draw over Forward, WP10's sampler, and
//! keyed CUDA graphs (cuda.graph_cache) for decode windows. Shape after ashhart/TensorFold@d1018b31:
//! zig/src/families/glm/cuda_engine.zig (Apache-2.0); greedy ties and candidate sampling follow
//! bertholomus/bertholomus/TensorFold@ddfad35 glm5_next/cuda/decode.py (Apache-2.0).
const std = @import("std");
const cuda = @import("cuda");
const forward = @import("forward.zig");
const sampler = @import("sampler.zig");
const state = @import("state.zig");
const uma = @import("uma.zig");
const mk_defer = @import("mk_defer");
pub const side_park = @import("park.zig");
pub const capture_mod = @import("capture.zig");
test { _ = capture_mod; }
test { _ = side_park; }

/// A captured graph's key includes every layout switch baked into its decode launch, along with the window rows and
/// chunk bucket (prefill: row bucket; decode: position bucket fixing indexer key count and dense/sparse split).
pub const GraphKey = struct {
    rows: u32,
    chunk_bucket: u32,
    ablate: bool = false,
    head_mode: u8 = 0,
    candidate_k: u32 = 0,
};

pub fn graphKey(rows: usize, chunk_rows: usize) !GraphKey {
    if (rows == 0 or rows > 65535 or chunk_rows == 0 or chunk_rows > 1 << 30) return error.BadGraphKey;
    return .{ .rows = @intCast(rows), .chunk_bucket = @intCast(std.math.ceilPowerOfTwo(usize, chunk_rows) catch return error.BadGraphKey) };
}

/// Prefill chunk buckets of prefill doc §3.7 (16k, 8k, 4k, and 1k for a short tail).
pub fn prefillBucket(rows: usize) !usize {
    for ([_]usize{ 1024, 4096, 8192, 16384 }) |b| if (rows <= b) return b;
    return error.BadGraphKey;
}

pub fn decodeKey(rows: usize, last_position: usize) !GraphKey {
    if (rows == 0 or rows > forward.max_decode_rows) return error.BadGraphKey;
    return .{ .rows = @intCast(rows), .chunk_bucket = @intCast(forward.decodeKeyBucket(last_position)) };
}

/// TF_GLM53_MEMSLIM=1: one live decode graph per row width (and ablate mode); a bucket change swaps it.
pub fn sameGraphSlot(a: GraphKey, b: GraphKey) bool {
    return a.rows == b.rows and a.ablate == b.ablate;
}

pub fn verifyRowsAllowed(rows: usize, head_rows: usize, out_rows: usize) bool {
    return rows != 0 and rows <= forward.max_decode_rows and rows <= head_rows and out_rows >= rows;
}

pub fn prefillKey(rows: usize) !GraphKey {
    return .{ .rows = @intCast(rows), .chunk_bucket = @intCast(try prefillBucket(rows)) };
}

fn withHeadKey(key_: GraphKey, head: forward.HeadMode) GraphKey {
    var key = key_;
    switch (head) {
        .none, .greedy => {},
        .candidates => |width| {
            key.head_mode = @intCast(@intFromEnum(std.meta.activeTag(head)));
            key.candidate_k = @intCast(width);
        },
    }
    return key;
}

fn drawFullFallback(allocator: std.mem.Allocator, candidates: []const sampler.Candidate, log_z: f32, seed: u64, position: u64, rule: sampler.Sampling) !u32 {
    if (rule.top_k == 0 and rule.temperature == 1 and rule.top_p > 0 and rule.top_p < 1)
        return sampler.drawNucleusWithFullLse(allocator, candidates, log_z, seed, position, rule);
    return sampler.drawCandidates(allocator, candidates, seed, position, rule);
}

fn drawNucleusWithFallback(allocator: std.mem.Allocator, candidates: []const sampler.Candidate, full_logits: []const f32, force_full: bool, log_z: f32, seed: u64, position: u64, rule: sampler.Sampling) !u32 {
    if (!force_full) {
        return sampler.drawNucleusWithLse(allocator, candidates, log_z, seed, position, rule) catch |err| {
            if (err != error.NucleusExceedsCandidateK) return err;
            return drawFullNucleusFromLogits(allocator, full_logits, log_z, seed, position, rule);
        };
    }
    return drawFullNucleusFromLogits(allocator, full_logits, log_z, seed, position, rule);
}

fn drawFullNucleusFromLogits(allocator: std.mem.Allocator, logits: []const f32, log_z: f32, seed: u64, position: u64, rule: sampler.Sampling) !u32 {
    if (logits.len == 0) return error.MissingFallbackVocabulary;
    const full = try allocator.alloc(sampler.Candidate, logits.len);
    defer allocator.free(full);
    var count: usize = 0;
    for (logits, 0..) |value, id| {
        if (std.math.isNan(value) or value == -std.math.inf(f32)) continue;
        full[count] = .{ .value = value, .id = @intCast(id) };
        count += 1;
    }
    return drawFullFallback(allocator, full[0..count], log_z, seed, position, rule);
}

fn nucleusFitsCandidates(allocator: std.mem.Allocator, candidates: []const sampler.Candidate, log_z: f32, rule: sampler.Sampling) !bool {
    _ = sampler.drawNucleusWithLse(allocator, candidates, log_z, 0, 0, rule) catch |err| {
        if (err == error.NucleusExceedsCandidateK) return false;
        return err;
    };
    return true;
}

fn verifyGraphKey(rows: usize, last_position: usize, ablate: bool, head: forward.HeadMode) !GraphKey {
    var key = try decodeKey(rows, last_position);
    key.ablate = ablate;
    return withHeadKey(key, head);
}

/// Append input and eagerly pass only complete pieces that have a following token.
/// Keeping this buffer operation separate lets the host tests exercise the same
/// flush rule without constructing a CUDA Forward.
fn appendFeed(comptime Context: type, gpa: std.mem.Allocator, pending: *std.ArrayList(u32), ids: []const u32, piece_rows: usize, ctx: *Context, comptime prefill_rows_fn: fn (*Context, []const u32) anyerror!void) !void {
    if (piece_rows == 0) return error.AppendNotAligned;
    try pending.appendSlice(gpa, ids);
    while (pending.items.len > piece_rows) {
        try prefill_rows_fn(ctx, pending.items[0..piece_rows]);
        const remain = pending.items.len - piece_rows;
        std.mem.copyForwards(u32, pending.items[0..remain], pending.items[piece_rows..]);
        pending.items.len = remain;
    }
}

/// WP10 marks an absent candidate (a shard with fewer than k finite logits) with id 0xFFFFFFFF and value -inf.
pub fn absentCandidate(c: sampler.Candidate) bool {
    return c.id == 0xFFFFFFFF and c.value == -std.math.inf(f32);
}

/// Device-to-host reads of head results. Production synchronizes the stream, then copies.
pub const Download = struct {
    ctx: ?*anyopaque = null,
    group: ?*const cuda.tp_group.Group = null,
    download_fn: *const fn (ctx: ?*anyopaque, stream: cuda.Stream, src: u64, dst: []u8) anyerror!void = deviceCopy,

    fn deviceCopy(_: ?*anyopaque, stream: cuda.Stream, src: u64, dst: []u8) anyerror!void {
        try stream.synchronize();
        try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "GLM-5.3 head results");
    }
};

/// The owned decode cache's settings. warm = 1: a key's first window runs eagerly and the next one is captured. The
/// first decode window is the communicator's first point-to-point exchange (the DCP2 attention partials; prefill has
/// none below its second chunk) and the first launch of the decode-only kernels: NCCL connects the peer lazily on that
/// first send/recv, which must not happen inside a stream capture (with warm = 0, decode 0 was the capture). Warm-up
/// keeps every lazy first use out of the graph; the replays are bit-equal to eager (tf-glm53-graph-replay).
// Keep the graph pool bounded by its existing admission reserve; head-mode variants beyond this count are evicted.
pub const decode_graphs: cuda.graph_cache.Settings = .{ .warm = 1, .max = 48 };

pub const Drafter = struct {
    ctx: *anyopaque,
    window: usize,
    taps_prefill: forward.taps_mod.Taps,
    taps_step: forward.taps_mod.Taps,
    restart_fn: *const fn (ctx: *anyopaque, from: usize) anyerror!void,
    prompt_fn: *const fn (ctx: *anyopaque, start: usize, rows: usize) anyerror!void,
    verify_fn: *const fn (ctx: *anyopaque, rows: usize, pos0: usize) anyerror!void,
    draft_fn: *const fn (ctx: *anyopaque, pos: usize, n: usize, anchor: u32, out: ?[]u32, probs: ?[]f64) anyerror!void,
    state_fn: ?*const fn (ctx: *anyopaque, dst: []u8) anyerror!i64 = null,
    adopt_fn: ?*const fn (ctx: *anyopaque, src: []const u8, from: i64) anyerror!void = null,
    ring_adopt_fn: ?*const fn (ctx: *anyopaque, from: i64) anyerror!void = null,
    /// TF_GLM53_SLOTS (each slot's Drafter): a shared window's part wrote `rows` tap rows at row `row0` of taps_step
    /// (`first`: the window's first part, the one verify every part shares). Null: shared windows tap nothing.
    verify_at_fn: ?*const fn (ctx: *anyopaque, row0: usize, rows: usize, pos0: usize, first: bool) anyerror!void = null,
    /// TF_GLM53_SLOTS: the `n` kept rows of the slot's last verify into its context, no drafts (a depth-0 round).
    absorb_fn: ?*const fn (ctx: *anyopaque, pos: usize, n: usize) anyerror!void = null,
};

pub const Agreement = struct {
    ctx: *anyopaque,
    call_fn: *const fn (ctx: *anyopaque, words: []u32) anyerror!void,
};

/// One save cut decoded without importing the follower module (also the glm_follow module root).
pub const SnapCut = struct { at: usize, kind: u32, key: u64, parent_key: u64, parent_at: usize, seq: u64 };

pub fn snapCutOf(h: anytype) SnapCut {
    return .{
        .at = @intCast(h.position),
        .kind = h.slot,
        .key = std.mem.readInt(u64, h.prefix_hash[0..8], .little),
        .parent_key = std.mem.readInt(u64, h.prefix_hash[8..16], .little),
        .parent_at = @intCast(std.mem.readInt(u64, h.prefix_hash[16..24], .little)),
        .seq = std.mem.readInt(u64, h.prefix_hash[24..32], .little),
    };
}

pub const SnapOps = struct {
    ctx: *anyopaque,
    save_fn: *const fn (ctx: *anyopaque, cut: SnapCut) anyerror!void,
    load_fn: *const fn (ctx: *anyopaque, r: usize, ids: []const u32, leaf_key: u64) anyerror!usize,
    commit_fn: *const fn (ctx: *anyopaque, n: usize) anyerror!void,
};

const DraftFill = union(enum) { restart: usize, prompt: struct { start: usize, rows: usize } };

/// TF_GLM53_SLOTS: request slots a shared decode window takes at most.
pub const max_shared = 8;

/// TF_GLM53_SLOTS: a request slot's sequence while another slot is bound. Engine.bind moves a slot's fields between
/// its Seq and the Engine, so every single-slot step runs unchanged on the bound slot.
pub const Seq = struct {
    tapped: usize = 0,
    tap_high: usize = 0,
    pos: usize = 0,
    frontier: usize = 0,
    history: std.ArrayList(u32) = .empty,
    last_reused: usize = 0,
    append_pending: std.ArrayList(u32) = .empty,
    draft_fill_pending: std.ArrayList(DraftFill) = .empty,
    window: [forward.max_decode_rows]u32 = undefined,
    window_rows: usize = 0,
    seed: u64 = 0,
    rule: sampler.Sampling = .{ .temperature = 0 },
    head: forward.HeadMode = .greedy,
    fallback_count: u64 = 0,
    drafter: ?Drafter = null,
};

/// One slot's window in a shared decode window: its input tokens and a drawn token per row.
pub const Part = struct { slot: usize, ids: []const u32, out: []u32 };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: ?std.Io = null,
    world: usize = 2,
    f: forward.Forward,
    mem: uma.MemTrace = .{},
    decode_graph_max: u32 = 48,
    graph_agreement_enabled: bool = false,
    graph_allow: []const GraphKey = &.{},
    graph_live: bool = false,
    abl: ?*const @import("ablate").Ablation = null,
    ablate_on: bool = false,
    drafter: ?Drafter = null,
    /// TF_GLM53_SLOTS: each request slot's cache view (f.st points at the bound one), each slot's sequence while it
    /// is not bound, and the bound slot. Empty without slots: slot 0 is the whole cache, always bound.
    views: []state.State = &.{},
    seqs: []Seq = &.{},
    bound: usize = 0,
    /// TF_GLM53_CAPTURE_DIR (rank 0, attached once serving starts): drafter training rows per request. Null: off.
    capture: ?*capture_mod.Capture = null,
    /// Drafter drafts in the next verify window (the adapter sets it; copy proposals count 0), for the capture.
    capture_held: u32 = 0,
    /// Fine resident prompt reuse grain; zero keeps the normal prefill piece grain.
    reuse_grain: usize = 0,
    agreement: ?Agreement = null,
    snap_ops: ?SnapOps = null,
    /// Set from TF_GLM53_DEFER_DRAFT_FILL during native setup; tests can select the path directly.
    defer_draft_fill: ?bool = null,
    draft_fill_pending: std.ArrayList(DraftFill) = .empty,
    tapped: usize = 0,
    tap_high: usize = 0,
    /// Next absolute position. The caches are position-addressed, so a rollback is setting it back.
    pos: usize = 0,
    /// Continuity (resume == fresh, bit for bit; review H1): positions below `frontier` hold KV written by the
    /// prefill path. Rows that decode/verify wrote (frontier..pos) are never kept for reuse: the next turn
    /// re-prefills them (reply tokens, indexer keys included) from the frontier, then the new turn.
    frontier: usize = 0,
    /// Tokens at positions 0..pos (prefilled, then kept decode rows).
    history: std.ArrayList(u32) = .empty,
    /// Rows reused by the most recent resident prefill.
    last_reused: usize = 0,
    resident: bool = false,
    park: ?side_park.Park = null,
    append_pending: std.ArrayList(u32) = .empty,
    /// The last verify window's input tokens (what `keep` appends).
    window: [forward.max_decode_rows]u32 = undefined,
    window_rows: usize = 0,
    seed: u64 = 0,
    rule: sampler.Sampling = .{ .temperature = 0 },
    download: Download = .{},
    graphs: ?*cuda.graph_cache.Cache(GraphKey) = null,
    /// The default decode path owns a graph cache. A caller-supplied cache still takes precedence; replayParity
    /// temporarily disables autoGraphs for its eager baseline.
    auto_graphs: bool = true,
    owned_graph_engine: ?cuda.graph_cache.CudaEngine = null,
    owned_graph_cache: ?cuda.graph_cache.Cache(GraphKey) = null,
    /// Host copy of the last head exchange: [world][rows][per-row bytes], rows of `head_first`.. in the window.
    head_host: []u8 = &.{},
    head_rows: usize = 0,
    head_per_rank: usize = 0,
    head_status: u32 = 0,

    fallback_device: ?cuda.DeviceBuffer = null,
    fallback_logits_host: []f32 = &.{},
    fallback_rows: []bool = &.{},
    fallback_log_z: []f32 = &.{},
    fallback_vocab: usize = 0,
    fallback_count: u64 = 0,

    mask_host: []u32 = &.{},
    mask_dev: ?cuda.DeviceBuffer = null,
    mask_words: usize = 0,
    mask_rows: usize = 0,
    head_masked: bool = false,

    /// How the last verify window ran: eager (graphs off, or a key's warm-up), captured or replayed.
    last_outcome: cuda.graph_cache.Outcome = .eager,
    verify_windows: u64 = 0,
    /// Every key this engine ran (graphed or eager), for the graph census.
    keys: std.AutoArrayHashMapUnmanaged(GraphKey, void) = .empty,
    /// MK8 deferred launch (TF_GLM53_MK=1): the decode sink, created on the first verify window that wants it.
    mk: ?*mk_defer.Executor = null,
    prof: ?forward.Profile = null,
    profile_checked: bool = false,
    profile_enabled: bool = false,
    profile_warned_graphs: bool = false,
    /// Profiled verify windows and their rows since the last report (TF_GLM53_PROFILE=1 logs every 32 windows).
    prof_windows: u64 = 0,
    prof_rows: u64 = 0,

    /// TF_GLM53_SLOTS: a slot's window in verifyShared (follow.zig names it through the engine type).
    pub const SharedPart = Part;

    /// Set on both ranks when the begin frame installs a request's sampling rule.
    pub fn setSamplingRule(e: *Engine, rule: sampler.Sampling) !void {
        const k = e.f.st.ws.candidate_k;
        if (rule.temperature > 0 and (rule.top_k == 0 or rule.top_k > k)) try e.ensureFallbackResources();
        e.rule = rule;
        e.fallback_count = 0;
        e.f.head = samplingHead(rule, k);
    }

    fn usesFallbackRows(e: *const Engine) bool {
        return switch (e.f.head) {
            .candidates => |k| e.rule.temperature > 0 and (e.rule.top_k == 0 or e.rule.top_k > k),
            .none, .greedy => false,
        };
    }

    fn ensureFallbackRows(e: *Engine, rows: usize) !void {
        if (e.fallback_rows.len >= rows) return;
        const next = try e.gpa.alloc(bool, rows);
        e.gpa.free(e.fallback_rows);
        e.fallback_rows = next;
    }

    fn ensureFallbackResources(e: *Engine) !void {
        const rows = e.f.st.ws.head_rows;
        try e.ensureFallbackRows(rows);
        if (e.fallback_log_z.len < rows) {
            const next = try e.gpa.alloc(f32, rows);
            e.gpa.free(e.fallback_log_z);
            e.fallback_log_z = next;
        }
        const full_vocab = e.f.w.config.vocab;
        e.fallback_vocab = full_vocab;
        const logits_count = try std.math.mul(usize, rows, full_vocab);
        if (e.fallback_logits_host.len < logits_count) {
            const next = try e.gpa.alloc(f32, logits_count);
            e.gpa.free(e.fallback_logits_host);
            e.fallback_logits_host = next;
        }
        const max_shard = if (e.world == 3) @as(usize, 51712) else (full_vocab / e.world);
        const max_shard_bytes = try std.math.mul(usize, max_shard, @sizeOf(f32));
        const fallback_bytes = max_shard_bytes + max_shard_bytes * e.world;
        if (e.fallback_device == null or e.fallback_device.?.len < fallback_bytes) {
            if (e.fallback_device) |*buffer| buffer.free();
            e.fallback_device = try cuda.DeviceBuffer.alloc(e.f.stream.d, fallback_bytes);
        }
    }

    pub fn candidateHeadK(e: *const Engine) usize {
        return e.f.st.ws.candidate_k;
    }

    fn deferDraftFill(e: *const Engine) bool {
        return e.defer_draft_fill orelse false;
    }

    fn restartDraft(e: *Engine, d: Drafter, from: usize) !void {
        if (e.deferDraftFill()) try e.draft_fill_pending.append(e.gpa, .{ .restart = from }) else try d.restart_fn(d.ctx, from);
    }

    fn promptDraft(e: *Engine, d: Drafter, start: usize, rows: usize) !void {
        if (e.deferDraftFill()) try e.draft_fill_pending.append(e.gpa, .{ .prompt = .{ .start = start, .rows = rows } }) else try d.prompt_fn(d.ctx, start, rows);
    }

    fn flushDraftFill(e: *Engine) !void {
        const d = e.drafter orelse return;
        for (e.draft_fill_pending.items) |op| switch (op) {
            .restart => |from| try d.restart_fn(d.ctx, from),
            .prompt => |range| try d.prompt_fn(d.ctx, range.start, range.rows),
        };
        e.draft_fill_pending.clearRetainingCapacity();
    }

    fn promptTaps(e: *Engine, start: usize, end: usize) !?forward.taps_mod.Taps {
        if (e.drafter == null) return null;
        const d = e.drafter.?;
        const from = @max(start, end -| d.window);
        if (start != e.tapped or e.tap_high > start) {
            try e.restartDraft(d, from);
            e.tap_high = 0;
        }
        var t = d.taps_prefill;
        t.from = from;
        return t;
    }

    fn promptTapped(e: *Engine, start: usize, end: usize) !void {
        if (e.drafter) |d| {
            const from = @max(start, end -| d.window);
            try e.promptDraft(d, from, end - from);
            e.tapped = end;
            e.tap_high = @max(e.tap_high, end);
        }
    }
    pub fn setAblate(e: *Engine, on: bool) !void {
        if (on and e.abl == null) return error.NoAblation;
        if (on != e.ablate_on and e.pos != 0) return error.AblateWithResidentRows;
        e.ablate_on = on;
    }

    pub fn deinit(e: *Engine) void {
        if (e.mk) |x| x.destroy();
        if (e.prof) |*p| p.deinit();
        if (e.owned_graph_cache) |*cache| cache.deinit();
        e.history.deinit(e.gpa);
        if (e.park) |*pk| pk.deinit(e.gpa);
        e.append_pending.deinit(e.gpa);
        e.draft_fill_pending.deinit(e.gpa);
        e.gpa.free(e.head_host);

        e.gpa.free(e.fallback_logits_host);
        e.gpa.free(e.fallback_log_z);
        e.gpa.free(e.fallback_rows);
        if (e.fallback_device) |*buffer| buffer.free();

        e.gpa.free(e.mask_host);
        if (e.mask_dev) |*buf| buf.free();

        // TF_GLM53_SLOTS: the other slots' lists (the bound slot's are the Engine's, freed above); views own nothing.
        for (e.seqs, 0..) |*q, i| if (i != e.bound) {
            q.history.deinit(e.gpa);
            q.append_pending.deinit(e.gpa);
            q.draft_fill_pending.deinit(e.gpa);
        };
        e.gpa.free(e.seqs);
        e.gpa.free(e.views);

        e.keys.deinit(e.gpa);
        e.* = undefined;
    }

    /// TF_GLM53_SLOTS: run every step below on request slot `slot` (its cache view, positions, history, sampling rule
    /// and drafter); the bound slot's fields go back to its Seq first. Without slots only slot 0 exists.
    pub fn bind(e: *Engine, slot: usize) !void {
        if (slot == e.bound) return;
        if (slot >= e.seqs.len or slot >= e.views.len) return error.UnknownSlot;
        e.stash(&e.seqs[e.bound]);
        e.load(&e.seqs[slot]);
        e.f.st = &e.views[slot];
        e.bound = slot;
    }

    fn stash(e: *const Engine, q: *Seq) void {
        q.* = .{
            .tapped = e.tapped,
            .tap_high = e.tap_high,
            .pos = e.pos,
            .frontier = e.frontier,
            .history = e.history,
            .last_reused = e.last_reused,
            .append_pending = e.append_pending,
            .draft_fill_pending = e.draft_fill_pending,
            .window = e.window,
            .window_rows = e.window_rows,
            .seed = e.seed,
            .rule = e.rule,
            .head = e.f.head,
            .fallback_count = e.fallback_count,
            .drafter = e.drafter,
        };
    }

    fn load(e: *Engine, q: *const Seq) void {
        e.tapped = q.tapped;
        e.tap_high = q.tap_high;
        e.pos = q.pos;
        e.frontier = q.frontier;
        e.history = q.history;
        e.last_reused = q.last_reused;
        e.append_pending = q.append_pending;
        e.draft_fill_pending = q.draft_fill_pending;
        e.window = q.window;
        e.window_rows = q.window_rows;
        e.seed = q.seed;
        e.rule = q.rule;
        e.f.head = q.head;
        e.fallback_count = q.fallback_count;
        e.drafter = q.drafter;
    }

    /// TF_GLM53_SLOTS: one decode window over several slots' windows (rows in part order, each slot's cache work on
    /// its own view), then each slot draws its rows under its own rule. As `verify`, positions do not move: each
    /// slot's rows are kept (or rolled back) on that slot afterwards. Eager: the segments' views bake no graph.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        if (parts.len == 0 or parts.len > max_shared) return error.BadSharedWindow;
        var positions: [forward.max_decode_rows]usize = undefined;
        var ids: [forward.max_decode_rows]u32 = undefined;
        var segs: [max_shared]forward.Seg = undefined;
        var pos0: [max_shared]usize = undefined;
        var head: forward.HeadMode = .greedy;
        var rows: usize = 0;
        for (parts, 0..) |pt, i| {
            for (parts[0..i]) |q| if (q.slot == pt.slot) return error.SlotTwice;
            try e.bind(pt.slot);
            const n = pt.ids.len;
            if (n == 0 or n > pt.out.len or rows + n > forward.max_decode_rows) return error.BadSharedWindow;
            if (e.mask_rows != 0) return error.UnexpectedMasks;
            if (i == 0) head = e.f.head else if (!std.meta.eql(head, e.f.head)) return error.SharedHeadsDiffer;
            if (e.pos + n > e.f.st.capacity) return error.ContextFull;
            try e.flushDraftFill();
            @memcpy(e.window[0..n], pt.ids);
            e.window_rows = n;
            pos0[i] = e.pos;
            for (0..n) |j| {
                positions[rows + j] = e.pos + j;
                ids[rows + j] = pt.ids[j];
            }
            segs[i] = .{ .st = &e.views[pt.slot], .row0 = rows, .rows = n };
            rows += n;
        }
        if (!verifyRowsAllowed(rows, e.f.st.ws.head_rows, rows)) return error.BadVerify;
        if (e.mem.on) e.verify_windows += 1;
        // Every part's slot drafter shares the one taps_step: the window's rows land there in row order (from the
        // first row's position), each part's at its segment's row0, so each slot's context takes its kept rows.
        var tapping = true;
        for (parts) |pt| {
            const d = e.slotDrafter(pt.slot) orelse {
                tapping = false;
                break;
            };
            if (d.verify_at_fn == null) tapping = false;
        }
        var f = e.f;
        f.segs = segs[0..parts.len];
        f.mask = .{};
        f.taps = if (tapping) blk: {
            var t = e.slotDrafter(parts[0].slot).?.taps_step;
            t.from = positions[0];
            break :blk t;
        } else null;
        f.ablate = null;
        f.prof = null;
        f.sink = null;
        f.mk = null;
        try f.prepare(.decode, positions[0..rows], ids[0..rows]);
        try f.launch(.decode, positions[0..rows], .{ .start = 0, .end = rows });
        e.last_outcome = .eager;
        if (tapping) for (parts, segs[0..parts.len], 0..) |pt, sg, i| {
            const d = e.slotDrafter(pt.slot).?;
            try d.verify_at_fn.?(d.ctx, sg.row0, sg.rows, pos0[i], i == 0);
        };
        try e.readHead(f.stream, f.st.ws.head_gather, rows, rows * f.head.rowBytes());
        var row: usize = 0;
        for (parts, 0..) |pt, i| {
            try e.bind(pt.slot);
            try e.draw(row, pt.ids.len, pos0[i], pt.out[0..pt.ids.len]);
            row += pt.ids.len;
        }
    }

    /// TF_GLM53_SLOTS: slot `slot`'s drafter (the engine's own while that slot is bound).
    fn slotDrafter(e: *const Engine, slot: usize) ?Drafter {
        if (slot == e.bound) return e.drafter;
        if (slot >= e.seqs.len) return null;
        return e.seqs[slot].drafter;
    }

    /// TF_GLM53_NO_GRAPHS=1: decode windows run eagerly (diagnosis of graph replay vs eager on the full model).
    fn graphsDisabled() bool {
        if (std.c.getenv("TF_GRAPH_CACHE")) |cache| if (cache[0] == '0') return true;
        const v = std.c.getenv("TF_GLM53_NO_GRAPHS") orelse return false;
        return v[0] == '1';
    }

    fn profileSwitch(e: *Engine) bool {
        if (!e.profile_checked) {
            e.profile_checked = true;
            if (std.c.getenv("TF_GLM53_PROFILE")) |v| e.profile_enabled = v[0] == '1' and v[1] == 0;
        }
        return e.profile_enabled;
    }

    fn ensureGraphCache(e: *Engine) void {
        if (e.graphs) |cache| {
            cache.name = "decode";
            cache.meter = uma.memAvailableGib;
            cache.room = uma.graphRoom;
            cache.agree = if (e.graph_agreement_enabled) .{ .ctx = e, .all = agreeGraphCapture } else null;
            cache.allowed = if (e.graph_allow.len != 0) e.graph_allow else null;
            cache.same_slot = if (e.graph_live) sameGraphSlot else null;
            return;
        }
        if (e.owned_graph_cache == null) {
            e.owned_graph_engine = .{ .d = e.f.stream.d };
            var settings = decode_graphs;
            settings.max = e.decode_graph_max;
            e.owned_graph_cache = cuda.graph_cache.Cache(GraphKey).init(e.gpa, e.owned_graph_engine.?.engine(), settings);
        }
        e.graphs = &e.owned_graph_cache.?;
        e.graphs.?.name = "decode";
        e.graphs.?.meter = uma.memAvailableGib;
        e.graphs.?.room = uma.graphRoom;
        e.graphs.?.agree = if (e.graph_agreement_enabled) .{ .ctx = e, .all = agreeGraphCapture } else null;
        e.graphs.?.allowed = if (e.graph_allow.len != 0) e.graph_allow else null;
        e.graphs.?.same_slot = if (e.graph_live) sameGraphSlot else null;
    }

    fn agreeGraphCapture(ctx: *anyopaque, local_room: bool) anyerror!bool {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const group: *const cuda.tp_group.Group = @ptrCast(@alignCast(e.f.coll.ctx));
        const stream = e.f.stream;
        var flag = try cuda.DeviceBuffer.alloc(stream.d, 1);
        defer flag.free();
        const local: u8 = @intFromBool(local_room);
        try flag.uploadAsync(0, std.mem.asBytes(&local), stream.handle);
        try group.allReduce(stream, flag.ptr, flag.ptr, 1, .u8, .min);
        try stream.synchronize();
        var agreed: u8 = 0;
        try flag.download(0, std.mem.asBytes(&agreed));
        return agreed != 0;
    }

    /// Reserve host result and key storage after weights and State are ready. Native startup should call this before
    /// the first request, so decode/verify never grow the head mirror or the graph-key map in steady state.
    pub fn reserveSteadyState(e: *Engine) !void {
        const sampled_head: forward.HeadMode = .{ .candidates = e.f.st.ws.candidate_k };
        const max_head_bytes = try std.math.mul(usize, e.f.st.ws.head_rows, sampled_head.rowBytes());
        const gathered_bytes = try std.math.mul(usize, max_head_bytes, e.world);
        if (e.head_host.len < gathered_bytes) {
            const next = try e.gpa.alloc(u8, gathered_bytes);
            e.gpa.free(e.head_host);
            e.head_host = next;
        }
        // 8 row widths (plus 9..16 when copy-draft verify lands) across the eight sparse position buckets.
        try e.keys.ensureTotalCapacity(e.gpa, 256);
        e.ensureGraphCache();
    }

    fn memPoint(e: *Engine, what: []const u8) !void {
        if (!e.mem.on) return;
        const io = e.io orelse return error.MissingMemTraceIo;
        var graphs: uma.GraphStats = .{};
        if (e.graphs) |cache| {
            cache.meter = uma.memAvailableGib;
            graphs = .{ .held = cache.count(), .captured = cache.stats.captured, .capture_mib = cache.stats.capture_mib };
        }
        try e.mem.point(e.gpa, io, e.f.stream, what, graphs);
    }

    fn observePrefillChunk(ctx: *anyopaque, start: usize, rows: usize) anyerror!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        const what = try std.fmt.allocPrint(e.gpa, "prefill chunk at {d} rows {d}", .{ start, rows });
        defer e.gpa.free(what);
        try e.memPoint(what);
    }

    fn collect(ctx: *anyopaque, stream: cuda.Stream, first: usize, rows: usize, gathered: u64, per_rank: usize) anyerror!void {
        _ = first;
        const e: *Engine = @ptrCast(@alignCast(ctx));
        try e.readHead(stream, gathered, rows, per_rank);
    }

    fn readHead(e: *Engine, stream: cuda.Stream, gathered: u64, rows: usize, per_rank: usize) !void {
        e.head_masked = e.mask_rows != 0;
        const n = per_rank * e.world;
        if (e.head_host.len < n) {
            e.gpa.free(e.head_host);
            e.head_host = &.{};
            e.head_host = try e.gpa.alloc(u8, n);
        }
        if (e.download.group) |group| if (group.hasRoce()) try group.syncChecked(stream);
        try e.download.download_fn(e.download.ctx, stream, gathered, e.head_host[0..n]);
        if (e.download.group) |group| {
            if (group.hasRoce()) {
                var status: [4]u8 = undefined;
                try e.download.download_fn(e.download.ctx, stream, e.f.st.ws.roce_status, &status);
                e.head_status = std.mem.readInt(u32, &status, .little);
            } else e.head_status = 0;
        } else e.head_status = 0;
        e.head_rows = rows;
        e.head_per_rank = per_rank;
        if (e.usesFallbackRows() and (rows > e.fallback_rows.len or rows > e.fallback_log_z.len)) return error.FallbackRowsTooSmall;
        if (e.usesFallbackRows()) {
            @memset(e.fallback_rows[0..rows], false);
            @memset(e.fallback_log_z[0..rows], std.math.nan(f32));
            for (0..rows) |r| {
                var fits = false;
                // T <= 1 nuclei are checked on the candidates first; only a row whose nucleus may reach past them
                // (or T > 1) downloads the full-vocabulary logits.
                if (e.rule.top_k == 0 and e.rule.temperature > 0 and e.rule.temperature <= 1 and e.rule.top_p > 0 and e.rule.top_p < 1) {
                    const info = try e.nucleusInfo(r);
                    e.fallback_log_z[r] = info.log_z;
                    fits = info.fits;
                }
                if (!fits) {
                    try e.fetchFullLogitsRow(stream, r);
                    e.fallback_rows[r] = true;
                    e.fallback_count += 1;
                }
            }
        }
    }

    const NucleusInfo = struct { fits: bool, log_z: f32 };

    fn nucleusInfo(e: *Engine, row: usize) !NucleusInfo {
        const k = switch (e.f.head) { .candidates => |width| width, .none, .greedy => return error.NoCandidateHead };
        const per = e.head_per_rank;
        const n = e.head_rows;
        var lists: [3][]const sampler.Candidate = undefined;
        var made: usize = 0;
        defer for (lists[0..made]) |list| e.gpa.free(list.ptr[0..k]);
        for (0..e.world) |rank| {
            const base = e.head_host[rank * per ..];
            const list = try e.gpa.alloc(sampler.Candidate, k);
            var len: usize = 0;
            for (0..k) |i| {
                const item: sampler.Candidate = .{
                    .value = std.mem.bytesToValue(f32, base[(row * k + i) * 4 ..][0..4]),
                    .id = std.mem.readInt(u32, base[n * k * 4 + (row * k + i) * 4 ..][0..4], .little),
                };
                if (absentCandidate(item)) continue;
                list[len] = item;
                len += 1;
            }
            lists[rank] = list[0..len];
            made += 1;
        }
        const merged = try sampler.mergeCandidates(e.gpa, lists[0..e.world], k);
        defer e.gpa.free(merged);
        var rank_lse: [3]sampler.Lse = undefined;
        for (0..e.world) |rank| {
            const base = e.head_host[rank * per ..];
            const at = n * k * 8 + row * 8;
            rank_lse[rank] = .{
                .max = std.mem.bytesToValue(f32, base[at..][0..4]),
                .sum = std.mem.bytesToValue(f32, base[at + 4 ..][0..4]),
            };
        }
        const log_z = sampler.mergeLse(rank_lse[0..e.world]);
        const fits = if (e.rule.temperature == 1) try nucleusFitsCandidates(e.gpa, merged, log_z, e.rule) else try sampler.temperedNucleusFits(e.gpa, merged, log_z, e.rule);
        return .{ .fits = fits, .log_z = log_z };
    }

    fn fetchFullLogitsRow(e: *Engine, stream: cuda.Stream, row: usize) !void {
        const device = e.fallback_device orelse return error.MissingFallbackBuffer;
        const full_vocab = e.f.w.config.vocab;
        const vocab_local = e.f.w.vocab.len();
        const local_bytes = try std.math.mul(usize, vocab_local, @sizeOf(f32));
        const max_shard = if (e.world == 3) @as(usize, 51712) else (full_vocab / e.world);
        const max_shard_bytes = max_shard * @sizeOf(f32);

        const send_buf = device.ptr;
        const gathered_buf = device.ptr + max_shard_bytes;
        const source = e.f.st.ws.logits + row * local_bytes;

        try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(send_buf, source, local_bytes, stream.handle), "copy full-vocabulary fallback logits");
        if (local_bytes < max_shard_bytes) {
            try stream.d.check(stream.d.api.cuMemsetD8Async(send_buf + local_bytes, 0, max_shard_bytes - local_bytes, stream.handle), "zero-pad fallback logits tail");
        }

        try e.f.coll.allGather(stream, send_buf, gathered_buf, max_shard_bytes, .head);

        const host_row = e.fallback_logits_host[row * full_vocab ..][0..full_vocab];
        for (0..e.world) |rank| {
            const span_start: usize = if (e.world == 3) (switch (rank) { 0 => 0, 1 => 51712, else => 103296 }) else rank * (full_vocab / e.world);
            const span_end: usize = if (e.world == 3) (switch (rank) { 0 => 51712, 1 => 103296, else => 154880 }) else (rank + 1) * (full_vocab / e.world);
            const dev_offset = gathered_buf + rank * max_shard_bytes;
            const host_dest = host_row[span_start..span_end];
            try e.download.download_fn(e.download.ctx, stream, dev_offset, std.mem.sliceAsBytes(host_dest));
        }
    }

    pub fn setMasks(e: *Engine, bits: []const u32, rows: usize) !void {
        if (rows == 0 or rows > forward.max_decode_rows) return error.BadMaskRows;
        if (bits.len % rows != 0) return error.BadMaskWords;
        const words = bits.len / rows;
        if (words == 0 or words > 1 << 16) return error.BadMaskWords;
        if (e.mask_rows != 0) return error.MasksPending;
        if (e.mask_words != 0 and e.mask_words != words) return error.MaskWidthChanged;
        if (e.mask_dev == null) {
            e.mask_host = try e.gpa.alloc(u32, forward.max_decode_rows * words);
            errdefer {
                e.gpa.free(e.mask_host);
                e.mask_host = &.{};
            }
            e.mask_dev = try cuda.DeviceBuffer.alloc(e.f.stream.d, forward.max_decode_rows * words * 4);
            errdefer if (e.mask_dev) |*buf| { buf.free(); e.mask_dev = null; };
        }
        @memcpy(e.mask_host[0..bits.len], bits);
        try e.mask_dev.?.uploadAsync(0, std.mem.sliceAsBytes(e.mask_host[0..bits.len]), e.f.stream.handle);
        e.mask_words = words;
        e.mask_rows = rows;
    }

    pub fn clearMasks(e: *Engine) void {
        e.mask_rows = 0;
    }

    /// Prompt rows from the prefill frontier: chunks at absolute positions, the head on the last row; returns the
    /// first token. Decode rows past the frontier must go through `nextTurn` (they are re-prefilled), never kept.
    pub fn prefill(e: *Engine, ids: []const u32) !u32 {
        defer e.mask_rows = 0;
        if (e.mask_rows != 0 and (e.mask_rows != 1 or e.f.gate_mode)) return error.BadPrefillMask;
        if (ids.len == 0) return error.EmptyPrompt;
        if (e.pos != e.frontier) return error.DecodeRowsNotReprefilled;
        // The drafter owns one prompt-tap buffer; consume an earlier deferred range before the next prefill overwrites it.
        if (e.deferDraftFill()) try e.flushDraftFill();
        const start = e.pos;
        const positions = try e.gpa.alloc(usize, ids.len);
        defer e.gpa.free(positions);
        for (positions, 0..) |*p, i| p.* = e.pos + i;
        var f = e.f;
        f.mask = if (e.mask_rows != 0) .{ .bits = e.mask_dev.?.ptr, .words = @intCast(e.mask_words) } else .{};
        f.taps = try e.promptTaps(start, start + ids.len);
        f.ablate = if (e.ablate_on) e.abl else null;
        f.sink = .{ .ctx = e, .sink_fn = collect };
        if (e.prof) |*p| p.resetBuckets(); // a new request: decode's per-row buckets start over
        // TF_GLM53_PROFILE_PREFILL=1 (native sets f.prof_prefill): time this prefill's stages on their own, then
        // drop any partial decode window counts the shared profile held, so neither report mixes the two.
        if (e.f.prof_prefill) {
            if (e.prof == null) e.prof = try forward.Profile.init(e.gpa, e.f.stream.d, e.f.w.config.layers + 1);
            e.prof.?.reset();
            e.prof_windows = 0;
            e.prof_rows = 0;
            f.prof = &e.prof.?;
        }
        var at: usize = 0;
        while (at < ids.len) {
            const c = try forward.chunkRows(positions, f.chunk_rows, at);
            // Prefill runs eagerly (its stage fetch depends on chunk_start); the key is recorded for the census.
            try e.keys.put(e.gpa, withHeadKey(try prefillKey(c.end - c.start), e.f.head), {});
            at = c.end;
        }
        if (e.mem.on) {
            e.parkBeforeWrite(start + ids.len);
            try f.pieced(.{ .positions = positions, .tokens = ids, .observer_ctx = e, .observer_fn = observePrefillChunk });
        } else {
            e.parkBeforeWrite(start + ids.len);
            try f.pieced(.{ .positions = positions, .tokens = ids });
        }
        if (e.f.prof_prefill) {
            try e.f.stream.synchronize();
            try e.prof.?.collect();
            if (e.f.rank == 0) {
                std.log.info("glm53 prefill profile: {d} rows from position {d} (stage ms for the whole prefill)", .{ ids.len, start });
                e.prof.?.report(1, ids.len, start);
            }
            e.prof.?.reset();
        }
        if (e.park) |*pk| pk.notePrefill(start, start + ids.len);
        try e.history.appendSlice(e.gpa, ids);
        e.pos += ids.len;
        e.frontier = e.pos;
        e.capturePrompt(start, e.pos);
        try e.promptTapped(start, e.pos);
        var out: [1]u32 = undefined;
        try e.draw(e.head_rows - 1, 1, e.pos - 1, &out);
        return out[0];
    }

    /// Longest bit-exact piece-aligned prefix available from a resident prefill frontier.
    pub fn reusableRows(history: []const u32, frontier: usize, ids: []const u32, piece_rows: usize) usize {
        if (piece_rows == 0 or ids.len == 0) return 0;
        const limit = @min(frontier, history.len);
        const max_common = @min(limit, ids.len);
        var common: usize = 0;
        while (common < max_common and history[common] == ids[common]) : (common += 1) {}
        const rows = @min(common, ids.len - 1);
        return rows / piece_rows * piece_rows;
    }

    pub fn reusable(e: *const Engine, ids: []const u32) usize {
        return reusableRows(e.history.items, e.frontier, ids, e.f.st.ws.piece_rows);
    }

    pub fn reusablePrompt(e: *const Engine, ids: []const u32) usize {
        const grain = if (e.reuse_grain == 0) e.f.st.ws.piece_rows else e.reuse_grain;
        return reusableRows(e.history.items, e.frontier, ids, grain);
    }

    /// TF_GLM53_SLOTS with resident reuse: `reusablePrompt` on slot `slot`, read from its Seq while another slot
    /// is bound (no bind: choosing a slot changes nothing on any rank).
    pub fn reusableOn(e: *const Engine, slot: usize, ids: []const u32) usize {
        const grain = if (e.reuse_grain == 0) e.f.st.ws.piece_rows else e.reuse_grain;
        if (slot == e.bound) return reusableRows(e.history.items, e.frontier, ids, grain);
        if (slot >= e.seqs.len) return 0;
        return reusableRows(e.seqs[slot].history.items, e.seqs[slot].frontier, ids, grain);
    }

    /// TF_GLM53_SLOTS: the prefill rows slot `slot` keeps (its frontier).
    pub fn frontierOn(e: *const Engine, slot: usize) usize {
        if (slot == e.bound) return e.frontier;
        if (slot >= e.seqs.len) return 0;
        return e.seqs[slot].frontier;
    }

    pub fn prefillResident(e: *Engine, ids: []const u32) !u32 {
        const r = e.reusablePrompt(ids);
        try e.rollback(r);
        e.last_reused = r;
        return e.prefill(ids[r..]);
    }

    pub fn settle(e: *Engine) !void {
        try e.rollback(e.frontier);
    }

    fn parkBeforeWrite(e: *Engine, end: usize) void {
        if (e.park) |*pk| {
            if (pk.beforeWrite(end)) std.log.info("glm53 side park: dropped, the request reaches position {d} (long tier from {d})", .{ end, pk.long_from });
        }
    }

    pub fn parkSave(e: *Engine, lo: usize, hi: usize) !void {
        if (e.park == null) return error.NoPark;
        const pk = &e.park.?;
        if (pk.long_from != 0 and hi > pk.long_from) {
            std.log.info("glm53 side park: not saved, rows reach the long tier", .{});
            return;
        }
        if (e.ablate_on) return error.ParkAblated;
        if (e.deferDraftFill()) try e.flushDraftFill();
        const s = try pk.save(e.gpa, e.history.items, e.pos, e.frontier, .{ .lo = lo, .hi = hi }, e.f.st.ws.piece_rows);
        try e.f.st.parkCopy(e.f.stream, .save, s.first, s.count);
        try e.f.stream.synchronize();
        std.log.info("glm53 side park: save rows [{d}, {d}) of frontier {d}", .{ lo, hi, e.frontier });
    }

    pub fn parkRestore(e: *Engine) !usize {
        if (e.park == null) return error.NoPark;
        const pk = &e.park.?;
        if (e.deferDraftFill()) try e.flushDraftFill();
        const r = try pk.restore(&e.history);
        if (r.copy) |s| {
            try e.f.st.parkCopy(e.f.stream, .restore, s.first, s.count);
            try e.f.stream.synchronize();
        }
        e.pos = r.frontier;
        e.frontier = r.frontier;
        e.window_rows = 0;
        e.tapped = @min(e.tapped, r.frontier);
        e.tap_high = std.math.maxInt(usize);
        std.log.info("glm53 side park: restore frontier {d} (parked {d})", .{ r.frontier, pk.frontier });
        return r.frontier;
    }
    /// The DFlash2 ring (and its `from`) at the current frontier, after any deferred prompt fill has run.
    pub fn saveDrafterState(e: *Engine, dst: []u8) !i64 {
        const d = e.drafter orelse return error.NoDrafter;
        const save = d.state_fn orelse return error.DrafterStateUnsupported;
        try e.flushDraftFill();
        return save(d.ctx, dst);
    }

    /// Both ranks call this after each saved-file operation, including local failures.
    pub fn agree(e: *Engine, ok: bool) !bool {
        var words = [_]u32{@intFromBool(ok)};
        try e.agreeWords(&words);
        return words[0] != 0;
    }

    /// Vector agreement is used by snapshot frames to keep a fixed collective count while checking each node.
    pub fn agreeWords(e: *Engine, words: []u32) !void {
        if (words.len > 128) return error.AgreementTooLarge;
        const a = e.agreement orelse return error.AgreementUnavailable;
        try a.call_fn(a.ctx, words);
    }

    pub fn snapSave(e: *Engine, h: anytype) !void {
        const ops = e.snap_ops orelse return error.SnapshotUnavailable;
        try ops.save_fn(ops.ctx, snapCutOf(h));
    }

    pub fn drainDraftFill(e: *Engine) !void { try e.flushDraftFill(); }

    pub fn snapLoad(e: *Engine, h: anytype, ids: []const u32) !usize {
        const ops = e.snap_ops orelse return error.SnapshotUnavailable;
        return ops.load_fn(ops.ctx, @intCast(h.position), ids, std.mem.readInt(u64, h.prefix_hash[0..8], .little));
    }

    pub fn snapCommit(e: *Engine, n: usize) !void {
        const ops = e.snap_ops orelse return error.SnapshotUnavailable;
        try ops.commit_fn(ops.ctx, n);
    }

    /// A failed local or remote disk restore returns to resident rows and forces a full drafter tap restart.
    pub fn restoreFailed(e: *Engine, r: usize) !void {
        try e.rollback(r);
        e.tap_high = e.pos + 1;
        e.tapped = @min(e.tapped, r);
        e.frontier = r;
        e.history.shrinkRetainingCapacity(@min(r, e.history.items.len));
        e.pos = r;
        e.window_rows = 0;
        e.draft_fill_pending.clearRetainingCapacity();
    }

    /// Adopt a validated, piece-aligned saved prefill. The caller restores the State planes first;
    /// this method restores only engine bookkeeping and the optional DFlash2 ring.
    pub fn restoreTo(e: *Engine, at: usize, ids: []const u32, ring: ?[]const u8, ring_from: ?i64) !void {
        if (at == 0 or at > ids.len or at % e.f.st.ws.piece_rows != 0) return error.BadRestorePosition;
        if (ring) |bytes| {
            const d = e.drafter orelse return error.NoDrafter;
            const adopt = d.adopt_fn orelse return error.DrafterStateUnsupported;
            try adopt(d.ctx, bytes, ring_from orelse return error.MissingDrafterState);
            e.tapped = at;
            e.tap_high = at;
        } else if (e.drafter != null or ring_from != null) return error.MissingDrafterState;
        e.history.clearRetainingCapacity();
        try e.history.appendSlice(e.gpa, ids[0..at]);
        e.pos = at;
        e.frontier = at;
        e.tapped = if (e.drafter == null) 0 else e.tapped;
        e.tap_high = if (e.drafter == null) 0 else e.tap_high;
        e.window_rows = 0;
        e.draft_fill_pending.clearRetainingCapacity();
    }

    /// Production restore adopts the saved ring directly into the drafter's device ring; B1 supplies ringAdopt.
    pub fn restoreToDevice(e: *Engine, at: usize, ids: []const u32, ring_from: ?i64) !void {
        if (at == 0 or at > ids.len or at % e.f.st.ws.piece_rows != 0) return error.BadRestorePosition;
        if (e.drafter) |d| {
            const adopt = d.ring_adopt_fn orelse return error.DrafterRingAdoptUnsupported;
            try adopt(d.ctx, ring_from orelse return error.MissingDrafterState);
            e.tapped = at;
            e.tap_high = at;
        } else if (ring_from != null) return error.MissingDrafterState;
        e.history.clearRetainingCapacity();
        try e.history.appendSlice(e.gpa, ids[0..at]);
        e.pos = at;
        e.frontier = at;
        if (e.drafter == null) {
            e.tapped = 0;
            e.tap_high = 0;
        }
        e.window_rows = 0;
        e.draft_fill_pending.clearRetainingCapacity();
    }

    /// The capture's prompt rows: the taps this prefill pass wrote (positions from..end at taps_prefill rows 0..,
    /// the `from` promptTaps chose).
    fn capturePrompt(e: *Engine, start: usize, end: usize) void {
        const c = e.capture orelse return;
        const d = e.drafter orelse return;
        c.prompt(e.f.stream, d.taps_prefill.out, start, @max(start, end -| d.window), end);
    }

    /// The adapter's release calls this first, before any rollback: the capture's end block holds the transcript.
    pub fn captureEnd(e: *Engine) void {
        const c = e.capture orelse return;
        c.end(e.history.items[0..@min(e.pos, e.history.items.len)], e.rule.temperature, e.rule.top_p, e.rule.top_k, e.seed);
    }

    pub fn requestEnd(e: *Engine) !void {
        if (e.f.rank == 0 and e.rule.temperature > 0) std.log.info("glm53 sample: full-vocab fallback rows {d}", .{e.fallback_count});
        try e.memPoint("request end");
    }

    pub fn prefillRows(e: *Engine, ids: []const u32) !void {
        if (ids.len == 0 or ids.len % e.f.st.ws.piece_rows != 0 or e.pos % e.f.st.ws.piece_rows != 0) return error.AppendNotAligned;
        if (e.pos != e.frontier) return error.DecodeRowsNotReprefilled;
        // Append prefill reuses taps_prefill for each row group, so finish any previous deferred ingest first.
        if (e.deferDraftFill()) try e.flushDraftFill();
        const start = e.pos;
        const positions = try e.gpa.alloc(usize, ids.len);
        defer e.gpa.free(positions);
        for (positions, 0..) |*p, i| p.* = e.pos + i;
        var f = e.f;
        f.taps = try e.promptTaps(start, start + ids.len);
        f.head = .none;
        f.sink = null;
        e.parkBeforeWrite(start + ids.len);
        try f.pieced(.{ .positions = positions, .tokens = ids });
        if (e.park) |*pk| pk.notePrefill(start, start + ids.len);
        try e.history.appendSlice(e.gpa, ids);
        e.pos += ids.len;
        e.frontier = e.pos;
        e.capturePrompt(start, e.pos);
        try e.promptTapped(start, e.pos);
    }

    pub const Append = struct { base: usize };

    pub fn appendBegin(e: *Engine, ids: []const u32) !void {
        const r = if (ids.len == 0) e.pos / e.f.st.ws.piece_rows * e.f.st.ws.piece_rows else e.reusable(ids);
        try e.rollback(r);
        e.last_reused = r;
        e.append_pending.clearRetainingCapacity();
        if (ids.len != 0) try e.append_pending.appendSlice(e.gpa, ids[r..]);
    }

    pub fn appendTokens(e: *Engine, ids: []const u32) !void {
        try appendFeed(Engine, e.gpa, &e.append_pending, ids, e.f.st.ws.piece_rows, e, appendPrefillRows);
    }

    fn appendPrefillRows(e: *Engine, ids: []const u32) anyerror!void {
        try e.prefillRows(ids);
    }

    pub fn appendFinish(e: *Engine) !u32 {
        if (e.mask_rows != 0) return error.UnexpectedMasks;
        if (e.append_pending.items.len == 0) return error.EmptyPrompt;
        const pending = e.append_pending.items;
        const token = try e.prefill(pending);
        e.append_pending.clearRetainingCapacity();
        return token;
    }

    /// A verify window: `ids` at e.pos.. (R <= verify max (8 or 16)), every row through the head, one drawn token per row in `out`.
    /// The position does not move; `keep` accepts rows.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32) !void {
        defer e.mask_rows = 0;
        const masked = e.mask_rows != 0;
        if (masked and e.mask_rows != ids.len) return error.MaskRowsMismatch;
        const rows = ids.len;
        // Every verify row must sit in one head block, so the gathered results hold them all (review L3).
        if (!verifyRowsAllowed(rows, e.f.st.ws.head_rows, out.len)) return error.BadVerify;
        try e.flushDraftFill();
        @memcpy(e.window[0..rows], ids);
        e.window_rows = rows;
        e.parkBeforeWrite(e.pos + rows);
        if (e.park) |*pk| pk.noteDecode(e.pos + rows);
        if (e.mem.on) e.verify_windows += 1;
        var positions: [forward.max_decode_rows]usize = undefined;
        for (positions[0..rows], 0..) |*p, i| p.* = e.pos + i;
        const key = try verifyGraphKey(rows, positions[rows - 1], e.ablate_on, e.f.head);
        const first_window = !e.keys.contains(key);
        try e.keys.put(e.gpa, key, {});
        const profile_requested = e.profileSwitch();
        const graphs_for_call = !masked and !graphsDisabled() and (e.graphs != null or e.auto_graphs);
        const profiling = profile_requested and !graphs_for_call;
        if (profile_requested and graphs_for_call and !e.profile_warned_graphs) {
            std.debug.print("WARNING: TF_GLM53_PROFILE needs TF_GRAPH_CACHE=0\n", .{});
            e.profile_warned_graphs = true;
        }
        if (profiling and e.prof == null) e.prof = try forward.Profile.init(e.gpa, e.f.stream.d, e.f.w.config.layers + 1);
        if (e.auto_graphs and !graphsDisabled() and !masked) e.ensureGraphCache();
        if (forward.megakernelSwitch() and e.mk == null) e.mk = try mk_defer.Executor.create(e.gpa, e.f.stream.d);
        var f = e.f;
        f.mask = if (masked) .{ .bits = e.mask_dev.?.ptr, .words = @intCast(e.mask_words) } else .{};
        f.taps = if (e.drafter) |d| blk: {
            var t = d.taps_step;
            t.from = e.pos;
            break :blk t;
        } else null;
        f.ablate = if (e.ablate_on) e.abl else null;
        f.prof = if (profiling) &e.prof.? else null;
        f.sink = null; // read after the (possibly graphed) launch
        if (forward.megakernelSwitch() and !e.ablate_on and e.rule.temperature == 0) if (e.mk) |x| {
            f.mk = x.sink();
            // The launch-shape policy's rows bucket for segments whose kernels do not show the row count.
            x.deferral.rows = @intCast(rows);
        };
        try f.prepare(.decode, positions[0..rows], ids);
        const Body = struct {
            f: *const forward.Forward,
            positions: []const usize,
            fn run(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
                const b: *const @This() = @ptrCast(@alignCast(ctx));
                var g = b.f.*;
                g.stream = stream;
                try g.launch(.decode, b.positions, .{ .start = 0, .end = b.positions.len });
            }
        };
        var body: Body = .{ .f = &f, .positions = positions[0..rows] };
        const active_cache = if (graphsDisabled() or masked) null else e.graphs;
        if (active_cache) |cache| {
            // Everything a capture bakes in: buffers, weights, kernels, transport, head mode and its k (review L2).
            const k: u64 = switch (f.head) {
                .candidates => |c| c,
                else => 0,
            };
            const print = cuda.graph_cache.fingerprintOf(&.{ @intFromPtr(f.st), @intFromPtr(f.w), @intFromPtr(f.k), @intFromPtr(f.coll.ctx), @backingInt(std.meta.activeTag(f.head)), k, @intFromPtr(f.mk), if (f.taps) |t| t.out else 0 });
            // MK8: a capture cannot upload a megakernel plan (no allocation or sync copy while capturing), so a
            // key's first window runs once eagerly to build its plans; the window is idempotent (same rows, same
            // KV writes, the head is read after the replay).
            if (f.mk != null and first_window) {
                try Body.run(&body, f.stream);
                // TF_GLM53_MK_CALIBRATE times one launch shape per eager window of a key: finish before the capture
                // fixes the choice (a capture would end it early with the best shape measured so far).
                if (e.mk) |x| {
                    var n: usize = 0;
                    while (x.deferral.calibrating() and n < mk_defer.max_calibration_windows) : (n += 1) try Body.run(&body, f.stream);
                }
            }
            e.last_outcome = try cache.run(key, f.stream, print, .{ .ctx = &body, .run = Body.run });
        } else {
            try Body.run(&body, f.stream);
            e.last_outcome = .eager;
        }
        if (profiling) {
            try f.stream.synchronize();
            try e.prof.?.collect();
            e.prof.?.bucketWindow(rows, e.pos);
            e.prof_windows += 1;
            e.prof_rows += rows;
            if (e.prof_windows == 32) {
                if (e.f.rank == 0) {
                    e.prof.?.report(e.prof_windows, e.prof_rows, e.pos);
                    e.prof.?.reportByRows();
                }
                e.prof.?.reset();
                e.prof_windows = 0;
                e.prof_rows = 0;
            }
        }
        if (e.mem.on and e.last_outcome == .captured) {
            const what = try std.fmt.allocPrint(e.gpa, "capture rows {d} bucket {d}", .{ rows, key.chunk_bucket });
            defer e.gpa.free(what);
            try e.memPoint(what);
        }
        if (e.mem.on and e.verify_windows % 256 == 0) {
            const what = try std.fmt.allocPrint(e.gpa, "decode window {d}", .{e.verify_windows});
            defer e.gpa.free(what);
            try e.memPoint(what);
        }
        if (e.drafter) |d| try d.verify_fn(d.ctx, rows, e.pos);
        try e.readHead(f.stream, f.st.ws.head_gather, rows, rows * f.head.rowBytes());
        try e.draw(0, rows, e.pos, out);
        // The window's tap rows (taps_step from e.pos) and final residual rows are still in place after the draw.
        if (e.capture) |c| if (e.drafter) |d| c.verify(f.stream, d.taps_step.out, f.st.ws.x, e.pos, e.capture_held, e.window[0..rows], out[0..rows]);
        e.capture_held = 0;
    }

    /// Spark parity hook: one verify window run eagerly, then through the graph cache twice (capture, replay); the
    /// fp32 logits rows and the gathered head results must be bit-identical (flare-local-inference's exported-graph
    /// defect zeroed a constant held in scratch: answers stayed fluent but wrong). The position does not move.
    pub fn replayParity(e: *Engine, ids: []const u32) !void {
        if (e.mask_rows != 0) return error.UnexpectedMasks;
        const cache = e.graphs orelse return error.NoGraphCache;
        const rows = ids.len;
        if (rows == 0 or rows > e.f.st.ws.head_rows) return error.BadVerify;
        var out: [forward.max_decode_rows]u32 = undefined;
        const n = rows * e.f.w.vocab.len() * 4;
        const live = try e.gpa.alloc(u8, n);
        defer e.gpa.free(live);
        const replay = try e.gpa.alloc(u8, n);
        defer e.gpa.free(replay);
        const auto_graphs = e.auto_graphs;
        e.auto_graphs = false;
        defer e.auto_graphs = auto_graphs;
        e.graphs = null;
        defer e.graphs = cache;
        try e.verify(ids, out[0..rows]);
        try e.download.download_fn(e.download.ctx, e.f.stream, e.f.st.ws.logits, live);
        const head = try e.gpa.dupe(u8, e.head_host[0 .. e.head_per_rank * e.world]);
        defer e.gpa.free(head);
        const tokens = out;
        e.graphs = cache;
        for (0..2) |_| {
            try e.verify(ids, out[0..rows]);
            try e.download.download_fn(e.download.ctx, e.f.stream, e.f.st.ws.logits, replay);
            if (!std.mem.eql(u8, live, replay) or !std.mem.eql(u8, head, e.head_host[0..head.len]) or
                !std.mem.eql(u32, tokens[0..rows], out[0..rows])) return error.GraphReplayMismatch;
        }
    }

    /// Accept the first `n` rows of the last verify window. Their decode-written KV serves this turn's decoding
    /// only; the frontier does not move, so the next turn re-prefills them.
    pub fn keep(e: *Engine, n: usize) !void {
        if (n > e.window_rows) return error.BadKeep;
        if (e.pos + n > e.f.st.capacity) return error.ContextFull;
        try e.history.appendSlice(e.gpa, e.window[0..n]);
        e.pos += n;
        e.window_rows = 0;
    }

    pub fn draftBlock(e: *Engine, pos: usize, n: usize, anchor: u32, out: ?[]u32, probs: ?[]f64) !void {
        const d = e.drafter orelse return error.NoDrafter;
        if (pos > e.pos or n > forward.max_decode_rows or n > pos) return error.BadDraftBlock;
        try e.flushDraftFill();
        if (pos - n != e.tapped) {
            try d.restart_fn(d.ctx, pos);
            e.tap_high = 0;
        }
        try d.draft_fn(d.ctx, pos, n, anchor, out, probs);
        e.tapped = pos;
        e.tap_high = @max(e.tap_high, pos);
    }

    /// TF_GLM53_SLOTS: draftBlock without the drafts: the kept rows go into the bound slot's drafter context only.
    pub fn absorbBlock(e: *Engine, pos: usize, n: usize) !void {
        const d = e.drafter orelse return error.NoDrafter;
        const absorb = d.absorb_fn orelse return error.NoDrafterAbsorb;
        if (pos > e.pos or n > forward.max_decode_rows or n > pos) return error.BadDraftBlock;
        try e.flushDraftFill();
        if (pos - n != e.tapped) {
            try d.restart_fn(d.ctx, pos);
            e.tap_high = 0;
        }
        try absorb(d.ctx, pos, n);
        e.tapped = pos;
        e.tap_high = @max(e.tap_high, pos);
    }

    /// Roll back to an earlier position: rows past it are overwritten by the next window. Below the frontier the
    /// prefill rows past `to` are dropped too.
    pub fn rollback(e: *Engine, to: usize) !void {
        if (to > e.pos) return error.BadRollback;
        if (e.deferDraftFill()) try e.flushDraftFill();
        e.pos = to;
        e.frontier = @min(e.frontier, to);
        e.tapped = @min(e.tapped, to);
        e.history.shrinkRetainingCapacity(to);
    }

    /// The tokens decode wrote past the frontier (this turn's reply), for the WP12 warm pass.
    pub fn decodedSinceFrontier(e: *const Engine) []const u32 {
        return e.history.items[e.frontier..e.pos];
    }

    /// Drop the decode-written rows: the position returns to the prefill frontier (their KV will be rewritten).
    pub fn rewindToFrontier(e: *Engine) void {
        e.pos = e.frontier;
        e.tapped = @min(e.tapped, e.frontier);
        e.history.shrinkRetainingCapacity(e.frontier);
    }

    /// The next turn: re-prefill this turn's reply (decode rows) from the frontier, then `turn`, in one prefill
    /// pass that is chunk-invariant with a fresh prefill of the whole transcript. Returns the first new token.
    /// `history` holds the verify *inputs*: the reply's last sampled token (EOS or end-of-turn) was never fed back,
    /// so `turn` must start with it, or the transcript loses it (review N6; WP12 asserts this).
    pub fn nextTurn(e: *Engine, turn: []const u32) !u32 {
        if (e.mask_rows != 0) return error.UnexpectedMasks;
        const reply = e.decodedSinceFrontier();
        const ids = try e.gpa.alloc(u32, reply.len + turn.len);
        defer e.gpa.free(ids);
        @memcpy(ids[0..reply.len], reply);
        @memcpy(ids[reply.len..], turn);
        e.rewindToFrontier();
        return e.prefill(ids);
    }

    /// Tokens for `rows` head rows from row `first` of the last head exchange; row r sits at position + r.
    /// Greedy: the ranks' top-1 with ties to the lower id (NaN never wins). Candidates: ranks merged
    /// (value desc, id asc), then WP10's keyed draw.
    pub fn draw(e: *Engine, first: usize, rows: usize, position: u64, out: []u32) !void {
        if (e.head_status != 0) return error.RoceExchangeFailed;
        if (first + rows > e.head_rows or out.len < rows) return error.BadDraw;
        for (0..rows) |j| {
            const r = first + j;

            out[j] = try e.drawRow(r, position + j);
        }
    }

    fn drawRow(e: *Engine, row: usize, position: u64) !u32 {
        const n = e.head_rows;
        switch (e.f.head) {
            .none => return error.NoHead,
            .greedy => {
                var best: ?sampler.Candidate = null;
                for (0..e.world) |rank| {
                    const base = e.head_host[rank * e.head_per_rank ..];
                    const value = std.mem.bytesToValue(f32, base[row * 4 ..][0..4]);
                    const id = std.mem.readInt(u32, base[n * 4 + row * 4 ..][0..4], .little);
                    if (std.math.isNan(value) or absentCandidate(.{ .value = value, .id = id }) or id == 0xFFFFFFFF) continue;
                    if (best == null or value > best.?.value or (value == best.?.value and id < best.?.id)) best = .{ .value = value, .id = id };
                }
                if (best == null) return error.NoFiniteLogit;
                if (e.head_masked and best.?.value == -std.math.inf(f32)) return error.NoAllowedToken;
                return best.?.id;
            },
            .candidates => |k| {
                var lists: [3][]const sampler.Candidate = undefined;
                var made: usize = 0;
                defer for (lists[0..made]) |list| e.gpa.free(list.ptr[0..k]);
                for (0..e.world) |rank| {
                    const base = e.head_host[rank * e.head_per_rank ..];
                    const list = try e.gpa.alloc(sampler.Candidate, k);
                    var len: usize = 0;
                    for (0..k) |i| {
                        const item: sampler.Candidate = .{ .value = std.mem.bytesToValue(f32, base[(row * k + i) * 4 ..][0..4]), .id = std.mem.readInt(u32, base[n * k * 4 + (row * k + i) * 4 ..][0..4], .little) };
                        if (absentCandidate(item)) continue;
                        list[len] = item;
                        len += 1;
                    }
                    lists[rank] = list[0..len];
                    made += 1;
                }
                const merged = try sampler.mergeCandidates(e.gpa, lists[0..e.world], k);
                defer e.gpa.free(merged);
                if (e.rule.top_k == 0 and e.rule.top_p < 1) {
                    if (e.rule.temperature == 1) {
                        var rank_lse: [3]sampler.Lse = undefined;
                        for (0..e.world) |rank| {
                            const base = e.head_host[rank * e.head_per_rank ..];
                            const at = n * k * 8 + row * 8;
                            rank_lse[rank] = .{ .max = std.mem.bytesToValue(f32, base[at..][0..4]), .sum = std.mem.bytesToValue(f32, base[at + 4 ..][0..4]) };
                        }
                        const log_z = sampler.mergeLse(rank_lse[0..e.world]);
                        var force_full = false;
                        if (e.usesFallbackRows()) {
                            if (row >= e.fallback_rows.len or row >= e.fallback_log_z.len) return error.FallbackRowsTooSmall;
                            force_full = e.fallback_rows[row];
                        }
                        if (force_full and e.fallback_vocab == 0) return error.MissingFallbackVocabulary;
                        const full_logits = if (e.fallback_vocab == 0) &.{} else e.fallback_logits_host[row * e.fallback_vocab ..][0..e.fallback_vocab];
                        return drawNucleusWithFallback(e.gpa, merged, full_logits, force_full, log_z, e.seed, position, e.rule);
                    }
                    // readHead downloaded the full row only when the candidates could not settle the nucleus
                    // (sampler.temperedNucleusFits); otherwise the candidates draw the same token.
                    if (e.usesFallbackRows()) {
                        if (row >= e.fallback_rows.len) return error.FallbackRowsTooSmall;
                        const full_logits = if (e.fallback_vocab == 0) &.{} else e.fallback_logits_host[row * e.fallback_vocab ..][0..e.fallback_vocab];
                        if (e.fallback_rows[row] and full_logits.len > 0) return drawFullNucleusFromLogits(e.gpa, full_logits, 0, e.seed, position, e.rule);
                    }
                    return e.drawCandidates(merged, position);
                }
                if (e.rule.top_k > 0 or e.rule.temperature > 0) {
                    if (e.usesFallbackRows()) {
                        const full_logits = if (e.fallback_vocab == 0) &.{} else e.fallback_logits_host[row * e.fallback_vocab ..][0..e.fallback_vocab];
                        if (full_logits.len > 0) return drawFullNucleusFromLogits(e.gpa, full_logits, 0, e.seed, position, e.rule);
                    }
                    return e.drawCandidates(merged, position);
                }
                return error.IncompleteCandidateCoverage;
            },
        }
    }

    /// WP10's deterministic candidate sampler after the ranks' candidates have been merged.
    fn drawCandidates(e: *const Engine, candidates: []const sampler.Candidate, position: u64) !u32 {
        return sampler.drawCandidates(e.gpa, candidates, e.seed, position, e.rule);
    }
};


pub fn samplingHead(rule: sampler.Sampling, k: usize) forward.HeadMode {
    return if (rule.temperature > 0) .{ .candidates = k } else .greedy;
}

test "sampled head selection is per rule and greedy remains greedy" {
    const sampled = samplingHead(.{ .temperature = 1, .top_p = 0.95 }, 64);
    try std.testing.expectEqual(@as(std.meta.Tag(forward.HeadMode), .candidates), std.meta.activeTag(sampled));
    try std.testing.expectEqual(@as(usize, 64), switch (sampled) { .candidates => |k| k, else => 0 });
    const greedy = samplingHead(.{ .temperature = 0 }, 64);
    try std.testing.expectEqual(@as(std.meta.Tag(forward.HeadMode), .greedy), std.meta.activeTag(greedy));
}

test "greedy sampled greedy graph lookup replays only matching head launches" {
    const FakeEngine = struct {
        head: forward.HeadMode = .greedy,
        pub fn setSamplingRule(e: *@This(), rule: sampler.Sampling) void {
            e.head = samplingHead(rule, 64);
        }
    };
    const allocator = std.testing.allocator;
    var graph_outputs: std.AutoArrayHashMapUnmanaged(GraphKey, u32) = .empty;
    defer graph_outputs.deinit(allocator);
    var engine: FakeEngine = .{};
    engine.setSamplingRule(.{ .temperature = 0 });
    const greedy = try verifyGraphKey(4, 2047, false, engine.head);
    engine.setSamplingRule(.{ .temperature = 1, .top_p = 0.9 });
    const sampled = try verifyGraphKey(4, 2047, false, engine.head);
    try graph_outputs.put(allocator, greedy, 7);
    try graph_outputs.put(allocator, sampled, 19);

    engine.setSamplingRule(.{ .temperature = 0 });
    const first_greedy = try verifyGraphKey(4, 2047, false, engine.head);
    engine.setSamplingRule(.{ .temperature = 1, .top_p = 0.9 });
    const sampled_again = try verifyGraphKey(4, 2047, false, engine.head);
    engine.setSamplingRule(.{ .temperature = 0 });
    const final_greedy = try verifyGraphKey(4, 2047, false, engine.head);
    const sequence = [_]GraphKey{ first_greedy, sampled_again, final_greedy };
    var greedy_only_engine: FakeEngine = .{};
    greedy_only_engine.setSamplingRule(.{ .temperature = 0 });
    const greedy_only = try verifyGraphKey(4, 2047, false, greedy_only_engine.head);
    var greedy_replays: usize = 0;
    for (sequence) |key| {
        const output = graph_outputs.get(key) orelse return error.MissingGraphLaunch;
        if (key.head_mode == greedy.head_mode) {
            try std.testing.expectEqual(@as(u32, 7), output);
            try std.testing.expectEqual(graph_outputs.get(greedy_only).?, output);
            greedy_replays += 1;
        } else {
            try std.testing.expectEqual(@as(u32, 19), output);
            try std.testing.expectEqual(graph_outputs.get(sampled).?, output);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), greedy_replays);
}

test "engine draw falls back when the full-vocabulary nucleus exceeds K" {
    const allocator = std.testing.allocator;
    const logits = [_]f32{ 0.0, -0.1, -0.4, -1.0, -2.0, -4.0 };
    const rule: sampler.Sampling = .{ .temperature = 1, .top_p = 0.99, .min_log = -1.5 };
    const seed: u64 = 19;
    const position: u64 = 41;

    // Independent reference: full softmax, descending sort, top_p cutoff, and keyed Gumbel scores.
    var order = [_]usize{ 0, 1, 2, 3, 4, 5 };
    var exp_values: [6]f64 = undefined;
    var norm: f64 = 0;
    for (logits, 0..) |value, i| {
        exp_values[i] = @exp(@as(f64, value));
        norm += exp_values[i];
    }
    for (0..order.len) |i| for (i + 1..order.len) |j| {
        if (logits[order[j]] > logits[order[i]]) std.mem.swap(usize, &order[i], &order[j]);
    };
    var nucleus_len: usize = 0;
    var mass: f64 = 0;
    while (nucleus_len < order.len) : (nucleus_len += 1) {
        mass += exp_values[order[nucleus_len]] / norm;
        if (mass >= rule.top_p) {
            nucleus_len += 1;
            break;
        }
    }
    try std.testing.expect(nucleus_len > 2 * forward.world);
    var expected: u32 = 0;
    var best = -std.math.inf(f64);
    for (order[0..nucleus_len]) |id| {
        if (@as(f64, logits[id]) < @as(f64, logits[order[0]]) + rule.min_log) continue;
        const uniform = sampler.keyedUniform(seed, position, @intCast(id));
        const score = @as(f64, logits[id]) - @log(-@log(uniform));
        if (score > best or (score == best and id < expected)) {
            best = score;
            expected = @intCast(id);
        }
    }

    const k: usize = 2;
    const per_rank = k * 8 + 8;
    const host = try allocator.alloc(u8, per_rank * forward.world);
    defer allocator.free(host);
    @memset(host, 0);
    for (0..forward.world) |rank| {
        const rank_logits = logits[rank * 3 ..][0..3];
        const candidates = try sampler.topCandidatesRef(allocator, rank_logits, @intCast(rank * 3), k);
        defer allocator.free(candidates);
        const base = host[rank * per_rank ..][0..per_rank];
        for (candidates, 0..) |candidate, i| {
            @memcpy(base[i * 4 ..][0..4], std.mem.asBytes(&candidate.value));
            std.mem.writeInt(u32, base[k * 4 + i * 4 ..][0..4], candidate.id, .little);
        }
        const lse = sampler.partialLse(rank_logits);
        const lse_off = k * 8;
        @memcpy(base[lse_off..][0..4], std.mem.asBytes(&lse.max));
        @memcpy(base[lse_off + 4 ..][0..4], std.mem.asBytes(&lse.sum));
    }
    var full_lse: f32 = -std.math.inf(f32);
    for (logits) |value| full_lse = @max(full_lse, value);
    var full_sum: f32 = 0;
    for (logits) |value| full_sum += @exp(value - full_lse);
    full_lse += @log(full_sum);
    const fallback_logits = try allocator.dupe(f32, &logits);
    defer allocator.free(fallback_logits);
    var fake_forward: forward.Forward = undefined;
    fake_forward.head = .{ .candidates = k };
    var e: Engine = .{
        .gpa = allocator,
        .f = fake_forward,
        .seed = seed,
        .rule = rule,
        .head_rows = 1,
        .head_per_rank = per_rank,
        .head_host = host,
        .fallback_logits_host = fallback_logits,
        .fallback_rows = try allocator.dupe(bool, &.{true}),
        .fallback_log_z = try allocator.dupe(f32, &.{full_lse}),
        .fallback_vocab = logits.len,
    };
    defer allocator.free(e.fallback_rows);
    defer allocator.free(e.fallback_log_z);
    var actual: [1]u32 = undefined;
    try e.draw(0, 1, position, &actual);
    try std.testing.expectEqual(expected, actual[0]);

    // The same row fits when all vocabulary candidates are present; it must draw the same token.
    const fit_k = logits.len / forward.world;
    const fit_per_rank = fit_k * 8 + 8;
    const fit_host = try allocator.alloc(u8, fit_per_rank * forward.world);
    defer allocator.free(fit_host);
    @memset(fit_host, 0);
    for (0..forward.world) |rank| {
        const rank_logits = logits[rank * fit_k ..][0..fit_k];
        const base = fit_host[rank * fit_per_rank ..][0..fit_per_rank];
        for (rank_logits, 0..) |value, i| {
            @memcpy(base[i * 4 ..][0..4], std.mem.asBytes(&value));
            std.mem.writeInt(u32, base[fit_k * 4 + i * 4 ..][0..4], @intCast(rank * fit_k + i), .little);
        }
        const lse = sampler.partialLse(rank_logits);
        @memcpy(base[fit_k * 8 ..][0..4], std.mem.asBytes(&lse.max));
        @memcpy(base[fit_k * 8 + 4 ..][0..4], std.mem.asBytes(&lse.sum));
    }
    var fit_forward: forward.Forward = undefined;
    fit_forward.head = .{ .candidates = fit_k };
    e.f = fit_forward;
    e.head_per_rank = fit_per_rank;
    e.head_host = fit_host;
    e.fallback_rows[0] = false;
    var fit: [1]u32 = undefined;
    try e.draw(0, 1, position, &fit);
    try std.testing.expectEqual(actual[0], fit[0]);
}

pub fn greedyRow(head_host: []const u8, per: usize, n: usize, r: usize) ?sampler.Candidate {
    var best: ?sampler.Candidate = null;
    const world = if (per > 0) head_host.len / per else 0;
    for (0..world) |q| {
        const base = head_host[q * per ..];
        const v = std.mem.bytesToValue(f32, base[r * 4 ..][0..4]);
        const id = std.mem.readInt(u32, base[n * 4 + r * 4 ..][0..4], .little);
        // NaN never wins; WP10's absent-entry sentinel is never a token (review L1).
        if (std.math.isNan(v) or absentCandidate(.{ .value = v, .id = id }) or id == 0xFFFFFFFF) continue;
        if (best == null or v > best.?.value or (v == best.?.value and id < best.?.id)) best = .{ .value = v, .id = id };
    }
    return best;

}

test "resident reuse rounds the common prefix down to whole pieces" {
    const history: [1100]u32 = @splat(0);
    var prompt: [1100]u32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 1100, &.{}, 512));
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 1100, &.{ 1, 2 }, 512));
    try std.testing.expectEqual(@as(usize, 512), Engine.reusableRows(&history, 1100, prompt[0..700], 512));
    prompt[700] = 1;
    try std.testing.expectEqual(@as(usize, 512), Engine.reusableRows(&history, 1100, &prompt, 512));
    // Identical prompts still leave the final prompt row for a head computation.
    try std.testing.expectEqual(@as(usize, 1024), Engine.reusableRows(&history, 1100, &history, 512));
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 1100, history[0..512], 512));
    try std.testing.expectEqual(@as(usize, 512), Engine.reusableRows(&history, 1100, history[0..513], 512));
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 1100, &prompt, 0));
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 511, &history, 512));
    // A frontier beyond history is safely clamped to the available rows.
    try std.testing.expectEqual(@as(usize, 1024), Engine.reusableRows(&history, 1300, &history, 512));
}


test "resident prompt reuse supports the selected grain and leaves the head row" {
    const history: [1100]u32 = @splat(0);
    var prompt: [1100]u32 = @splat(0);
    try std.testing.expectEqual(@as(usize, 699), Engine.reusableRows(&history, 1100, prompt[0..700], 1));
    try std.testing.expectEqual(@as(usize, 640), Engine.reusableRows(&history, 1100, prompt[0..700], 64));
    prompt[700] = 1;
    try std.testing.expectEqual(@as(usize, 700), Engine.reusableRows(&history, 1100, prompt[0..701], 1));
    try std.testing.expectEqual(@as(usize, 1000), Engine.reusableRows(&history, 1000, &history, 1));
    try std.testing.expectEqual(@as(usize, 960), Engine.reusableRows(&history, 1000, &history, 64));
    try std.testing.expectEqual(@as(usize, 0), Engine.reusableRows(&history, 0, &history, 1));
}

test "greedyRow merges masked shards, keeps lower-id ties and ignores NaN" {
    var host: [16]u8 = @splat(0);
    std.mem.writeInt(u32, host[0..4], @bitCast(@as(f32, 2.0)), .little);
    std.mem.writeInt(u32, host[4..8], 5, .little);
    std.mem.writeInt(u32, host[8..12], @bitCast(-std.math.inf(f32)), .little);
    std.mem.writeInt(u32, host[12..16], 9, .little);
    try std.testing.expectEqual(sampler.Candidate{ .value = 2, .id = 5 }, greedyRow(&host, 8, 1, 0).?);
    std.mem.writeInt(u32, host[0..4], @bitCast(@as(f32, 2.0)), .little);
    std.mem.writeInt(u32, host[4..8], 12, .little);
    std.mem.writeInt(u32, host[8..12], @bitCast(@as(f32, 2.0)), .little);
    try std.testing.expectEqual(@as(u32, 9), greedyRow(&host, 8, 1, 0).?.id);
    std.mem.writeInt(u32, host[0..4], @bitCast(std.math.nan(f32)), .little);
    std.mem.writeInt(u32, host[4..8], 1, .little);
    try std.testing.expectEqual(@as(u32, 9), greedyRow(&host, 8, 1, 0).?.id);

}

const AppendProbe = struct {
    flushed: usize = 0,
    fn prefill(p: *AppendProbe, rows: []const u32) !void {
        try std.testing.expectEqual(@as(usize, 512), rows.len);
        p.flushed += rows.len;
    }
};

test "append feed flushes whole pieces only when another token follows" {
    const incoming = [_]usize{ 1, 511, 512, 513, 1024, 1025 };
    for ([_]usize{ 0, 300 }) |initial| {
        for (incoming) |count| {
            var pending: std.ArrayList(u32) = .empty;
            defer pending.deinit(std.testing.allocator);
            try pending.appendNTimes(std.testing.allocator, 7, initial);
            const ids = try std.testing.allocator.alloc(u32, count);
            defer std.testing.allocator.free(ids);
            @memset(ids, 9);
            var probe: AppendProbe = .{};
            try appendFeed(AppendProbe, std.testing.allocator, &pending, ids, 512, &probe, AppendProbe.prefill);
            const total = initial + count;
            const expected_flush = if (total == 0) 0 else (total - 1) / 512 * 512;
            try std.testing.expectEqual(expected_flush, probe.flushed);
            try std.testing.expectEqual(total - expected_flush, pending.items.len);
            try std.testing.expect(pending.items.len >= 1);
        }
    }
}

test "settle drops decode-written rows and keeps the prefill frontier" {
    var e: Engine = .{ .gpa = std.testing.allocator, .f = undefined, .pos = 7, .frontier = 4 };
    try e.history.appendSlice(e.gpa, &.{ 10, 11, 12, 13, 20, 21, 22 });
    try e.settle();
    try std.testing.expectEqual(@as(usize, 4), e.pos);
    try std.testing.expectEqual(@as(usize, 4), e.frontier);
    try std.testing.expectEqualSlices(u32, &.{ 10, 11, 12, 13 }, e.history.items);
    e.history.deinit(e.gpa);
}

test "agree always enters the fake two-rank reduction for local success and failure" {
    const Fake = struct {
        calls: usize = 0,
        peer_ok: bool = true,
        fn call(ctx: *anyopaque, words: []u32) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.calls += 1;
            for (words) |*word| word.* = @intFromBool(word.* != 0 and self.peer_ok);
        }
    };
    var fake = Fake{};
    var e: Engine = .{ .gpa = std.testing.allocator, .f = undefined, .agreement = .{ .ctx = &fake, .call_fn = Fake.call } };
    try std.testing.expect(try e.agree(true));
    try std.testing.expect(!try e.agree(false));
    try std.testing.expectEqual(@as(usize, 2), fake.calls);
}

test "append finishing rejects an empty tail" {
    var e: Engine = .{ .gpa = std.testing.allocator, .f = undefined };
    try std.testing.expectError(error.EmptyPrompt, e.appendFinish());
    e.append_pending.deinit(e.gpa);
    e.history.deinit(e.gpa);
}

test "CUDA graph keys distinguish verify rows and bucket prefill chunks and decode positions" {
    // These are the supported verify widths: 1..8, then the copy-draft widths 12 and 16.
    // At last_position 4096, the key bucket is ceil_pow2(4096 + 1) = 8192.
    for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 }) |rows| {
        const key = try decodeKey(rows, 4096);
        try std.testing.expectEqual(@as(u32, @intCast(rows)), key.rows);
        try std.testing.expectEqual(@as(u32, 8192), key.chunk_bucket);
        for ([_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 12, 16 }) |other| {
            if (rows != other) try std.testing.expect(!std.meta.eql(key, try decodeKey(other, 4096)));
        }
    }
    try std.testing.expectEqual(GraphKey{ .rows = 8, .chunk_bucket = 4096 }, try graphKey(8, 4096));
    try std.testing.expectEqual(GraphKey{ .rows = 1, .chunk_bucket = 8192 }, try graphKey(1, 5000));
    try std.testing.expectError(error.BadGraphKey, graphKey(0, 4096));
    try std.testing.expectEqual(GraphKey{ .rows = 808, .chunk_bucket = 1024 }, try prefillKey(808));
    try std.testing.expectEqual(GraphKey{ .rows = 4096, .chunk_bucket = 4096 }, try prefillKey(4096));
    try std.testing.expectEqual(GraphKey{ .rows = 4, .chunk_bucket = 2048 }, try decodeKey(4, 2047));
    try std.testing.expectEqual(GraphKey{ .rows = 4, .chunk_bucket = 4096 }, try decodeKey(4, 2048));
    try std.testing.expectEqual(GraphKey{ .rows = 1, .chunk_bucket = 262144 }, try decodeKey(1, 200000));
    try std.testing.expectError(error.BadGraphKey, decodeKey(17, 100));
    const plain = GraphKey{ .rows = 4, .chunk_bucket = 4096 };
    const ablated = GraphKey{ .rows = 4, .chunk_bucket = 4096, .ablate = true };
    try std.testing.expect(!std.meta.eql(plain, ablated));
    var map: std.AutoArrayHashMapUnmanaged(GraphKey, void) = .empty;
    defer map.deinit(std.testing.allocator);
    try map.put(std.testing.allocator, plain, {});
    try map.put(std.testing.allocator, ablated, {});
    try std.testing.expectEqual(@as(usize, 2), map.count());
}

test "MEMSLIM graph slots follow row width and ablate mode" {
    try std.testing.expect(sameGraphSlot(.{ .rows = 8, .chunk_bucket = 2048 }, .{ .rows = 8, .chunk_bucket = 4096 }));
    try std.testing.expect(!sameGraphSlot(.{ .rows = 8, .chunk_bucket = 2048 }, .{ .rows = 1, .chunk_bucket = 2048 }));
    try std.testing.expect(!sameGraphSlot(.{ .rows = 8, .chunk_bucket = 2048 }, .{ .rows = 8, .chunk_bucket = 2048, .ablate = true }));
}

test "verify staging accepts 12 and 16 rows and rejects 17" {
    try std.testing.expect(verifyRowsAllowed(12, 16, 16));
    try std.testing.expect(verifyRowsAllowed(16, 16, 16));
    try std.testing.expect(!verifyRowsAllowed(17, 16, 17));
}

test "setAblate requires vectors and refuses mode changes with resident rows" {
    var e: Engine = .{ .gpa = std.testing.allocator, .f = undefined };
    try std.testing.expectError(error.NoAblation, e.setAblate(true));
    try e.setAblate(false);
    e.abl = @ptrFromInt(@alignOf(@import("ablate").Ablation));
    e.pos = 1;
    try std.testing.expectError(error.AblateWithResidentRows, e.setAblate(true));
    try e.setAblate(false);
    e.pos = 0;
    try e.setAblate(true);
    try std.testing.expect(e.ablate_on);
}

test "absent WP10 candidates are recognised only by the id and -inf pair" {
    try std.testing.expect(absentCandidate(.{ .value = -std.math.inf(f32), .id = 0xFFFFFFFF }));
    try std.testing.expect(!absentCandidate(.{ .value = -std.math.inf(f32), .id = 3 }));
    try std.testing.expect(!absentCandidate(.{ .value = 0, .id = 0xFFFFFFFF }));
}

test "RoCE-poisoned head status refuses token draws" {
    var e: Engine = .{ .gpa = std.testing.allocator, .f = undefined, .head_status = 1 };
    var out: [1]u32 = undefined;
    try std.testing.expectError(error.RoceExchangeFailed, e.draw(0, 1, 0, &out));
}

var graph_test_available_gib: f64 = 0;

fn graphTestMeter() ?f64 {
    return graph_test_available_gib;
}

fn graphTestRoom() bool {
    // 4 GiB guard floor + half of the 4 GiB margin + one 64 MiB graph.
    return graph_test_available_gib >= 6.0625;
}

test "decode graph cache stays eager at 5.9 GiB and captures at 6.1 GiB" {
    const FakeBackend = struct {
        captures: u32 = 0,
        launches: u32 = 0,
        slots: [8]u8 = @splat(0),
        fn engine(f: *@This()) cuda.graph_cache.Engine {
            return .{ .ctx = f, .capture = capture, .launch = launch, .free = free };
        }
        fn capture(ctx: *anyopaque, _: cuda.Stream, _: cuda.graph_cache.Body) anyerror!*anyopaque {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            const slot: usize = @intCast(f.captures % @as(u32, @intCast(f.slots.len)));
            f.captures += 1;
            return @ptrCast(&f.slots[slot]);
        }
        fn launch(ctx: *anyopaque, _: *anyopaque, _: cuda.Stream) anyerror!void {
            const f: *@This() = @ptrCast(@alignCast(ctx));
            f.launches += 1;
        }
        fn free(_: *anyopaque, _: *anyopaque) void {}
    };
    const BodyProbe = struct {
        runs: u32 = 0,
        fn body(b: *@This()) cuda.graph_cache.Body {
            return .{ .ctx = b, .run = run };
        }
        fn run(ctx: *anyopaque, _: cuda.Stream) anyerror!void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.runs += 1;
        }
    };
    var backend: FakeBackend = .{};
    var body: BodyProbe = .{};
    var cache = cuda.graph_cache.Cache(GraphKey).init(std.testing.allocator, backend.engine(), .{ .warm = 0, .max = 2 });
    cache.meter = graphTestMeter;
    cache.room = graphTestRoom;
    defer cache.deinit();

    graph_test_available_gib = 5.9;
    try std.testing.expectEqual(cuda.graph_cache.Outcome.eager, try cache.run(.{ .rows = 1, .chunk_bucket = 2048 }, .{ .d = undefined, .handle = null }, 1, body.body()));
    try std.testing.expectEqual(@as(u32, 0), backend.captures);
    try std.testing.expectEqual(@as(u32, 1), body.runs);

    graph_test_available_gib = 6.1;
    try std.testing.expectEqual(cuda.graph_cache.Outcome.captured, try cache.run(.{ .rows = 1, .chunk_bucket = 2048 }, .{ .d = undefined, .handle = null }, 1, body.body()));
    try std.testing.expectEqual(@as(u32, 1), backend.captures);
    try std.testing.expectEqual(@as(usize, 1), cache.count());
}

test "force semantic analysis of every engine declaration" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Engine);
    std.testing.refAllDecls(Download);
}
