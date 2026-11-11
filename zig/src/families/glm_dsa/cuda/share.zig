// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! Tensor-parallel spans for glm_moe_dsa CUDA.
const std = @import("std");
const Config = @import("config").Config;

pub const Span = struct {
    lo: usize,
    hi: usize,

    pub fn len(s: Span) usize {
        return s.hi - s.lo;
    }
};

pub const Share = struct {
    rank: usize,
    world: usize,
    heads: Span,
    moe: Span,
    shared: Span,
    dense: Span,
    vocab: Span,

    pub fn of(c: *const Config, rank: usize, world: usize) !Share {
        if (world == 0 or rank >= world) return error.BadRank;
        return .{
            .rank = rank,
            .world = world,
            .heads = try spanHeads(c.heads, rank, world),
            .moe = try spanMoe(c.moe_inter, rank, world),
            .shared = try spanMoe(c.moe_inter, rank, world),
            .dense = try span128Channels(c.dense_inter, rank, world),
            .vocab = try spanVocab(c.vocab, rank, world),
        };
    }

    pub fn moeForLayer(s: Share, layer: usize) Span {
        if (s.world == 3) {
            return spanMoeRotated(s.rank, layer % 3);
        }
        return s.moe;
    }

    /// Routed expert `expert`'s channels on this rank under TF_GLM53_MOE_BALANCE: `.expert` rotates the six-block
    /// share by expert as well as by layer (k = (expert + layer) % 3), so the eight experts a row picks spread their big
    /// shares over the three ranks instead of all landing on layer % 3; `.layer` keeps moeForLayer's split.
    pub fn moeForExpert(s: Share, layer: usize, expert: usize, balance: MoeBalance) Span {
        if (s.world == 3 and balance == .expert) return spanMoeRotated(s.rank, (expert + layer) % 3);
        return s.moeForLayer(layer);
    }

    pub fn sharedForLayer(s: Share, layer: usize) Span {
        if (s.world == 3) {
            return spanMoeRotated(s.rank, (layer + 1) % 3);
        }
        return s.shared;
    }
};

/// TF_GLM53_MOE_BALANCE: unset or "off" loads the routed experts as before (one span a layer); "layer" loads the same
/// spans through the per-expert path (span table and per-expert kernels; must reproduce "off" bit for bit); "expert"
/// rotates each expert's six-block share (Share.moeForExpert). Three ranks only.
pub const MoeBalance = enum { off, layer, expert };

pub fn moeBalanceSelected(value: ?[]const u8) !MoeBalance {
    const raw = value orelse return .off;
    if (raw.len == 0 or std.mem.eql(u8, raw, "off")) return .off;
    if (std.mem.eql(u8, raw, "layer")) return .layer;
    if (std.mem.eql(u8, raw, "expert")) return .expert;
    return error.BadMoeBalance;
}

pub fn moeBalanceFromEnv() !MoeBalance {
    const v = std.c.getenv("TF_GLM53_MOE_BALANCE") orelse return .off;
    return moeBalanceSelected(std.mem.span(v));
}

/// Rotated TP=3 MoE expert intermediate channels across layers (k = (layer + offset) % 3).
pub fn spanMoeRotated(rank: usize, k: usize) Span {
    if (k == 0) {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 768 },
            1 => .{ .lo = 768, .hi = 1408 },
            2 => .{ .lo = 1408, .hi = 2048 },
            else => .{ .lo = 0, .hi = 0 },
        };
    } else if (k == 1) {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 640 },
            1 => .{ .lo = 640, .hi = 1408 },
            2 => .{ .lo = 1408, .hi = 2048 },
            else => .{ .lo = 0, .hi = 0 },
        };
    } else {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 640 },
            1 => .{ .lo = 640, .hi = 1280 },
            2 => .{ .lo = 1280, .hi = 2048 },
            else => .{ .lo = 0, .hi = 0 },
        };
    }
}

/// Split heads across ranks, supporting uneven TP=3 (22, 22, 20).
pub fn spanHeads(total_heads: usize, rank: usize, world: usize) !Span {
    if (world == 0 or rank >= world) return error.BadSplit;
    if (world == 3 and total_heads == 64) {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 22 },
            1 => .{ .lo = 22, .hi = 44 },
            2 => .{ .lo = 44, .hi = 64 },
            else => error.BadRank,
        };
    }
    return spanChannels(total_heads, rank, world);
}

/// Split MoE expert intermediate channels (in multiples of 128), supporting uneven TP=3 (6, 5, 5 halves).
pub fn spanMoe(total_channels: usize, rank: usize, world: usize) !Span {
    if (world == 0 or rank >= world) return error.BadSplit;
    if (world == 3 and total_channels == 2048) {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 768 },
            1 => .{ .lo = 768, .hi = 1408 },
            2 => .{ .lo = 1408, .hi = 2048 },
            else => error.BadRank,
        };
    }
    return span128Channels(total_channels, rank, world);
}

/// Split vocabulary across ranks, supporting uneven TP=3 (51712, 51584, 51584).
pub fn spanVocab(total_vocab: usize, rank: usize, world: usize) !Span {
    if (world == 0 or rank >= world) return error.BadSplit;
    if (world == 3 and total_vocab == 154880) {
        return switch (rank) {
            0 => .{ .lo = 0, .hi = 51712 },
            1 => .{ .lo = 51712, .hi = 103296 },
            2 => .{ .lo = 103296, .hi = 154880 },
            else => error.BadRank,
        };
    }
    return spanChannels(total_vocab, rank, world);
}

/// Split a channel axis evenly and return global channel coordinates.
pub fn spanChannels(total_channels: usize, rank: usize, world: usize) !Span {
    if (world == 0 or world > 256 or total_channels % world != 0) return error.BadSplit;
    const width = total_channels / world;
    return .{ .lo = rank * width, .hi = (rank + 1) * width };
}

/// Split an axis on 128-channel quantization units, returning channel coordinates.
pub fn span128Channels(total_channels: usize, rank: usize, world: usize) !Span {
    if (world == 0 or total_channels % 128 != 0 or (total_channels / 128) % world != 0) return error.BadSplit;
    return spanChannels(total_channels, rank, world);
}

test "TP2 owns 32 heads and halves of expert, dense and vocabulary axes" {
    const c = Config{};
    const r0 = try Share.of(&c, 0, 2);
    const r1 = try Share.of(&c, 1, 2);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 32 }, r0.heads);
    try std.testing.expectEqual(Span{ .lo = 32, .hi = 64 }, r1.heads);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 1024 }, r0.moe);
    try std.testing.expectEqual(Span{ .lo = 1024, .hi = 2048 }, r1.shared);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 6144 }, r0.dense);
    try std.testing.expectEqual(Span{ .lo = 77440, .hi = 154880 }, r1.vocab);
    try std.testing.expectError(error.BadRank, Share.of(&c, 2, 2));
}

test "TP3 owns 22/22/20 heads, 6/5/5 halves, 4096 dense, and 51712/51584/51584 vocab" {
    const c = Config{};
    const r0 = try Share.of(&c, 0, 3);
    const r1 = try Share.of(&c, 1, 3);
    const r2 = try Share.of(&c, 2, 3);
    try std.testing.expectEqual(Span{ .lo = 0, .hi = 22 }, r0.heads);
    try std.testing.expectEqual(Span{ .lo = 22, .hi = 44 }, r1.heads);
    try std.testing.expectEqual(Span{ .lo = 44, .hi = 64 }, r2.heads);

    try std.testing.expectEqual(Span{ .lo = 0, .hi = 768 }, r0.moe);
    try std.testing.expectEqual(Span{ .lo = 768, .hi = 1408 }, r1.moe);
    try std.testing.expectEqual(Span{ .lo = 1408, .hi = 2048 }, r2.moe);

    try std.testing.expectEqual(Span{ .lo = 0, .hi = 4096 }, r0.dense);
    try std.testing.expectEqual(Span{ .lo = 4096, .hi = 8192 }, r1.dense);
    try std.testing.expectEqual(Span{ .lo = 8192, .hi = 12288 }, r2.dense);

    try std.testing.expectEqual(Span{ .lo = 0, .hi = 51712 }, r0.vocab);
    try std.testing.expectEqual(Span{ .lo = 51712, .hi = 103296 }, r1.vocab);
    try std.testing.expectEqual(Span{ .lo = 103296, .hi = 154880 }, r2.vocab);
    try std.testing.expectError(error.BadRank, Share.of(&c, 3, 3));
}

test "TF_GLM53_MOE_BALANCE=expert: every expert's three spans tile 2048 in 128 blocks, big shares a third a rank" {
    const c = Config{};
    var sh: [3]Share = undefined;
    for (&sh, 0..) |*s, r| s.* = try Share.of(&c, r, 3);
    for (0..78) |layer| {
        var big: [3]usize = .{ 0, 0, 0 };
        for (0..256) |e| {
            var at: usize = 0;
            for (sh, 0..) |s, r| {
                const span = s.moeForExpert(layer, e, .expert);
                try std.testing.expectEqual(at, span.lo); // ranks in order, no gap, no overlap
                try std.testing.expect(span.lo % 128 == 0 and span.hi % 128 == 0);
                try std.testing.expect(span.len() == 768 or span.len() == 640);
                if (span.len() == 768) big[r] += 1;
                at = span.hi;
            }
            try std.testing.expectEqual(@as(usize, 2048), at);
        }
        for (big) |b| try std.testing.expect(b == 85 or b == 86);
        // .layer and .off keep the layer's one span for every expert
        for (sh) |s| for ([_]usize{ 0, 7, 255 }) |e| try std.testing.expectEqual(s.moeForLayer(layer), s.moeForExpert(layer, e, .layer));
    }
    try std.testing.expectEqual(MoeBalance.off, try moeBalanceSelected(null));
    try std.testing.expectEqual(MoeBalance.off, try moeBalanceSelected("off"));
    try std.testing.expectEqual(MoeBalance.layer, try moeBalanceSelected("layer"));
    try std.testing.expectEqual(MoeBalance.expert, try moeBalanceSelected("expert"));
    try std.testing.expectError(error.BadMoeBalance, moeBalanceSelected("experts"));
}
