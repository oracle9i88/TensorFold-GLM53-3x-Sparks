// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
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

    pub fn sharedForLayer(s: Share, layer: usize) Span {
        if (s.world == 3) {
            return spanMoeRotated(s.rank, (layer + 1) % 3);
        }
        return s.shared;
    }
};

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
