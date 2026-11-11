// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 the glm53-tp3-3x-sparks contributors
//! TF_GLM53_FILL_*: with request slots, a prompt's pass runs in pieces between the other streams' rounds (the lane
//! host's fill queue), so a long prompt no longer stops every reply for its whole pass. Pieces end on multiples of the
//! piece rows, a multiple of the engine's chunk rows: the engine cuts its chunks from absolute positions, so the pieces
//! run exactly the chunks one pass would (forward.chunkAt), and the cache rows and the first token are the same.
//! While other streams decode, a piece is followed by their rounds for a while (the fill share); alone, pieces run
//! back to back. Host arithmetic only.
const std = @import("std");

/// Where the next piece of a prompt of `len` tokens ends, from position `pos`: the next multiple of `rows` (0: the rest).
pub fn pieceEnd(pos: usize, len: usize, rows: usize) usize {
    if (rows == 0 or pos >= len) return len;
    return @min(len, (pos / rows + 1) * rows);
}

/// How long the decoding streams run after a piece that took `spent_ns`, so the pass takes `share` of the time.
pub fn waitAfter(spent_ns: u64, share: f64) u64 {
    if (!(share < 1)) return 0;
    if (!(share > 0)) return std.math.maxInt(u64);
    return @intFromFloat(@as(f64, @floatFromInt(spent_ns)) * (1 - share) / share);
}

/// Piece rows must be whole chunks of the engine's grid (0: the whole prompt in one pass).
pub fn validRows(rows: usize, chunk_rows: usize) bool {
    return rows == 0 or (chunk_rows != 0 and rows % chunk_rows == 0);
}

pub fn parseShare(raw: ?[]const u8) error{BadFillShare}!f64 {
    const v = raw orelse return 0.5;
    const x = std.fmt.parseFloat(f64, std.mem.trim(u8, v, " ")) catch return error.BadFillShare;
    if (!(x > 0 and x <= 1)) return error.BadFillShare;
    return x;
}

test "fill: pieces end on the absolute grid, the last one at the prompt's end" {
    try std.testing.expectEqual(@as(usize, 4096), pieceEnd(0, 68_000, 4096));
    try std.testing.expectEqual(@as(usize, 69_632), pieceEnd(67_584, 80_000, 4096)); // a reused prefix mid-chunk: to the chunk's end
    try std.testing.expectEqual(@as(usize, 68_000), pieceEnd(65_536, 68_000, 4096));
    try std.testing.expectEqual(@as(usize, 68_000), pieceEnd(12_288, 68_000, 0));
    try std.testing.expectEqual(@as(usize, 16_384), pieceEnd(8_192, 68_000, 8192));
    // every piece boundary is a multiple of the engine's chunk rows (4096) whatever the reused start
    var pos: usize = 1536;
    while (pos < 50_000) {
        const e = pieceEnd(pos, 50_000, 8192);
        try std.testing.expect(e == 50_000 or e % 4096 == 0);
        pos = e;
    }
}

test "fill: the decoding streams get the time the share leaves them" {
    try std.testing.expectEqual(@as(u64, 8_000_000_000), waitAfter(8_000_000_000, 0.5));
    try std.testing.expectEqual(@as(u64, 24_000_000_000), waitAfter(8_000_000_000, 0.25));
    try std.testing.expectEqual(@as(u64, 0), waitAfter(8_000_000_000, 1.0));
    try std.testing.expect(validRows(4096, 4096) and validRows(8192, 4096) and validRows(0, 4096));
    try std.testing.expect(!validRows(6144, 4096) and !validRows(2048, 4096));
    try std.testing.expectEqual(@as(f64, 0.5), try parseShare(null));
    try std.testing.expectEqual(@as(f64, 0.25), try parseShare("0.25"));
    try std.testing.expectError(error.BadFillShare, parseShare("0"));
    try std.testing.expectError(error.BadFillShare, parseShare("1.5"));
}
