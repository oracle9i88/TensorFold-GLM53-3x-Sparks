// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Kristian Aaron (Krishando)
//! Adapted from upstream TensorFold #1381: a drafter pass can skip holds for a stream with a copy ahead.
const std = @import("std");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;

test "copy_first suppresses the drafter hold while leaving default copy behavior alone" {
    const empty: Table = .{};
    const Copy = struct { fn yes(_: *anyopaque, _: *const depth.Who) bool { return true; } };
    var probe_byte: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &probe_byte, .copy = Copy.yes };
    var st: ?depth.State = null;
    defer if (st) |*s| s.deinit(std.testing.allocator);
    const who: depth.Who = .{ .state = &st, .draft_room = 32, .finished = false, .forced = false, .granted = 2 };
    var ordinary: depth.Rule = .{ .gpa = std.testing.allocator, .prior = &.{0.8}, .most_drafts = 7, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .batch_rows = 12 };
    defer ordinary.deinit();
    var opt_in: depth.Rule = .{ .gpa = std.testing.allocator, .k = .{ .copy_first = true }, .prior = &.{0.8}, .most_drafts = 7, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 12 };
    defer opt_in.deinit();
    try std.testing.expectEqual(@as(i64, 1), try ordinary.headDepth(&who, null, probe));
    try std.testing.expectEqual(@as(i64, 0), try opt_in.headDepth(&who, null, probe));
}

test "copy arbitration leaves the drafter at its normal held depth" {
    const empty: Table = .{};
    const Copy = struct { fn yes(_: *anyopaque, _: *const depth.Who) bool { return true; } };
    var probe_byte: u8 = 0;
    const probe: depth.Probe = .{ .ptr = &probe_byte, .copy = Copy.yes };
    var state: ?depth.State = null;
    defer if (state) |*s| s.deinit(std.testing.allocator);
    const stream: depth.Who = .{ .state = &state, .draft_room = 32, .finished = false, .forced = false, .granted = 2 };
    var normal: depth.Rule = .{ .gpa = std.testing.allocator, .prior = &.{0.8}, .most_drafts = 7, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 12 };
    defer normal.deinit();
    var arbitration: depth.Rule = .{ .gpa = std.testing.allocator, .k = .{ .copy_arbitrate = true }, .prior = &.{0.8}, .most_drafts = 7, .family_costs = &empty, .shared_costs = &empty, .mtp_step_ms = 0.5, .plain_guard = false, .node_probabilities = true, .granted_depth = true, .batch_rows = 12 };
    defer arbitration.deinit();
    try std.testing.expectEqual(@as(i64, 1), try normal.headDepth(&stream, null, probe));
    try std.testing.expectEqual(@as(i64, 3), try arbitration.headDepth(&stream, null, probe));
}
