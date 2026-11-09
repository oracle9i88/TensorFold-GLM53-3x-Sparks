// SPDX-License-Identifier: Apache-2.0
//! GLM-5.3 drafter training capture (TF_GLM53_CAPTURE_DIR, rank 0 only): one file per request with the rows a
//! drafter trains on, read from buffers the serving path already fills (no extra kernels, outputs unchanged).
//! - prompt block: the drafter tap rows of the prompt's last `window` positions (what the drafter's ring holds);
//! - verify block: a verify window's tap rows and final residual rows (after the last layer, before the final norm),
//!   its input ids, the target's drawn token per row, and how many rows were drafter drafts (copy proposals: 0);
//! - end block: the transcript (prompt and kept tokens) and the sampling rule.
//! A position written again by a later window was rejected there: the transcript row is the last write, and the
//! earlier writes are on-policy rows (the target's distribution over the drafter's own proposals).
//! File: "TFCAP001", then blocks: u32 kind, u32 JSON bytes, JSON, u64 payload bytes, payload (little-endian; bf16 rows
//! laid out [row][layer][hidden] for taps, [row][hidden] for the final residual).
const std = @import("std");
const cuda = @import("cuda");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, count: usize, stream: *anyopaque) usize;
extern "c" fn fclose(stream: *anyopaque) c_int;

pub const schema = "tf.glm53.capture/v1";
pub const magic = "TFCAP001";
pub const Kind = enum(u32) { header = 0, prompt = 1, verify = 2, end = 3 };

/// Rows per device copy: prompt blocks go out in slices of this many rows; a verify window must fit in one.
pub const slice_rows: usize = 64;

pub const Capture = struct {
    gpa: std.mem.Allocator,
    dir: []u8,
    layers: []usize,
    hidden: usize,
    window: usize,
    drafter: []u8,
    pid: std.c.pid_t,
    seq: u64 = 0,
    file: ?*anyopaque = null,
    /// Host staging for one slice: slice_rows * (layers + 1) * hidden bf16 values.
    stage: []u8,
    json: std.ArrayList(u8) = .empty,
    warned: bool = false,

    pub fn init(gpa: std.mem.Allocator, dir: []const u8, layers: []const usize, hidden: usize, window: usize, drafter: []const u8) !Capture {
        const stage = try gpa.alloc(u8, slice_rows * (layers.len + 1) * hidden * 2);
        errdefer gpa.free(stage);
        const own_dir = try gpa.dupe(u8, dir);
        errdefer gpa.free(own_dir);
        const own_layers = try gpa.dupe(usize, layers);
        errdefer gpa.free(own_layers);
        const own_drafter = try gpa.dupe(u8, drafter);
        return .{ .gpa = gpa, .dir = own_dir, .layers = own_layers, .hidden = hidden, .window = window, .drafter = own_drafter, .pid = std.c.getpid(), .stage = stage };
    }

    pub fn deinit(c: *Capture) void {
        c.close();
        c.json.deinit(c.gpa);
        c.gpa.free(c.stage);
        c.gpa.free(c.dir);
        c.gpa.free(c.layers);
        c.gpa.free(c.drafter);
    }

    /// After a prefill pass: the prompt rows [from, end) sit at rows 0.. of the drafter's prompt tap buffer.
    /// A pass from position 0 starts a new request file.
    pub fn prompt(c: *Capture, stream: cuda.Stream, taps: u64, start: usize, from: usize, end_pos: usize) void {
        c.promptRows(stream, taps, start, from, end_pos) catch |err| c.fail("prompt rows", err);
    }

    /// After a verify window's draw: its tap rows and final residual rows sit at rows 0.. of their buffers.
    pub fn verify(c: *Capture, stream: cuda.Stream, taps: u64, x: u64, pos: usize, held: u32, ids: []const u32, sampled: []const u32) void {
        c.verifyRows(stream, taps, x, pos, held, ids, sampled) catch |err| c.fail("verify rows", err);
    }

    /// Request end (before any rollback): the transcript, then the file closes.
    pub fn end(c: *Capture, transcript: []const u32, temperature: f64, top_p: f64, top_k: usize, seed: u64) void {
        c.endBlock(transcript, temperature, top_p, top_k, seed) catch |err| c.fail("end block", err);
        c.close();
    }

    fn promptRows(c: *Capture, stream: cuda.Stream, taps: u64, start: usize, from: usize, end_pos: usize) !void {
        if (start == 0) try c.open();
        if (c.file == null) return; // this request's file failed, or a continuation arrived without its start
        if (end_pos < from) return error.CaptureRange;
        const rows = end_pos - from;
        const row_bytes = c.layers.len * c.hidden * 2;
        c.json.clearRetainingCapacity();
        try c.json.print(c.gpa, "{{\"start\":{d},\"from\":{d},\"end\":{d}}}", .{ start, from, end_pos });
        try c.head(.prompt, rows * row_bytes);
        var at: usize = 0;
        while (at < rows) : (at += slice_rows) {
            const n = @min(slice_rows, rows - at);
            const dst = c.stage[0 .. n * row_bytes];
            try download(stream, taps + at * row_bytes, dst);
            try c.raw(dst);
        }
    }

    fn verifyRows(c: *Capture, stream: cuda.Stream, taps: u64, x: u64, pos: usize, held: u32, ids: []const u32, sampled: []const u32) !void {
        if (c.file == null) return;
        const rows = ids.len;
        if (rows == 0 or rows > slice_rows or sampled.len != rows) return error.CaptureRows;
        const tap_bytes = rows * c.layers.len * c.hidden * 2;
        const x_bytes = rows * c.hidden * 2;
        try download(stream, taps, c.stage[0..tap_bytes]);
        try download(stream, x, c.stage[tap_bytes..][0..x_bytes]);
        c.json.clearRetainingCapacity();
        try c.json.print(c.gpa, "{{\"pos\":{d},\"rows\":{d},\"held\":{d},\"ids\":", .{ pos, rows, held });
        try writeU32s(&c.json, c.gpa, ids);
        try c.json.appendSlice(c.gpa, ",\"sampled\":");
        try writeU32s(&c.json, c.gpa, sampled);
        try c.json.append(c.gpa, '}');
        try c.head(.verify, tap_bytes + x_bytes);
        try c.raw(c.stage[0 .. tap_bytes + x_bytes]);
    }

    fn endBlock(c: *Capture, transcript: []const u32, temperature: f64, top_p: f64, top_k: usize, seed: u64) !void {
        if (c.file == null) return;
        c.json.clearRetainingCapacity();
        try c.json.print(c.gpa, "{{\"len\":{d},\"temperature\":{d},\"top_p\":{d},\"top_k\":{d},\"seed\":{d}}}", .{ transcript.len, temperature, top_p, top_k, seed });
        try c.head(.end, transcript.len * 4);
        try c.raw(std.mem.sliceAsBytes(transcript));
    }

    fn open(c: *Capture) !void {
        c.close();
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&buf, "{s}/req-{d}-{d:0>6}.tfcap", .{ c.dir, c.pid, c.seq }, 0);
        c.seq += 1;
        c.file = fopen(path.ptr, "wb") orelse return error.CaptureOpen;
        try c.raw(magic);
        c.json.clearRetainingCapacity();
        try c.json.print(c.gpa, "{{\"schema\":\"{s}\",\"hidden\":{d},\"window\":{d},\"final\":\"residual after the last layer, before the final norm\",\"layers\":", .{ schema, c.hidden, c.window });
        try writeUsizes(&c.json, c.gpa, c.layers);
        try c.json.appendSlice(c.gpa, ",\"drafter\":");
        const quoted = try std.json.Stringify.valueAlloc(c.gpa, c.drafter, .{});
        defer c.gpa.free(quoted);
        try c.json.appendSlice(c.gpa, quoted);
        try c.json.append(c.gpa, '}');
        try c.head(.header, 0);
    }

    fn close(c: *Capture) void {
        if (c.file) |f| _ = fclose(f);
        c.file = null;
    }

    /// A failed write stops this request's file (logged once per process); serving goes on.
    fn fail(c: *Capture, what: []const u8, err: anyerror) void {
        if (!c.warned) std.log.warn("glm53 capture: {s} failed ({s}); this request's file stops here", .{ what, @errorName(err) });
        c.warned = true;
        c.close();
    }

    fn head(c: *Capture, kind: Kind, payload: usize) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], @intFromEnum(kind), .little);
        std.mem.writeInt(u32, b[4..8], @intCast(c.json.items.len), .little);
        try c.raw(&b);
        try c.raw(c.json.items);
        var p: [8]u8 = undefined;
        std.mem.writeInt(u64, &p, @intCast(payload), .little);
        try c.raw(&p);
    }

    fn raw(c: *Capture, bytes: []const u8) !void {
        const f = c.file orelse return error.CaptureClosed;
        if (bytes.len == 0) return;
        if (fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.CaptureWrite;
    }
};

/// Production download (as engine.Download): finish the stream's work, then copy.
fn download(stream: cuda.Stream, src: u64, dst: []u8) !void {
    try stream.synchronize();
    try stream.d.check(stream.d.api.cuMemcpyDtoH_v2(dst.ptr, src, dst.len), "GLM-5.3 capture rows");
}

fn writeU32s(list: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: []const u32) !void {
    try list.append(gpa, '[');
    for (xs, 0..) |x, i| {
        if (i != 0) try list.append(gpa, ',');
        try list.print(gpa, "{d}", .{x});
    }
    try list.append(gpa, ']');
}

fn writeUsizes(list: *std.ArrayList(u8), gpa: std.mem.Allocator, xs: []const usize) !void {
    try list.append(gpa, '[');
    for (xs, 0..) |x, i| {
        if (i != 0) try list.append(gpa, ',');
        try list.print(gpa, "{d}", .{x});
    }
    try list.append(gpa, ']');
}

test "glm53 capture: id lists are JSON arrays" {
    const gpa = std.testing.allocator;
    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    try writeU32s(&list, gpa, &.{ 7, 8, 9 });
    try std.testing.expectEqualStrings("[7,8,9]", list.items);
    list.clearRetainingCapacity();
    try writeUsizes(&list, gpa, &.{});
    try std.testing.expectEqualStrings("[]", list.items);
}
