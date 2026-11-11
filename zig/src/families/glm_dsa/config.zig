// SPDX-License-Identifier: Apache-2.0
// Modified 2026 by Kristian Aaron (Krishando). Original: ashhart/TensorFold@d1018b318 (Apache-2.0).
// Modified 2026 in the glm53-tp3-3x-sparks branch (GLM-5.3 on three DGX Sparks); see docs/tp3-changes.txt.
//! GLM-5.3's shape (model_type glm_moe_dsa) from config.json, and one Mac's share of it under tensor parallelism.
const std = @import("std");

pub const max_layers = 96;

pub const KvFormat = enum { bf16, fp8 };
pub const RoutedCodebook = enum { mcg, mcg_e4m3, mul1 };
pub const BackboneFormat = enum { bf16, fp8_block, other };
pub const Backend = enum { metal, cuda };

/// Cache layout and distribution are serving choices, not checkpoint shape.
pub const ServingOptions = struct {
    dcp: u32,
    kv: KvFormat,

    pub fn defaults(backend: Backend) ServingOptions {
        return .{ .dcp = if (backend == .metal) 1 else 2, .kv = .bf16 };
    }
};

pub const Config = struct {
    hidden: u32 = 6144,
    layers: u32 = 78,
    vocab: u32 = 154880,
    eps: f32 = 1e-5,
    /// LayerNorm epsilon for the DSA indexer's k_norm.
    indexer_eps: f32 = 1e-6,
    dense_layers: u32 = 3,
    dense_inter: u32 = 12288,
    experts: u32 = 256,
    topk: u32 = 8,
    moe_inter: u32 = 2048,
    routed_scale: f32 = 2.5,
    heads: u32 = 64,
    nope: u32 = 192,
    rope: u32 = 64,
    v_dim: u32 = 256,
    q_lora: u32 = 2048,
    kv_lora: u32 = 512,
    rope_theta: f64 = 8e6,
    i_heads: u32 = 32,
    i_dim: u32 = 128,
    i_topk: u32 = 2048,
    /// MTP layers advertised by the source HF config.
    configured_mtp: u32 = 0,
    /// Preserved for the Metal drafter; the CUDA serving path may ignore MTP.
    mtp: u32 = 0,
    /// Effective serving choice, applied after parsing via `applyServingOptions`.
    dcp: u32 = 1,
    kv: KvFormat = .bf16,
    routed_codebook: RoutedCodebook = .mcg,
    backbone: BackboneFormat = .fp8_block,
    n_shared_experts: u32 = 1,
    norm_topk_prob: bool = true,
    eos: [3]u32 = .{ 154820, 154827, 154829 },
    eos_n: u32 = 3,
    /// Layer i runs its own indexer when set; otherwise it reuses the last full layer's top-k.
    full: std.StaticBitSet(max_layers) = .empty,
    /// This Mac's rank of `ranks`: its attention heads, dense rows, expert halves and vocabulary rows.
    rank: u32 = 0,
    ranks: u32 = 1,

    pub fn isMoe(c: *const Config, i: u32) bool {
        return i >= c.dense_layers;
    }
    pub fn qHead(c: *const Config) u32 {
        return c.nope + c.rope;
    }
    /// The attention heads this Mac runs: [head0, head0 + localHeads).
    pub fn localHeads(c: *const Config) u32 {
        if (c.ranks == 3 and c.heads == 64) {
            return if (c.rank == 2) 20 else 22;
        }
        return c.heads / c.ranks;
    }
    pub fn head0(c: *const Config) u32 {
        if (c.ranks == 3 and c.heads == 64) {
            return switch (c.rank) {
                0 => 0,
                1 => 22,
                2 => 44,
                else => 0,
            };
        }
        return c.rank * c.localHeads();
    }
    /// Each routed or shared expert's 128-channel halves this Mac holds: [half0, half0 + localHalves).
    pub fn localHalves(c: *const Config) u32 {
        if (c.ranks == 3 and c.moe_inter == 2048) {
            return if (c.rank == 0) 6 else 5;
        }
        return c.moe_inter / 128 / c.ranks;
    }
    pub fn half0(c: *const Config) u32 {
        if (c.ranks == 3 and c.moe_inter == 2048) {
            return switch (c.rank) {
                0 => 0,
                1 => 6,
                2 => 11,
                else => 0,
            };
        }
        return c.rank * c.localHalves();
    }
    pub fn localDense(c: *const Config) u32 {
        return c.dense_inter / c.ranks;
    }
    /// The head's vocabulary rows [lo, hi) this Mac computes.
    pub fn vocabPart(c: *const Config) [2]u32 {
        if (c.ranks == 3 and c.vocab == 154880) {
            return switch (c.rank) {
                0 => .{ 0, 51712 },
                1 => .{ 51712, 103296 },
                2 => .{ 103296, 154880 },
                else => .{ 0, 0 },
            };
        }
        const n = c.vocab / c.ranks;
        return .{ c.rank * n, if (c.rank + 1 == c.ranks) c.vocab else (c.rank + 1) * n };
    }
    /// The full layer whose top-k layer i uses: itself when it has an indexer.
    pub fn indexFrom(c: *const Config, i: u32) u32 {
        var l = i;
        while (l > 0 and !c.full.isSet(l)) l -= 1;
        return l;
    }

    pub fn applyServingOptions(c: *Config, options: ServingOptions) !void {
        if (options.dcp < 1 or options.dcp > 2) return error.UnsupportedDcp;
        c.dcp = options.dcp;
        c.kv = options.kv;
    }

    /// CUDA supports only the checkpoint's native FP8 128x128 backbone scales.
    pub fn admitCuda(c: *const Config) !void {
        if (c.backbone != .fp8_block) return error.UnsupportedBackbone;
    }
};

fn int(o: std.json.ObjectMap, key: []const u8) !i64 {
    const v = o.get(key) orelse return error.BadConfig;
    return if (v == .integer) v.integer else error.BadConfig;
}

fn float(o: std.json.ObjectMap, key: []const u8) !f64 {
    const v = o.get(key) orelse return error.BadConfig;
    return switch (v) {
        .float => |f| f,
        .integer => |i| @floatFromInt(i),
        else => error.BadConfig,
    };
}

fn str(o: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const v = o.get(key) orelse return null;
    return if (v == .string) v.string else null;
}

/// Keep room for the optional MTP layer in the fixed IndexShare bitset.
pub fn validateLayerCapacity(layers: u32, mtp: u32) !void {
    if (layers > max_layers or mtp > max_layers - layers) return error.BadConfig;
}

fn backboneFormat(quant: ?std.json.ObjectMap) BackboneFormat {
    const q = quant orelse return .bf16;
    const block = q.get("weight_block_size");
    if (std.mem.eql(u8, str(q, "quant_method") orelse "", "fp8") and
        std.mem.eql(u8, str(q, "fmt") orelse "", "e4m3") and block != null and block.? == .array and
        block.?.array.items.len == 2 and block.?.array.items[0] == .integer and block.?.array.items[1] == .integer and
        block.?.array.items[0].integer == 128 and block.?.array.items[1].integer == 128)
    {
        return .fp8_block;
    }
    // A serving checkpoint assembled from EXL3 routed packs over the official FP8 parent: quant_method "exl3"
    // with backbone_format "fp8" keeps the parent's FP8 E4M3 128x128 block-scaled backbone tensors unchanged.
    if (std.mem.eql(u8, str(q, "quant_method") orelse "", "exl3") and
        std.mem.eql(u8, str(q, "backbone_format") orelse "", "fp8"))
    {
        return .fp8_block;
    }
    return .other;
}

/// config.json of a glm_moe_dsa checkpoint; a shape the kernels do not take is refused.
pub fn parse(gpa: std.mem.Allocator, json: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
    defer parsed.deinit();
    const t = parsed.value.object;
    if (!std.mem.eql(u8, str(t, "model_type") orelse "", "glm_moe_dsa")) return error.NotGlmMoeDsa;
    var c: Config = .{};
    c.hidden = @intCast(try int(t, "hidden_size"));
    c.layers = @intCast(try int(t, "num_hidden_layers"));
    c.vocab = @intCast(try int(t, "vocab_size"));
    c.eps = @floatCast(try float(t, "rms_norm_eps"));
    c.dense_layers = @intCast(try int(t, "first_k_dense_replace"));
    c.dense_inter = @intCast(try int(t, "intermediate_size"));
    c.experts = @intCast(try int(t, "n_routed_experts"));
    c.topk = @intCast(try int(t, "num_experts_per_tok"));
    c.moe_inter = @intCast(try int(t, "moe_intermediate_size"));
    c.routed_scale = @floatCast(try float(t, "routed_scaling_factor"));
    c.heads = @intCast(try int(t, "num_attention_heads"));
    c.nope = @intCast(try int(t, "qk_nope_head_dim"));
    c.rope = @intCast(try int(t, "qk_rope_head_dim"));
    c.v_dim = @intCast(try int(t, "v_head_dim"));
    c.q_lora = @intCast(try int(t, "q_lora_rank"));
    c.kv_lora = @intCast(try int(t, "kv_lora_rank"));
    c.i_heads = @intCast(try int(t, "index_n_heads"));
    c.i_dim = @intCast(try int(t, "index_head_dim"));
    c.i_topk = @intCast(try int(t, "index_topk"));
    c.configured_mtp = @intCast(try int(t, "num_nextn_predict_layers"));
    c.mtp = c.configured_mtp;
    try validateLayerCapacity(c.layers, c.mtp);
    const rp = t.get("rope_parameters") orelse return error.BadConfig;
    if (rp != .object or !std.mem.eql(u8, str(rp.object, "rope_type") orelse "", "default")) return error.UnsupportedRope;
    c.rope_theta = try float(rp.object, "rope_theta");
    const types = t.get("indexer_types") orelse return error.BadConfig;
    if (types != .array or types.array.items.len != c.layers) return error.BadConfig;
    for (types.array.items, 0..) |v, i| {
        if (v != .string) return error.BadConfig;
        if (std.mem.eql(u8, v.string, "full")) c.full.set(i) else if (!std.mem.eql(u8, v.string, "shared")) return error.BadConfig;
    }
    if (c.mtp > 0) c.full.set(c.layers);
    if (!c.full.isSet(0)) return error.BadConfig;
    c.n_shared_experts = @intCast(try int(t, "n_shared_experts"));
    const norm = t.get("norm_topk_prob") orelse return error.BadConfig;
    if (norm != .bool) return error.BadConfig;
    c.norm_topk_prob = norm.bool;
    const quant_value = t.get("quantization_config");
    const quant = if (quant_value) |q| (if (q == .object) q.object else null) else null;
    c.backbone = if (quant_value != null and quant == null) .other else backboneFormat(quant);
    if (str(t, "routed_codebook")) |v| {
        if (std.mem.eql(u8, v, "mcg")) c.routed_codebook = .mcg else if (std.mem.eql(u8, v, "mcg_e4m3")) c.routed_codebook = .mcg_e4m3 else if (std.mem.eql(u8, v, "mul1")) c.routed_codebook = .mul1 else return error.UnsupportedCodebook;
    } else if (quant) |q| if (str(q, "routed_codebook")) |v| {
        if (std.mem.eql(u8, v, "mcg")) c.routed_codebook = .mcg else if (std.mem.eql(u8, v, "mcg_e4m3")) c.routed_codebook = .mcg_e4m3 else if (std.mem.eql(u8, v, "mul1")) c.routed_codebook = .mul1 else return error.UnsupportedCodebook;
    };
    if (t.get("eos_token_id")) |v| if (v == .array and v.array.items.len <= 3) {
        c.eos_n = @intCast(v.array.items.len);
        for (v.array.items, 0..) |e, i| c.eos[i] = @intCast(e.integer);
    };
    const plain = (if (t.get("n_group")) |v| v.integer else 1) == 1 and (if (t.get("topk_group")) |v| v.integer else 1) == 1;
    const sigmoid = std.mem.eql(u8, str(t, "scoring_func") orelse "sigmoid", "sigmoid");
    const shaped = c.hidden == 6144 and c.kv_lora == 512 and c.rope == 64 and c.nope == 192 and c.v_dim == 256 and
        c.q_lora == 2048 and c.i_heads == 32 and c.i_dim == 128 and c.i_topk == 2048 and c.moe_inter == 2048 and
        c.n_shared_experts == 1 and c.norm_topk_prob;
    if (!plain or !sigmoid or !shaped) return error.UnsupportedGlmShape;
    return c;
}

/// Tensor parallel over `ranks` Macs: heads, dense rows, expert halves and vocabulary rows split evenly.
pub fn split(c: *Config, rank: u32, ranks: u32) !void {
    if (ranks == 0 or rank >= ranks) return error.BadSplit;
    if (ranks == 3) {
        if (c.heads != 64 or c.moe_inter != 2048 or c.dense_inter % (128 * ranks) != 0 or c.vocab != 154880)
            return error.BadSplit;
    } else {
        if (c.heads % ranks != 0 or (c.moe_inter / 128) % ranks != 0 or c.dense_inter % (128 * ranks) != 0)
            return error.BadSplit;
    }
    c.rank = rank;
    c.ranks = ranks;
}

test "GLM-5.3's indexer pattern: layers 0-2 full, then every fourth from 6; shared layers reuse the last full one" {
    var c: Config = .{};
    for ([_]u32{ 0, 1, 2, 6, 10, 74 }) |i| c.full.set(i);
    try std.testing.expectEqual(@as(u32, 2), c.indexFrom(5));
    try std.testing.expectEqual(@as(u32, 6), c.indexFrom(9));
    try std.testing.expectEqual(@as(u32, 74), c.indexFrom(77));
    try split(&c, 1, 2);
    try std.testing.expectEqual(@as(u32, 32), c.head0());
    try std.testing.expectEqual(@as(u32, 8), c.half0());
    try std.testing.expectEqual([2]u32{ 77440, 154880 }, c.vocabPart());
    try split(&c, 0, 3);
    try std.testing.expectEqual(@as(u32, 22), c.localHeads());
    try std.testing.expectEqual(@as(u32, 0), c.head0());
    try std.testing.expectEqual(@as(u32, 6), c.localHalves());
    try std.testing.expectEqual(@as(u32, 0), c.half0());
    try std.testing.expectEqual([2]u32{ 0, 51712 }, c.vocabPart());
    try split(&c, 1, 3);
    try std.testing.expectEqual(@as(u32, 22), c.localHeads());
    try std.testing.expectEqual(@as(u32, 22), c.head0());
    try std.testing.expectEqual(@as(u32, 5), c.localHalves());
    try std.testing.expectEqual(@as(u32, 6), c.half0());
    try std.testing.expectEqual([2]u32{ 51712, 103296 }, c.vocabPart());
    try split(&c, 2, 3);
    try std.testing.expectEqual(@as(u32, 20), c.localHeads());
    try std.testing.expectEqual(@as(u32, 44), c.head0());
    try std.testing.expectEqual(@as(u32, 5), c.localHalves());
    try std.testing.expectEqual(@as(u32, 11), c.half0());
    try std.testing.expectEqual([2]u32{ 103296, 154880 }, c.vocabPart());
    try std.testing.expectError(error.BadSplit, split(&c, 0, 5));
}

test "config declarations are fully analyzed and unknown quantization is labeled other" {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(Config);
    std.testing.refAllDecls(ServingOptions);
    try validateLayerCapacity(78, 1);
    try validateLayerCapacity(max_layers, 0);
    try std.testing.expectError(error.BadConfig, validateLayerCapacity(max_layers, 1));
    try std.testing.expectError(error.BadConfig, validateLayerCapacity(max_layers + 1, 0));

    var quant: std.json.ObjectMap = .empty;
    defer quant.deinit(std.testing.allocator);
    try quant.put(std.testing.allocator, "quant_method", .{ .string = "unknown" });
    var cfg: Config = .{ .backbone = backboneFormat(quant) };
    try std.testing.expectEqual(BackboneFormat.other, cfg.backbone);
    try std.testing.expectError(error.UnsupportedBackbone, Config.admitCuda(&cfg));
}

test "an EXL3 serving checkpoint over the FP8 parent keeps the FP8 block backbone" {
    var quant: std.json.ObjectMap = .empty;
    defer quant.deinit(std.testing.allocator);
    try quant.put(std.testing.allocator, "quant_method", .{ .string = "exl3" });
    try quant.put(std.testing.allocator, "routed_codebook", .{ .string = "mcg_e4m3" });
    try quant.put(std.testing.allocator, "backbone_format", .{ .string = "fp8" });
    try std.testing.expectEqual(BackboneFormat.fp8_block, backboneFormat(quant));
    try quant.put(std.testing.allocator, "backbone_format", .{ .string = "bf16" });
    try std.testing.expectEqual(BackboneFormat.other, backboneFormat(quant));
}
