# GLM-5.3 on three DGX Sparks (TP3)

Branch `glm53-tp3-3x-sparks`. It takes this fork's GLM-5.3 CUDA family (`zig/src/families/glm_dsa/cuda`, written
for two Sparks) to three GB10 nodes, then adds request slots: up to three requests decode together from one shared
KV pool, and a long prompt is read in pieces between the other replies' rounds. Everything here was built and
measured on one three-Spark cluster in November 2026; numbers are from that cluster only.

## Hardware and layout

- Three DGX Spark (GB10, 128 GB unified memory each), cabled as a triangle: each pair of nodes is joined directly by
  a ConnectX-7 QSFP link, RoCE v2.
- Tensor parallel over three ranks (TP3). Every rank keeps the whole KV window (DCP=1): the two-Spark design's
  half-window cache (DCP=2) does not split over three ranks.
- An EXL3 GLM-5.3 checkpoint with an FP8 backbone view, a DSpark drafter (`--drafter`, 4-bit head).

## What this branch changes

In commit order (`git log a73f5fd4..HEAD`); every switch is off unless set, so each step can be compared on its own.

| Change | Switch |
|---|---|
| A first TP3 port (an agent's draft), then made correct: the KV and indexer slots sized for the full window, the DCP=2-only paths refused at startup | (TP3 with `TF_GLM_WORLD=3`) |
| Sampling at T<1 settles the nucleus on the candidates before fetching full logits | (always) |
| NVFP4 backbone (o_proj, q_b, shared and dense MLP), and for the replicated query projections | `TF_GLM53_BACKBONE_FP4=1`, `TF_GLM53_BACKBONE_FP4_QA=1` |
| RoCE gathers over the cabled triangle (two neighbours exchanged in one kernel, probed against NCCL at startup) | `TF_ROCE=1 TF_ROCE_TRIANGLE=1` |
| Routed experts with a span table rotated by expert | `TF_GLM53_MOE_BALANCE=expert` |
| Block-FP8 decode head (lossy; the bf16 rows stay for the drafter's head) | `TF_GLM53_HEAD_FP8=1` |
| The drafter's MLP split three ways | `TF_GLM53_DRAFT_SPLIT3=1` |
| Decode absorb v3 feeds each kv_b weight to all rows | `TF_GLM53_ABSORB_ROWS=1` |
| DFlash2 chance temperature | `TF_GLM53_DRAFT_CHANCE_TEMP` |
| Drafter training capture (rank 0 writes tap rows and residuals) | `TF_GLM53_CAPTURE_DIR` |
| Profiles: decode stages by verify rows, prefill stages | `TF_GLM53_PROFILE=1`, `TF_GLM53_PROFILE_PREFILL=1`, `TF_GLM53_PROFILE_DECODE_SUB=1` |
| Request slots: N requests share decode windows (`--context` becomes a slot's window) | `TF_GLM53_SLOTS=N` |
| A stream alone drafts; shared rounds feed each slot's drafter ring | `TF_GLM53_SLOT_DRAFTS=solo` (default), `all`, `0` |
| Per-slot resident reuse: a prompt takes the free slot that keeps its prefix | on with slots; `TF_GLM53_SLOT_REUSE=0` |
| Slots of different windows | `TF_GLM53_SLOT_SIZES=a,b,..` |
| **One shared KV pool**: a request claims prompt + reply + 64 rows; long claims from the bottom, short ones from the top; idle kept prefixes give way (fewest kept rows first); a long kept prefix waits to grow in place rather than being read again | `TF_GLM53_POOL=P` with `TF_GLM53_SLOTS=N`; `TF_GLM53_POOL_BIG`, `_EVICT`, `_WAIT` |
| **Prompts in pieces** between the other streams' rounds (the lane host's fill queue); pieces end on the engine's absolute chunk grid, so the cache and the first token are the same as one pass | on with slots; `TF_GLM53_FILL_ROWS` (4096), `_ALONE` (8192), `_SHARE` (0.5); `TF_GLM53_FILL_ROWS=0` reads prompts whole |

## Running it

Build as the fork does (ReleaseFast, sm_121):

```sh
zig build native -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121
```

Each rank runs the same binary with `TF_GLM_RANK=r TF_GLM_WORLD=3 TF_GLM53_MK=0` and the fork's NCCL/RoCE settings:

```sh
tensorfold-native serve <model-dir> --host 0.0.0.0 --port 8100 --name glm-5.3 \
  --drafter <drafter-dir> --drafter-bits 4 --kv-cache bf16 --context 131072 --metrics-open --no-update-check
```

The switches of the everyday server on that cluster (three slots, one 139,264-position pool, pieces on):

```
TENSORFOLD_MEMORY_RESERVE_GIB=9 TF_GLM53_SERVE_MARGIN_MIB=1024 CONTEXT=131072
TF_GLM53_BACKBONE_FP4=1 TF_GLM53_BACKBONE_FP4_QA=1 TF_GLM53_PRE_FUSE=0 TF_GLM53_PREP_FUSE=0
TF_GLM53_COPYD=1 TF_GLM53_COPY_ROWS=8 TF_GLM53_DRAFT_CHANCES=1 TF_GLM53_LIVE_COSTS=1
TF_ROCE=1 TF_ROCE_TRIANGLE=1 TF_GLM53_MOE_BALANCE=expert
TF_GLM53_ATTN_PIPE=1 TF_GLM53_PIECE_PIPE=1 TF_GLM53_HEAD_FP8=1 TF_GLM53_DRAFT_SPLIT3=1 TF_GLM53_ABSORB_ROWS=1
TF_GLM53_SLOTS=3 TF_GLM53_POOL=139264
```

Leave out the last line for one stream with the whole 131,072-token window and no pool.

## What it measured

Greedy unless noted; one cluster, November 2026.

| Measure | Result |
|---|---|
| One stream, single row per round | 50.0 ms; each extra verify row +14.5 ms |
| Short reply alone (183 tokens) | 22.2–22.7 tok/s |
| Two / three replies at once | 30.5–31.0 / 33.6 tok/s in total |
| Cold prefill | 32K tokens 59.5 s to the first token; 68K about 141 s |
| Follow-up on a kept 68K prefix | first token 1.5 s |
| Three replies at once vs each alone (temperature 0) | the same text, 3/3 |
| A reply running when a 34K prompt arrives | 42.6 s (20.4 alone); 85.1 s when prompts were read whole |
| A short request arriving 5 s into a 68K read | first token 9.8 s; 139.8 s when prompts were read whole |
| A 68K prompt read in pieces vs whole | the same output tokens (the server's token sha matches); about 1% slower alone |

## Limits

- One request holds at most 131,072 tokens: the indexer's exact top-k kernel keeps 135,168 keys per rank (8 CTAs ×
  512 threads × 33 entries), sized for the two-Spark half window. A multi-block selection would lift it, but memory
  binds first: bf16 KV is about 95 KB per position per rank.
- The pool is sized to the tightest node's free memory. On this cluster 139,264 positions left the two other nodes
  6.7–7.1 GB free.
- Pieces alternate with decode rounds, so a long read and the running replies share the GPU (half each by default).
  A family round planner (prompt pieces and decode rows in one forward) would remove that trade-off.

## Credits

The GLM-5.3 CUDA family, its two-Spark design and serving are Kristian Aaron's fork of TensorFold (see `NOTICE`);
TensorFold is Ash Hart's. The shared pool's semantics follow ciprianveg/gb10-tensorfold (credited there to Mia's
design); no code was copied. The first TP3 draft came from an Antigravity agent session.
