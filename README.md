# kv-cache-envelope

**Given one 40 GB A100 and a context length L, how many sequences can you actually serve
at once?**

This repository answers that question three ways and reconciles them: an analytical
calculator that turns any HuggingFace config into predicted KV cache bytes, a measured
sweep that pushes a real vLLM engine to four distinct failure boundaries, and a
term-by-term decomposition of why the two differ.

The short version: the per-token arithmetic is *exact*, the whiteboard capacity estimate
is wrong by 4%, and getting `num_key_value_heads` wrong instead of
`num_attention_heads` would have made it wrong by 600%.

---

## The headline number

Qwen2.5-7B-Instruct, bf16, TP=1, `gpu_memory_utilization=0.90`, on one
A100-SXM4-40GB. Measured KV pool: **390,720 tokens = 20.87 GiB = 24,420 blocks of 16**.
N=3 non-warmup repetitions, zero spread.

| context length L | concurrent sequences | source |
|---:|---:|---|
| 256 | 1,526 | measured |
| 512 | 763 | measured |
| 1,024 | 381 | measured |
| 2,048 | 190 | measured |
| 4,096 | 95 | measured |
| 8,192 | 47 | measured |
| 16,384 | 23 | measured |
| 32,768 | 11 | measured |

Each row is a separate engine launch at that `max_model_len`, not a division of one
measurement: the pool came out at **exactly 390,720 tokens at every single one of the
eight lengths**. `max_model_len` is a per-request limit, not a memory budget. It changes
how the pool is divided, never how big it is.

With `--kv-cache-dtype fp8` every figure doubles, measured at a token ratio of 2.0000 to
2.0003 across four block sizes. On A100
(sm80) there are no native fp8 tensor cores, so this is purely a storage win and the
attention kernel dequantises on read.

**Regenerate every figure and table in this repository with one command:**

```bash
sbatch slurm/run_analysis.sh
```

---

## Hardware and software this was measured on

| | |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB, **39.67 GiB usable**, full NVLink `NV4` mesh |
| node | MeluXina `gpu` partition at ICHEC, 4x A100, 128 cores, 480 GB RAM, diskless |
| driver / CUDA | 595.91.07 / 13.2 |
| engine | vLLM 0.27.1, image digest `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| models | Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28`, Qwen2.5-32B-Instruct `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` |
| account | `p201466`, `gpu` partition only |

Every number in this README traces to a committed artifact under `results/raw/`.
`analysis/verify_identifiers.py` fails the build if any identifier in prose does not
appear in the lockfile or an artifact.

---

## 1. The analytical calculator

```
kv_bytes = 2 * num_hidden_layers * num_key_value_heads * head_dim * dtype_bytes * seq_len * batch
```

The factor 2 is K and V. Everything hard is in getting the four structural terms out of a
config that may not state them.

```bash
python -m kv_cache_envelope.cli predict --model Qwen/Qwen2.5-7B-Instruct \
    --seq-len 8192 --budget-gib 20.87
```

**`num_key_value_heads` is the trap.** Under grouped-query attention the query projection
has `num_attention_heads` heads but only K and V are cached, and there are
`num_key_value_heads` of those. Using the query head count overstates the cache by exactly
the GQA ratio: **7x on Qwen2.5-7B** (28 query heads to 4 KV heads), 5x on Qwen2.5-32B.

**The MHA fallback is a second trap, and Falcon is where it bites.** HuggingFace documents
an absent `num_key_value_heads` as meaning MHA. `tiiuae/falcon-7b` omits the field *and*
declares `multi_query: true` with 71 attention heads and one KV head. Taking the documented
fallback there overstates its cache **71x**. The calculator applies Falcon's own derivation
rules before the fallback, and logs every use of the fallback so it can be audited.

Validated against 11 model configs committed under `configs/model_configs/` with their
revisions: 10 the calculator answers for, spanning **6 distinct GQA ratios and three
attention kinds**, plus one it refuses.

| attention | models | GQA ratio |
|---|---|---|
| MHA | Phi-3-mini-4k, OLMo-2-7B, SmolLM2-1.7B | 1x |
| GQA | Mistral-7B-v0.2, Granite-3.1-8B | 4x |
| GQA | Qwen2.5-32B | 5x |
| GQA | Qwen2.5-7B, Qwen2.5-0.5B | 7x |
| GQA | starcoder2-7b | 9x |
| MQA | falcon-7b | 71x |

Head dimension is not uniform either: Phi-3-mini uses 96 and SmolLM2 and falcon-7b use 64,
so a calculator that hard-codes 128 is wrong on three of the ten.

`deepseek-ai/DeepSeek-V2-Lite` is also committed, and the calculator **refuses** it:
multi-head latent attention caches a compressed latent per token, not per-head K and V, so
the formula does not describe it. Refusing beats returning a plausible wrong number.

Full table: `results/tables/calculator_validation.md`. 46 unit tests in
`tests/test_calculator.py`.

**Tensor parallelism divides the per-GPU cache**, because KV heads are sharded across
ranks. Measured: Qwen2.5-32B at TP=2 reported a 4.04 GiB pool for 33,072 tokens, exactly
half of what the unsharded formula predicts. With fewer KV heads than ranks vLLM replicates
instead of splitting, so the per-GPU cost stops falling and the aggregate starts rising:
Qwen2.5-7B at TP=8 holds the same per-GPU cache as at TP=4 on twice the hardware.

---

## 2. The measured envelope, and four real boundaries

162 committed artifacts across 8 sweeps. Failed launches are artifacts too: the failure
point is the capacity limit the project exists to find.

Sweeping `gpu_memory_utilization` at `max_model_len=32768` fails at **both** ends, in four
distinguishable ways:

| gmu | outcome | what the engine said |
|---|---|---|
| 0.38 | `oom_no_cache_blocks` | "No available memory for the cache blocks" |
| **0.40** | `kv_cache_too_small` | needs 1.75 GiB for one sequence, has 1.03; engine's own estimate: max model length **19,280** |
| 0.42 - 0.95 | `ok` | pool from 34,144 to 427,872 tokens, exactly linear in gmu |
| 0.97 | `oom_profiling` | CUDA OOM during graph capture: wanted 150 MiB, 100.94 MiB free |
| **0.99** | `budget_exceeds_free` | requested 39.28 GiB, only **39.26 GiB** was free. Missed by 0.02 GiB |

Separately, and not a memory boundary at all: `block_size=8` is rejected outright by the
attention backend on sm80 (`backend_unsupported`). It is plotted as its own category,
because filing it with the OOMs would be a lie about what limits capacity.

The pool is **exactly linear** in `gpu_memory_utilization`, and the slope of that line
recovers the usable device memory: a least-squares fit over the nine working settings gives
**39.67 GiB of pool per unit gmu**, against a card that measures **39.67 GiB** usable, 0.01%
apart. The fit crosses zero at **gmu 0.374**: that fraction of the card is consumed by fixed
overhead before any cache exists at all, which is why the low-end boundary sits where it
does.

---

## 3. Predicted vs measured, and the residual

**The per-token formula is exact.** The engine reports its pool twice, independently: a
token count and a size in GiB. Multiplying the first by the analytical per-token size
reproduces the second across **91 successful launches** with a **maximum relative error of
0.192%**, which is the rounding in the engine's own two-decimal GiB print. That holds for
fp8 and for TP=2 as well.

So the residual is not in the KV arithmetic. It is entirely in **how big the budget turns
out to be**, and that decomposes into five named terms. Four of the five are read out of
the engine's own startup accounting rather than attributed by us:

| term | GiB | note |
|---|---:|---|
| card size on the datasheet | 40.00 | |
| usable device memory | **39.67** | the datasheet overstates by 0.33 |
| budget at `gmu=0.90` | 35.71 | `0.90 x 39.67` |
| weights + non-torch | 14.58 | the loader reports 14.29 for weights alone |
| peak activation (profile run) | 0.26 | measured by a real forward pass |
| CUDA graph capture | 0.53 | spent *after* the pool is sized |
| **KV pool** | **20.87** | what is left |

Reading it:

- **The whiteboard estimate** `0.90 x 40 - 14.29 = 21.7 GiB` is **4.0% high** at
  `gmu=0.90`, and its error grows as the pool shrinks: across the measured range
  `gmu` 0.42 to 0.95 it overstates the pool by **3.8% to 37.9%**. Starting from measured
  usable memory and subtracting the terms the engine reports gives
  `39.67 x 0.90 - 14.58 - 0.26 = 20.83 GiB`, which is **0.2% low**. That is a capacity
  model you can plan a fleet with.
- **`gpu_memory_utilization` is not a cap on total usage.** The terms sum to 36.24 GiB
  against a 35.71 GiB budget: graph capture is spent after the pool is sized, so the real
  high-water mark sits **0.53 GiB above** the limit you asked for, at every one of the nine
  utilisation settings measured. That is why the failure at `gmu=0.97` is an OOM during
  capture rather than a clean refusal.
- **Block-granularity rounding is negligible for the pool and real for the sequence.**
  The pool loses 64 tokens of 390,720 (0.016%) at `block_size=128`. A single 1-token
  sequence, meanwhile, occupies a whole 128-token block. Those are different costs and
  conflating them is easy.
- **Allocator fragmentation is *not* a major term here, and this repository does not claim
  it is.** The pool is one contiguous allocation divided into fixed-size blocks, so there
  is no fragmentation within it. The only fragmentation visible anywhere in the data is
  58 MiB reported in a single OOM message, 0.15% of the card.

Figure: `results/figures/predicted_vs_measured.png`. Table:
`results/tables/predicted_vs_measured.csv`.

### The first engine you start gets 8.5% less cache

Same command line, same node. The first launch after a node is allocated gets
**357,520 tokens**; every launch after it gets **390,720**. The cold torch.compile cache
means compilation runs during the profiling forward pass, inflating both the non-torch
allocation (15.58 vs 14.58 GiB) and the measured activation peak (1.04 vs 0.26 GiB).

Operationally: the first replica after a cold start has 8.5% less KV capacity than its
identical siblings, and nothing in its logs says so unless you compare. It is also enough
to change *which* boundary a marginal configuration hits.

---

## 4. PagedAttention under deliberate oversubscription

Qwen2.5-7B at `max_model_len=2048`, `max_num_seqs=256`, every request 512 prompt tokens
and 1,536 generated so it holds a full context by the end. Short prompts are the point:
they sail through admission and only become a capacity problem as they grow. N=3, and the
preemption counts were **identical across all three repetitions** at every point.

| gmu | pool (tokens) | predicted capacity | preemption onset | preemptions at c=256 | wall at c=256 |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 167,856 | 82 | c=96 | 291 | 110.8 s |
| 0.70 | 242,144 | 118 | c=128 | 138 | 90.0 s |
| 0.80 | 316,432 | 155 | c=192 | 101 | 83.9 s |
| 0.90 | 390,720 | 191 | c=256 | 64 | 79.2 s |

**The analytical capacity brackets the preemption onset in all four cases.** At every
utilisation the highest preemption-free concurrency is below `pool / max_model_len` and
the lowest preempting concurrency is above it. That is a sharp falsifiable prediction,
confirmed four times, and it is what makes the calculator an operational tool rather than
an arithmetic exercise.

**Nothing ever failed.** Zero request errors at up to 3.1x oversubscription. PagedAttention
converts memory exhaustion into latency, never into an error: KV usage pins at 1.000 and
the scheduler preempts. The cost is wall-clock, and it tracks the preemption count
monotonically: 40% longer to do identical work at gmu 0.60 than at 0.90.

**Preemption is recompute, and there is no swap.** Reading the scheduler at the pinned
version, `_preempt_request` frees every block and sets `num_computed_tokens = 0`; the
sequence restarts from its first token. There is **no occurrence of "swap" anywhere** in
vLLM V1's scheduler, block pool or cache manager at 0.27.1: the V0 CPU-swap path is gone.
So the recompute-vs-swap ratio is not a measurable quantity at this version, and every
preemption pays the full prefill cost again. The knob that prevents that is
`--max-num-seqs`, not a bigger `--swap-space`.

Figure: `results/figures/paged_attention.png`. The `gpu_memory_utilization` and
`block_size` panels also draw on 2,410 records mined from the sibling project's already
committed sweeps (`results/mined_p1/`), rather than re-measuring what had already been
paid for in node-hours.

---

## Capacity planning, put together

To size a deployment for context length L on a 40 GB A100:

1. **Compute the per-token cost.** `2 * layers * num_key_value_heads * head_dim *
   dtype_bytes`, divided by the tensor-parallel degree. For Qwen2.5-7B in bf16 at TP=1
   that is 56 KiB per token. Do not use `num_attention_heads`.
2. **Estimate the pool.** `usable_memory x gmu - weights - activation_peak`. Use measured
   usable memory (39.67 GiB, not 40), and expect the engine to exceed the budget by the
   CUDA graph capture size.
3. **Divide.** `pool_bytes / (per_token x L)`, floored. A partially resident sequence is a
   preemption waiting to happen, not a servable sequence.
4. **Set `--max-num-seqs` at or below that number.** Above it the engine does not fail, it
   preempts, and each preemption throws away all the prefill that sequence had done.
5. **Warm the node first**, or budget 8.5% less capacity for the first replica.

Worked table for the models measured here: `results/tables/capacity_planning.md`.

---

## Layout

```
src/kv_cache_envelope/
  calculator.py     HF config -> predicted KV bytes. GQA/MQA/MLA, TP sharding, block rounding
  hf_config.py      config bundles with their revisions, read never typed
  engine.py         launch vLLM in the pinned container, scrape /metrics, classify failures
  occupancy.py      drive a live engine to a target occupancy, sample /metrics throughout
  sweep.py          YAML -> engine launches -> artifacts, failures included
  provenance.py     the only writer into results/raw/; refuses incomplete provenance
configs/            sweep definitions, plus 11 committed model config bundles
slurm/              run_prep.sh, run_sweep.sh, run_analysis.sh
analysis/           mine_p1.py and the plot/table scripts; read results/, never write them
results/raw/        162 artifacts, one per engine launch, append-only, chmod 440
results/mined_p1/   2,410 records derived from the sibling project, kept separate on purpose
docs/notes.md       source reading, findings, and an honest definition-of-done audit
docs/vllm_source/   vLLM 0.27.1 sources as captured from inside the pinned image
```

## What is not measured

Stated explicitly, because an unexplained gap looks like a choice:

- **fp8 output quality.** The concurrency half of the fp8 tradeoff is measured exactly
  (2.0000x). No quality proxy was run, so no claim is made about its effect on outputs.
- **`max_num_seqs` as a swept axis.** Its effect on pool size is suspected but confounded
  with cold-start in the data collected, so it is not reported.
- **Models beyond the Qwen2.5 family, on a running engine.** Nine of the eleven validated
  configs are validated against config files, not against launched engines.
- **GPUs other than A100-SXM4-40GB.** The `block_size=8` rejection in particular is an
  attention-backend property and may not transfer.
