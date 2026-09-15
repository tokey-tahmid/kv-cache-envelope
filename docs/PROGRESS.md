# PROGRESS — kv-cache-envelope

Status as of 2026-09-15. Written for a human picking this up, and honest about what is
not measured.

## Where it stands

**Substantially complete against the PROJECT 2 definition of done**, with two stated gaps
(fp8 output quality, and `max_num_seqs` as a swept axis). Full audit table at the end of
`docs/notes.md`.

- 164+ raw artifacts across 10 sweeps, every one from a real engine launch on a real A100.
- Four distinct failure boundaries measured at both ends of the memory axis, N=3 each.
- The residual between the whiteboard estimate and the measured pool decomposed term by
  term, with four of the five terms read from the engine's own startup accounting.
- Calculator validated against 11 committed model configs, 10 answered spanning 6 GQA
  ratios and 3 attention kinds plus 1 refused, and against running engines on two of them.
- vLLM's block manager and scheduler read inside the pinned container, cited by file and
  function in `docs/notes.md`.
- 6.97 node-hours consumed of the ~15 allocated (`docs/budget.md`, every row from `sacct`).

## What was measured

All on A100-SXM4-40GB (**39.67 GiB usable**, not 40), driver 595.91.07, CUDA 13.2,
vLLM 0.27.1 at the digest in `provision.lock.json`, Qwen2.5-7B-Instruct at
`a09a35458c702b33eeacc393d103063234e8bc28` unless stated. N=3 non-warmup repetitions.

### The capacity envelope

At `gpu_memory_utilization=0.90`, TP=1: **390,720 tokens = 20.87 GiB = 24,420 blocks of
16**, and that pool is *identical to the token* at all eight log-spaced `max_model_len`
values from 256 to 32,768. Concurrency is therefore exactly `390,720 / L`: 1,526 sequences
at 256 tokens of context down to **11 at 32,768**.

`block_size` does not change the pool either (390,720 at 16/32/64, 390,656 at 128, a
0.016% spread), it only changes how it is divided. fp8 doubles it at exactly 2.0000x.

### The OOM boundary, measured at both ends

Sweeping `gpu_memory_utilization` at `max_model_len=32768` produces four distinguishable
failures, none of them extrapolated:

| gmu | outcome |
|---|---|
| 0.38 | `oom_no_cache_blocks` — profiling left nothing at all for the cache |
| **0.40** | `kv_cache_too_small` — needs 1.75 GiB for one sequence, has 1.03; the engine's own estimate of the longest servable model length is **19,280** |
| 0.42 - 0.95 | works; pool rises linearly from 34,144 to 427,872 tokens |
| 0.97 | `oom_profiling` — CUDA OOM during graph capture, wanted 150 MiB with 100.94 MiB free |
| **0.99** | `budget_exceeds_free` — asked for 39.28 GiB, 39.26 GiB was free. **Missed by 0.02 GiB** |

Plus one boundary that is not a memory boundary at all: `block_size=8` is rejected by the
attention backend on sm80 (`backend_unsupported`), and is reported as its own category
rather than filed with the OOMs.

### Preemption under deliberate oversubscription

`max_model_len=2048`, `max_num_seqs=256`, 512-token prompts generating 1,536 tokens.
Preemption counts were **identical across all three repetitions** at every point.

| gmu | pool (tokens) | predicted capacity | preemption onset | preemptions at c=256 | wall at c=256 |
|---:|---:|---:|---:|---:|---:|
| 0.60 | 167,856 | 82 | c=96 | 291 | 110.8 s |
| 0.70 | 242,144 | 118 | c=128 | 138 | 90.0 s |
| 0.80 | 316,432 | 155 | c=192 | 101 | 83.9 s |
| 0.90 | 390,720 | 191 | c=256 | 64 | 79.2 s |

**The analytically predicted capacity brackets the preemption onset in all four cases.**
Zero request failures anywhere, at up to 3.1x oversubscription: PagedAttention converts
memory exhaustion into latency, not into errors. The cost tracks the preemption count
monotonically, 40% more wall-clock for identical work at gmu 0.60 than at 0.90.

## The residual, and the explanation

**The per-token formula is exact.** The engine reports its pool twice independently, as a
token count and as GiB. `tokens x (2 x layers x kv_heads x head_dim x dtype_bytes)`
reproduces the GiB figure across **91 successful launches with a maximum relative error of
0.192%**, which is the rounding in the engine's own two-decimal print. That includes fp8 and
TP=2. Nothing in the KV arithmetic is approximate.

The residual is entirely in **how large the budget turns out to be**, and it decomposes
into five named terms:

| term | GiB |
|---|---:|
| datasheet card size | 40.00 |
| usable device memory | **39.67** |
| budget at gmu 0.90 | 35.71 |
| weights + non-torch | 14.58 (loader reports 14.29 for weights alone) |
| peak activation | 0.26 |
| CUDA graph capture | 0.53 |
| **KV pool** | **20.87** |

- Whiteboard estimate `0.90 x 40 - 14.29 = 21.7 GiB` is **4.0% high** at gmu 0.90, and
  3.8% to 37.9% high across the measured gmu range.
- Corrected estimate `39.67 x 0.90 - 14.58 - 0.26 = 20.83 GiB` is **0.2% low**.
- The `num_attention_heads` mistake would have been **600%** wrong.
- The terms sum to 36.24 GiB against a 35.71 GiB budget: **graph capture is spent after
  the pool is sized**, so peak usage exceeds the limit you asked for. That is why gmu 0.97
  OOMs during capture rather than being refused at startup.
- **Block rounding is negligible for the pool** (64 tokens of 390,720) and real per
  sequence (a 1-token sequence occupies a whole block).
- **Allocator fragmentation is explicitly not claimed as a major term.** The pool is a
  single contiguous allocation divided into fixed-size blocks. The only fragmentation
  anywhere in the data is 58 MiB reported inside one OOM message, 0.15% of the card.
  Saying "fragmentation" would have been a plausible-sounding guess, and the data does not
  support it.

## What failed, and what that bought

Four things went wrong, and three of them produced findings.

1. **The failure classifier guessed vLLM's error wordings and matched almost none of
   them.** Three genuinely different boundaries were filed as generic `launch_failed`
   whose recorded error was the API server's outer traceback, naming no cause. Recovered
   only because every artifact stores `engine_log_path` and the logs were preserved, so
   the root causes were still on disk. Patterns are now copied from those logs;
   `configs/boundary_recheck*.yaml` re-measured the three points.

2. **The calculator assumed the per-GPU cache equals the model's cache.** Qwen2.5-32B at
   TP=2 reported 4.04 GiB for 33,072 tokens, exactly half the unsharded prediction.
   Tensor parallelism shards KV heads across ranks. Now modelled in `kv_heads_per_rank`,
   including the case with fewer KV heads than ranks, where vLLM replicates instead of
   splitting so the per-GPU cost stops falling and the aggregate starts rising.

3. **The probe returned from `stop()` before the driver had reclaimed device memory.**
   The first 32B sweep had its first launch succeed and every repetition after it fail
   with `budget_exceeds_free`, reporting 2.78 GiB free on a card that should have been
   empty. Invisible at 7B, where 14.58 GiB of weights leaves enough slack to absorb a late
   reclaim. **Caveat on the fix:** the re-run succeeded, but the recorded wait was only
   0.05 s, so the data does not conclusively establish that the wait is what fixed it
   rather than the race simply not recurring. The wait is recorded in every artifact so a
   future recurrence will be diagnosable.

4. **`pandas.to_markdown` needs `tabulate`, absent from the shared venv.** Replaced with a
   local formatter rather than installing into a venv that the sibling project's 4,573
   artifacts were measured against.

## The finding I did not expect

**The first engine launched on a freshly allocated node gets 8.5% less KV cache than every
launch after it**: 357,520 tokens against 390,720, for an identical command line. The cold
torch.compile cache means compilation runs during the profiling forward pass, inflating
both the non-torch allocation (15.58 vs 14.58 GiB) and the measured activation peak (1.04
vs 0.26 GiB). Apptainer maps the host `/tmp` into the container, so the second launch on a
node reuses the compiled artefacts.

It holds across every job in the sweep and it changes *which boundary a marginal
configuration hits*: at gmu 0.40 the warmup launch failed with "no cache blocks" while the
three measured launches failed with "cache too small for one sequence".

This is the workspace's warmup-exclusion rule earning its place in a *memory* measurement
rather than a timing one. A mean over all repetitions would have reported a capacity that
no configuration actually achieves.

## Not measured, stated plainly

- **fp8 output quality.** The concurrency half of the tradeoff is exact (2.0000x). No
  quality proxy was run. The spec marks this "budget permits"; it was not done.
- **`max_num_seqs` as a swept axis.** Suspected to affect the pool through CUDA graph
  capture, but the only two values in the data (64 and 256) are confounded with cold-start,
  so nothing is claimed.
- **Recompute-vs-swap ratio.** Not measurable at this version: vLLM V1 at 0.27.1 has **no
  swap path at all** (no occurrence of "swap" in the scheduler, block pool or cache
  manager). Every preemption is a full recompute. Reported as a negative result rather
  than as a gap.
- **Nine of the eleven validated models** are validated against config files, not against
  running engines.
- **One GPU generation.** Everything is A100-SXM4-40GB.

## For a human to decide

1. **Is the fp8 quality proxy worth the node-hours?** It is the one explicitly optional
   item in the spec that was not done. Roughly 1-2 node-hours would buy a perplexity or
   greedy-agreement comparison at matched configs, turning "fp8 doubles capacity" into a
   tradeoff curve. Budget is available (6.97 of ~15 used).
2. **Should `max_num_seqs` get its own sweep?** It is the remaining axis where the data
   hints at an effect but cannot support a claim. About 1 node-hour.
3. **The 32B teardown fix is not conclusively validated** (point 3 above). If the 32B
   numbers are going to carry weight in the writeup, one deliberate stress test of the
   teardown path would settle it.
4. **P1's `engine_log_path` join depends on logs living on project storage**, outside git.
   The parsed breakdown is copied into each artifact so the artifacts stand alone, but the
   raw logs will eventually be cleaned up. If they should be kept, they need archiving.

## Reproducing

```bash
sbatch slurm/run_prep.sh       # tests, config bundles, vLLM source from the pinned image
sbatch slurm/run_sweep.sh configs/envelope_len.yaml    # and the other configs/
sbatch slurm/run_analysis.sh   # every figure and table, one command
```
