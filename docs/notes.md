# Notes: source reading, findings, surprises

Running log. Every entry states what was actually observed and the job ID or file
that observed it, so anything here can be re-checked rather than trusted.

All source citations are read **inside the pinned container**, not from GitHub:

```
image  vllm/vllm-openai   digest sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967
vllm   0.27.1             package /usr/local/lib/python3.12/dist-packages/vllm
```

captured by `slurm/run_prep.sh` (job 5211409) into `docs/vllm_source/`, with
`docs/vllm_source/VERSION.txt` recording the version and path the interpreter
itself reported. Line numbers below are line numbers in those captured files.

---

## 2026-09-15 — Source reading: how vLLM V1 actually manages KV blocks

### The three layers, and which one owns what

`vllm/v1/core/` splits the job across three objects, and knowing which one
answers which question is most of understanding the behaviour:

| file | object | owns |
|---|---|---|
| `block_pool.py` | `BlockPool` | the flat array of blocks, the free list, the prefix-cache hash map, reference counts |
| `kv_cache_manager.py` | `KVCacheManager` | per-request block bookkeeping, the admission arithmetic, the watermark |
| `sched/scheduler.py` | `Scheduler` | which requests run this step, and who gets preempted when they cannot all fit |

There is a fourth, `kv_cache_coordinator.py`, which fans the above across *KV
cache groups* so that a model with mixed attention types (full attention plus
sliding window, or attention plus Mamba state) can have different block sizes per
group. For a uniform model like Qwen2.5 there is one group and the coordinator is
a pass-through, but its presence is why `KVCacheManager` talks about
`kv_cache_group_ids` everywhere.

### Allocation: one flat array, a doubly linked free list, and refcounts

`BlockPool.__init__` (`block_pool.py:162`) builds the whole cache up front:

```python
self.blocks: list[KVCacheBlock] = [KVCacheBlock(idx) for idx in range(num_gpu_blocks)]
self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
...
self.null_block = self.free_block_queue.popleft()
self.null_block.is_null = True
```

Three things fall out of those five lines:

* **`num_gpu_blocks` is fixed at startup and never changes.** Everything this
  project measures about the envelope is decided before the first request
  arrives. That is why the measurement harness is built around engine *launches*
  rather than around requests.
* **Block 0 is stolen as a null block** and is never allocatable. It is the
  placeholder written into a request's block table for positions that are not
  actually resident (outside a sliding window, for instance). So the usable pool
  is `num_gpu_blocks - 1`, which `get_usage` (`block_pool.py:807`) corrects for
  explicitly: `total_gpu_blocks = self.num_gpu_blocks - 1`. A capacity
  calculation that uses `num_gpu_blocks` is off by one block, which is invisible
  at 24,420 blocks and would not be at 100.
* **The free list is ordered, not a set.** `FreeKVCacheBlockQueue` is a doubly
  linked list maintained in eviction order, which is what makes eviction O(1)
  and makes the *order* blocks are freed in semantically meaningful (see below).

`BlockPool.get_new_blocks` (`block_pool.py:647`) is the allocation path and it is
deliberately dumb:

```python
if num_blocks > self.get_num_free_blocks():
    raise ValueError(f"Cannot get {num_blocks} free blocks from the pool")
ret = self.free_block_queue.popleft_n(num_blocks)
...
    self._maybe_evict_cached_block(block)
    assert block.ref_cnt == 0
    block.ref_cnt += 1
```

It does not consult the prefix cache and it does not decide whether allocating is
a good idea. Both of those are the caller's job. What it does do is
**evict-on-allocate**: a block taken from the free list may still be sitting in
the prefix-cache hash map, and `_maybe_evict_cached_block`
(`block_pool.py:679`) removes its hash entries at the moment it is handed out.
Eviction is therefore lazy and free until the block is actually needed, which is
why a cached block can be reused by a later request right up until the moment the
pool runs dry.

### Freeing: the order is the eviction policy

`BlockPool.free_blocks` (`block_pool.py:719`) is where the cache's replacement
policy actually lives, and it is not a plain LRU:

```python
for block in ordered_blocks:
    block.ref_cnt -= 1
    if block.ref_cnt == 0 and not block.is_null:
        if block.block_hash is None and self.enable_caching:
            blocks_without_hash.append(block)
        else:
            blocks_with_hash.append(block)
self.free_block_queue.prepend_n(blocks_without_hash)
self.free_block_queue.append_n(blocks_with_hash)
```

Blocks that carry a hash (full blocks, eligible for a future prefix-cache hit)
go to the **tail** of the free queue; blocks with no hash (partial tail blocks
that can never match anything) go to the **head**. Since allocation pops from the
head, the useless blocks are spent first and the cacheable ones survive longest.
So the policy is "LRU among cacheable blocks, with never-cacheable blocks
prioritised for immediate reuse", and it is implemented entirely by which end of
a linked list a block is pushed onto.

`KVCacheManager.free` (`kv_cache_manager.py:567`) frees a request's blocks *in
reverse order* for the same reason, stated in its own docstring: "We free the
blocks in reverse order so that the tail blocks are evicted first when caching is
enabled." The head of a sequence is the part most likely to be a shared system
prompt, so it should be the last thing evicted.

### Sharing and reference counting: `touch` is the whole mechanism

Prefix-cache sharing is reference counting and nothing more.
`KVCacheManager.get_computed_blocks` (`kv_cache_manager.py:229`) asks the
coordinator for the longest cache hit, and the blocks it returns are already
owned by whoever cached them. `BlockPool.touch` (`block_pool.py:702`) is what
makes a second owner:

```python
for block in blocks:
    # ref_cnt=0 means this block is in the free list (i.e. eviction
    # candidate), so remove it.
    if block.ref_cnt == 0 and not block.is_null:
        self.free_block_queue.remove(block)
    block.ref_cnt += 1
```

A cached block with `ref_cnt == 0` is simultaneously *in the free list* and
*in the prefix-cache hash map*. That dual membership is the design: it is
available for reuse by a matching prefix (a hit) and available for reallocation
to anyone (an eviction), and whichever happens first wins. `touch` resurrects it
out of the free list; `get_new_blocks` steals it and drops its hash. There is no
separate eviction pass and no background reclaim.

One detail with a real performance consequence, from `get_computed_blocks`:

```python
# NOTE: When all tokens hit the cache, we must recompute the last token
# to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
max_cache_hit_length = request.num_tokens - 1
```

A request that is a byte-for-byte repeat of a cached one still recomputes, and
because `allocate_slots` needs `num_computed_tokens` to be block-aligned, it
recomputes **an entire block**, not one token. At `block_size=128` that is up to
128 tokens of prefill that a naive model of prefix caching would predict away.
This is one of the named reasons a measured cache hit rate sits below the
theoretical bound.

### Admission: the watermark, and the full-sequence gate

`KVCacheManager.allocate_slots` (`kv_cache_manager.py:344`) is the admission
decision and it returns `None` rather than raising when the request does not fit.
Two guards matter for the envelope.

**The watermark** (`kv_cache_manager.py:171`, `:463`):

```python
self.watermark_blocks = int(watermark * kv_cache_config.num_blocks)
...
watermark_blocks = 0
# The watermark is applied to waiting/preempted requests only, and only
# when there's at least one request already scheduled.
if has_scheduled_reqs and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
    watermark_blocks = self.watermark_blocks
```

New and preempted requests must leave a reserve; already-running requests may
spend it. That asymmetry is deliberate anti-livelock: a running sequence is
allowed to consume the last blocks in order to finish and release everything it
holds, whereas admitting a *new* sequence into the last blocks would deadlock a
full pool. The watermark defaults to 0.0 in this build, so on these measurements
it is not binding, but the asymmetric rule still is.

**The full-sequence gate** (`kv_cache_manager.py:471`), which is newer and is
what stops chunked prefill from over-admitting:

```python
if full_sequence_must_fit:
    full_num_tokens = min(request.num_tokens, self.max_model_len)
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)
    required_blocks = num_blocks_to_allocate + watermark_blocks
    if required_blocks > self.block_pool.get_num_free_blocks():
        return None
```

Without it, chunked prefill checks only whether the *first chunk* fits, admits
the request, and discovers the problem thousands of tokens later, by which time
the only remedy is preemption. This is exactly the mechanism the occupancy sweep
is designed to provoke, and it is why that sweep uses short prompts and long
generations: a short prompt sails through admission and only becomes a capacity
problem as it grows.

### Cache exhaustion: preemption is recompute, and **there is no swap**

`Scheduler.schedule` (`sched/scheduler.py:439`) runs `allocate_slots` in a
`while True` and treats `None` as "make room" (`sched/scheduler.py:576`):

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break                      # the request can be scheduled
    # The request cannot be scheduled.
    # Preempt the lowest-priority request.
    if self.policy == SchedulingPolicy.PRIORITY:
        preempted_req = max(self.running, key=lambda r: (r.priority, r.arrival_time))
        ...
    else:
        preempted_req = self.running.pop()          # FCFS: the newest running request
    self._preempt_request(preempted_req, ...)
    if preempted_req == request:
        break                      # nothing left to preempt; give up on this request
```

Under the default FCFS policy `self.running.pop()` takes the **last** request in
the running list, which is the most recently admitted one. So the eviction victim
is the newest work, not the largest or the oldest: a long-running sequence that
got in early is protected, and the marginal arrival pays. The loop's exit
condition is the request preempting *itself*, which is how a single request too
large for the entire pool terminates rather than spinning.

`Scheduler._preempt_request` (`sched/scheduler.py:1274`) is the part that
answers the "recompute vs swap" question outright:

```python
self._free_request_blocks(request)
self.encoder_cache_manager.free(request)
...
request.status = RequestStatus.PREEMPTED
request.num_computed_tokens = 0
...
request.num_preemptions += 1
...
self.waiting.prepend_request(request)
```

**Every block is freed and `num_computed_tokens` is reset to zero.** The
sequence will be recomputed from its first token when it is readmitted. Nothing
is copied to host memory.

I then checked whether a swap path exists at all at this version:

```
grep -in "swap" v1_core_sched_scheduler.py  -> no matches
grep -in "swap" v1_core_block_pool.py v1_core_kv_cache_manager.py  -> no matches
grep -rn "swap_space\|num_cpu_blocks" v1_kv_cache_interface.py v1_core_kv_cache_utils.py  -> no matches
```

**vLLM V1 has no swap-to-CPU preemption.** The V0 engine had
`--swap-space` and a CPU block pool; in 0.27.1's V1 core the concept is gone from
the scheduler, the block pool and the cache manager alike. The workspace
`CLAUDE.md` asks for "recompute-vs-swap events under deliberate
oversubscription"; the honest answer is that **at this pinned version the ratio is
not a measurable quantity, because swap is not implemented**. Every preemption is
a recompute. That is a stronger result than a ratio would have been, and it is
also visible in the metrics surface: `vllm:num_preemptions_total` exists and
there is no counterpart counting swapped blocks.

The consequence for capacity planning is direct. Preemption is not a graceful
degradation; it throws away all the work a sequence has done. A deployment that
preempts is paying its prefill cost twice, and `--max-num-seqs` is the knob that
prevents it, not a larger `--swap-space`.

`request.num_preemptions += 1` is also the exact source of the Prometheus
counter used throughout the measurements below, which is worth stating: the
counter is per *preemption event*, not per preempted request, so one unlucky
request can contribute many.

---

## 2026-09-15 — The per-token formula is exact; the budget is where the error is

Two questions get conflated under "predicted vs measured", and separating them is
the whole analysis.

**Question 1: is `2 * layers * kv_heads * head_dim * dtype_bytes` right?**
Yes: maximum relative error **0.192% over 91 successful launches**, which is the rounding
in the engine's own two-decimal GiB print. The engine reports its pool in two
independent ways: a token count (`vllm:cache_config_info`'s
`kv_cache_size_tokens`) and a size in GiB ("Available KV cache memory"). Multiply
the first by the analytical per-token size and the second should come out:

| model | TP | tokens | x per-token | = GiB | engine said | source |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 1 | 390,720 | 57,344 B | 20.869 | 20.87 | `envelope-len`, job 5211478 |
| Qwen2.5-7B (fp8) | 1 | 781,456 | 28,672 B | 20.870 | 20.87 | `envelope-block`, job 5211480 |
| Qwen2.5-32B | 2 | 33,072 | 131,072 B | 4.036 | 4.04 | `envelope-32b`, job 5213931 |

The formula is not approximately right, it is right. **Everything interesting in
this project is therefore about the size of the budget, not about the arithmetic
of the cache.** Which is worth saying plainly, because "predicted vs measured
will not match" was the expectation going in, and the part that does not match is
not the part one would guess.

### The 32B row is where tensor parallelism bit

That third row nearly went in wrong. The unsharded per-token size for Qwen2.5-32B
is `2 * 64 * 8 * 128 * 2 = 262,144` bytes, which puts 33,072 tokens at 8.07 GiB
against an engine that said 4.04. Exactly a factor of two, at TP=2.

**Tensor parallelism shards the KV heads across ranks**, so the per-GPU cache is
the model's cache divided by the TP degree. The calculator now models this in
`kv_heads_per_rank`, including the case that surprises people: when there are
fewer KV heads than ranks, vLLM replicates rather than splitting a head, so the
per-GPU cost stops falling and the *aggregate* cost across the node starts
rising. Qwen2.5-7B has 4 KV heads, so TP=8 stores the same per-GPU cache as TP=4
while using twice the cards for it.

This is a capacity-planning trap with the same shape as the GQA trap: an answer
that is exactly wrong by a clean integer factor looks plausible enough to ship.

---

## 2026-09-15 — Where the budget actually goes, term by term

The estimate an engineer makes on a whiteboard is
`gpu_memory_utilization x card size - weights`. At `gmu=0.90` on a 40 GB A100
with a 7B model that is `0.90 x 40 - 14.29 = 21.7 GiB`. The measured pool is
**20.87 GiB**. The 0.8 GiB gap is not one effect, it is five, and four of them
are read straight out of the engine's own startup accounting rather than
attributed by us. Every number below is the median of N=3 non-warmup launches
from `envelope-len` (job 5211478), which had zero spread.

| term | GiB | where it comes from |
|---|---|---|
| card size on the datasheet | 40.00 | the part number |
| **usable device memory** | **39.67** | `nvidia-smi` total minus ECC and driver reserve |
| free at process start | 39.26 | 0.41 GiB already gone to the CUDA context |
| budget at `gmu=0.90` | 35.71 | `0.90 x 39.67`, the engine's own echo |
| weights + non-torch | 14.58 | loader reports 14.29 for weights alone |
| peak activation (profile run) | 0.26 | measured by a forward pass at `max_num_batched_tokens` |
| CUDA graph capture | 0.53 | taken *after* the pool is sized |
| **KV pool** | **20.87** | what is left |

Reading the residual term by term:

1. **The datasheet lies by 0.33 GiB.** A "40 GB" A100 reports 39.67 GiB usable.
   Starting the estimate from 40 overstates the pool by 0.30 GiB at `gmu=0.90`
   before anything else happens. Small, but it is pure arithmetic error and it is
   free to remove.

2. **"Weights" is not what the weights weigh.** The loader reports 14.29 GiB;
   the profiler charges 14.58 GiB for "weights + non-torch". The extra 0.29 GiB
   is the CUDA context, cuBLAS and NCCL workspaces, and driver-side allocations
   that never appear in `torch.cuda.memory_allocated`. Non-torch memory is real
   memory and the budget pays for it.

3. **Activation peak is small here and is not a constant.** 0.26 GiB, measured
   by an actual profiling forward pass rather than estimated. It scales with
   `max_num_batched_tokens`, which is derived from `max_model_len`, so it is a
   term that a capacity model has to keep rather than fold into a constant.

4. **CUDA graph capture is spent after the pool is already sized, and the budget
   is overshot as a result.** The terms sum to
   `14.58 + 0.26 + 0.53 + 20.87 = 36.24 GiB` against a requested budget of
   **35.71 GiB**: the engine ends up 0.53 GiB over its own stated limit, which is
   exactly the graph capture. vLLM knows, and says so in the same log line that
   reports the breakdown, offering a `--kv-cache-memory` value that would fit.
   For capacity planning the consequence is that `gpu_memory_utilization` is not
   a cap on total usage, it is a cap on the part of usage that is budgeted, and
   the real high-water mark sits above it. That is why the failure at `gmu=0.97`
   is an OOM during capture rather than a refusal at startup.

5. **Block-granularity rounding is real but negligible at these sizes.** The pool
   is sized in bytes and then divided into blocks, and the remainder is dropped:
   at `block_size=128` the pool holds 390,656 tokens against 390,720 at
   `block_size=16`, a loss of 64 tokens, or **0.016%**. Rounding matters per
   *sequence*, not for the pool: a 1-token sequence occupies a whole block, so at
   `block_size=128` it wastes 127 tokens of cache. That is a workload-shape cost,
   not a capacity cost, and conflating the two is easy.

6. **Allocator fragmentation does not appear in the pool at all.** The pool is a
   single contiguous allocation made once at startup, so there is no
   fragmentation *within* it: `BlockPool` hands out fixed-size blocks from a flat
   array. Fragmentation appears only in the PyTorch caching allocator around it,
   and it is visible exactly once in this dataset: the `gmu=0.97` OOM reports
   "58.02 MiB is reserved by PyTorch but unallocated" while failing to find
   150 MiB. At 0.15% of the card it is not a headline term, and **claiming
   fragmentation as a major residual component would not be supported by this
   data.**

### The corrected estimate

Starting from measured usable memory rather than the datasheet, and subtracting
the terms the engine reports:

```
39.67 x 0.90 - 14.58 - 0.26 = 20.83 GiB   vs 20.87 GiB measured   (0.2% low)
```

The remaining 0.04 GiB is that the engine sizes the pool before capture and the
rounding lands where it lands. A 0.2% capacity model is good enough to plan a
fleet with; the 4% error from the whiteboard version is not, and the
`num_attention_heads` error would have been 600%.

---

## 2026-09-15 — The first engine you start gets a smaller cache than the next one

This is the finding that most surprised me, and it came out of the warmup
discipline rather than out of a hypothesis.

Every sweep marks repetition 0 as warmup and excludes it. In `envelope-len`
(job 5211478) the warmup launch and the three measured launches are the same
configuration on the same node, and they do not agree:

| | weights + non-torch | peak activation | KV pool tokens |
|---|---|---|---|
| repetition 0 (warmup) | 15.58 GiB | 1.04 GiB | 357,520 |
| repetitions 1-3 | 14.58 GiB | 0.26 GiB | 390,720 |

**The first launch on a freshly allocated node gets 33,200 fewer tokens of KV
cache, 8.5% less capacity**, for exactly the same command line. The pattern holds
across every job in the sweep: checking each job's launches in time order, only
the first shows `(15.58, 1.04)` and every later one shows `(14.58, 0.26)`. In job
5211479 the first *successful* launch already shows the warm figures, because the
two launches before it failed and paid the cold cost on its behalf.

The mechanism is the torch.compile and inductor cache. The container points every
compile cache at `/tmp`, which Apptainer maps to the host, so the first launch on
a node compiles and every later launch on that node reuses the result. The
compilation is in flight during the profiling forward pass, so it inflates both
the non-torch allocation and the measured activation peak, and the pool is sized
from what is left.

Two consequences worth stating:

* **Operationally**, the first server you start after a node reboot has 8.5% less
  KV capacity than its identical siblings, and nothing in its logs says so unless
  you compare. If a deployment sizes `--max-num-seqs` from a measurement taken on
  a warm node, the first replica after a cold start will preempt where the others
  do not.
* **Methodologically**, this is the warmup rule from the workspace `CLAUDE.md`
  earning its place in a *memory* measurement, not a timing one. The rule is
  usually justified by CUDA graph capture and autotuning affecting latency. Here
  it moved a capacity number by 8.5%, and a sweep that reported mean-of-all-reps
  would have reported a number that describes no achievable configuration.

It also changes *which boundary you hit*. At `gmu=0.40` and `max_model_len=32768`
(`boundary-recheck`, job 5213929) the warmup launch failed with
`No available memory for the cache blocks` while the three measured launches
failed with `kv_cache_too_small`. The extra gigabyte of cold-start overhead was
the difference between "there is a cache but it is too small for one sequence"
and "there is no cache at all".

---

## 2026-09-15 — Four failure modes, and why naming them mattered

The `gpu_memory_utilization` sweep pins `max_model_len` at the model's maximum so
that both ends of the axis fail, and they fail in genuinely different ways.
Measured at `max_model_len=32768`, Qwen2.5-7B, block_size 16, N=3:

| gmu | status | what the engine said |
|---|---|---|
| 0.38 | `oom_no_cache_blocks` | "No available memory for the cache blocks" |
| **0.40** | **`kv_cache_too_small`** | needs 1.75 GiB for one sequence, has 1.03 GiB; **"estimated maximum model length is 19280"** |
| 0.42 - 0.95 | `ok` | pool from 34,144 to 427,872 tokens |
| 0.97 | `oom_profiling` | `torch.OutOfMemoryError`: wanted 150 MiB, 100.94 MiB free of 39.67 GiB |
| **0.99** | **`budget_exceeds_free`** | requested 39.28 GiB, only 39.26 GiB was free before allocation |

And separately, off the memory axis entirely (`boundary-recheck-block`, job
5213930):

| block_size | status | what the engine said |
|---|---|---|
| 8 | `backend_unsupported` | "No valid attention backend found for cuda with AttentionSelectorConfig(head_size=128, ..., block_size=8, ...)" |

The `gmu=0.99` case is a nice one: the request misses by **0.02 GiB**. 39.28
requested against 39.26 free, because the CUDA context took 0.41 GiB before vLLM
asked for anything. The refusal is pre-flight, so nothing was ever allocated;
this is a different event from the 0.97 OOM, which got as far as graph capture
before running out.

### The methodology point

The first version of `classify_failure` used regexes I had written from memory of
what vLLM's errors look like. They matched the 0.97 OOM and nothing else, so
three distinct boundaries were filed as a generic `launch_failed` whose recorded
`error` was the API server's outer traceback, which names no cause at all. The
sweep *looked* complete: 162 artifacts, failures recorded, nothing dropped.

What recovered it was that the harness stores `engine_log_path` in every artifact
and the engine logs are preserved, so the root causes were still on disk and
could be read out after the fact. The patterns are now copied from those logs.
The original artifacts are kept exactly as written, because raw artifacts are
append-only and "launch_failed with this text" was an honest record of what the
harness knew; `configs/boundary_recheck*.yaml` re-measure the three points under
their own sweep names with the correct classification.

The general lesson is the same one the sibling project kept learning: **a
check that recognises only what it expected will report a clean result on a
surprise.** The insurance is storing more than you plan to parse.

---

## 2026-09-15 — The pool does not care about max_model_len or block_size

Two null results, both sharp, both from N=3 with zero spread.

**`max_model_len` does not change the pool.** Across eight log-spaced values from
256 to 32,768 (`envelope-len`, job 5211478) the pool is **390,720 tokens at every
single one**, not approximately but exactly. Capacity therefore falls as exactly
`390,720 / L`, from 1,526.25 concurrent sequences at 256 tokens to 11.92 at
32,768. `max_model_len` is a per-request limit, not a memory budget: it changes
how the pool is *divided*, never how large it is.

The one caveat this dataset cannot rule out: the profile run uses
`max_num_batched_tokens`, which is derived from `max_model_len`, so the activation
peak could in principle scale with it. Measured here it does not move at all
(0.26 GiB at every L), so on this model and this card the term is flat. That is a
measurement, not a guarantee for a model with a larger activation footprint.

**`block_size` does not change the pool either.** 390,720 tokens at block sizes
16, 32 and 64, and 390,656 at 128, a spread of 0.016%. `num_gpu_blocks` moves
inversely and exactly: 24,420 / 12,210 / 6,105 / 3,052. The pool is sized in bytes
and then divided, so granularity is free at the pool level. Where it is not free
is per sequence, through the rounding described above.

**fp8 doubles the tokens and nothing else.** 781,456 tokens against 390,720, a
ratio of 2.0000, at an identical 20.87 GiB pool. On A100 (sm80) there are no
native fp8 tensor cores, so this is purely a storage win: the same bytes hold
twice the tokens and the attention kernel dequantises on read. The only other
thing that moved was CUDA graph capture, 0.88 GiB against 0.53, and notably the
pool did *not* shrink to pay for it, which confirms the ordering: the pool is
sized before capture, and capture overshoots the budget.

---

## 2026-09-15 — A harness defect that only a big model could find

The first Qwen2.5-32B sweep (`envelope-32b`, job 5213931) measured the harness
rather than the model. Its first launch at TP=2 succeeded; every repetition after
it failed with `budget_exceeds_free` reporting **2.78 GiB free of 39.67** on a
card that should have been empty. The previous engine's 31 GiB of weights were
still resident: `EngineProbe.stop()` kills the process group and returns, but the
driver reclaims device memory asynchronously.

This never showed up in the 7B sweeps. At 14.58 GiB of weights there is enough
slack that a late reclaim still fits under the budget, so 142 artifacts came back
clean while the bug was there the whole time. It took the configuration that
could not absorb it to make it visible, which is the usual shape: the defect is
not in the code that changed, it is in the margin that was always being consumed.

Fixed by `EngineProbe.wait_for_gpu_memory_release`, which polls `nvidia-smi` until
the card is actually idle and **records the wait in the artifact**, so a launch
that began against a dirty card is visible in its own provenance rather than only
in its failure. The failed artifacts are kept: they are the evidence.

**Caveat, and it should be stated rather than glossed.** The re-run
(`envelope-32b-v2`, job 5214082) succeeded where the original failed, but the
recorded wait was **0.05 s**: by the time the poll ran, the card already read as
idle. So the data does not establish that waiting is what fixed it, as opposed to
the race simply not recurring, or the `nvidia-smi` round trip itself being enough
of a barrier. What the fix reliably provides is *diagnosis*: every artifact now
carries how long its launch waited and what the card reported, so a recurrence
will be visible in the record rather than only in a confusing failure. Calling
this "fixed" without the caveat would be the same class of overclaim as calling
fragmentation a major residual term.

---

## Definition-of-done audit against the workspace CLAUDE.md

Checked against the PROJECT 2 section. Honest about what is not met.

| requirement | status | evidence |
|---|---|---|
| Analytical calculator, any HF config to predicted KV bytes, with unit tests | **MET** | `src/kv_cache_envelope/calculator.py`, 46 tests in `tests/test_calculator.py` (job 5214085) |
| Calculator validated against >=3 models with differing GQA configurations | **MET** | 11 committed config bundles in `configs/model_configs/`; 10 answered, spanning **6 distinct GQA ratios** (1, 4, 5, 7, 9, 71) across MHA, GQA and MQA, plus DeepSeek-V2-Lite which the calculator refuses. Head dimension varies too (64, 96, 128), so a hard-coded 128 would be wrong on three of the ten. Table in `results/tables/calculator_validation.md`. Two of them (7B, 32B) additionally validated against a *running engine*. |
| GQA handled via `num_key_value_heads` | **MET**, and the trap turned out to have a second form | Falcon-7B omits the field entirely and is multi-query; the HF-documented MHA fallback would have overstated its cache 71x. See `_resolve_kv_heads`. |
| Measured envelope: sweep sequence length 256 to model max, log-spaced | **MET** | `envelope-len`, 8 points, N=3 |
| ... x batch size | **PARTIAL** | Batch is swept as *concurrency* in the `occupancy` sweep (32 to 256 at four utilisation settings), which is the operationally meaningful axis, and `max_num_seqs` was varied only incidentally. A dedicated `max_num_seqs` axis at fixed concurrency is **not measured**. |
| Record allocated KV blocks, total GPU memory, max concurrent sequences | **MET** | every artifact carries `num_gpu_blocks`, `kv_cache_size_tokens`, `kv_cache_max_concurrency`, the full memory breakdown and `nvidia-smi` occupancy |
| Sweep to actual OOM and record the boundary | **MET** | four distinct boundaries measured at both ends of the `gpu_memory_utilization` axis, N=3 each, all committed as artifacts with status |
| Residual quantified and explained, not hand-waved | **MET** | the term-by-term table above; four of the five terms are read from the engine's own accounting |
| PagedAttention behaviour vs `gpu_memory_utilization` and `block_size` | **MET** | `envelope-gmu`, `envelope-block`, plus P1's `paged-attention` and `memory-utilisation` sweeps mined rather than re-measured |
| Preemption count under deliberate oversubscription | **MET** | `occupancy` sweep, plus mined P1 data |
| Recompute-vs-swap events | **MET as a negative result** | vLLM V1 at 0.27.1 has **no swap path at all**: no occurrence of "swap" in the scheduler, block pool or cache manager. Every preemption is a recompute. The ratio the spec asks for is not a measurable quantity at this version, and saying so is the honest answer. |
| Source reading with file paths and function names at a pinned version | **MET** | this document's first section, read inside the pinned image |
| README as a capacity-planning result | **MET** | `README.md` |
| Every figure regenerable by one command | **MET** | `sbatch slurm/run_analysis.sh` |
| KV cache quantization: fp16 vs fp8 on achievable concurrency | **MET for concurrency** | exactly 2.0000x tokens, measured |
| ... and on an output quality proxy | **NOT MEASURED** | The spec marks this "budget permits". No quality proxy was run, so no claim is made about fp8's effect on output quality. The concurrency half is measured; the quality half is absent, not estimated. |

### Known gaps, stated plainly

* **No fp8 output-quality measurement.** The memory-vs-quality tradeoff curve the
  spec describes as "a strong addition" is not in this repository.
* **`max_num_seqs` is not a swept axis.** The `validate` sweep used 64 and
  everything else used 256, and the two differ in pool size, but the comparison is
  confounded because the `validate` launches were also cold-cache. The effect of
  `max_num_seqs` on the pool is therefore **not measured** here, only suspected.
* **Single model family for the measured envelope.** Qwen2.5-7B carries almost
  all of the measurements; Qwen2.5-32B contributes the tensor-parallel validation
  and little else. The calculator's cross-model validation is against config
  files, not against eleven running engines.
* **One GPU generation.** Everything is A100-SXM4-40GB. The block-size null
  result in particular may not transfer to a card with different attention
  backend availability, given that `block_size=8` is rejected here by the backend
  rather than by the cache.
