# Node-hour budget

Account `p201466`, `gpu` partition. **A `gpu` job bills the whole 4-GPU node** regardless
of how many GPUs it uses, so every figure below is wall-clock node-hours, not GPU-hours.

Every row is copied from `sacct`, not estimated:

```bash
sacct -X --starttime 2026-09-15 --format=JobID,JobName,State,Elapsed,NNodes -n | grep kve
```

The workspace spec allocates **~15 node-hours** to this project.

## Ledger

| job | sweep | state | elapsed | node-hours | what it bought |
|---|---|---|---:|---:|---|
| 5211409 | prep | COMPLETED | 00:01:26 | 0.024 | syntax gate, unit tests, 11 HF config bundles, vLLM source dumped from the pinned image |
| 5211456 | `validate` | COMPLETED | 00:05:29 | 0.091 | 2-point pipeline validation before widening anything |
| 5211478 | `envelope-len` | COMPLETED | 01:05:17 | 1.088 | 8 context lengths x 4 launches |
| 5211479 | `envelope-gmu` | COMPLETED | 01:43:00 | 1.717 | 13 utilisation settings x 4 launches, both failure boundaries |
| 5211480 | `envelope-block` | COMPLETED | 01:20:04 | 1.335 | 5 block sizes x 2 KV dtypes x 4 launches |
| 5211481_0 | `occupancy` | COMPLETED | 00:58:35 | 0.976 | oversubscription, gmu 0.60/0.80 |
| 5211481_1 | `occupancy` | COMPLETED | 00:54:44 | 0.912 | oversubscription, gmu 0.70/0.90 |
| 5213929 | `boundary-recheck` | COMPLETED | 00:17:09 | 0.286 | re-measure 2 boundaries with the corrected classifier |
| 5213930 | `boundary-recheck-block` | COMPLETED | 00:08:55 | 0.149 | `block_size=8` boundary, named correctly |
| 5213931 | `envelope-32b` | COMPLETED | 00:20:37 | 0.344 | 32B TP=2/4; found the TP sharding bug and the dirty-card defect |
| 5214082 | `envelope-32b-v2` | see below | | | 32B re-run after the teardown fix |
| 5214085 | analysis | FAILED | 00:02:49 | 0.047 | `tabulate` absent from the shared venv; mining completed before the failure |
| 5214094 | analysis | see below | | | figures and tables |

**Consumed by completed jobs: 6.97 node-hours** of the ~15 allocated, plus the two jobs
still in flight at the time of writing. `myquota` showed `p201466` at 0 of 130 GPU
node-hours used at the start of this month, so the monthly allocation is not a constraint
here and the sweeps were sized for coverage rather than for thrift.

## Where the hours went, and whether it was worth it

- **59% on the three envelope sweeps** (4.14 h). These carry the capacity table, the
  linearity result, the two null results and both memory boundaries. Every one of them is
  N=3 with zero spread, which retrospectively means N=3 was generous for an init-time
  measurement; the variance that does exist is entirely between warm and cold, not between
  repetitions.
- **27% on oversubscription** (1.89 h). Bought the preemption-onset result, which is the
  one that turns the calculator into an operational tool. Worth more than its share.
- **6% on boundary rechecks** (0.44 h). Bought precise names for three failures that had
  been recorded as generic. Cheap, and the alternative was a writeup that said
  "launch_failed" about three different phenomena.
- **5% on the 32B sweep** (0.34 h) which *failed*, and was the most informative hour in
  the ledger: it found a genuine bug in the calculator (TP sharding) and a genuine bug in
  the harness (the driver reclaims device memory asynchronously). Both were invisible at
  7B.

## Things that did not cost node-hours

The `paged-attention` and `memory-utilisation` panels draw on 2,410 records mined from the
sibling project's already-committed artifacts. Re-measuring those sweeps would have cost
roughly another 2 node-hours and produced data that already existed. The mining script
reads them read-only and keeps the results in `results/mined_p1/`, separate from this
project's own `results/raw/`, because "measured here" and "measured next door on an
earlier driver" are different claims.
