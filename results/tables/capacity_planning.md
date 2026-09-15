# Capacity planning: sequences servable at a given context length

Concurrent sequences that fit in the **measured** KV pool at
`gpu_memory_utilization=0.90`, each holding a full context of the stated length.
The pool size is measured; the division is the analytical per-token size, which
the residual analysis shows to be exact. Floor, not round: a partially resident
sequence is a preemption waiting to happen, not a servable sequence.

| model                     | TP | kv dtype | kv KiB/token/GPU | measured pool (tokens) | measured pool (GiB/GPU) | seqs @ 1024 | seqs @ 2048 | seqs @ 4096 | seqs @ 8192 | seqs @ 16384 | seqs @ 32768 |
|---------------------------|----|----------|------------------|------------------------|-------------------------|-------------|-------------|-------------|-------------|--------------|--------------|
| Qwen/Qwen2.5-32B-Instruct | 2  | auto     | 128              | 37744                  | 4.61                    | 36          | 18          | 9           | 4           | 2            | 1            |
| Qwen/Qwen2.5-7B-Instruct  | 1  | auto     | 56               | 390720                 | 20.87                   | 381         | 190         | 95          | 47          | 23           | 11           |
| Qwen/Qwen2.5-7B-Instruct  | 1  | fp8      | 28               | 781440                 | 20.87                   | 763         | 381         | 190         | 95          | 47           | 23           |

