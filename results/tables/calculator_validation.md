# Calculator validation against real model configs

Every row is computed by `kv_cache_envelope.calculator` from the committed
`config.json` bundle named by its revision. The last column is the error the
`num_attention_heads` mistake would have produced for that model.

| model                               | revision     | kind | layers | q heads | kv heads | head_dim | GQA ratio | kv KiB/token | GiB @ 8k ctx | kv heads from      | head_dim from | if q heads used |
|-------------------------------------|--------------|------|--------|---------|----------|----------|-----------|--------------|--------------|--------------------|---------------|-----------------|
| Qwen/Qwen2.5-0.5B-Instruct          | 7ae557604adf | GQA  | 24     | 14      | 2        | 64       | 7x        | 12           | 0.094        | config             | derived       | 7x too large    |
| Qwen/Qwen2.5-32B-Instruct           | 5ede1c97bbab | GQA  | 64     | 40      | 8        | 128      | 5x        | 256          | 2            | config             | derived       | 5x too large    |
| Qwen/Qwen2.5-7B-Instruct            | a09a35458c70 | GQA  | 28     | 28      | 4        | 128      | 7x        | 56           | 0.438        | config             | derived       | 7x too large    |
| bigcode/starcoder2-7b               | bb9afde76d79 | GQA  | 32     | 36      | 4        | 128      | 9x        | 64           | 0.25         | config             | derived       | 9x too large    |
| ibm-granite/granite-3.1-8b-instruct | 4009206d5fc9 | GQA  | 40     | 32      | 8        | 128      | 4x        | 160          | 1.25         | config             | derived       | 4x too large    |
| mistralai/Mistral-7B-Instruct-v0.2  | 63a8b0818953 | GQA  | 32     | 32      | 8        | 128      | 4x        | 128          | 1            | config             | derived       | 4x too large    |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | 31b70e2e869a | MHA  | 24     | 32      | 32       | 64       | 1x        | 192          | 1.5          | config             | derived       | 1x too large    |
| allenai/OLMo-2-1124-7B-Instruct     | 470b1fba1ae0 | MHA  | 32     | 32      | 32       | 128      | 1x        | 512          | 4            | config             | derived       | 1x too large    |
| microsoft/Phi-3-mini-4k-instruct    | f39ac1d28e92 | MHA  | 32     | 32      | 32       | 96       | 1x        | 384          | 0.75         | config             | derived       | 1x too large    |
| tiiuae/falcon-7b                    | ec89142b67d7 | MQA  | 32     | 71      | 1        | 64       | 71x       | 8            | 0.062        | falcon_multi_query | derived       | 71x too large   |

**10 models, 6 distinct GQA ratios, 3 attention kinds.**

## Configs the calculator refuses to answer for

Refusing beats returning a plausible wrong number. These architectures do not
cache per-head K and V, so `2*L*H_kv*D` does not describe them.

| model                        | revision                                 | refused because                                                                                                                                                                   |
|------------------------------|------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| deepseek-ai/DeepSeek-V2-Lite | 604d5664dddd88a0433dbae533b7fe9472482de0 | deepseek-ai/DeepSeek-V2-Lite: multi-head latent attention (keys ['kv_lora_rank']); the cache is a compressed latent per token, not per-head K and V, so 2*L*H_kv*D does not apply |

