# CPT Benchmark Report

本报告按 family 拆分展示：`Native Family` 内部可比，`DeepSpeed Family` 内部可比。
DeepSpeed 章节中的 `native_ddp_bridge_ref` 仅作跨 family 参考，不参与严格同协议结论。

## Native Family

| Experiment | Backend | Benchmark Group | Runtime (s) | Samples/s | Steps/s | Tokens/s | Train Loss | Max Alloc (MB) | Last Save (s) | Resume |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cpt_bench_native_single | single_gpu | cpt_bench_native_batch8_tf32 | 542.2 | 1.475 | 0.184 | 3021.578 | 2.872 | 16965.300 | 7.430 | - |
| cpt_bench_native_ddp | ddp | cpt_bench_native_batch8_tf32 | 321.7 | 2.487 | 0.311 | 5093.503 | 2.872 | 20247.350 | 7.269 | - |

## DeepSpeed Family

注：`native_ddp_bridge_ref` 为跨 family 参考行，不能直接与 ZeRO family 得出严格同协议结论。

| Experiment | Backend | Benchmark Group | Runtime (s) | Samples/s | Steps/s | Tokens/s | Train Loss | Max Alloc (MB) | Last Save (s) | Resume |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| native_ddp_bridge_ref (cross-family reference only) | ddp_bridge_ref | cpt_bench_native_batch8_tf32 | 321.7 | 2.487 | 0.311 | 5093.503 | 2.872 | 20247.350 | 7.269 | - |
| cpt_bench_zero2 | deepspeed_zero2 | cpt_bench_deepspeed_batch16_fp16 | 1293.3 | 1.237 | 0.077 | 2533.623 | 2.765 | 19717.950 | 16.121 | - |
| cpt_bench_zero3_offload | deepspeed_zero3_offload | cpt_bench_deepspeed_batch16_fp16 | 2330.5 | 0.687 | 0.043 | 1406.075 | 2.765 | 6491.930 | 11.974 | - |

## Analysis

### Native Family

- 在相同 `effective_batch_size=8` 条件下，`ddp` 的 `tokens/s` 从 `3021.578` 提升到 `5093.503`，约 `1.69x`；`runtime` 从 `542.2s` 降到 `321.7s`，约 `40.7%` 更快。
- 两者 `train_loss` 基本一致：`2.872` vs `2.872`，差值约 `0.000009`。
- 代价是峰值显存从 `16965.300` MB 增到 `20247.350` MB，约 `19.3%`。
- 结论：在当前双卡 `3090` 的 native 轨道里，`ddp` 是默认首选训练后端。

### DeepSpeed Family

- 在相同 `effective_batch_size=16` 的 `fp16` 协议下，`zero2` 的 `tokens/s` 为 `2533.623`，`zero3_offload` 为 `1406.075`；`zero2` 约快 `1.80x`。
- `zero3_offload` 的峰值显存从 `19717.950` MB 降到 `6491.930` MB，约减少 `67.1%`。
- 两者 `train_loss` 仍基本一致：`2.765` vs `2.765`，差值约 `0.000001`；`last save` 为 `16.121`s vs `11.974`s。
- 结论：显存扛得住时优先 `zero2`；需要为更大模型、更长上下文或更小显存预算让路时再选 `zero3_offload`。

### Cross-Family Note

- `native_ddp_bridge_ref` 仍然只作参考，因为它和 DeepSpeed family 的 `benchmark_group`、precision、effective batch 都不同，不能直接拿来下严格性能结论。
