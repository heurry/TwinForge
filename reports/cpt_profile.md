# CPT Profiling Report

本报告汇总 `torch profiler` 与 DeepSpeed Flops Profiler 产物，用于解释 benchmark 中的主要 hotspot。

| Experiment | Backend | Runtime (s) | Tokens/s | Max Alloc (MB) | Top Hotspot | Hotspot Share | Flops Profile |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| cpt_profile_native_single | single_gpu | 65.0 | 3023.864 | 16965.300 | Command Buffer Full | 37.3 | - |
| cpt_profile_native_ddp | ddp | 43.0 | 4573.024 | 20247.350 | Command Buffer Full | 31.3 | - |
| cpt_profile_zero2 | deepspeed_zero2 | 203.1 | 1936.404 | 19717.950 | ProfilerStep* | 45.5 | yes |
| cpt_profile_zero3_offload | deepspeed_zero3_offload | 417.1 | 314.226 | 6415.900 | - | - | - |

## Analysis

- 这些 profile 结果带 instrumentation 开销，绝对吞吐只用于解释瓶颈，不替代正式 benchmark 结论。
- Native family：`ddp` 相比 `single` 的 profile 吞吐从 `3023.864` 提升到 `4573.024`，约 `1.51x`；运行时从 `65.0s` 降到 `43.0s`，变化 `-33.9%`；峰值显存从 `16965.3 MB` 增到 `20247.3 MB`，变化 `+19.3%`。
- Native family：两条路的 top hotspot 都集中在 `Command Buffer Full` / `Command Buffer Full` 一类 launch 或同步事件，说明当前 native 路线更像 launch-bound / sync-bound，而不是显式通信主导。
- DeepSpeed family：`zero2` 的 profile 吞吐为 `1936.404`，`zero3_offload` 为 `314.226`，前者约是后者的 `6.16x`；峰值显存分别为 `19718.0 MB` 和 `6415.9 MB`，`zero3_offload` 约只用到 `zero2` 的 `32.5%`。
- DeepSpeed family：`zero2` 的 top hotspot 是 `ProfilerStep*`，且保留了 DeepSpeed flops profile；`zero3_offload` 当前未启用 torch profiler，只保留 runtime、显存和 comms summary，因为该路线在本机上叠加 torch profiler 不稳定。
- Interpretation：如果目标是正式训练吞吐，当前仍应优先参考 benchmark 里的 `native ddp` 和 `zero2`；如果目标是解释 profile 现象，这份报告更支持“native 侧主要受 kernel launch / sync 约束，DeepSpeed 侧主要受 optimizer / partition orchestration 约束”的判断。

## cpt_profile_native_single

- Torch profile dir: `runs/cpt/profile/native/qwen3_1_7b/single/profiling/torch`
- DeepSpeed flops profile: `-`
- Top 5 non-communication hotspots:
- `Command Buffer Full` count=`71149` time=`9385585.4`us share=`37.3`
- `cudaLaunchKernel` count=`148677` time=`8262090.6`us share=`32.9`
- `ProfilerStep*` count=`3` time=`2232508.8`us share=`8.9`
- `cudaStreamSynchronize` count=`198` time=`1920548.4`us share=`7.6`
- `cudaMemsetAsync` count=`10800` time=`1055923.3`us share=`4.2`

## cpt_profile_native_ddp

- Torch profile dir: `runs/cpt/profile/native/qwen3_1_7b/ddp/profiling/torch`
- DeepSpeed flops profile: `-`
- Top 5 non-communication hotspots:
- `Command Buffer Full` count=`35654` time=`4739157.9`us share=`31.3`
- `cudaLaunchKernel` count=`75222` time=`4294619.7`us share=`28.4`
- `cudaStreamSynchronize` count=`108` time=`3163981.2`us share=`20.9`
- `ProfilerStep*` count=`3` time=`1146522.8`us share=`7.6`
- `cudaMemsetAsync` count=`5400` time=`381877.9`us share=`2.5`

## cpt_profile_zero2

- Torch profile dir: `runs/cpt/profile/deepspeed/qwen3_1_7b/zero2/profiling/torch`
- DeepSpeed flops profile: `runs/cpt/profile/deepspeed/qwen3_1_7b/zero2/profiling/deepspeed/deepspeed_flops_profile.txt`
- Top 5 non-communication hotspots:
- `ProfilerStep*` count=`3` time=`35528013.6`us share=`45.5`
- `cudaDeviceSynchronize` count=`7663` time=`33081697.5`us share=`42.4`
- `torch::autograd::AccumulateGrad` count=`7440` time=`4044355.3`us share=`5.2`
- `cudaLaunchKernel` count=`146522` time=`501361.6`us share=`0.6`
- `Command Buffer Full` count=`3242` time=`379063.7`us share=`0.5`

## cpt_profile_zero3_offload

- Torch profile dir: `-`
- DeepSpeed flops profile: `-`
- Top 5 non-communication hotspots:
- `-`
