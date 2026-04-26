# CPT Optimization Report

本报告汇总 `Optimization Track` 的第一版 sweep，当前聚焦已验证主线 `native ddp` 与 `zero2`，用于筛选进入正式长训的推荐配置。

| Family | Variant | Backend | Workers | GC | Eff Batch | Runtime (s) | Tokens/s | Max Alloc (MB) | Status | Exit Code |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| native | ddp_base | ddp | 2 | True | 8 | 159.1 | 5148.575 | 20247.3 | completed | 0 |
| native | ddp_workers4 | ddp | 4 | True | 8 | 153.3 | 5342.209 | 20247.3 | completed | 0 |
| native | ddp_no_gc | ddp | 2 | False | 8 | - | - | - | failed | 1 |
| deepspeed | zero2_base | deepspeed_zero2 | 2 | True | 16 | 624.3 | 2624.357 | 19718.0 | completed | 0 |
| deepspeed | zero2_workers4 | deepspeed_zero2 | 4 | True | 16 | 629.2 | 2603.874 | 19718.0 | completed | 0 |
| deepspeed | zero2_no_gc | deepspeed_zero2 | 2 | False | 16 | - | - | - | failed | 1 |

## Analysis

- 当前 Optimization Track 的第一版 sweep 只覆盖当前 `train_cpt.py` 已稳定接入且可直接控制的旋钮：`dataloader_num_workers` 和 `gradient_checkpointing`。
- `max_seq_length` 与更激进的 batch-density sweep 仍需要新的 tokenized slice 或额外显存验证，因此暂不纳入这一轮自动 sweep。
- Native DDP：baseline `ddp_base` 的吞吐为 `5148.575`，峰值显存 `20247.3 MB`。
- Native DDP：当前已完成变体中吞吐最高的是 `ddp_workers4`，达到 `5342.209`，相对 baseline 变化 `+3.8%`。
- Native DDP：以下变体在本轮 sweep 中已确认失败：`ddp_no_gc`。
- DeepSpeed ZeRO-2：baseline `zero2_base` 的吞吐为 `2624.357`，峰值显存 `19718.0 MB`。
- DeepSpeed ZeRO-2：当前已完成变体里，baseline 仍是吞吐最高或唯一可比样本。
- DeepSpeed ZeRO-2：以下变体在本轮 sweep 中已确认失败：`zero2_no_gc`。
- Recommendation：先用这轮结果筛出 `native ddp` 和 `zero2` 各自的推荐配置，再进入更长 step 的正式长训和中断恢复验证。

## Native Variants

- `ddp_base` backend=`ddp` workers=`2` gc=`True` tokens/s=`5148.575` max_alloc=`20247.3` status=`completed` path=`runs/cpt/optimize/native/qwen3_1_7b/ddp_base`
- `ddp_workers4` backend=`ddp` workers=`4` gc=`True` tokens/s=`5342.209` max_alloc=`20247.3` status=`completed` path=`runs/cpt/optimize/native/qwen3_1_7b/ddp_workers4`
- `ddp_no_gc` backend=`ddp` workers=`2` gc=`False` tokens/s=`-` max_alloc=`-` status=`failed` path=`runs/cpt/optimize/native/qwen3_1_7b/ddp_no_gc`

## Deepspeed Variants

- `zero2_base` backend=`deepspeed_zero2` workers=`2` gc=`True` tokens/s=`2624.357` max_alloc=`19718.0` status=`completed` path=`runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_base`
- `zero2_workers4` backend=`deepspeed_zero2` workers=`4` gc=`True` tokens/s=`2603.874` max_alloc=`19718.0` status=`completed` path=`runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_workers4`
- `zero2_no_gc` backend=`deepspeed_zero2` workers=`2` gc=`False` tokens/s=`-` max_alloc=`-` status=`failed` path=`runs/cpt/optimize/deepspeed/qwen3_1_7b/zero2_no_gc`
