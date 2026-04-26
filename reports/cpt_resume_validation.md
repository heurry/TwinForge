# CPT Resume Validation Report

本报告验证推荐训练配置的短程 `resume` 能力，确保后续 SFT / eval / serving 可基于稳定 checkpoint 继续推进。

| Family | Variant | Status | Stage1 Step | Stage2 Step | Resume Requested | Resume Success | Ckpt-10 | Ckpt-12 | Stage1 Tokens/s | Stage2 Tokens/s |
| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: |
| native | ddp_recommended | passed | 10 | 12 | True | True | True | True | 4256.425 | 13513.785 |
| deepspeed | zero2_recommended | passed | 10 | 12 | True | True | True | True | 2397.247 | 9480.110 |

## Analysis

- 本报告用于验证推荐训练配置能否稳定完成 `checkpoint -> resume -> continue training`，而不是比较最终收敛。
- `ddp_recommended`：resume 验证通过，已确认 `step 10 -> checkpoint-10 -> resume -> step 12 -> checkpoint-12` 的完整链路成立。
- `zero2_recommended`：resume 验证通过，已确认 `step 10 -> checkpoint-10 -> resume -> step 12 -> checkpoint-12` 的完整链路成立。
- 只有当推荐配置的 resume 验证通过后，再用它去承接后续 SFT / eval / serving，才能避免在下游阶段暴露 checkpoint 不可恢复的问题。

## ddp_recommended

- stage1 max alloc: `20247.3 MB`
- stage2 max alloc: `20247.3 MB`
- resume checkpoint: `runs/cpt/resume_validation/native/qwen3_1_7b/ddp_recommended/checkpoint-10`
- artifacts dir: `runs/cpt/resume_validation/native/qwen3_1_7b/ddp_recommended/resume_validation`

## zero2_recommended

- stage1 max alloc: `19718.0 MB`
- stage2 max alloc: `19718.0 MB`
- resume checkpoint: `runs/cpt/resume_validation/deepspeed/qwen3_1_7b/zero2_recommended/checkpoint-10`
- artifacts dir: `runs/cpt/resume_validation/deepspeed/qwen3_1_7b/zero2_recommended/resume_validation`
