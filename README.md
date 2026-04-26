<p align="center">
  <img src="assets/twinforge-logo.svg" alt="TwinForge" width="560">
</p>

<p align="center">
  <a href="README.md"><img alt="status" src="https://img.shields.io/badge/status-work%20in%20progress-yellow"></a>
  <a href="configs/serve/vllm.yaml"><img alt="serving" src="https://img.shields.io/badge/serving-vLLM-00A3E0"></a>
  <a href="src/serve/openai_client.py"><img alt="api" src="https://img.shields.io/badge/API-OpenAI--compatible-brightgreen"></a>
  <a href="reports/cpt_benchmark.md"><img alt="benchmark" src="https://img.shields.io/badge/CPT-benchmark%20ready-blue"></a>
  <a href="reports/serving_benchmark.md"><img alt="latency" src="https://img.shields.io/badge/mean%20TTFT-20ms-success"></a>
  <a href="configs/train/ds"><img alt="training" src="https://img.shields.io/badge/training-DDP%20%7C%20DeepSpeed-6F42C1"></a>
  <a href="configs/model/qwen3_1_7b_base.yaml"><img alt="model" src="https://img.shields.io/badge/model-Qwen3--1.7B-orange"></a>
</p>

<p align="center">
  <strong>Single-node GPU LLM training and serving performance lab.</strong><br>
  PyTorch · DeepSpeed · PEFT · vLLM · OpenAI-compatible API · RTX 3090
</p>

## Latest News

- [2026/04] Added vLLM OpenAI-compatible serving, SSE streaming client, and minimal serving benchmark with TTFT / latency / decode tokens/s reports.
- [2026/04] Added CPT benchmark, profiling, communication, optimization, and resume validation tracks for Native DDP and DeepSpeed.
- [2026/04] Added SFT LoRA dataset processing, training, merge, and minimal GSM8K / MMLU-mini evaluation pipeline.
- [2026/04] Split training entrypoints into sanity, benchmark, profile, optimize, resume, and longtrain families.

## Overview

TwinForge 是一个面向单机双卡 `RTX 3090` 的轻量级大模型训练与推理实验平台，目标是围绕 `Qwen3-1.7B`、`DeepSpeed`、`PEFT` 和 `vLLM`，把数据准备、CPT、SFT、评测、部署、profiling 逐步做成可复现、可比较、可扩展的完整闭环。

> 项目状态：`Work in Progress`
>
> 当前仓库已经完成 CPT 基线、benchmark / profiling / optimization / resume validation 的主脚手架，并补齐了最小 `SFT / LoRA -> merge -> eval -> vLLM serving -> serving benchmark` 代码链路；当前仍待完成的是这条链路的端到端实跑验证，而不是继续扩张长训或 TP/PP。

TwinForge 不是一个追求“大而全”的训练框架，而是一个围绕双卡工作站的实验平台。它强调三件事：

- 可复现：数据、配置、训练入口、日志、报告尽量落成明确文件。
- 可比较：把 `single_gpu`、`ddp`、`zero`、后续 `tp` 与不同推理后端放到统一协议下比较。
- 可扩展：在主链路跑通后，继续接入 DeepSpeed 的 profiling、monitoring、communication logging、inference、AutoTP 等能力。

## Current Snapshot

当前已经完成或验证的部分：

- [x] 环境基线检查与独立 `.venv` 运行环境
- [x] 基于 manifest 的原始数据下载入口
- [x] CPT 清洗、抽样、packing、tokenize 主链路
- [x] `Qwen3-1.7B` CPT 训练入口
- [x] `scripts/train/sanity` 与 `scripts/train/bench/{native,deepspeed}` 两层训练入口
- [x] `smoke / single_gpu / ddp / zero2 / zero3_offload` CPT sanity 启动脚本
- [x] `single_gpu` 与 `ddp` 两档训练实际跑通
- [x] CUDA 显存 callback，支持按 step 记录 `allocated / reserved / max_allocated`
- [x] `train_summary.json`，记录 tokens/s、checkpoint 保存耗时/大小、resume 状态
- [x] CPT benchmark 切片脚本、slice spec、索引清单与 benchmark 汇总脚本
- [x] 实施计划、操作日志、错误日志
- [x] `zero2 / zero3_offload` 正式 benchmark、profiling、communication、optimization 主线
- [x] 短程 `resume validation` 轨道与报告
- [x] SFT cleaned/tokenized 数据链路、LoRA 训练入口与 merge 脚本
- [x] 最小评测链路：`GSM8K + MMLU_mini`
- [x] `vLLM` 服务入口、OpenAI-compatible 客户端与最小服务压测脚本
- [x] profiling / communication / monitor 主线脚手架接入
- [ ] `resume -> SFT -> merge -> eval -> vLLM` 端到端实跑验证
- [ ] DeepSpeed Inference / HF eager 服务基线补齐

当前已经验证过的 CPT 结果：

- `single_gpu`：完成 `200` steps，训练日志与显存日志已落盘
- `ddp`：完成 `300` steps，主进程最终保存逻辑已验证

这些结果目前只代表 `sanity` 级验证，不代表性能结论。正式性能对比会在受控 benchmark 配置下完成。
当前 benchmark 已拆为 `Native Family` 与 `DeepSpeed Family` 两组，不再输出单表式 `single/ddp/zero` 严格可比结论。

## Why This Repo

- 轻量：默认目标就是单机双卡 `3090`，不把多节点超大规模能力作为第一优先级。
- 透明：核心能力优先落成可读脚本和配置，而不是藏在复杂框架抽象里。
- 可追踪：每次实施动作和异常都要求写入日志，方便复盘和继续推进。
- 可演进：当前主线是数据链路和 CPT，后续再逐步接入 SFT、评测、推理、压测和 profiling。

## DeepSpeed Scope

当前仓库把 DeepSpeed 能力分成三层：

### 近期主线

- ZeRO-1 / ZeRO-2 / ZeRO-3 Offload 训练对比
- DeepSpeed Inference
- Automatic Tensor Parallelism inference
- Flops Profiler
- PyTorch Profiler
- Monitoring
- Communication Logging

### 独立 POC

- AutoTP training
- Pipeline parallelism
- Ulysses / ALST sequence parallelism
- Universal checkpointing
- Autotuning

### 暂不优先

- ZeRO++
- 多节点分布式能力
- MoE 等超出当前 `Qwen3-1.7B` 主线的教程能力

详细能力矩阵、优先级和阶段拆分见 [Implementation Plan](reports/implementation_plan.md)。

## What Works Today

当前已经不是占位的主要模块：

- `scripts/00_check_env.py`
- `scripts/01_download_data.py`
- `scripts/02_build_cpt_corpus.py`
- `scripts/04_tokenize_cpt.py`
- `scripts/04_build_cpt_benchmark_slice.py`
- `scripts/12_report_cpt_benchmark.py`
- `scripts/03_build_sft_dataset.py`
- `scripts/05_tokenize_sft.py`
- `scripts/08_merge_lora.py`
- `scripts/09_eval_all.sh`
- `scripts/11_benchmark_serving.py`
- `src/data/cleaners.py`
- `src/data/samplers.py`
- `src/data/packers.py`
- `src/data/sft.py`
- `src/train/train_cpt.py`
- `src/train/train_sft.py`
- `src/train/callbacks.py`
- `src/eval/*`
- `src/serve/*`
- `configs/model/qwen3_1_7b_base.yaml`
- `configs/data/cpt_benchmark_slice.yaml`
- `configs/train/sft_lora.yaml`
- `configs/train/sanity/*`
- `configs/train/bench/native/*`
- `configs/train/bench/deepspeed/*`
- `configs/train/ds/*`
- `scripts/train/sanity/*`
- `scripts/train/bench/native/*`
- `scripts/train/bench/deepspeed/*`
- `configs/train/profile/*`
- `configs/train/longtrain/*`
- `configs/train/resume/*`
- `scripts/train/profile/*`
- `scripts/train/longtrain/*`
- `scripts/train/resume/*`
- `scripts/13_report_cpt_profile.py`
- `scripts/14_report_cpt_comms.py`
- `scripts/16_report_cpt_resume_validation.py`
- `scripts/06_train_cpt*.sh`
- `scripts/07_train_sft_lora.sh`
- `scripts/10_serve_vllm.sh`

当前仍以占位或半成品为主的部分：

- `src/prof/*`
- `src/serve/deepspeed_generate.py`
- `scripts/10_serve_deepspeed.sh`

## Local Model And Artifacts

当前默认模型配置指向本地目录：

```yaml
model_name_or_path: model/Qwen3-1.7B
tokenizer_name_or_path: model/Qwen3-1.7B
```

也就是说：

- 基座模型权重不包含在仓库里
- 训练输出、数据产物、日志和本地模型目录都应视为本地产物
- 仓库本身提交的是源码、配置、脚本、报告和 Markdown 日志

## Quick Start

### Prerequisites

- Linux
- CUDA 可用的 PyTorch 环境
- 建议双卡 `RTX 3090`
- 本地准备好 `model/Qwen3-1.7B`

### Install

推荐使用项目内虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip setuptools wheel
pip install "torch>=2.2" torchvision torchaudio
pip install "transformers>=4.37.0" "datasets>=2.18.0" "accelerate>=0.28.0" deepspeed peft trl
pip install sentencepiece safetensors scikit-learn numpy pandas pyarrow ujson jsonlines tqdm pyyaml tensorboard evaluate jieba nltk
```

### Check Environment

```bash
python scripts/00_check_env.py
```

### Download Datasets

```bash
python scripts/01_download_data.py \
  --manifest configs/dataset_manifest.json \
  --output_root data/raw
```

### Build Cleaned CPT Corpus

```bash
python scripts/02_build_cpt_corpus.py \
  --manifest configs/dataset_manifest.json \
  --input_root data/raw/cpt \
  --output_root data/cleaned/cpt
```

### Tokenize CPT Corpus

```bash
python scripts/04_tokenize_cpt.py \
  --manifest configs/dataset_manifest.json \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --input_root data/cleaned/cpt \
  --output_path data/tokenized/cpt/train
```

### Build Deterministic CPT Benchmark Slice

```bash
python scripts/04_build_cpt_benchmark_slice.py \
  --slice_config configs/data/cpt_benchmark_slice.yaml \
  --overwrite
```

默认会同时写出：

- `data/tokenized/cpt/bench/`
- `data/tokenized/cpt/bench_summary.json`
- `data/tokenized/cpt/bench_indices.json`

其中 `bench_summary.json` 会记录数据集 fingerprint、tokenizer 路径和 tokenizer 指纹，`bench_indices.json` 会保存实际使用的样本索引清单。

### Run CPT Training

最小验证：

```bash
bash scripts/06_train_cpt_smoke.sh
```

分层训练入口：

```bash
bash scripts/06_train_cpt_single_gpu.sh
bash scripts/06_train_cpt_ddp.sh
bash scripts/06_train_cpt_zero2.sh
```

兼容入口：

```bash
bash scripts/06_train_cpt.sh
```

默认等价于 `zero2` 档位。

新的分轨入口：

```bash
bash scripts/train/sanity/cpt_single.sh
bash scripts/train/sanity/cpt_ddp.sh
bash scripts/train/sanity/cpt_zero2.sh
bash scripts/train/sanity/cpt_zero3_offload.sh

bash scripts/train/bench/native/cpt_single.sh
bash scripts/train/bench/native/cpt_ddp.sh

bash scripts/train/bench/deepspeed/cpt_zero2.sh
bash scripts/train/bench/deepspeed/cpt_zero3_offload.sh
```

生成 benchmark 汇总表：

```bash
python scripts/12_report_cpt_benchmark.py --output reports/cpt_benchmark.md
```

说明：

- `Native Family` 只比较 `single` 与 `ddp`
- `DeepSpeed Family` 只比较 `zero2` 与 `zero3_offload`
- 报告中的 `native_ddp_bridge_ref` 仅作跨 family 参考，不参与严格同协议结论

### Run Profiling

执行 profiling / communication 基线：

```bash
bash scripts/train/profile/run_all.sh
```

或按单项串行执行：

```bash
bash scripts/train/profile/native/cpt_single.sh
bash scripts/train/profile/native/cpt_ddp.sh
bash scripts/train/profile/deepspeed/cpt_zero2.sh
bash scripts/train/profile/deepspeed/cpt_zero3_offload.sh
```

生成 profiling 报告：

```bash
python scripts/13_report_cpt_profile.py --output reports/cpt_profile.md
python scripts/14_report_cpt_comms.py --output reports/cpt_comms.md
```

### Run Optimization

执行第一版 `Optimization Track` sweep：

```bash
bash scripts/train/optimize/run_all.sh
```

或按单项执行：

```bash
bash scripts/train/optimize/native/cpt_ddp_base.sh
bash scripts/train/optimize/native/cpt_ddp_workers4.sh
bash scripts/train/optimize/native/cpt_ddp_no_gc.sh

bash scripts/train/optimize/deepspeed/cpt_zero2_base.sh
bash scripts/train/optimize/deepspeed/cpt_zero2_workers4.sh
bash scripts/train/optimize/deepspeed/cpt_zero2_no_gc.sh
```

生成优化报告：

```bash
python scripts/15_report_cpt_optimization.py --output reports/cpt_optimization.md
```

说明：

- 当前第一版优化 sweep 只覆盖已经稳定接入的旋钮：`dataloader_num_workers` 与 `gradient_checkpointing`
- 这些优化 run 默认 `save_model_artifacts: false`，只保留 `train_summary.json`、`train_log_history.json` 和日志
- `max_seq_length` 与更激进的 batch-density sweep 暂未进入自动 sweep，因为当前 benchmark tokenized slice 固定在 `2048`，需要单独准备新的 tokenized 视图后再做

### Run Formal Long-Train Candidates

基于当前 `Optimization Track` 的结论，仓库里保留了两条正式长训候选入口：

- `native ddp`: `workers=4`, `gradient_checkpointing=true`
- `zero2`: `workers=2`, `gradient_checkpointing=true`

对应启动入口：

```bash
bash scripts/train/longtrain/native/cpt_ddp.sh
bash scripts/train/longtrain/deepspeed/cpt_zero2.sh
```

说明：

- 两条长训入口默认都使用全量 tokenized 数据：`data/tokenized/cpt/train`
- 两条配置都恢复 `save_model_artifacts: true`，会保存 checkpoint 和最终模型
- 当前默认 `max_steps=3000`、`save_steps=500`，可通过脚本透传 `--resume_from_checkpoint ...` 继续训练
- `native ddp` 是吞吐优先候选；`zero2` 是显存更稳的正式 DeepSpeed 候选
- 当前阶段这两条入口默认不作为优先执行项；它们只是在链路闭环完成后的保留长训入口

### Run Resume Validation

在正式进入 `SFT / eval / serving` 前，建议先验证推荐配置的短程 `resume` 能力：

```bash
bash scripts/train/resume/native/cpt_ddp_validate.sh
bash scripts/train/resume/deepspeed/cpt_zero2_validate.sh
```

或串行跑完两条：

```bash
bash scripts/train/resume/run_all.sh
```

生成 resume 报告：

```bash
python scripts/16_report_cpt_resume_validation.py --output reports/cpt_resume_validation.md
```

说明：

- `resume validation` 默认使用 `data/tokenized/cpt/bench`
- 两条验证都会执行 `stage1 -> checkpoint-10 -> stage2 resume -> checkpoint-12`
- 该阶段目标是验证 checkpoint 可恢复，不是验证长训收敛

### Build Cleaned SFT Dataset

```bash
python scripts/03_build_sft_dataset.py \
  --manifest configs/dataset_manifest.json \
  --input_root data/raw/sft \
  --output_root data/cleaned/sft \
  --overwrite
```

默认会读取：

- `data/raw/sft/ultrachat_200k_mini.jsonl`
- `data/raw/sft/wildchat_mini.jsonl`

并稳定输出：

- `data/cleaned/sft/train.jsonl`
- `data/cleaned/sft/val.jsonl`
- `data/cleaned/sft/summary.json`

### Tokenize SFT Dataset

```bash
python scripts/05_tokenize_sft.py \
  --manifest configs/dataset_manifest.json \
  --model_config configs/model/qwen3_1_7b_base.yaml \
  --input_root data/cleaned/sft \
  --output_root data/tokenized/sft \
  --overwrite
```

说明：

- 默认使用 `Qwen3-1.7B` chat template
- 默认只训练 assistant 响应，system/user token 会被 mask 为 `-100`
- 默认启用 packing，`seq_length=2048`

### Run SFT LoRA

```bash
bash scripts/07_train_sft_lora.sh
```

当前默认配置：

- `LoRA + DeepSpeed ZeRO-2`
- `r=64`
- `lora_alpha=128`
- `lora_dropout=0.05`
- 目标是先完成短程闭环，不是正式长训

### Merge LoRA

```bash
python scripts/08_merge_lora.py \
  --base_model model/Qwen3-1.7B \
  --adapter_path runs/sft/qwen3_1_7b_miniv2_sft_lora \
  --output_dir runs/sft/qwen3_1_7b_miniv2_sft_merged
```

### Run Minimal Eval

```bash
bash scripts/09_eval_all.sh \
  --model_path runs/sft/qwen3_1_7b_miniv2_sft_merged
```

默认只跑：

- `gsm8k`
- `mmlu_mini`

并最终聚合到：

- `runs/eval/summary.json`

### Serve With vLLM

```bash
bash scripts/10_serve_vllm.sh runs/sft/qwen3_1_7b_miniv2_sft_merged
```

默认配置读取：

- `configs/serve/vllm.yaml`

### Run Minimal Serving Benchmark

```bash
python scripts/11_benchmark_serving.py \
  --base_url http://127.0.0.1:8000 \
  --model qwen3-1.7b-miniv2
```

这里的 `--model` 传的是 vLLM 暴露出来的 OpenAI `model id`，默认来自 `configs/serve/vllm.yaml` 里的 `served_model_name`，不是本地模型目录路径。

当前最小服务报告会输出：

- `TTFT`
- `end-to-end latency`
- `output tokens`
- `decode tokens/s`

## Recommended Workflow

建议按下面顺序推进，而不是直接跳到长训或部署：

1. 用 `scripts/00_check_env.py` 确认 Python / CUDA / PyTorch / DeepSpeed 基线。
2. 下载原始数据并生成 `cleaned`、`tokenized` CPT 数据。
3. 用 `configs/data/cpt_benchmark_slice.yaml` 生成固定的 `data/tokenized/cpt/bench` 数据切片，并保留 `bench_summary.json` 与 `bench_indices.json`。
4. 先跑 `scripts/train/sanity/*` 完成后端验证。
5. 先跑 `scripts/train/bench/native/*` 完成 native family 对比，再跑 `scripts/train/bench/deepspeed/*` 完成 DeepSpeed family 对比，最后生成双章节 `reports/cpt_benchmark.md`。
6. 基于 benchmark 结果先跑 `scripts/train/profile/*`，生成 `reports/cpt_profile.md` 与 `reports/cpt_comms.md`，完成吞吐/显存差异归因。
7. 在 profile 结果基础上先跑 `scripts/train/optimize/*`，筛出 `native ddp` 与 `zero2` 各自的推荐训练配置，并生成 `reports/cpt_optimization.md`。
8. 在优化结果基础上，先跑 `scripts/train/resume/*`，验证推荐配置的 checkpoint / resume 能力。
9. 用通过 resume 验证的推荐配置推进 `scripts/03_build_sft_dataset.py -> scripts/05_tokenize_sft.py -> scripts/07_train_sft_lora.sh -> scripts/08_merge_lora.py`。
10. 对 merged SFT 模型执行 `scripts/09_eval_all.sh`，先拿到 `GSM8K + MMLU_mini` 的最小结果，再启动 `scripts/10_serve_vllm.sh` 与 `scripts/11_benchmark_serving.py`。
11. 只有当这条最小链路稳定闭合后，再考虑执行 `scripts/train/longtrain/*`，以及后续的 TP / PP / 更完整 serving family。

## Repository Layout

```text
.
├── configs/   # 模型、训练、DeepSpeed、服务、数据配置
├── data/      # 本地数据产物目录，不随仓库提交
├── logs/      # 操作日志、错误日志、下载日志
├── model/     # 本地基座模型目录，不随仓库提交
├── reports/   # 实施计划、基线报告、训练与服务报告
├── runs/      # 本地训练、评测、压测输出，不随仓库提交
├── scripts/   # 可直接执行的入口脚本
└── src/       # 数据处理、训练、评测、服务、profiling 代码
```

核心目录说明：

- `configs/`
  模型配置、训练超参数、DeepSpeed 配置、服务配置、数据 manifest。
- `scripts/`
  面向实验流程的 CLI 入口，适合串联整个链路。
- `src/data/`
  数据清洗、抽样、packing、chat template 相关逻辑。
- `src/train/`
  CPT / SFT 训练逻辑及训练辅助模块。
- `src/eval/`
  benchmark 评测与结果聚合。
- `src/serve/`
  推理调用、服务验证、客户端脚本。
- `src/prof/`
  profiling、communication、kernel benchmark 相关实验代码。

## Documentation

关键文档与记录入口：

- [Implementation Plan](reports/implementation_plan.md)
- [Baseline Report](reports/baseline.md)
- [Operation Log](logs/operation_log.md)
- [Error Log](logs/error_log.md)
- [Training Log](reports/training_log.md)
- [CPT Sanity Report](reports/cpt_sanity.md)
- [CPT Benchmark](reports/cpt_benchmark.md)
- [CPT Profile](reports/cpt_profile.md)
- [CPT Comms](reports/cpt_comms.md)
- [CPT Optimization](reports/cpt_optimization.md)
- [CPT Resume Validation](reports/cpt_resume_validation.md)
- [Serving Benchmark](reports/serving_benchmark.md)

记录约定：

- 每次代码、配置、文档、实验操作后，更新 `logs/operation_log.md`
- 每次失败、阻塞、异常后，更新 `logs/error_log.md`
- 每次阶段目标变化后，回写 `reports/implementation_plan.md`

## Roadmap

当前实施路线已经从“补骨架”切换为“能力分轨”：

1. `Sanity Track`
2. `Benchmark Track`
3. `Optimization Track`
4. `Serving Track`
5. `Profiling / Observability Track`
6. `AutoTP / Pipeline / Sequence Parallel POC Track`

完整阶段说明、优先级和可行性矩阵见 [Implementation Plan](reports/implementation_plan.md)。

## Known Constraints

- `liwu/MNBVC` 这类旧式 dataset script 仓库不能再直接依赖 `datasets.load_dataset(...)`，当前已改为 Hugging Face 仓库原始文件直拉。
- 当前 `single_gpu` 与 `ddp` 结果已经验证链路，但还不是正式 benchmark 结论。
- `SFT / eval / serve` 的最小脚手架已完成，但当前仓库里还没有把 `resume -> SFT -> merge -> eval -> vLLM` 整条链路做一次正式端到端实跑验证。
- 仓库当前主线仍是 `Qwen3-1.7B` 的 CPT / DeepSpeed 实验，不覆盖更大规模分布式训练的全部场景。

## Contributing

欢迎提交 issue 或 PR，但建议遵守以下约定：

- 先说明目标、范围和影响模块。
- 不要把未验证流程标记为“已完成”。
- 涉及实验、命令、调参、报错处理时，同步更新日志文件。
- 保持配置、脚本、报告和 README 之间的一致性。
