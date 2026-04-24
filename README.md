# TwinForge

一个面向单机双卡 `RTX 3090` 的轻量级大模型训练与推理实验平台，目标是围绕 `Qwen3-1.7B`、`DeepSpeed`、`PEFT` 和 `vLLM`，把数据准备、CPT、SFT、评测、部署、profiling 逐步做成可复现、可比较、可扩展的完整闭环。

> 项目状态：`Work in Progress`
>
> 当前仓库已经跑通 CPT 数据链路，并验证了 `single_gpu` 与 `ddp` 两档训练；`zero2`、受控 benchmark、DeepSpeed inference、SFT、评测和服务压测仍在推进中。

## Overview

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
- [x] `smoke / single_gpu / ddp / zero2` 四档 CPT 启动脚本
- [x] `single_gpu` 与 `ddp` 两档训练实际跑通
- [x] CUDA 显存 callback，支持按 step 记录 `allocated / reserved / max_allocated`
- [x] 实施计划、操作日志、错误日志
- [ ] `zero2` 重新验证与正式 benchmark
- [ ] SFT 数据链路与 LoRA 训练
- [ ] 统一评测链路与结果聚合
- [ ] DeepSpeed inference / vLLM serving benchmark
- [ ] profiling / communication / monitor 主线接入

当前已经验证过的 CPT 结果：

- `single_gpu`：完成 `200` steps，训练日志与显存日志已落盘
- `ddp`：完成 `300` steps，主进程最终保存逻辑已验证

这些结果目前只代表 `sanity` 级验证，不代表性能结论。正式性能对比会在受控 benchmark 配置下完成。

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
- `src/data/cleaners.py`
- `src/data/samplers.py`
- `src/data/packers.py`
- `src/train/train_cpt.py`
- `src/train/callbacks.py`
- `configs/model/qwen3_1_7b_base.yaml`
- `configs/train/cpt*.yaml`
- `configs/train/deepspeed_zero2.json`
- `scripts/06_train_cpt*.sh`

当前仍以占位或半成品为主的部分：

- `scripts/03_build_sft_dataset.py`
- `scripts/05_tokenize_sft.py`
- `src/train/train_sft.py`
- `src/eval/*`
- `src/serve/*`
- `src/prof/*`
- `scripts/11_benchmark_serving.py`

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

## Recommended Workflow

建议按下面顺序推进，而不是直接跳到长训或部署：

1. 用 `scripts/00_check_env.py` 确认 Python / CUDA / PyTorch / DeepSpeed 基线。
2. 下载原始数据并生成 `cleaned`、`tokenized` CPT 数据。
3. 先跑 `smoke -> single_gpu -> ddp -> zero2` 的 `sanity` 验证。
4. 将训练配置拆成 `sanity / benchmark / optimized` 三类，不再混用。
5. 在固定有效全局 batch、相同步数和相同数据切片下做正式 benchmark。
6. 再推进 ZeRO 优化、SFT、评测、DeepSpeed inference / vLLM serving 和 profiling。

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
- `merge`、`serve` 与服务压测脚本默认依赖 SFT 或 merged 模型产物，因此在 SFT 链路完成前不能视为完整可用。
- 仓库当前主线仍是 `Qwen3-1.7B` 的 CPT / DeepSpeed 实验，不覆盖更大规模分布式训练的全部场景。

## Contributing

欢迎提交 issue 或 PR，但建议遵守以下约定：

- 先说明目标、范围和影响模块。
- 不要把未验证流程标记为“已完成”。
- 涉及实验、命令、调参、报错处理时，同步更新日志文件。
- 保持配置、脚本、报告和 README 之间的一致性。
