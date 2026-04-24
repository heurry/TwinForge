# Operation Log

记录规则：

- 每次代码、配置、文档、脚本、实验操作都要追加一条记录
- 每条记录至少包含日期、操作内容、影响文件、结果
- 一次操作失败时，这里记录动作摘要，详细异常写入 `logs/error_log.md`

## 2026-04-23

- 操作：启动项目实施阶段。
  影响文件：`reports/implementation_plan.md`
  结果：新增分阶段实施计划，明确目标、任务、交付物和验收标准。

- 操作：建立项目操作日志。
  影响文件：`logs/operation_log.md`
  结果：新增统一操作记录入口，后续所有实施动作在此追加。

- 操作：建立项目错误日志。
  影响文件：`logs/error_log.md`
  结果：新增统一异常记录入口，后续错误和阻塞在此追加。

- 操作：补充 README 中的实施计划与日志入口。
  影响文件：`README.md`
  结果：项目文档增加计划文件与日志文件索引。

- 操作：重构 README 信息架构，使其更符合开源项目首页风格。
  影响文件：`README.md`
  结果：重写项目概览、状态说明、功能矩阵、快速开始、路线图、贡献约定和文档入口，区分已实现能力与实施中能力。

- 操作：执行第一步环境基线检查。
  影响文件：`reports/baseline.md`
  结果：确认当前 `base` 环境缺少训练与数据链路关键依赖，`scripts/01_download_data.py` 与 `src/train/train_cpt.py` 均无法启动。

- 操作：检查现有 conda 环境可复用性。
  影响文件：`reports/baseline.md`
  结果：快速抽查 `nanovllm`、`deepseek`、`simlingo`，均不满足当前项目最小依赖集。

- 操作：检查 `/home/xdu/LLM/vllm-ai_infra/.venv` 是否可复用。
  影响文件：`reports/baseline.md`
  结果：确认用户给出的 `.enve` 路径不存在，实际 `.venv` 虽可用但依赖不全且未识别可用 CUDA，不适合直接复用。

- 操作：新增环境检查脚本。
  影响文件：`scripts/00_check_env.py`
  结果：提供可重复执行的环境基线检查入口，后续可用于复查依赖、GPU 和核心脚本启动状态。

- 操作：为 TwinForge 创建专用虚拟环境。
  影响文件：`.venv/`
  结果：使用 Python 3.10.18 在项目根目录创建独立运行环境，不再依赖 `base` 或其他项目环境。

- 操作：安装 TwinForge 运行依赖。
  影响文件：`.venv/`
  结果：完成 GPU 版 `torch/torchvision/torchaudio` 与 `requirements.txt` 依赖安装，`deepspeed` 与 `jieba` 构建成功。

- 操作：验证新环境可用性。
  影响文件：`reports/baseline.md`
  结果：确认 `.venv` 中关键依赖均可导入，`scripts/01_download_data.py --help` 与 `src/train/train_cpt.py --help` 均可正常启动。

- 操作：验证新环境的真实 CUDA 可见性。
  影响文件：`reports/baseline.md`
  结果：在非沙箱环境下确认 `torch.cuda.is_available()==True`，设备数为 2，GPU 型号为 `NVIDIA GeForce RTX 3090`。

- 操作：更新 Git 忽略规则。
  影响文件：`.gitignore`
  结果：新增 `.venv/`，避免将本地专用环境提交到仓库。

- 操作：修正 CPT 数据源配置。
  影响文件：`configs/dataset_manifest.json`
  结果：将 `MNBVC` 改为 `liwu/MNBVC` 仓库文件直拉模式，并将代码语料从 `bigcode/the-stack-v2` 调整为 `codeparrot/codeparrot-clean`。

- 操作：增强数据下载脚本以兼容 Hugging Face 仓库原始文件。
  影响文件：`scripts/01_download_data.py`
  结果：新增对 `hf_repo_files` 源类型的支持，可直接下载并解析 `.jsonl.gz` 文本文件。

- 操作：统一代码语料下载路径。
  影响文件：`scripts/01_download_data.py`、`configs/dataset_manifest.json`
  结果：将 `codeparrot/codeparrot-clean` 也切换为仓库文件直拉模式，避免 `load_dataset(..., streaming=True)` 在当前网络下的不稳定行为。

- 操作：同步更新项目文档与实施计划。
  影响文件：`README.md`、`reports/implementation_plan.md`
  结果：记录 `MNBVC` 的脚本仓库兼容问题及当前已采用的绕过方案。

- 操作：启动后台数据下载任务。
  影响文件：`data/raw/`、`configs/dataset_manifest.json`
  结果：已通过 `.venv/bin/python scripts/01_download_data.py --manifest configs/dataset_manifest.json --output_root data/raw --seed 42` 启动下载会话；当前会话已进入 `fineweb_sample` 下载阶段，`data/raw/cpt/fineweb_sample.jsonl` 已开始增长。

- 操作：完成 Phase 1 的 CPT 数据清洗、抽样和 packing 实现。
  影响文件：`scripts/02_build_cpt_corpus.py`、`scripts/04_tokenize_cpt.py`、`src/data/cleaners.py`、`src/data/samplers.py`、`src/data/packers.py`、`src/train/train_cpt.py`
  结果：补齐 `raw -> cleaned -> tokenized -> train` 主链路，新增清洗统计、语言 quota 分配、token packing 与训练入口的 tokenized 数据消费逻辑。

- 操作：同步更新 Phase 1 文档状态。
  影响文件：`README.md`、`reports/implementation_plan.md`
  结果：README 的功能矩阵、Quick Start 和项目状态已反映新的 CPT 数据链路；实施计划已补充 Phase 1 当前进展和验收说明。

- 操作：执行 Phase 1 小样本验收。
  影响文件：`data/cleaned/cpt/`、`data/tokenized/cpt/`
  结果：基于现有 `data/raw/cpt/fineweb_sample.jsonl` 成功生成 `data/cleaned/cpt/fineweb_sample.jsonl`、`data/cleaned/cpt/summary.json`、`data/tokenized/cpt/train` 与 `data/tokenized/cpt/summary.json`；并验证 tokenized dataset 可被 `DataCollatorForLanguageModeling` 直接消费。

- 操作：增强原始数据下载脚本以支持续跑。
  影响文件：`scripts/01_download_data.py`
  结果：新增 `--skip_existing` 参数；当目标输出文件已存在且非空时，脚本会跳过该数据集，避免重新覆盖已完成下载的数据。

- 操作：重新启动原始数据下载任务并进入续跑模式。
  影响文件：`data/raw/`、`logs/download_data_20260423_resume.log`
  结果：已通过 `.venv/bin/python scripts/01_download_data.py --manifest configs/dataset_manifest.json --output_root data/raw --seed 42 --skip_existing` 启动持久下载会话；当前已确认跳过 `fineweb_sample` 并开始下载 `mnbvc_gov_subset`。

- 操作：修正 SFT 下载慢路径与 Hugging Face Xet 下载问题。
  影响文件：`scripts/01_download_data.py`、`configs/dataset_manifest.json`
  结果：为 SFT 数据集增加流式抽样逻辑，并在下载脚本内默认禁用 `HF_HUB_DISABLE_XET=1`；`ultrachat_200k_mini` 与 `wildchat_mini` 已切换为流式下载配置。

- 操作：中断旧的慢速下载会话并按新逻辑续跑。
  影响文件：`data/raw/sft/`、`data/raw/eval/`
  结果：停止了原先对 `WildChat` 全量 parquet 分片的下载，随后重新启动会话；已确认 `wildchat_mini` 在新逻辑下成功产出 `10000` 条样本，并继续进入 `eval` 数据下载阶段。

- 操作：完成改造后下载会话的结果核对。
  影响文件：`data/raw/sft/wildchat_mini.jsonl`、`data/raw/eval/gsm8k.jsonl`、`data/raw/eval/mmlu_mini.jsonl`、`data/raw/eval/humaneval.jsonl`
  结果：`wildchat_mini` 已成功产出 `10000` 条样本；`gsm8k`、`mmlu_mini`、`humaneval` 已正常落盘；`mbpp` 因旧式 dataset script 兼容问题未成功下载，已单独记录到错误日志。

- 操作：为 `mbpp` 增加可下载方案并完成验收。
  影响文件：`scripts/01_download_data.py`、`configs/dataset_manifest.json`、`data/raw/eval/mbpp.jsonl`
  结果：将 `mbpp` 从旧式 dataset script 路径切换为 Hugging Face 仓库原始文件 `data/mbpp.jsonl` 直拉模式；重新执行下载后成功产出 `974` 条评测样本。

- 操作：将训练启动脚本切换为 DeepSpeed launcher。
  影响文件：`scripts/06_train_cpt.sh`、`scripts/07_train_sft_lora.sh`
  结果：将原先的 `torchrun --nproc_per_node=2` 改为 `deepspeed --num_gpus=2` 启动方式，并优先使用仓库内 `.venv/bin/deepspeed`，保证 CPT 与 SFT 入口风格一致。

- 操作：统一模型配置到 `Qwen3-1.7B`。
  影响文件：`configs/model/qwen3_1_7b_base.yaml`、`configs/train/cpt.yaml`、`configs/train/sft_lora.yaml`、`configs/serve/vllm.yaml`、`configs/dataset_manifest.json`、`scripts/06_train_cpt.sh`、`scripts/07_train_sft_lora.sh`、`scripts/08_merge_lora.py`、`scripts/09_eval_all.sh`、`scripts/10_serve_vllm.sh`、`README.md`
  结果：删除旧的模型配置文件，统一改为本地 `model/Qwen3-1.7B`，并同步更新训练输出目录、merge/eval/serve 路径和文档示例。

- 操作：基于完整 CPT 原始数据重建 `cleaned` 与 `tokenized` 产物。
  影响文件：`data/cleaned/cpt/`、`data/tokenized/cpt/`
  结果：重新执行 `scripts/02_build_cpt_corpus.py --overwrite` 与 `scripts/04_tokenize_cpt.py --overwrite`；`data/cleaned/cpt/summary.json` 现已覆盖英文、中文、代码三路数据，`data/tokenized/cpt/summary.json` 已更新为 `model/Qwen3-1.7B` tokenizer，并生成 `49285` 条长度为 `2048` 的训练样本。

- 操作：落地 CPT 分层训练配置与启动脚本。
  影响文件：`src/train/train_cpt.py`、`configs/train/cpt.yaml`、`configs/train/cpt_smoke.yaml`、`configs/train/cpt_single_gpu.yaml`、`configs/train/cpt_ddp.yaml`、`configs/train/cpt_zero2.yaml`、`scripts/06_train_cpt.sh`、`scripts/06_train_cpt_smoke.sh`、`scripts/06_train_cpt_single_gpu.sh`、`scripts/06_train_cpt_ddp.sh`、`scripts/06_train_cpt_zero2.sh`、`README.md`、`reports/implementation_plan.md`
  结果：新增 `smoke / single_gpu / ddp / zero2` 四档 CPT 训练入口；保留 `scripts/06_train_cpt.sh` 作为默认 `zero2` 兼容脚本；并将训练覆盖策略、checkpoint 保留数量、DDP 未使用参数检查和 gradient checkpointing 切换改为由训练配置控制。

- 操作：修正 CPT 训练入口对 `transformers 5.6.1` 的兼容性与精度探测逻辑。
  影响文件：`src/train/train_cpt.py`、`configs/model/qwen3_1_7b_base.yaml`、`configs/train/cpt_smoke.yaml`、`configs/train/cpt_single_gpu.yaml`、`configs/train/cpt_ddp.yaml`
  结果：将模型加载改为按 checkpoint dtype 自动解析，兼容 `TrainingArguments` / `Trainer` 新签名，改用 `warmup_steps`，并在运行时根据真实 CUDA 能力自动降级 `bf16/fp16/tf32`，避免在参数初始化阶段因精度或 API 变更直接报错。

## 2026-04-24

- 操作：完成 CPT `single_gpu` 基线训练验收。
  影响文件：`runs/cpt/qwen3_1_7b_miniv2_cpt_single_gpu/`、`reports/implementation_plan.md`
  结果：执行 `bash scripts/06_train_cpt_single_gpu.sh` 成功完成 `200` steps 单卡训练；最终 `train_runtime` 约 `1234s`、`train_loss` 约 `2.19`，并确认输出目录已包含最终模型、`train_log_history.json`、TensorBoard 日志与 `checkpoint-100` / `checkpoint-200`。

- 操作：完成 CPT `ddp` 基线训练验收并修正多 rank 最终保存逻辑。
  影响文件：`src/train/train_cpt.py`、`runs/cpt/qwen3_1_7b_miniv2_cpt_ddp/`、`reports/implementation_plan.md`
  结果：执行 `bash scripts/06_train_cpt_ddp.sh` 成功完成 `300` steps 双卡 DDP 训练；最终 `train_runtime` 约 `1932s`、`train_loss` 约 `2.153`，并确认输出目录已包含最终模型、`train_log_history.json`、TensorBoard 日志与 `checkpoint-100` / `checkpoint-200` / `checkpoint-300`。同时将 `train_cpt.py` 调整为仅主进程写最终模型、tokenizer 与训练日志，避免多 rank 并发保存同一产物。

- 操作：复验 CPT `single_gpu` 训练并确认显存日志回调生效。
  影响文件：`logs/operation_log.md`、`runs/cpt/qwen3_1_7b_miniv2_cpt_single_gpu/`
  结果：执行 `bash scripts/06_train_cpt_single_gpu.sh` 成功完成 `200` steps 单卡训练；最终 `train_runtime=1132s`、`train_loss=2.191`、`train_steps_per_second=0.177`。`train_log_history.json` 已按 `10` steps 间隔记录 `cuda_memory_allocated_mb`、`cuda_memory_reserved_mb`、`cuda_memory_max_allocated_mb`，本次运行样本中分别约为 `9862.53 MB`、`17934.0 MB`、`16965.3 MB`；输出目录已包含最终模型、`train_log_history.json`、日志目录与 `checkpoint-100` / `checkpoint-200`。

- 操作：复验 CPT `ddp` 训练并确认显存日志回调生效。
  影响文件：`logs/operation_log.md`、`runs/cpt/qwen3_1_7b_miniv2_cpt_ddp/`
  结果：执行 `bash scripts/06_train_cpt_ddp.sh` 成功完成 `300` steps 双卡 DDP 训练；最终 `train_runtime=1896s`、`train_loss=2.153`、`train_steps_per_second=0.158`。`train_log_history.json` 已按 `10` steps 间隔记录 `cuda_memory_allocated_mb`、`cuda_memory_reserved_mb`、`cuda_memory_max_allocated_mb`，本次运行样本中分别约为 `13144.27 MB`、`21806.0 MB`、`20247.54 MB`；输出目录已包含最终模型、`train_log_history.json`、日志目录与 `checkpoint-100` / `checkpoint-200` / `checkpoint-300`。

- 操作：基于官方 DeepSpeed 文档重构实施计划。
  影响文件：`reports/implementation_plan.md`、`logs/operation_log.md`
  结果：参考 `Getting Started`、`Config JSON`、`ZeRO Tutorial` 与 `Inference Tutorial`，将计划从“按阶段补骨架”改为“sanity / benchmark / optimized / serving”四层实验体系；明确当前 `single_gpu` 与 `ddp` 结果仅作为工程 sanity，不再作为性能结论；新增受控 benchmark、ZeRO 优化路径以及 `HF eager / DeepSpeed Inference / vLLM` 推理对比要求，确保后续训练、评测、部署与压测形成可比较的完整闭环。

- 操作：继续基于官方 DeepSpeed Tutorials 细化实施计划并加入能力矩阵。
  影响文件：`reports/implementation_plan.md`、`logs/operation_log.md`
  结果：进一步阅读 `Tutorials Index`、`AutoTP Training`、`Automatic Tensor Parallelism`、`Pipeline Parallelism`、`Ulysses / ALST`、`Flops Profiler`、`PyTorch Profiler`、`Autotuning`、`Monitoring`、`Communication Logging`、`Universal Checkpointing`、`ZeRO-Offload` 与 `ZeRO++`，将计划细化为“能力筛选 + 分轨实施”版本；新增双卡 `3090` 场景下的可行性矩阵，明确 `ZeRO / DeepSpeed Inference / AutoTP inference / Profiler / Monitor / Comms` 为近期主线，`AutoTP training` 为独立 POC，`Pipeline` 与 `Ulysses` 为后置高级探索，`ZeRO++` 暂不进入近期计划。

- 操作：更新 README 并补充仓库忽略规则，准备发布当前源码状态。
  影响文件：`README.md`、`.gitignore`、`logs/operation_log.md`
  结果：README 已改为反映当前真实项目状态、训练验证结果、DeepSpeed 主线能力和实施路线；同时将 `data/`、`model/`、`runs/`、下载日志与临时文件加入忽略规则，避免把本地数据、权重和训练产物错误推送到 GitHub。

- 操作：修正 GitHub 远端并重新推送当前项目源码。
  影响文件：`logs/operation_log.md`
  结果：将 `origin` 从错误的 `TwinForge-.git` 修正为 `git@github.com:heurry/TwinForge.git`，随后提交当前源码快照 `61e5376`（`Build CPT pipeline and refresh project docs`）并成功推送到远端 `main` 分支；本地数据、模型目录和训练产物未纳入提交范围。
