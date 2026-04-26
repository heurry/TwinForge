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

- 操作：落地 Phase 0 与 Phase 2/3 的训练协议、sanity/benchmark 结构和对比脚手架。
  影响文件：`src/train/train_cpt.py`、`configs/train/sanity/`、`configs/train/bench/`、`configs/train/ds/`、`scripts/train/`、`scripts/04_build_cpt_benchmark_slice.py`、`scripts/12_report_cpt_benchmark.py`、`scripts/06_train_cpt*.sh`、`reports/implementation_plan.md`、`reports/cpt_sanity.md`、`reports/cpt_benchmark.md`、`README.md`
  结果：新增 `sanity` 与 `bench` 两套 CPT 配置和启动脚本，补齐 `zero3_offload` sanity/benchmark 入口，并将旧的 `06_train_cpt*.sh` 改为兼容包装层；`train_cpt.py` 新增 `train_summary.json` 输出，统一记录 `train_tokens_per_second`、checkpoint 保存耗时/大小和 `resume_success`；新增固定 benchmark 切片脚本并已基于 `data/tokenized/cpt/train` 生成 `1024` 条样本的 `data/tokenized/cpt/bench`；新增 benchmark 汇总脚本与报告占位文件，为后续正式对比提供直接入口。

- 操作：补齐 Phase 1 的 benchmark slice 复现契约。
  影响文件：`scripts/04_build_cpt_benchmark_slice.py`、`configs/data/cpt_benchmark_slice.yaml`、`reports/implementation_plan.md`、`README.md`、`logs/operation_log.md`
  结果：将 benchmark 切片流程从“固定前 `1024` 条样本”升级为“spec + indices + fingerprint”模式；新增 `configs/data/cpt_benchmark_slice.yaml` 统一管理切片规则，切片脚本现可同步输出 `bench_summary.json` 与 `bench_indices.json`，并在 summary 中记录数据集 fingerprint、tokenizer 路径和 tokenizer 指纹，确保后续 `single_gpu / ddp / zero / tp` benchmark 可追溯到同一份数据视图和同一套 tokenizer 元数据。

- 操作：修正 CPT `zero2 / zero3_offload` 的 DeepSpeed benchmark 稳定性配置。
  影响文件：`configs/train/ds/ds_zero2_bench.json`、`configs/train/ds/ds_zero2_sanity.json`、`configs/train/ds/ds_zero3_offload_bench.json`、`configs/train/ds/ds_zero3_offload_sanity.json`、`scripts/train/bench/cpt_zero2.sh`、`scripts/train/sanity/cpt_zero2.sh`、`scripts/train/bench/cpt_zero3_offload.sh`、`scripts/train/sanity/cpt_zero3_offload.sh`
  结果：针对双卡 `3090` 上 `zero2` benchmark 首步反向传播因 `200000000` bucket 配置触发约 `762 MiB` 通信张量额外分配而 OOM 的问题，将 ZeRO 配置改为与 Hugging Face `Trainer` 自动同步的 `"auto"` micro-batch / grad accumulation / clipping，并将 `zero2` 的 `reduce_bucket_size` 和 `allgather_bucket_size` 下调到 `50000000`、关闭 `overlap_comm`；同时将 `zero3_offload` 改为 `"auto"` bucket 参数并关闭 `overlap_comm`。DeepSpeed 启动脚本也去掉了多余的 `--num_gpus`，避免与 `CUDA_VISIBLE_DEVICES` 重复指定时出现无意义警告。

- 操作：将 CPT 正式 benchmark family 从 `tf32` 调整为 `fp16`。
  影响文件：`configs/train/bench/cpt_bench_single.yaml`、`configs/train/bench/cpt_bench_ddp.yaml`、`configs/train/bench/cpt_bench_zero2.yaml`、`configs/train/bench/cpt_bench_zero3_offload.yaml`、`reports/implementation_plan.md`、`logs/operation_log.md`
  结果：在双卡 `3090` 上验证发现，`ZeRO-2` 即使收缩 bucket 参数，在 `tf32/float32` 路线下仍会于 optimizer step 为 FP32 梯度分区额外申请约 `3.21 GiB` 显存，导致 benchmark family 无法完整覆盖 `single / ddp / zero2 / zero3_offload`。因此将整组 `bench` 统一切换为 `fp16=true, tf32=false`，并同步将 `benchmark_group` 重命名为 `cpt_bench_core_batch16_fp16`，保证这四个后端处于同一组真正可执行、可比较的协议下。

- 操作：修正 `fp16` benchmark 配置下模型仍按 `bfloat16` 加载导致的单卡训练报错。
  影响文件：`src/train/train_cpt.py`、`logs/error_log.md`
  结果：用户在执行 `CUDA_VISIBLE_DEVICES=0 bash scripts/train/bench/cpt_single.sh` 时，于首个 step 的梯度裁剪阶段触发 `NotImplementedError: _amp_foreach_non_finite_check_and_unscale_cuda not implemented for 'BFloat16'`。根因是 benchmark YAML 已切到 `fp16=true`，但模型仍按 checkpoint 默认 `bfloat16` 加载，导致 `GradScaler` 试图对 `bf16` 梯度做 `unscale`。现已将 `train_cpt.py` 改为根据训练精度显式选择模型加载 dtype：`bf16 -> torch.bfloat16`、`fp16 -> torch.float16`、否则再回退到模型配置默认 dtype。

- 操作：修正上一轮单卡 `fp16` bench 修复方向错误的问题。
  影响文件：`src/train/train_cpt.py`、`logs/error_log.md`
  结果：用户按上一轮修复重跑 `CUDA_VISIBLE_DEVICES=0 bash scripts/train/bench/cpt_single.sh` 后，又在 `GradScaler.unscale_` 阶段触发 `ValueError: Attempting to unscale FP16 gradients.`。这说明原生 Hugging Face `Trainer` 的 `fp16` 路径不能直接把模型参数加载为 `fp16`，而必须保留 `fp32` 参数并依赖 `autocast + GradScaler`。现已将 `train_cpt.py` 的模型 dtype 解析进一步修正为：DeepSpeed 混精才显式按 `bf16/fp16` 加载，single/DDP 的原生 Trainer 混精统一按 `fp32` 参数加载。

- 操作：确认当前统一 `fp16` benchmark family 与原生单卡全参数训练的显存边界冲突。
  影响文件：`logs/error_log.md`
  结果：用户在完成 single 路径精度修复后再次执行 `CUDA_VISIBLE_DEVICES=0 bash scripts/train/bench/cpt_single.sh`，训练已不再报精度 API 错误，但在第一个 optimizer step 初始化 Adam `exp_avg` 时因仅剩约 `42.62 MiB` 空闲显存而 OOM。该现象表明：在单张 `24 GB` `3090` 上，原生 Hugging Face `Trainer` 的 `fp16` 路径虽然实现正确，但其 `fp32` 参数 + 梯度 + Adam 状态的显存账本已经超出硬件上限。由此确认问题已从“训练代码 bug”转为“benchmark 设计冲突”，后续应拆分 native family 与 DeepSpeed family，而不是继续强推单卡/ZeRO 共用同一 `fp16` benchmark 协议。

- 操作：将 CPT benchmark 正式拆分为 `Native Family` 与 `DeepSpeed Family`。
  影响文件：`configs/train/bench/`、`scripts/train/bench/`、`scripts/12_report_cpt_benchmark.py`、`reports/cpt_benchmark.md`、`README.md`、`reports/implementation_plan.md`
  结果：删除旧的 `scripts/train/bench/cpt_*.sh` 与顶层 `configs/train/bench/cpt_bench_*.yaml`，改为 `configs/train/bench/native/`、`configs/train/bench/deepspeed/` 以及对应的 `scripts/train/bench/native/`、`scripts/train/bench/deepspeed/` 两组显式入口；native family 固定为 `single + ddp + tf32 + effective batch 8`，DeepSpeed family 固定为 `zero2 + zero3_offload + fp16 + effective batch 16`。同时重写 benchmark 报告脚本为双章节输出，并引入 `native_ddp_bridge_ref` 作为跨 family 参考行，README 与实施计划已同步改为“native 内部比较、DeepSpeed 内部比较、bridge 仅作参考”的结构。

- 操作：修正 `ZeRO-3 offload` benchmark 在最终模型保存阶段的多 rank 同步错误。
  影响文件：`src/train/train_cpt.py`、`logs/error_log.md`
  结果：根据用户提供的 `bash scripts/train/bench/deepspeed/cpt_zero3_offload.sh` 运行日志确认，训练 `100/100` steps 已完成，失败点并非训练本体，而是结束后仅由 `world_process_zero` 调用 `trainer.save_model(output_dir)`，导致 rank0 进入 `DeepSpeed ZeRO-3` 的 `_zero3_consolidated_16bit_state_dict()` 参数聚合，而 rank1 提前进入 `accelerator.wait_for_everyone()`，最终触发 NCCL watchdog 超时。现已将最终保存逻辑改为：启用 DeepSpeed 时所有 rank 都进入 `trainer.save_model(output_dir)`，仅主进程负责实际写文件；非 DeepSpeed 路径仍保持主进程单独保存。`python -m py_compile src/train/train_cpt.py` 已通过，待用户重跑 `scripts/train/bench/deepspeed/cpt_zero3_offload.sh` 验证。

- 操作：增强 CPT benchmark 报告生成脚本，自动输出结论性分析段。
  影响文件：`scripts/12_report_cpt_benchmark.py`、`reports/cpt_benchmark.md`
  结果：在保留 family 分表结构的基础上，为报告新增 `Analysis` 章节，自动根据 `train_summary.json` 计算 `Native Family` 的 `single vs ddp` 吞吐/显存变化，以及 `DeepSpeed Family` 的 `zero2 vs zero3_offload` 吞吐/显存变化，并在末尾补充 `cross-family` 解释说明。随后重新生成 `reports/cpt_benchmark.md`，使报告从“表格展示”升级为“表格 + 自动结论”。

- 操作：落地 CPT Phase 4 的 profiling / communication / monitoring 脚手架。
  影响文件：`src/train/callbacks.py`、`src/train/train_cpt.py`、`src/prof/torch_profile.py`、`configs/train/profile/`、`configs/train/ds/ds_zero2_profile.json`、`scripts/train/profile/`、`scripts/13_report_cpt_profile.py`、`scripts/14_report_cpt_comms.py`、`reports/cpt_profile.md`、`reports/cpt_comms.md`、`README.md`、`reports/implementation_plan.md`
  结果：在现有 `train_cpt.py` 路径上新增可选 `TorchProfilerCallback` 与 DeepSpeed comms summary 导出逻辑，不再额外分叉训练主脚本；新增 `native single / native ddp / deepspeed zero2` 三条短程 profile 配置与启动脚本；为 `zero2` 增加启用 `flops_profiler`、`comms_logger`、`tensorboard`、`csv_monitor` 的 DeepSpeed profile 配置；新增两条报告生成脚本，可从 profile 产物自动生成 `reports/cpt_profile.md` 与 `reports/cpt_comms.md`。静态校验已通过：相关 Python 文件 `py_compile` 通过、profile shell 脚本 `bash -n` 通过、两条报告脚本可在无 profile 产物时生成占位报告。

- 操作：为 CPT 训练新增“只记录训练过程、不保存模型产物”的显式开关。
  影响文件：`src/train/train_cpt.py`、`configs/train/profile/native/cpt_profile_native_single.yaml`、`configs/train/profile/native/cpt_profile_native_ddp.yaml`、`configs/train/profile/deepspeed/cpt_profile_zero2.yaml`
  结果：`train_cpt.py` 现支持训练配置项 `save_model_artifacts`；当其为 `false` 时，训练过程中不再保存 checkpoint，也不会在训练结束时保存最终模型与 tokenizer，但仍会正常输出 `train_log_history.json`、`train_summary.json`、profiling/comms 产物和控制台日志。当前三条 Phase 4 profile 配置已默认设置为 `save_model_artifacts: false`，避免 profiling run 因最终保存模型额外消耗时间和磁盘空间。

- 操作：补齐 `zero3_offload` 的 Phase 4 profiling 覆盖。
  影响文件：`configs/train/profile/deepspeed/cpt_profile_zero3_offload.yaml`、`configs/train/ds/ds_zero3_offload_profile.json`、`scripts/train/profile/deepspeed/cpt_zero3_offload.sh`、`scripts/13_report_cpt_profile.py`、`reports/cpt_profile.md`、`README.md`、`reports/implementation_plan.md`
  结果：新增 `zero3_offload` 的 profile 配置、DeepSpeed profile JSON 和启动脚本，并将其纳入默认 profiling 报告输入；同时为该配置默认开启 `torch_profiler` 与 `deepspeed_comms_logging`，并设置 `save_model_artifacts: false`、`stage3_gather_16bit_weights_on_model_save: false`，避免 profile run 在最终保存阶段做不必要的权重聚合与模型写盘。相关静态校验已通过，`reports/cpt_profile.md` 也已更新为包含 `zero3_offload` 占位行的四后端版本。

- 操作：新增一键串行启动 Phase 4 profiling 的总控脚本。
  影响文件：`scripts/train/profile/run_all.sh`、`README.md`
  结果：新增 `scripts/train/profile/run_all.sh`，默认按 `native_single -> native_ddp -> zero2 -> zero3_offload -> 生成 cpt_profile.md -> 生成 cpt_comms.md` 的顺序串行执行当前全部 profile 入口，不主动清理历史输出目录；README 的 `Run Profiling` 章节已同步加入一键启动命令。脚本已完成 `chmod +x` 和 `bash -n` 校验。

- 操作：稳定化 `zero3_offload` 的 profiling 配置，并增强总控脚本的容错能力。
  影响文件：`configs/train/profile/deepspeed/cpt_profile_zero3_offload.yaml`、`configs/train/ds/ds_zero3_offload_profile.json`、`src/train/callbacks.py`、`scripts/train/profile/run_all.sh`、`logs/error_log.md`
  结果：根据用户在 `bash scripts/train/profile/run_all.sh` 中的真实终端日志，确认 `zero3_offload` 不是 GPU OOM，而是 `ZeRO-3 CPU offload + torch profiler + DeepSpeed flops profiler + comms/wall_clock` 叠加过重导致的系统级 `SIGKILL`。现已将 `zero3_offload` profile 收敛为 lite 模式：`max_steps=4`、`dataloader_num_workers=0`、torch profiler 仅保留 `wait=1/warmup=1/active=1` 的轻量 key averages，不再记录 `record_shapes/profile_memory/flops/chrome trace`；DeepSpeed 侧关闭 `flops_profiler`、`tensorboard`、`csv_monitor`，并关闭 CPU offload 的 `pin_memory`。同时将 `TorchProfilerCallback` 改为只在 `world_process_zero` 启动 profiler，并在当前 PyTorch 支持时启用 `acc_events=True`；`scripts/train/profile/run_all.sh` 现即使某一条 profile 失败，也会继续生成 `reports/cpt_profile.md` 与 `reports/cpt_comms.md`，最后再汇总失败步骤并以非零状态退出。

- 操作：将 `zero3_offload` profile 从“lite torch profiler”进一步降级为“仅保留 DeepSpeed comms/wall_clock”。
  影响文件：`configs/train/profile/deepspeed/cpt_profile_zero3_offload.yaml`、`scripts/14_report_cpt_comms.py`、`logs/error_log.md`
  结果：在用户单独执行 `bash scripts/train/profile/deepspeed/cpt_zero3_offload.sh` 的复验中，确认即使 torch profiler 已极度收缩，`zero3_offload` 仍会在第三个 step 左右被系统 `SIGKILL`，且输出目录只有 TensorBoard event，没有 `train_summary.json` 或 profiler 导出文件。由此进一步收敛为：当前机器上的 `zero3_offload` profiling 不适合再启用 torch profiler。现已在 `cpt_profile_zero3_offload.yaml` 中将 `torch_profiler.enabled` 直接关闭，仅保留 DeepSpeed `wall_clock_breakdown + comms_logger`，并同步扩展 `scripts/14_report_cpt_comms.py`，使后续成功跑完后能把 `zero3_offload` 的 comms summary 纳入 `reports/cpt_comms.md`。

- 操作：清理 `runs/` 下的 CPT benchmark 模型与 checkpoint 权重，仅保留分析产物。
  影响文件：`runs/cpt/bench/native/qwen3_1_7b/single/`、`runs/cpt/bench/native/qwen3_1_7b/ddp/`、`runs/cpt/bench/deepspeed/qwen3_1_7b/zero2/`、`runs/cpt/bench/deepspeed/qwen3_1_7b/zero3_offload/`
  结果：按“只保留分析内容”的口径，删除了四个 benchmark run 目录中的全部 `checkpoint-*` 目录，以及根目录下的最终模型与训练产物文件，包括 `model.safetensors`、`config.json`、`generation_config.json`、`tokenizer*.json`、`chat_template.jinja`、`training_args.bin` 等；保留了 `train_summary.json`、`train_log_history.json`、`logs/` 事件文件及 profiling 目录。`runs/cpt` 体积已从清理前约 `156G` 降到约 `6.0G`，释放空间约 `150G`。

- 操作：为 profiling 与 communication 报告脚本新增自动 `Analysis` 段并重生成报告。
  影响文件：`scripts/13_report_cpt_profile.py`、`scripts/14_report_cpt_comms.py`、`reports/cpt_profile.md`、`reports/cpt_comms.md`
  结果：将本轮 profile/comms 的关键结论写回报告生成器，而不是手工编辑 markdown。`cpt_profile.md` 现在会自动生成 `Native family`、`DeepSpeed family` 与 `Interpretation` 三类分析，概括 `ddp vs single` 的 launch/sync 特征以及 `zero2 vs zero3_offload` 的吞吐/显存取舍；`cpt_comms.md` 则会自动生成 `DDP`、`ZeRO-2`、`ZeRO-3 offload` 的通信形态分析，明确 `zero3_offload` 主要由 `all_gather_into_tensor + reduce_scatter_tensor` 主导。相关脚本已通过 `py_compile`，并已重新生成两份报告。

- 操作：落地 Phase 5 的第一版 `Optimization Track` 脚手架。
  影响文件：`src/train/train_cpt.py`、`configs/train/optimize/`、`scripts/train/optimize/`、`scripts/15_report_cpt_optimization.py`、`reports/cpt_optimization.md`、`README.md`、`reports/implementation_plan.md`
  结果：围绕已经验证有价值的两条主线 `native ddp` 与 `zero2`，新增了第一版优化 sweep：`base / workers4 / no_gc` 三类变体，统一默认 `save_model_artifacts: false`，只保留训练分析产物；同时新增串行总控脚本 `scripts/train/optimize/run_all.sh` 和自动汇总脚本 `scripts/15_report_cpt_optimization.py`。为支撑优化报告，`train_summary.json` 现在额外记录 `dataloader_num_workers`、`gradient_checkpointing` 与 `max_seq_length`。第一版 sweep 当前只覆盖 `train_cpt.py` 已稳定接入且可直接控制的旋钮：`dataloader_num_workers` 与 `gradient_checkpointing`；`max_seq_length` 与更激进的 batch-density sweep 暂未自动化，因为现有 `bench` tokenized slice 固定在 `2048`。相关 Python 脚本已通过 `py_compile`，shell 脚本已通过 `bash -n`，并已成功生成占位版 `reports/cpt_optimization.md`。

- 操作：完成第一轮 `Optimization Track` 实跑并修正优化 runner/report 的失败状态记录。
  影响文件：`runs/cpt/optimize/`、`reports/cpt_optimization.md`、`scripts/train/optimize/run_all.sh`、`scripts/15_report_cpt_optimization.py`、`logs/error_log.md`
  结果：用户实际执行 `bash scripts/train/optimize/run_all.sh` 后，确认 `native ddp` 路线中 `ddp_workers4` 优于 `ddp_base`，吞吐从约 `5148.6` 提升到约 `5342.2 tokens/s`，提升约 `3.8%`，峰值显存基本不变；`zero2` 路线中 `zero2_workers4` 略低于 `zero2_base`，吞吐从约 `2624.4` 下降到约 `2603.9 tokens/s`，因此当前仍推荐 `zero2_base`。`native_ddp_no_gc` 与 `zero2_no_gc` 均在本轮 sweep 中因 `torch.OutOfMemoryError` 失败，说明关闭 gradient checkpointing 在当前双卡 `3090` 上不可行。针对本次运行暴露的问题，已修复 `scripts/train/optimize/run_all.sh` 中失败步骤被误记为 `exit code 0` 的 bug，并新增 `runs/cpt/optimize/run_status.tsv` 作为状态元数据；`scripts/15_report_cpt_optimization.py` 现会读取该状态文件，将失败变体显示为 `failed` 而不再混入 `missing`，并在报告中补充 `Exit Code` 列。

- 操作：将 `Optimization Track` 的两条推荐路线固化为正式长训配置和启动脚本。
  影响文件：`configs/train/longtrain/native/cpt_longtrain_native_ddp.yaml`、`configs/train/longtrain/deepspeed/cpt_longtrain_zero2.yaml`、`scripts/train/longtrain/native/cpt_ddp.sh`、`scripts/train/longtrain/deepspeed/cpt_zero2.sh`、`README.md`、`reports/implementation_plan.md`
  结果：新增 `longtrain/` 分轨目录，把当前最优候选 `ddp_workers4` 与 `zero2_base` 固化为正式长训入口。两条配置均切回全量数据 `data/tokenized/cpt/train`，恢复 `save_model_artifacts: true`，默认 `max_steps=3000`、`save_steps=500`，用于后续正式 CPT 长训与中断恢复验证；README 与实施计划已同步加入新入口和推荐说明。

- 操作：调整当前阶段优先级，明确“链路跑通 + 参数搜索”高于正式长训。
  影响文件：`README.md`、`reports/implementation_plan.md`
  结果：将当前阶段目标改为先打通 `CPT 推荐配置 -> resume -> SFT / LoRA -> eval -> serving` 整条链路，并继续完成受控参数搜索；`longtrain` 入口保留，但明确标记为后置项，不再作为近期默认执行步骤。

- 操作：新增短程 `resume validation` 轨道，用于验证推荐训练配置的 checkpoint 可恢复性。
  影响文件：`src/train/train_cpt.py`、`configs/train/resume/`、`scripts/train/resume/`、`scripts/16_report_cpt_resume_validation.py`、`README.md`、`reports/implementation_plan.md`
  结果：为 `native ddp` 与 `zero2` 两条推荐配置新增 `stage1/stage2` 短程 resume 配置和验证脚本，固定执行 `step 10 -> checkpoint-10 -> resume -> step 12 -> checkpoint-12`；同时新增 `reports/cpt_resume_validation.md` 的自动生成脚本。`train_summary.json` 现额外记录 `global_step`，便于 resume 报告核验 stage1/stage2 的步数推进。

- 操作：补齐“先闭合 SFT 链路，再考虑长训”的最小实现路径。
  影响文件：`src/data/sft.py`、`scripts/03_build_sft_dataset.py`、`scripts/05_tokenize_sft.py`、`src/train/train_sft.py`、`configs/train/sft_lora.yaml`、`configs/train/ds/ds_zero2_sft.json`、`scripts/07_train_sft_lora.sh`、`scripts/08_merge_lora.py`、`src/eval/eval_gsm8k.py`、`src/eval/eval_mmlu.py`、`src/eval/aggregate.py`、`scripts/09_eval_all.sh`、`src/serve/hf_generate.py`、`src/serve/openai_client.py`、`src/serve/vllm_generate.py`、`configs/serve/vllm.yaml`、`scripts/10_serve_vllm.sh`、`scripts/11_benchmark_serving.py`、`README.md`、`reports/implementation_plan.md`
  结果：实现了最小 `resume -> SFT / LoRA -> merge -> eval -> vLLM serving -> serving benchmark` 代码链路。SFT 侧新增 cleaned/tokenized 构建与 assistant-only label masking，训练侧补齐了与 `train_cpt.py` 风格一致的 `train_sft.py` 和 LoRA + ZeRO-2 配置，merge 脚本改为显式 CLI；评测侧实现 `GSM8K + MMLU_mini` 最小闭环和 `runs/eval/summary.json` 聚合；服务侧实现了 `vLLM` 服务脚本、OpenAI-compatible 客户端与最小压测脚本。当前已完成 `py_compile` / `bash -n` 级静态校验，下一步重点是对这条链路做一次真实端到端实跑，而不是继续进入正式长训或 TP / PP。

- 操作：修复 `05_tokenize_sft.py` 在首次实跑中暴露的 `BatchEncoding -> token ids` 提取错误，并同步消除 assistant masking 的低效实现。
  影响文件：`src/data/sft.py`、`logs/error_log.md`
  结果：将 `tokenize_sft_messages()` 改为显式从 `BatchEncoding` 中提取 `input_ids`，避免 `list(BatchEncoding)` 误得到键名序列；同时把 assistant-only label 的构造改成“整条对话只 tokenize 一次 + 通过 `offset_mapping` 与 chat template 渲染区间做 span 标注”，不再对每个 assistant turn 反复执行整段 tokenization。随后用用户原始命令完整复跑 `scripts/05_tokenize_sft.py`，成功生成 `data/tokenized/sft/train`、`data/tokenized/sft/val` 与 `data/tokenized/sft/summary.json`；最终统计为 `train=17121`、`val=359` 条 packed sequence，速度恢复到约 `240-255 examples/s`。

- 操作：修正 `scripts/10_serve_vllm.sh` 在未安装 `vllm` 时只报 shell `command not found` 的问题。
  影响文件：`scripts/10_serve_vllm.sh`、`logs/error_log.md`
  结果：将 vLLM 启动脚本改为优先使用 `.venv/bin/vllm`，其次在 module 已安装但 CLI 缺失时回退到 `python -m vllm.entrypoints.openai.api_server`，若当前虚拟环境完全未安装 `vllm`，则显式输出安装提示 `.venv/bin/pip install vllm`，不再让用户看到模糊的 `vllm: 未找到命令`。同时补上 `trust_remote_code` 配置向服务命令的透传。

- 操作：增强 `scripts/10_serve_vllm.sh` 的端口冲突处理。
  影响文件：`scripts/10_serve_vllm.sh`
  结果：脚本现支持显式解析 `--host`、`--port`、`--served-model-name`、`--api-key` 覆盖项，并在真正启动 `vllm serve` 前先用 Python 做一次本地 bind 预检查。若端口已被占用，会直接输出清晰错误和可执行示例 `bash scripts/10_serve_vllm.sh <model> --port 8001`，不再把用户抛进 vLLM 内部的长 traceback。
