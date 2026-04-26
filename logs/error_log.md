# Error Log

记录规则：

- 每次命令失败、脚本异常、训练报错、数据源不可用、环境阻塞都要追加记录
- 每条记录至少包含日期、阶段、现象、初步原因、处理动作、当前状态
- 已解决问题标记为 `resolved`，未解决问题标记为 `open`

## 2026-04-23

- 当前状态：暂无项目实施阶段错误记录。
  阶段：初始化
  现象：无
  初步原因：无
  处理动作：创建错误日志文件，等待后续追加。
  状态：resolved

- 当前状态：核心项目环境未就绪。
  阶段：Phase 0 / Environment Baseline
  现象：当前 `base` 环境缺少 `torch`、`datasets`、`accelerate`、`deepspeed`、`peft`、`trl`、`jsonlines`，`scripts/01_download_data.py` 和 `src/train/train_cpt.py` 无法启动。
  初步原因：当前 Python 解释器不是为 TwinForge 单独准备的训练环境。
  处理动作：记录基线结果，新增 `scripts/00_check_env.py`，随后在项目根目录创建 `.venv` 并完成依赖安装。
  状态：resolved

- 当前状态：`/home/xdu/LLM/vllm-ai_infra/.venv` 不能直接复用。
  阶段：Phase 0 / Environment Baseline
  现象：用户给出的 `.enve` 路径不存在；实际 `.venv` 中缺少 `datasets`、`accelerate`、`deepspeed`、`peft`、`trl`、`jsonlines`，且 `torch` 未识别可用 CUDA 设备。
  初步原因：该环境是另一个项目的运行环境，不是针对 TwinForge 训练链路准备的完整依赖集合。
  处理动作：将其判定为不可直接复用，保留为依赖参考环境。
  状态：resolved

- 当前状态：沙箱内 CUDA 自检结果与真实终端不一致。
  阶段：Phase 0 / Environment Validation
  现象：在受限沙箱内执行 `torch.cuda.is_available()` 返回 `False`，但在非沙箱环境下返回 `True`，并识别到 2 张 RTX 3090。
  初步原因：当前代理执行沙箱会影响 `torch` 的设备探测结果。
  处理动作：改用非沙箱校验重新生成 `reports/baseline.md`，以真实终端结果为准。
  状态：resolved

- 当前状态：`liwu/MNBVC` 不能通过标准 `datasets.load_dataset(...)` 直接加载。
  阶段：Phase 0 / Dataset Source Validation
  现象：`datasets` 报错 `Dataset scripts are no longer supported, but found MNBVC.py`。
  初步原因：该仓库仍依赖旧式 dataset script，而当前 `datasets` 版本不再支持这一路径。
  处理动作：将下载逻辑改为从 Hugging Face 数据仓库中的 `.jsonl.gz` 文件直接拉取并解析。
  状态：resolved

- 当前状态：Phase 1 tokenization 初次验收时 Hugging Face datasets 缓存目录不可写。
  阶段：Phase 1 / CPT Tokenization
  现象：执行 `scripts/04_tokenize_cpt.py` 时，`Dataset.from_generator(...)` 尝试在 `/home/xdu/.cache/huggingface/datasets/` 创建 lock 文件并报错 `OSError: [Errno 30] Read-only file system`。
  初步原因：当前代理沙箱对用户主目录下的默认 Hugging Face cache 路径只读。
  处理动作：为 `scripts/04_tokenize_cpt.py` 增加 `--cache_dir` 和仓库内默认缓存目录 `data/tokenized/cpt/.hf_cache`，随后重新执行小样本验收并通过。
  状态：resolved

- 当前状态：SFT 数据下载速率异常，`WildChat` 长时间停留在 parquet 分片 `0.00/xxxM`。
  阶段：Phase 0 / Dataset Download
  现象：旧版下载逻辑对 `WildChat` 使用非流式 `load_dataset(...)` 后再抽样，导致先连续拉取多个大体积 parquet 分片；traceback 同时显示底层进入 `huggingface_hub.file_download.xet_get(...)`。
  初步原因：下载脚本对 SFT 数据采用了“全量下载后抽样”的低效路径，且 Hugging Face Hub 的 Xet 下载路径在当前网络环境下表现不稳定。
  处理动作：将 `scripts/01_download_data.py` 改为支持 SFT 流式抽样，并在脚本初始化阶段设置 `HF_HUB_DISABLE_XET=1`；中断旧会话后重新续跑，`wildchat_mini` 已成功写出 `10000` 条样本。
  状态：resolved

- 当前状态：`mbpp` 评测集已恢复可下载。
  阶段：Phase 0 / Dataset Download
  现象：最初下载 `Muennighoff/mbpp` 时，`datasets` 报错 `Dataset scripts are no longer supported, but found mbpp.py`。
  初步原因：该仓库仍依赖旧式 dataset script，而当前 `datasets` 版本不再支持这一路径。
  处理动作：将 `configs/dataset_manifest.json` 中的 `mbpp` 切换为 `hf_repo_files` 模式，并在 `scripts/01_download_data.py` 中补齐通用仓库原始记录文件下载逻辑；随后成功下载 `data/raw/eval/mbpp.jsonl`，共 `974` 条。
  状态：resolved

- 当前状态：CPT smoke training 的主阻塞已转为训练资源，而不是训练入口兼容性。
  阶段：Phase 2 / CPT Smoke Training
  现象：训练过程中先后出现 `TrainingArguments` / `Trainer` 参数不兼容、`bf16/gpu` 不支持、`Attempting to unscale FP16 gradients` 以及单卡 OOM。
  初步原因：本地 `.venv` 使用的是 `transformers 5.6.1` / `accelerate 1.13.0`，与旧 Trainer 写法存在 API 差异；同时模型 checkpoint 默认 dtype、Trainer AMP 设置和实际 GPU / 进程占用状态未对齐。
  处理动作：将 `src/train/train_cpt.py` 改为兼容 `Trainer` 新签名，并让模型 dtype 与 `bf16/fp16/tf32` 在运行时按真实环境自动决策；另将单卡脚本默认 GPU 调整到更空闲的 `1` 号卡。用户终端日志显示 `smoke` 已成功跑通，当前单卡 `200` steps 仍主要受显存占用影响。
  状态：resolved

## 2026-04-24

- 当前状态：旧的 `tf32` CPT benchmark family 已确认不适用于双卡 `RTX 3090` 上的 `DeepSpeed ZeRO-2`，新的 `fp16` benchmark family 已切换完成但仍待用户实际重跑验证。
  阶段：Phase 3 / CPT Benchmark / ZeRO-2
  现象：执行 `CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/bench/cpt_zero2.sh` 时发生两次连续失败。第一次使用旧 `ds_zero2_bench.json` 配置时，在首个 backward 的 `reduce_ipg_grads -> allreduce_bucket` 路径申请约 `762 MiB` 通信张量时报 `torch.OutOfMemoryError`；将 `reduce_bucket_size` / `allgather_bucket_size` 下调并关闭 `overlap_comm` 后再次运行，仍在首个 optimizer step 的 `single_partition_of_fp32_groups` / `flatten(self.averaged_gradients[i]).to(...)` 路径申请约 `3.21 GiB` 显存时报 `torch.OutOfMemoryError`。日志中同时出现 `Gradient accumulation steps mismatch: GradientAccumulationPlugin has 1, DeepSpeed config has 8`，但该提示并非本次 OOM 的主因。
  初步原因：当时的正式 benchmark family 采用 `fp16=false, bf16=false, tf32=true`。其中 `tf32` 只是 `float32` matmul 的计算模式，不会降低参数或优化器状态的存储开销；本地 `Qwen3-1.7B` checkpoint 本身按 `bfloat16` 加载，但 `DeepSpeed ZeRO-2` 在 optimizer step 仍需维护和拼接 `fp32` 主权重分区、Adam 状态以及梯度分区。双卡 `24 GB` 场景下，这条 `tf32/float32` 基准路线即使收缩通信 bucket，仍会在 ZeRO-2 的更新阶段因额外 `fp32` 分区而爆显存。
  处理动作：先将 `configs/train/ds/ds_zero2_bench.json` 与 `configs/train/ds/ds_zero2_sanity.json` 改为与 Hugging Face `Trainer` 自动同步的 `"auto"` micro-batch / grad accumulation / clipping，缩小 bucket 到 `50000000` 并关闭 `overlap_comm`；随后确认根因不在 bucket，而在正式 benchmark 精度族本身，于是把 `configs/train/bench/cpt_bench_single.yaml`、`cpt_bench_ddp.yaml`、`cpt_bench_zero2.yaml`、`cpt_bench_zero3_offload.yaml` 全部改为 `fp16=true, tf32=false`，并将 `benchmark_group` 从 `cpt_bench_core_batch16_tf32` 重命名为 `cpt_bench_core_batch16_fp16`，以形成一组单卡、DDP、ZeRO-2、ZeRO-3 offload 都有机会完整跑通的统一 benchmark 协议。
  状态：open

- 当前状态：`single` benchmark 在切换到 `fp16` 家族后的首轮运行已定位到代码级精度对齐问题，修复已完成但仍待用户实际重跑确认。
  阶段：Phase 3 / CPT Benchmark / Single GPU
  现象：执行 `CUDA_VISIBLE_DEVICES=0 bash scripts/train/bench/cpt_single.sh` 时先后出现两种精度相关报错。第一次训练在第一个 step 的梯度裁剪阶段报错 `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`；针对该问题将模型直接改为 `fp16` 加载后，再次运行又在同一位置报错 `ValueError: Attempting to unscale FP16 gradients.`。两次调用栈都位于 `accelerate.Accelerator.clip_grad_norm_ -> unscale_gradients -> torch.amp.grad_scaler._unscale_grads_`。
  初步原因：虽然 benchmark YAML 已切换为 `fp16=true, tf32=false`，但原生 Hugging Face `Trainer` 的 `fp16` 路径并不是“`fp16` 参数直训”，而是“`fp32` 参数 + autocast + GradScaler”。第一次失败是因为模型仍按 checkpoint 的 `torch_dtype=auto` 加载为 `bfloat16`，导致 `GradScaler` 试图对 `bf16` 梯度执行未实现的 `unscale`；第二次失败说明把原生 Trainer 路径直接改成 `fp16` 参数也不对，因为 `GradScaler` 同样禁止对 `fp16` 梯度执行 `unscale`。
  处理动作：将 `src/train/train_cpt.py` 的模型 dtype 解析改为按训练后端区分：若启用 DeepSpeed 混精，则模型显式按 `bf16/fp16` 加载；若使用原生 `Trainer` 的 single/DDP 混精，则统一按 `fp32` 参数加载，让 `autocast + GradScaler` 按预期工作，避免出现 “`fp16` 配置 + `bf16/fp16` 参数直训” 这种不兼容组合。
  状态：open

- 当前状态：当前统一的 `fp16` benchmark family 仍不适合原生 `Trainer` 单卡全参数训练，问题已从“精度配置错误”转为“内存账本本身不成立”。
  阶段：Phase 3 / CPT Benchmark / Single GPU
  现象：在完成 single/DDP 路径的精度对齐修复后，再次执行 `CUDA_VISIBLE_DEVICES=0 bash scripts/train/bench/cpt_single.sh`，训练进入第一个 optimizer step 时于 `torch.optim.adam.Adam._init_group` 初始化 `state["exp_avg"] = torch.zeros_like(...)` 报 `torch.OutOfMemoryError`；日志显示 GPU 0 总显存 `23.56 GiB`，当时仅剩 `42.62 MiB` 空闲，PyTorch 已分配约 `22.75 GiB`，额外申请 `48 MiB` Adam 状态即失败。
  初步原因：这说明在当前硬件和软件栈下，原生 Hugging Face `Trainer` 的 `fp16` 训练路径虽然要求 `fp32` 参数 + `autocast + GradScaler` 才是正确实现，但 `1.7B` 全参数模型在单张 `24 GB` `3090` 上以这种方式训练时，参数、梯度与 Adam 状态的显存总和已经超过可用显存。此前最开始能跑通的单卡训练并不是这套受控 benchmark 配方，而是旧的 `fp16=false, tf32=true` sanity 路线；那一路径实际按 checkpoint 默认 `bfloat16` 持有模型参数，显存开销远小于当前 `fp32` 参数的原生 `fp16` benchmark 路线。
  处理动作：已将该问题归因从“代码 bug”升级为“benchmark 设计冲突”：当前仓库不能再继续把 `single / ddp / zero2 / zero3_offload` 强行塞进同一个原生 `fp16` benchmark family 里对比。后续应在 benchmark 设计上做拆分，例如将 `single / ddp` 归为 native family，将 `zero2 / zero3_offload` 归为 DeepSpeed family，或重新定义单卡 baseline（例如更换优化器、引入 offload、或明确取消单卡全参 benchmark 项）。
  状态：open

- 当前状态：`DeepSpeed ZeRO-3 offload` benchmark 训练主体已完成，但最终模型保存阶段发生跨 rank collective 失配，修复已完成但仍待用户复验。
  阶段：Phase 3 / CPT Benchmark / DeepSpeed Family / ZeRO-3 Offload
  现象：执行 `bash scripts/train/bench/deepspeed/cpt_zero3_offload.sh` 时，`100/100` steps 训练与中途 `checkpoint-50` 保存均成功；最终打印 `train_runtime=2292s`、`train_loss=2.765` 后，在训练结束阶段卡住约 `30` 分钟，随后 rank0 报 `_ALLGATHER_BASE` timeout、rank1 报 `ALLREDUCE` timeout，堆栈分别落在 `deepspeed.runtime.engine._zero3_consolidated_16bit_state_dict()` 与 `accelerate.wait_for_everyone()`。日志同时显示 rank0 正在执行 `trainer.save_model()`，而 rank1 已进入最终 barrier。
  初步原因：`src/train/train_cpt.py` 在训练结束后只让 `world_process_zero` 调用 `trainer.save_model(output_dir)`。但 `ZeRO-3` 的最终模型保存需要所有 rank 共同参与参数 gather/consolidation；当 rank0 进入 `save_model()` 内部的 `_zero3_consolidated_16bit_state_dict()`，而 rank1 提前进入 `accelerator.wait_for_everyone()` 时，分布式 collective 顺序被打乱，最终触发 NCCL watchdog 超时。这不是训练本体 OOM，也不是 `zero3_offload` 无法收敛，而是最终保存阶段的多进程调用时序错误。
  处理动作：已修改 `src/train/train_cpt.py` 的最终保存逻辑：若启用 DeepSpeed，则所有 rank 都进入 `trainer.save_model(output_dir)`，仅由主进程实际写文件；非 DeepSpeed 路径仍保留仅主进程保存模型与 tokenizer 的行为。已通过 `python -m py_compile src/train/train_cpt.py` 静态校验；待用户重新执行 `bash scripts/train/bench/deepspeed/cpt_zero3_offload.sh` 验证最终保存是否恢复正常。
  状态：open

## 2026-04-25

- 当前状态：`scripts/10_serve_vllm.sh` 在当前环境下直接报 `vllm: 未找到命令`，已改为显式检测并输出可执行的安装提示。
  阶段：Serving Track / vLLM
  现象：执行 `bash scripts/10_serve_vllm.sh runs/sft/qwen3_1_7b_miniv2_sft_merged` 时，shell 在脚本末尾直接报 `vllm: 未找到命令`。进一步检查确认 `.venv/bin/` 下不存在 `vllm` CLI，且 `.venv/bin/python` 也无法导入 `vllm` module。
  初步原因：当前仓库虚拟环境中尚未安装 `vllm`，旧脚本只在 `.venv/bin/vllm` 不存在时盲目回退到系统 `vllm`，因此最终变成普通的 shell `command not found`，错误不够明确。
  处理动作：已将 `scripts/10_serve_vllm.sh` 改为三档启动逻辑：优先使用 `.venv/bin/vllm`；若 CLI 不存在但 Python 环境可导入 `vllm`，则回退到 `python -m vllm.entrypoints.openai.api_server`；若两者都不存在，则明确输出当前环境未安装 `vllm`，并提示执行 `.venv/bin/pip install vllm`。同时补上 `trust_remote_code` 配置透传。
  状态：resolved

- 当前状态：SFT tokenization 首次实跑时因 `BatchEncoding` 处理错误导致 Arrow dataset 生成失败，现已修复并复验通过。
  阶段：SFT Chain / Tokenization
  现象：执行 `python scripts/05_tokenize_sft.py --manifest configs/dataset_manifest.json --model_config configs/model/qwen3_1_7b_base.yaml --input_root data/cleaned/sft --output_root data/tokenized/sft --overwrite` 时，`datasets.Dataset.from_generator` 在写入 Arrow split 过程中报 `pyarrow.lib.ArrowInvalid: Failed to parse string: 'attention_mask' as a scalar of type int32`，随后抛出 `datasets.exceptions.DatasetGenerationError`。同时控制台显示生成速度异常缓慢，约 `6s/example`。
  初步原因：`src/data/sft.py` 中的 `tokenize_sft_messages()` 直接对 `tokenizer.apply_chat_template(..., tokenize=True)` 与 `tokenizer(..., return_offsets_mapping=True)` 的返回值执行 `list(...)`。在当前 `Qwen3` tokenizer / transformers 版本下，这些接口返回的是 `BatchEncoding`，而不是纯 token id 列表；对其执行 `list(...)` 得到的是键名序列 `['input_ids', 'attention_mask', ...]` 或 `['input_ids', 'offset_mapping', ...]`，最终被错误写入 `input_ids`，在 Arrow cast 阶段失败。另一方面，旧实现为了给 assistant-only label 打 mask，会对同一条对话的多个前缀反复执行整段 tokenization，导致 tokenization 速度严重退化。
  处理动作：已在 `src/data/sft.py` 中改为显式从 `BatchEncoding` 提取 `input_ids`，并将 assistant mask 计算改为“整条对话只 tokenize 一次 + 基于 `offset_mapping` 与渲染后字符区间做 span 标注”；前缀部分只保留 `chat_template(tokenize=False)` 字符串渲染，不再重复做完整 tokenization。修复后已使用同一条命令完成复验，`data/tokenized/sft/summary.json` 成功生成，训练集产出 `17121` 条 packed sequence，验证集产出 `359` 条，生成速度恢复到约 `240-255 examples/s`。
  状态：resolved

- 当前状态：Optimization Track 中关闭 gradient checkpointing 的两条变体已确认在当前双卡 `RTX 3090` 环境下不可行。
  阶段：Phase 5 / Optimization Track / Native DDP + DeepSpeed ZeRO-2
  现象：执行 `bash scripts/train/optimize/run_all.sh` 时，`native_ddp_no_gc` 在第 `1/50` step 附近即报 `torch.OutOfMemoryError`，双 rank 都在额外申请约 `1.16 GiB` 显存时失败；`zero2_no_gc` 在第 `4/50` step 左右也报相同级别的 `torch.OutOfMemoryError`，同样为双 rank 同步失败。与此同时，`ddp_base`、`ddp_workers4`、`zero2_base`、`zero2_workers4` 均已完整跑通。
  初步原因：当前 `Qwen3-1.7B`、`max_seq_length=2048`、现有 effective batch 组合下，关闭 gradient checkpointing 会显著抬高激活与反向阶段显存占用；对原生 `DDP` 与 `ZeRO-2` 来说，双卡 `24 GB` 显存都已经不足以容纳该开销，因此该旋钮在当前硬件与配方下不是“待验证优化项”，而是已被实测否决。
  处理动作：保留 `gradient_checkpointing=true` 作为 `native ddp` 与 `zero2` 的默认推荐设置；后续优化主线仅继续比较已跑通的变体。并修正 `scripts/train/optimize/run_all.sh` 的失败状态记录逻辑，避免再把失败步骤误记成 `exit code 0`。
  状态：resolved

- 当前状态：`run_all.sh` 中的 `zero3_offload` profile 在第三个 step 后被系统以 `SIGKILL` 终止，根因已收敛到“profile 组合过重”，修复已落地但仍待复验。
  阶段：Phase 4 / Profiling / DeepSpeed / ZeRO-3 Offload
  现象：执行 `bash scripts/train/profile/run_all.sh` 时，`native_single`、`native_ddp`、`zero2` 均顺利完成，`zero3_offload` 在 `step 1 -> 3` 期间持续变慢，`train_runtime` 从约 `52s` 增长到约 `311s`，随后 DeepSpeed launcher 通过 `sigkill_handler` 回收两个子进程，rank1 最终返回码为 `-9`。整个过程中 GPU 侧 `cuda_memory_max_allocated_mb` 仅约 `6428 MB`，没有出现 GPU OOM 或 Python traceback。
  初步原因：`zero3_offload` profile 同时叠加了 `ZeRO-3 CPU offload`、`torch profiler(record_shapes/profile_memory/with_flops/export trace)`、`DeepSpeed flops profiler`、`wall_clock_breakdown`、`comms_logger`、`pin_memory=true` 的 CPU offload 配置，导致主机侧开销和/或内存压力远高于 `zero2`，最终更像是被系统级 `SIGKILL` 杀掉，而不是训练逻辑异常或显存不足。
  处理动作：已将 `configs/train/profile/deepspeed/cpt_profile_zero3_offload.yaml` 调整为轻量 profile 协议：`max_steps` 从 `12` 降到 `4`，`dataloader_num_workers` 改为 `0`，`torch_profiler.active` 改为 `1`，并关闭 `record_shapes`、`profile_memory`、`with_flops`、`export_chrome_trace`；同时将 `configs/train/ds/ds_zero3_offload_profile.json` 中的 `flops_profiler`、DeepSpeed `tensorboard/csv_monitor` 关闭，并将 `offload_optimizer/offload_param.pin_memory` 改为 `false`，优先保证这一路 profile 能稳定完成。另对 `TorchProfilerCallback` 改为仅在 `world_process_zero` 启动 profiler，并启用 `acc_events=True`（若当前 PyTorch 支持），进一步降低多进程 profile 额外开销。
  状态：open

- 当前状态：`zero3_offload` 在切换到轻量 torch profiler 后仍于第三个 step 前后被 `SIGKILL`，因此已判定这一路不适合再叠加 torch profiler。
  阶段：Phase 4 / Profiling / DeepSpeed / ZeRO-3 Offload
  现象：用户单独执行 `bash scripts/train/profile/deepspeed/cpt_zero3_offload.sh` 时，即使配置已收缩到 `max_steps=4`、`active=1`、`record_shapes/profile_memory/flops/chrome trace` 全关，训练仍在 `step 1 -> 3` 间持续减速，随后 rank0 提前退出，rank1 从 `TCPStore` 收到连接关闭告警，最终由 DeepSpeed launcher 以返回码 `-9` 终止。输出目录依旧只有 TensorBoard event，没有 `train_summary.json` 或 profiler 结果文件。
  初步原因：`ZeRO-3 CPU offload` 本身已经显著依赖 host 内存与 CPU copy；在当前双卡 `3090` 环境下，即使 torch profiler 已缩到最轻量，只要继续叠加它，仍可能触发系统层面的资源压力并导致 rank0 被杀。问题已从“torch profiler 过重”进一步收敛为“zero3_offload profile 不应再启用 torch profiler”。
  处理动作：已将 `configs/train/profile/deepspeed/cpt_profile_zero3_offload.yaml` 的 `torch_profiler.enabled` 直接关闭，后续 `zero3_offload` profile 仅保留 DeepSpeed `wall_clock_breakdown + comms_logger` 作为稳定可交付的分析数据源；同时扩展 `scripts/14_report_cpt_comms.py`，使其能够把 `zero3_offload` 的 DeepSpeed comms summary 纳入 `reports/cpt_comms.md`。
  状态：open
