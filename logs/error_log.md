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
