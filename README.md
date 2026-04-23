# LLM Train Platform (Mini v2)

本项目是一个面向 **双卡 RTX 3090** 的大模型训练与推理学习平台骨架，目标是用 **Qwen2.5-3B-Base + DeepSpeed + vLLM + Mini v2 数据方案** 跑通以下闭环：

- Mini continued pretraining（CPT）
- LoRA/QLoRA 指令微调（SFT）
- 基础评测（GSM8K / MMLU 子集 / HumanEval / MBPP）
- vLLM 推理部署与压测
- 后续可扩展的 kernel / profiling / fusion 实验

## 当前打包内容

已包含：

- `requirements.txt`
- `configs/` 下的模型、训练、服务、数据配置
- `scripts/01_download_data.py` 完整实现
- `src/train/train_cpt.py` 完整实现
- 其余脚本与模块的占位文件/接口说明
- 启动脚本、LoRA merge 脚本、vLLM 启动脚本

说明：

1. **MNBVC** 在不同镜像/整理版本中的 dataset id 可能不同，因此 `dataset_manifest.json` 中的 `source` 先保留为占位值，必要时请手动改成你本地可用的实际数据源。
2. `scripts/02_build_cpt_corpus.py`、`scripts/03_build_sft_dataset.py`、`scripts/04_tokenize_cpt.py`、`scripts/05_tokenize_sft.py`、`src/train/train_sft.py` 当前为结构化占位版本，便于继续扩展。
3. 第一版建议先跑通 `download -> CPT -> merge/serve` 主链路，再继续补 SFT 和评测细节。

## 目录结构

```text
llm_train_platform/
├── README.md
├── requirements.txt
├── configs/
│   ├── dataset_manifest.json
│   ├── model/
│   │   └── qwen25_3b_base.yaml
│   ├── train/
│   │   ├── cpt.yaml
│   │   ├── sft_lora.yaml
│   │   ├── deepspeed_zero2.json
│   │   └── deepspeed_zero3_offload.json
│   └── serve/
│       └── vllm.yaml
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── tokenized/
│   └── manifests/
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_cpt_corpus.py
│   ├── 03_build_sft_dataset.py
│   ├── 04_tokenize_cpt.py
│   ├── 05_tokenize_sft.py
│   ├── 06_train_cpt.sh
│   ├── 07_train_sft_lora.sh
│   ├── 08_merge_lora.py
│   ├── 09_eval_all.sh
│   ├── 10_serve_vllm.sh
│   └── 11_benchmark_serving.py
├── src/
│   ├── data/
│   │   ├── cleaners.py
│   │   ├── samplers.py
│   │   ├── packers.py
│   │   └── chat_templates.py
│   ├── train/
│   │   ├── train_cpt.py
│   │   ├── train_sft.py
│   │   ├── callbacks.py
│   │   └── utils.py
│   ├── eval/
│   │   ├── eval_gsm8k.py
│   │   ├── eval_mmlu.py
│   │   ├── eval_humaneval.py
│   │   ├── eval_mbpp.py
│   │   └── aggregate.py
│   ├── prof/
│   │   ├── torch_profile.py
│   │   ├── benchmark_attention.py
│   │   └── compare_kernels.py
│   └── serve/
│       ├── hf_generate.py
│       ├── vllm_generate.py
│       └── openai_client.py
├── runs/
│   ├── cpt/
│   ├── sft/
│   ├── eval/
│   └── benchmark/
└── reports/
    ├── baseline.md
    ├── training_log.md
    └── serving_benchmark.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -U pip setuptools wheel
pip install "torch>=2.2" torchvision torchaudio
pip install "transformers>=4.37.0" "datasets>=2.18.0" "accelerate>=0.28.0" deepspeed peft trl
pip install sentencepiece safetensors scikit-learn numpy pandas pyarrow ujson jsonlines tqdm pyyaml tensorboard evaluate jieba nltk
```

### 2. 下载 / 抽样数据

```bash
python scripts/01_download_data.py --manifest configs/dataset_manifest.json --output_root data/raw
```

### 3. 训练 mini CPT

```bash
bash scripts/06_train_cpt.sh
```

### 4. 训练 SFT（后续补齐 `train_sft.py` 后使用）

```bash
bash scripts/07_train_sft_lora.sh
```

### 5. 合并 LoRA

```bash
python scripts/08_merge_lora.py
```

### 6. 启动 vLLM 服务

```bash
bash scripts/10_serve_vllm.sh
```

## 建议执行顺序

1. 先确认 `01_download_data.py` 能把公开数据正常拉下来。
2. 先只跑 `CPT smoke test`，验证 loss 下降、checkpoint 可保存。
3. 再补齐 `train_sft.py` 和 SFT 数据处理链路。
4. 最后再加评测、profiling、kernel/fusion 实验。

## 后续建议

下一步优先补齐：

- `scripts/03_build_sft_dataset.py`
- `scripts/05_tokenize_sft.py`
- `src/train/train_sft.py`
- `src/eval/*`
- `src/prof/*`
