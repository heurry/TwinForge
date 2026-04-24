# Baseline Report

- Date: 2026-04-23 19:49:19
- Project root: `/home/xdu/LLM/llm_train_platform_miniv2`
- Python executable: `/home/xdu/LLM/llm_train_platform_miniv2/.venv/bin/python`
- Python version: `3.10.18`
- Platform: `Linux-6.8.0-110-generic-x86_64-with-glibc2.35`

## Summary

- Status: ready for smoke validation.
- All critical Python modules are importable.
- Torch runtime: version `2.11.0+cu128`, cuda_available=`True`, device_count=`2`

## Module Check

| Module | Status | Detail |
| --- | --- | --- |
| `torch` | `ok` | `2.11.0+cu128` |
| `transformers` | `ok` | `5.6.1` |
| `datasets` | `ok` | `4.8.4` |
| `accelerate` | `ok` | `1.13.0` |
| `deepspeed` | `ok` | `0.18.9` |
| `peft` | `ok` | `0.19.1` |
| `trl` | `ok` | `1.2.0` |
| `jsonlines` | `ok` | `unknown` |

## Command Startup Check

- `python scripts/01_download_data.py --help`: exit_code=`0`
```text
usage: 01_download_data.py [-h] [--manifest MANIFEST]
                           [--output_root OUTPUT_ROOT] [--seed SEED]

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST
  --output_root OUTPUT_ROOT
  --seed SEED
```
- `python src/train/train_cpt.py --help`: exit_code=`0`
```text
usage: train_cpt.py [-h] --model_config MODEL_CONFIG --train_config
                    TRAIN_CONFIG --dataset_config DATASET_CONFIG
                    [--deepspeed DEEPSPEED]
                    [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]

options:
  -h, --help            show this help message and exit
  --model_config MODEL_CONFIG
  --train_config TRAIN_CONFIG
  --dataset_config DATASET_CONFIG
  --deepspeed DEEPSPEED
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
```

## GPU Check

- `nvidia-smi`: exit_code=`0`
```text
NVIDIA GeForce RTX 3090, 580.126.09, 24576 MiB
NVIDIA GeForce RTX 3090, 580.126.09, 24576 MiB
```
