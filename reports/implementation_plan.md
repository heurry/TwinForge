# Implementation Plan

日期：2026-04-24

## Reference Basis

本计划参考以下官方文档，并结合当前仓库和本地双卡 `RTX 3090` 环境进行裁剪：

- [DeepSpeed Tutorials Index](https://www.deepspeed.ai/tutorials/)
- [DeepSpeed Getting Started](https://www.deepspeed.ai/getting-started/)
- [DeepSpeed Configuration JSON](https://www.deepspeed.ai/docs/config-json/)
- [DeepSpeed ZeRO Tutorial](https://www.deepspeed.ai/tutorials/zero/)
- [DeepSpeed ZeRO-Offload Tutorial](https://www.deepspeed.ai/tutorials/zero-offload/)
- [DeepSpeed AutoTP Training Tutorial](https://www.deepspeed.ai/tutorials/autotp-training/)
- [DeepSpeed Automatic Tensor Parallelism Inference Tutorial](https://www.deepspeed.ai/tutorials/automatic-tensor-parallelism/)
- [DeepSpeed Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)
- [DeepSpeed Ulysses / ALST Sequence Parallelism Tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)
- [DeepSpeed Flops Profiler Tutorial](https://www.deepspeed.ai/tutorials/flops-profiler/)
- [DeepSpeed PyTorch Profiler Tutorial](https://www.deepspeed.ai/tutorials/pytorch-profiler/)
- [DeepSpeed Autotuning Tutorial](https://www.deepspeed.ai/tutorials/autotuning/)
- [DeepSpeed Monitoring Tutorial](https://www.deepspeed.ai/tutorials/monitor/)
- [DeepSpeed Communication Logging Tutorial](https://www.deepspeed.ai/tutorials/comms-logging/)
- [DeepSpeed Universal Checkpointing Tutorial](https://www.deepspeed.ai/tutorials/universal-checkpointing/)
- [DeepSpeed Inference Tutorial](https://www.deepspeed.ai/tutorials/inference-tutorial/)
- [DeepSpeed ZeRO++ Tutorial](https://www.deepspeed.ai/tutorials/zeropp/)

## Local Constraints

当前计划以如下约束为前提：

- 单机双卡 `RTX 3090`，每卡 `24 GB`
- 当前主训练入口是 Hugging Face `Trainer`，已接入 `TrainingArguments(deepspeed=...)`
- 当前主模型是本地 `Qwen3-1.7B`
- 当前主训练链路是 `raw -> cleaned -> tokenized -> CPT`
- 当前仓库已经验证 `single_gpu` 与 `ddp` 训练入口可跑通
- 当前仓库已经接入显存 callback，并可写入 `train_log_history.json`

本地模型与 `transformers` 实现给出两个重要信号：

- 本地模型配置中 `hidden_size=2048`、`num_attention_heads=16`、`num_key_value_heads=8`，因此 `tp_size=2` 在结构上是合法的。[model/Qwen3-1.7B/config.json](/home/xdu/LLM/llm_train_platform_miniv2/model/Qwen3-1.7B/config.json:11)
- 当前 `transformers` 里的 `qwen3` 已内建 `base_model_tp_plan`、`base_model_pp_plan` 以及 `lm_head` 的 `_tp_plan/_pp_plan`，说明 AutoTP 和 Pipeline 方向都不是“模型结构不支持”，而是“工程接入成本不同”。[configuration_qwen3.py](/home/xdu/LLM/llm_train_platform_miniv2/.venv/lib/python3.10/site-packages/transformers/models/qwen3/configuration_qwen3.py:44) [modeling_qwen3.py](/home/xdu/LLM/llm_train_platform_miniv2/.venv/lib/python3.10/site-packages/transformers/models/qwen3/modeling_qwen3.py:442)

## Diagnosis

当前仓库最大的计划问题不是“缺少更多训练脚本”，而是“实验层次没有拆开”：

- 现在的 `single_gpu`、`ddp`、`zero2` 配置同时改变了步数、精度、并行方式、累积步数和显存策略。
- 这些结果只能证明链路可跑通，不能证明哪种后端更优。
- 继续在这个基础上叠新教程能力，会让训练、优化、推理全部混成不可比较的结果。

所以后续计划必须同时覆盖三件事：

1. 工程打通：先把每一种后端和工具真正跑起来。
2. 受控对比：把后端、并行方式、ZeRO、TP、推理引擎放到同一套协议里对比。
3. 优化闭环：基于对比结论继续推进长训、SFT、评测和部署，而不是凭感觉叠功能。

## DeepSpeed Capability Matrix

下表是对官方 tutorials 的落地筛选，目标是回答“当前双卡 `3090` 能不能做”“值不值得现在做”“需要改多大”。

| 能力 | 官方来源 | 双卡 3090 可行性 | 当前仓库接入成本 | 当前优先级 | 结论 |
|---|---|---:|---:|---:|---|
| ZeRO-1 / ZeRO-2 训练 | ZeRO Tutorial | 高 | 低 | P0 | 立即纳入 benchmark 主线 |
| ZeRO-3 / CPU Offload | ZeRO / ZeRO-Offload | 高 | 中 | P1 | 作为容量优化与长训候选，不先混入主 benchmark |
| AutoTP 训练 | AutoTP Training | 中 | 中到高 | P1 | 可做 POC；先验证和 `Trainer` 路径兼容性 |
| AutoTP 推理 | Automatic Tensor Parallelism | 高 | 中 | P0 | 直接列入推理 benchmark 主线 |
| DeepSpeed Inference | Inference Tutorial | 高 | 中 | P0 | 与 `HF eager`、`vLLM` 做正式对比 |
| Pipeline Parallelism | Pipeline Tutorial | 中 | 高 | P2 | 需要单独训练脚本，不适合先塞进现有 `Trainer` 主线 |
| Ulysses / ALST Sequence Parallel | Ulysses / ALST | 中 | 高 | P2 | 适合长上下文探索，不作为近期主线 |
| Flops Profiler | Flops Profiler | 高 | 低 | P0 | 立即纳入训练 profiling 主线 |
| PyTorch Profiler | PyTorch Profiler | 高 | 低到中 | P0 | 立即纳入训练 profiling 主线 |
| Monitoring | Monitoring | 高 | 低 | P0 | 立即纳入训练与推理的常驻指标采集 |
| Communication Logging | Comms Logging | 高 | 低 | P0 | 用于解释 DDP / ZeRO 通信差异 |
| Autotuning | Autotuning | 中 | 中 | P1 | 在 benchmark 稳定后启用，不先替代人工 benchmark |
| Universal Checkpointing | Universal Checkpointing | 中到高 | 中 | P1 | 在 ZeRO / TP 产物切换前引入 |
| ZeRO++ | ZeRO++ | 低 | 高 | P3 | 单机双卡场景收益有限，暂不列入近期计划 |

### Why These Decisions

#### 立即可做并且值得做

- `ZeRO-1 / ZeRO-2 / ZeRO-3 / Offload`：
  当前训练入口已经走 `deepspeed` 配置文件，接入成本最低，最适合形成“显存节省 vs 吞吐损失”的系统比较。
- `Flops Profiler / PyTorch Profiler / Monitoring / Communication Logging`：
  这些不是“锦上添花”，而是 benchmark 的解释层。没有它们，就只能看到 `steps/s`，看不到通信、算子和 checkpoint 的瓶颈。
- `AutoTP 推理 / DeepSpeed Inference`：
  用户已经明确希望覆盖完整训练到推理过程。当前双卡环境对 `tp_size=2` 友好，而 DeepSpeed 推理后端可以与 `vLLM` 做直接对比。

#### 可以做，但不该现在就塞进主线

- `AutoTP 训练`：
  官方教程说明它可以与 Hugging Face 模型协作，且当前本地 `qwen3` 已有 TP plan；但当前仓库是基于 `Trainer` 的训练骨架，不应直接把它混入现有 benchmark。应先单独做 `tp2` POC，确认 loss、resume、checkpoint 与日志链路是否成立。
- `Autotuning`：
  适合在训练主线稳定后，进一步搜索 `micro_batch`、ZeRO stage、bucket 参数，但它不应该替代人工设计的受控 benchmark。
- `Universal Checkpointing`：
  只有在后续确实要跨 `ddp / zero / tp` 切换 checkpoint 产物时，价值才会显著。

#### 技术上可做，但当前不该抢优先级

- `Pipeline Parallelism`：
  官方教程基于 `PipelineModule` 和 `engine.train_batch()`，与当前 `Trainer` 主线差异太大。它不是不能做，而是需要单独开一条训练脚本和数据装载方式。
- `Ulysses / ALST Sequence Parallelism`：
  官方教程对 `transformers` 接入友好，但它本质上服务于更长上下文训练。当前项目更缺 benchmark 和训练闭环，而不是先冲长上下文。

#### 当前不建议投入

- `ZeRO++`：
  官方示例聚焦更大规模通信环境；对当前单机双卡 `3090` 来说，不是最优先的收益来源。
- 其他大规模分布式导向教程：
  如多节点、MoE、超大规模优化器压缩等，不是当前项目主矛盾。

## Plan Structure

后续实施分成四条并行但有先后依赖的主线：

1. `Sanity Track`
2. `Benchmark Track`
3. `Optimization Track`
4. `Serving Track`

这四条主线共同支撑最终闭环：

`raw -> cleaned -> tokenized -> CPT -> SFT -> eval -> serving -> benchmark -> profiling`

## Naming Rules

后续配置和脚本命名改为按职责划分，而不是按后端裸命名：

- `configs/train/sanity/*.yaml`
- `configs/train/bench/*.yaml`
- `configs/train/opt/*.yaml`
- `configs/train/ds/*.json`
- `scripts/train/sanity/*.sh`
- `scripts/train/bench/*.sh`
- `scripts/train/opt/*.sh`

如果暂不移动目录，至少文件名要满足以下规则：

- `cpt_sanity_single.yaml`
- `cpt_sanity_ddp.yaml`
- `cpt_sanity_zero2.yaml`
- `cpt_bench_single_fp16.yaml`
- `cpt_bench_ddp_fp16.yaml`
- `cpt_bench_zero2_fp16.yaml`
- `cpt_opt_zero2_longseq.yaml`
- `cpt_opt_zero3_offload_longrun.yaml`

DeepSpeed 配置也按职责拆开：

- `ds_zero0_bench.json`
- `ds_zero2_bench.json`
- `ds_zero3_offload_opt.json`
- `ds_tp2_train_poc.json`
- `ds_infer_tp2.json`
- `ds_profile_flops.json`
- `ds_profile_comms.json`
- `ds_monitor.json`
- `ds_autotune_zero.json`

## Metrics Contract

### Training Metrics

所有训练实验统一记录：

- `train_runtime`
- `train_samples_per_second`
- `train_steps_per_second`
- `train_tokens_per_second`
- `loss`
- `grad_norm`
- `learning_rate`
- `cuda_memory_allocated_mb`
- `cuda_memory_reserved_mb`
- `cuda_memory_max_allocated_mb`
- `checkpoint_save_seconds`
- `checkpoint_size_bytes`
- `resume_success`
- `resume_loss_continuity`
- `oom_or_nan`

### Profiling Metrics

- `flops`
- `macs`
- `params`
- top hot ops
- attention / matmul / layernorm hotspot time
- communication op summary
- all_reduce / all_gather / reduce_scatter time

### Serving Metrics

- load time
- warmup time
- peak GPU memory
- TTFT
- decode tokens/s
- request latency p50 / p95
- concurrency throughput
- output correctness spot check

## Benchmark Contract

任何横向 benchmark 都必须满足：

- 相同模型
- 相同 tokenized 数据切片
- 相同 seed
- 相同 `max_steps`
- 相同 `logging_steps`
- 相同 `save_steps`
- 相同精度族
- 相同有效全局 batch
- 相同评测节点

有效全局 batch 定义：

`effective_batch = per_device_train_batch_size * gradient_accumulation_steps * data_parallel_world_size`

对于 TP 训练 POC，额外要求：

- 明确 TP 只改变模型切分方式，不允许同时改变训练步数和数据切片
- 若 TP 依赖新的精度或新的训练脚本，则必须单独归为新的 benchmark family

## Phase 0

阶段名称：实验协议与目录重构

目标：

- 把当前“能跑几个脚本”改成“有层次的实验体系”

任务：

- 将现有 `single_gpu`、`ddp`、`zero2` 配置重分类为 `sanity`
- 新增 `bench` 与 `opt` 配置命名约定
- 统一 `runs/` 输出目录结构
- 在训练日志中补充：
  - `train_tokens_per_second`
  - checkpoint 保存耗时
  - resume 结果
- 约束报告命名：
  - `reports/cpt_sanity.md`
  - `reports/cpt_benchmark.md`
  - `reports/cpt_optimization.md`
  - `reports/serving_benchmark.md`

交付物：

- 更新后的 `reports/implementation_plan.md`
- 重命名后的训练配置与脚本
- 日志字段定义落地

验收标准：

- 任意一次训练都能被明确归类为 `sanity / benchmark / optimized`

## Phase 1

阶段名称：CPT 数据链路与 benchmark 数据切片

目标：

- 保持现有 CPT 主链路稳定
- 为所有 benchmark 固定数据视图

任务：

- 保持现有 `raw -> cleaned -> tokenized` 链路
- 增加 benchmark 专用数据切片：
  - 固定前 `N` 个 packed sample
  - 固定采样索引
  - 固定 tokenizer 版本
- 在 `summary.json` 中标记：
  - 训练全集规模
  - benchmark 切片规模
  - benchmark 切片生成规则

交付物：

- `data/tokenized/cpt/bench/`
- `data/tokenized/cpt/bench_summary.json`

验收标准：

- `single_gpu / ddp / zero / tp` 的 benchmark 都能吃同一份数据切片

## Phase 2

阶段名称：Training Sanity Track

目标：

- 先把最重要、最现实的训练后端全部跑通

实验包：

1. `sanity_single`
2. `sanity_ddp`
3. `sanity_zero2`
4. `sanity_zero3_offload`

任务：

- 单卡 `sanity`
- 双卡 DDP `sanity`
- 双卡 ZeRO-2 `sanity`
- 双卡 ZeRO-3 或 CPU Offload `sanity`
- 每个后端都验证：
  - 首次启动
  - checkpoint 保存
  - resume
  - 显存日志
  - 最终模型保存

输出文件建议：

- `configs/train/sanity/cpt_sanity_single.yaml`
- `configs/train/sanity/cpt_sanity_ddp.yaml`
- `configs/train/sanity/cpt_sanity_zero2.yaml`
- `configs/train/sanity/cpt_sanity_zero3_offload.yaml`
- `configs/train/ds/ds_zero2_sanity.json`
- `configs/train/ds/ds_zero3_offload_sanity.json`

验收标准：

- 四条 sanity 后端都完成一次成功训练
- 至少 `zero2` 和 `zero3_offload` 各完成一次恢复

当前优先级说明：

- `zero2` 是当前最近主线
- `zero3_offload` 是为了回答“显存还能不能继续省”

## Phase 3

阶段名称：Training Benchmark Track

目标：

- 在同一协议下比较 `single_gpu / ddp / zero2 / zero3_offload`

Benchmark Family A：Data Parallel Family

- `bench_single_fp16`
- `bench_ddp_fp16`

Benchmark Family B：ZeRO Family

- `bench_zero1_fp16`
- `bench_zero2_fp16`
- `bench_zero3_offload_fp16`

任务：

- 固定有效全局 batch
- 固定 `max_steps`
- 固定 `fp16`
- 固定 benchmark 数据切片
- 固定日志与保存频率
- 输出正式对比表

必须回答的问题：

1. DDP 相比单卡，吞吐提升多少。
2. ZeRO-2 相比 DDP，显存节省多少，吞吐损失多少。
3. ZeRO-3 / Offload 是否值得在双卡 `3090` 上承担额外复杂度。

交付物：

- `reports/cpt_benchmark.md`

验收标准：

- 形成第一版正式 benchmark 结论
- 给出“正式长训后端推荐”

## Phase 4

阶段名称：Profiling and Observability Track

目标：

- 给 benchmark 和优化结果提供归因

子任务：

### 4.1 Flops Profiler

- 在 benchmark family 中挑 1 到 2 个后端接入 `Flops Profiler`
- 记录 FLOPs、MACs、参数量和 hotspot layer

### 4.2 PyTorch Profiler

- 对 `single_gpu` 与 `zero2` 做对比 profile
- 关注 attention、matmul、optimizer step、dataloader

### 4.3 Monitoring

- 接入训练过程中的常驻监控输出
- 保留 TensorBoard 或其他监控后端统一出口

### 4.4 Communication Logging

- 对 `ddp` 与 `zero2` 比较通信模式
- 重点记录 all-reduce / reduce-scatter / all-gather

交付物：

- `reports/cpt_profile.md`
- `reports/cpt_comms.md`

验收标准：

- 至少能解释 benchmark 中最主要的 2 个瓶颈

## Phase 5

阶段名称：Optimization Track

目标：

- 把 benchmark 中节省出来的显存和吞吐转化为真实训练收益

优化顺序：

1. 先确定最终训练后端。
2. 再逐项尝试：
   - 增大 `micro_batch`
   - 增大 `gradient_accumulation_steps`
   - 增大 `max_seq_length`
   - 延长 `max_steps`
3. 再考虑更重的 DeepSpeed 优化：
   - ZeRO-3
   - CPU Offload
   - bucket 参数调优
   - activation checkpointing 组合

任务：

- 设计 `opt` 配置族
- 做一次正式长训
- 做一次中断恢复
- 输出推荐参数

交付物：

- `reports/cpt_optimization.md`
- 推荐训练配置

验收标准：

- 有一套可解释的正式 CPT 长训参数

## Phase 6

阶段名称：AutoTP Training POC Track

目标：

- 基于官方 AutoTP 教程，验证 `tp_size=2` 在当前 `Qwen3-1.7B` 上是否可作为训练候选

为什么放在独立 Phase：

- 当前模型结构允许 `tp_size=2`
- 当前 `transformers` 已有 `tp_plan`
- 但当前主训练脚本是 `Trainer` 路径，需要单独验证兼容性

任务：

- 新增 `tp2` 训练 POC 配置
- 验证：
  - 模型加载
  - 前向 / 反向
  - loss 是否下降
  - checkpoint 是否保存
  - resume 是否可用
- 如果 `Trainer` 路径不稳定，则单独建立 `deepspeed.initialize` 版训练脚本

交付物：

- `configs/train/poc/cpt_tp2_poc.yaml`
- `configs/train/ds/ds_tp2_train_poc.json`
- 可选：`src/train/train_cpt_tp.py`
- `reports/cpt_tp_poc.md`

验收标准：

- 能明确回答 `AutoTP Training` 是否进入后续主线

## Phase 7

阶段名称：Pipeline / Sequence Parallel Advanced POC

目标：

- 只在主训练和 benchmark 主线稳定后，探索更激进的并行方式

子方向：

### 7.1 Pipeline Parallelism POC

- 基于官方 `PipelineModule` 教程单独建脚本
- 不污染现有 `Trainer` 主线
- 只做最小可运行验证

### 7.2 Ulysses / ALST POC

- 目标是更长上下文训练
- 先做 `seq_parallel_size=2` 的最小接入
- 验证对 dataloader、loss 和 checkpoint 的影响

交付物：

- `src/train/train_cpt_pipeline.py`
- `src/train/train_cpt_ulysses.py`
- `reports/cpt_pipeline_poc.md`
- `reports/cpt_ulysses_poc.md`

验收标准：

- 明确这两条路线是否值得进入正式优化阶段

备注：

- 这两条路线技术上可做，但当前不应抢在 benchmark 前面

## Phase 8

阶段名称：Autotuning and Config Search

目标：

- 在 benchmark 结论之后，用 DeepSpeed 官方 autotuning 做局部搜索

任务：

- 先人工确定搜索边界：
  - `micro_batch`
  - `gradient_accumulation_steps`
  - `zero_stage`
  - bucket size
- 再启用 DeepSpeed autotuning
- 将 autotuning 结果与人工 benchmark 对照，防止过拟合某一项指标

交付物：

- `configs/train/ds/ds_autotune_zero.json`
- `reports/cpt_autotune.md`

验收标准：

- autotuning 给出的推荐配置能被人工 benchmark 复现

## Phase 9

阶段名称：SFT、评测与 checkpoint 选择

目标：

- 用正式 CPT 产物或 base 模型完成 SFT、评测与 checkpoint 选择闭环

任务：

- 完成 SFT 数据处理
- 完成 LoRA / merge
- 对 base、CPT、SFT checkpoint 做统一评测
- 形成服务前的模型选择结论

交付物：

- `runs/sft/*`
- `runs/eval/*`
- `reports/model_selection.md`

验收标准：

- 至少有一个可部署 checkpoint 被选出

## Phase 10

阶段名称：Serving Track

目标：

- 把训练产物接入两个以上推理后端
- 对比不同推理引擎在双卡 `3090` 上的真实表现

Serving Family A：Local Eager Baseline

- `HF eager` 单卡
- `HF eager` 双卡手工切分可选

Serving Family B：DeepSpeed Inference

- `DeepSpeed init_inference`
- `mp_size=2`
- AutoTP inference

Serving Family C：vLLM

- `vllm serve`
- `tensor_parallel_size=2`

任务：

- 新增 `src/serve/deepspeed_generate.py`
- 新增 `scripts/10_serve_deepspeed.sh`
- 保留 `vLLM` 服务基线
- 对三类后端做统一压测

交付物：

- `src/serve/hf_generate.py`
- `src/serve/deepspeed_generate.py`
- `src/serve/vllm_generate.py`
- `scripts/10_serve_deepspeed.sh`
- `scripts/10_serve_vllm.sh`
- `scripts/11_benchmark_serving.py`
- `reports/serving_benchmark.md`

验收标准：

- 至少两种推理后端能成功加载训练产物
- 有正式的服务 benchmark 报告

## Phase 11

阶段名称：Checkpoint Portability and Lifecycle

目标：

- 解决不同训练后端和推理后端之间的 checkpoint 可移植性问题

任务：

- 接入 Universal Checkpointing
- 验证 checkpoint 在 `ddp / zero / tp` 之间的可恢复性
- 明确哪些产物用于：
  - 继续训练
  - merge
  - 推理
  - 服务

交付物：

- `reports/checkpoint_portability.md`

验收标准：

- 至少完成一次跨训练后端或跨推理后端的 checkpoint 验证

## Deprioritized Items

以下内容当前不列入近期主线：

- ZeRO++
- 多节点训练
- MoE 训练与推理
- 超大规模 optimizer compression
- 与当前 `Qwen3-1.7B` 主线无关的专用 kernel 教程

这些方向不是不能做，而是对当前单机双卡项目不是最优先收益来源。

## Execution Rules

- 每次代码、配置、文档、实验操作都追加记录到 `logs/operation_log.md`
- 每次失败、异常、阻塞都追加记录到 `logs/error_log.md`
- 每个阶段结束后更新一次本文件状态
- 未完成受控 benchmark 前，不得把 `sanity` 结果写成性能结论
- 未完成 profile / comms 归因前，不得把吞吐差异直接归因于某个后端
- 未完成两个以上推理后端验证前，不得把训练产物视为完整部署交付

## Immediate Next Actions

1. 把现有 `single_gpu`、`ddp`、`zero2` 配置与脚本重新命名并归类到 `sanity`。
2. 补齐 `zero2` 与 `zero3_offload` 的 sanity run，并验证 resume。
3. 新建第一组受控 benchmark：`single_fp16 / ddp_fp16 / zero2_fp16`，固定相同有效全局 batch 与相同步数。
4. 给 benchmark 主线接入 `Flops Profiler`、`PyTorch Profiler`、`Monitoring` 和 `Communication Logging`。
5. 在训练主线稳定后，单独开 `AutoTP Training tp2 POC`。
6. 并行推进 `DeepSpeed Inference tp2` 与 `vLLM tp2` 的 serving baseline。
