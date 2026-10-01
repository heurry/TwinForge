"""Structured diagnosis.

Two modes, one contract (DiagnoseResponse):
  * stub —— 确定性、纯函数：直接从 Go 取来的证据推导诊断（无 GPU 也能端到端跑，可单测）
  * live —— 调网关上的 LLM，要求其输出结构化 JSON，再解析回 DiagnoseResponse
`run_diagnose` 按 cfg.stub_mode 调度：auto 时试 live，失败回退 stub。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from aiservice.config import Config
from aiservice.schemas import (
    DiagnoseRequest,
    DiagnoseResponse,
    EvidenceItem,
    RecommendedAction,
    RelatedResource,
)

# ---- 阈值（与 Go 取证侧/前端语义一致）----
ERROR_RATE_THRESHOLD = 0.02
P95_LATENCY_THRESHOLD_MS = 3000.0


def run_diagnose(req: DiagnoseRequest, cfg: Config) -> DiagnoseResponse:
    start = time.perf_counter()
    resp = _dispatch(req, cfg)
    resp.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return resp


def _dispatch(req: DiagnoseRequest, cfg: Config) -> DiagnoseResponse:
    mode = cfg.stub_mode
    force_stub = mode == "on" or (mode != "off" and not cfg.llm_base_url)
    if force_stub:
        resp = stub_diagnose(req)
        resp.mode = "stub"
        return resp
    try:
        resp = live_diagnose(req, cfg)
        # 成功的 LLM JSON 为 live；不完整 JSON 时会显式切到规则化降级，不能伪装成 live。
        if resp.mode == "stub":
            resp.mode = "live"
        return resp
    except Exception as exc:  # noqa: BLE001 - 上游不稳定时优雅降级
        if mode == "off":
            return DiagnoseResponse(
                status="failed", error=f"{type(exc).__name__}: {exc}",
                mode="live", model_id=cfg.llm_model,
            )
        resp = stub_diagnose(req)
        resp.mode = "stub"
        resp.error = f"llm fallback: {type(exc).__name__}: {exc}"
        return resp


# ========================= stub（确定性，纯函数）=========================


def stub_diagnose(req: DiagnoseRequest) -> DiagnoseResponse:
    """Rule-based diagnosis derived from the evidence bundle. Deterministic."""
    evidence = req.evidence or {}
    if evidence.get("scope") == "inference" and evidence.get("inference"):
        return _inference_diagnose(evidence)
    if evidence.get("scope") == "training" and evidence.get("training"):
        return _training_diagnose(evidence)
    metrics: Dict[str, Any] = evidence.get("metrics") or {}
    incidents: List[Dict[str, Any]] = evidence.get("incidents") or []
    deployments: List[Dict[str, Any]] = evidence.get("deployments") or []

    error_rate = _as_float(metrics.get("error_rate"))
    p95 = _as_float(metrics.get("p95_latency_ms"))
    p99 = _as_float(metrics.get("p99_latency_ms"))
    qps = _as_float(metrics.get("qps"))
    req_count = _as_int(metrics.get("request_count"))
    active_incidents = [i for i in incidents if str(i.get("status")) != "resolved"]

    evidence_items: List[EvidenceItem] = []
    if metrics:
        evidence_items.append(EvidenceItem(
            label="请求量", source="metrics",
            detail=f"近 10 分钟 {req_count} 次请求，QPS≈{qps:.3f}",
        ))
        evidence_items.append(EvidenceItem(
            label="错误率", source="metrics",
            detail=f"error_rate={error_rate:.3f}（阈值 {ERROR_RATE_THRESHOLD}）",
        ))
        evidence_items.append(EvidenceItem(
            label="延迟", source="metrics",
            detail=f"P95={p95:.0f}ms，P99={p99:.0f}ms（阈值 {P95_LATENCY_THRESHOLD_MS:.0f}ms）",
        ))
    if active_incidents:
        evidence_items.append(EvidenceItem(
            label="未解决故障", source="incidents",
            detail=f"{len(active_incidents)} 起进行中事件，最高级别 {active_incidents[0].get('severity')}",
        ))

    root_cause, impact, confidence, actions = _reason(error_rate, p95, active_incidents)

    related: List[RelatedResource] = []
    for dep in deployments[:3]:
        related.append(RelatedResource(
            type="deployment", id=_as_str(dep.get("id")), name=_as_str(dep.get("name")),
        ))
    for inc in active_incidents[:3]:
        related.append(RelatedResource(
            type="incident", id=_as_str(inc.get("id")), name=_as_str(inc.get("title")),
        ))

    return DiagnoseResponse(
        status="completed",
        root_cause=root_cause,
        confidence=round(confidence, 3),
        impact=impact,
        evidence=evidence_items,
        recommended_actions=actions,
        related_resources=related,
        model_id="stub",
        endpoint_id="stub",
        category="general",
        severity="critical" if error_rate >= ERROR_RATE_THRESHOLD else "warning" if p95 >= P95_LATENCY_THRESHOLD_MS or active_incidents else "info",
    )


def _inference_diagnose(evidence: Dict[str, Any]) -> DiagnoseResponse:
    inference = evidence.get("inference") or {}
    benchmark = inference.get("benchmark") or {}
    baseline = inference.get("baseline") or {}
    runtime_logs = inference.get("runtime_logs") or {}
    model_id = _inference_model_id(inference, benchmark)
    log_signatures = runtime_logs.get("signatures") or {}
    scenarios = [
        item for item in (benchmark.get("summary") or {}).get("scenarios", [])
        if isinstance(item, dict)
    ]
    if not scenarios:
        return DiagnoseResponse(
            status="completed", category="insufficient_evidence", severity="warning",
            root_cause="推理压测缺少场景指标，无法对 TTFT/TPOT 做可靠归因。",
            confidence=0.35, impact="当前无法判断推理性能是否满足目标。",
            evidence=[EvidenceItem(label="压测证据", source="benchmark", detail="scenarios 为空")],
            recommended_actions=[RecommendedAction(
                action="重新运行完整 1K/2K × 1--16 并发矩阵", risk="low", impact="补齐诊断证据",
            )],
            model_id=model_id, endpoint_id=_as_str(benchmark.get("endpoint_id")),
        )

    worst = max(scenarios, key=lambda item: _as_float(item.get("p95_ttft_ms")))
    min_success = min(_as_float(item.get("success_rate")) for item in scenarios)
    min_quality = min(_as_float(item.get("quality_gate_pass_rate")) for item in scenarios)
    max_memory = max(_scenario_memory(item) for item in scenarios)
    labels = {
        str(label)
        for item in scenarios
        for label in ((item.get("bottleneck") or {}).get("labels") or [])
    }
    ttft = _as_float(worst.get("p95_ttft_ms"))
    tpot = _as_float(worst.get("p95_tpot_ms"))
    e2e = _as_float(worst.get("p95_ms"))
    context_length = _as_int(worst.get("context_length"))
    concurrency = _as_int(worst.get("concurrency"))
    max_num_seqs = _benchmark_param(benchmark, "max_num_seqs")

    if min_success < 0.99:
        category, severity = "request_failure", "critical"
        root = f"推理请求成功率最低 {min_success:.1%}，未达到 99% 可用性门禁。"
        impact = "部分客服请求失败，直接影响服务可用性。"
        actions = [
            RecommendedAction(action="检查 vLLM 错误日志、OOM 与请求超时，并先降低并发", risk="low", impact="定位并缓解失败请求"),
            RecommendedAction(action="修复后重跑相同 context/concurrency 矩阵验证成功率", risk="low", impact="确认恢复"),
        ]
    elif min_quality < 0.99:
        category, severity = "quality_regression", "critical"
        root = f"输出质量门禁最低 {min_quality:.1%}，性能实验存在正确性回归。"
        impact = "客服回复可能为空、截断或违反安全约束，不能接受该优化配置。"
        actions = [RecommendedAction(action="回滚最近参数变更并核对 finish_reason、输出有效性和安全规则", risk="low", impact="恢复输出正确性")]
    elif "scheduler-saturation" in labels or (max_num_seqs > 0 and concurrency > max_num_seqs):
        category, severity = "scheduler_saturation", "warning"
        root = (
            f"{context_length // 1024}K 上下文、并发 {concurrency} 时 P95 TTFT={ttft:.0f}ms，"
            f"请求并发超过 max_num_seqs={max_num_seqs}，主要瓶颈是调度排队。"
        )
        impact = f"高并发客服请求首字等待显著增加，P95 端到端延迟达到 {e2e:.0f}ms。"
        actions = [
            RecommendedAction(action="TTFT/吞吐优先时切换 max_num_seqs=16、max_num_batched_tokens=8192 高并发档", risk="medium", impact="降低调度排队，但需接受并复核 TPOT 上升"),
            RecommendedAction(action="TPOT/端到端优先时保持 max_num_seqs=8、max_num_batched_tokens=4096 均衡档", risk="low", impact="避免过大的 decode batch"),
            RecommendedAction(action="保留 prefix caching，并按相同请求数、上下文和输出长度做回归", risk="low", impact="保证对比可复现"),
        ]
    elif max_memory >= 95 or "memory-pressure" in labels:
        category, severity = "memory_pressure", "warning"
        root = f"压测期间 GPU 显存峰值 {max_memory:.1f}%，KV cache 与运行时工作区余量偏低。"
        impact = "继续提高并发或上下文长度可能引发抢占、OOM 或长尾抖动。"
        actions = [RecommendedAction(action="将 gpu_memory_utilization 下调 0.03--0.05 后复测", risk="medium", impact="留出运行时显存余量")]
    elif tpot >= 100 or "decode-bound" in labels:
        category, severity = "decode_bottleneck", "warning"
        root = f"高负载场景 P95 TPOT={tpot:.1f}ms，decode 阶段成为主要瓶颈。"
        impact = "长回复生成速度下降。"
        actions = [RecommendedAction(action="降低同批序列数并评估量化 kernel/算子路径", risk="medium", impact="改善单 token 生成速度")]
    elif ttft >= 3000 or "prefill-bound" in labels:
        category, severity = "prefill_bottleneck", "warning"
        root = f"{context_length // 1024}K 上下文 P95 TTFT={ttft:.0f}ms，prefill 计算仍是瓶颈。"
        impact = "客服请求首字响应偏慢。"
        actions = [RecommendedAction(action="保持 prefix caching 与 chunked prefill，调整 batch token 预算", risk="low", impact="降低 prefill 等待")]
    else:
        category, severity = "inference_healthy", "info"
        root = "请求成功率、输出质量、TTFT 与 TPOT 均未触发异常门禁。"
        impact = "当前推理服务无明显性能或正确性异常。"
        actions = [RecommendedAction(action="保持配置并持续采集相同矩阵趋势", risk="low", impact="建立稳定基线")]

    evidence_items = [
        EvidenceItem(label="压测矩阵", source="benchmark", detail=f"run={benchmark.get('run_id')}，{len(scenarios)} 个场景"),
        EvidenceItem(label="最差 TTFT", source="benchmark", detail=f"context={context_length}, concurrency={concurrency}, P95 TTFT={ttft:.0f}ms"),
        EvidenceItem(label="Decode", source="benchmark", detail=f"P95 TPOT={tpot:.1f}ms，P95 E2E={e2e:.0f}ms"),
        EvidenceItem(label="正确性门禁", source="benchmark", detail=f"success={min_success:.1%}，quality={min_quality:.1%}"),
        EvidenceItem(label="GPU 显存", source="gpu", detail=f"场景峰值={max_memory:.1f}%"),
    ]
    comparison = _matching_baseline_scenario(baseline, context_length, concurrency)
    if comparison:
        base_ttft = _as_float(comparison.get("p95_ttft_ms"))
        if base_ttft > 0:
            gain = (base_ttft - ttft) / base_ttft
            evidence_items.append(EvidenceItem(
                label="Baseline 对比", source="benchmark",
                detail=f"TTFT {base_ttft:.0f}ms -> {ttft:.0f}ms，改善 {gain:.1%}",
            ))

    signature_actions = {
        "marlin_fp8_fallback": RecommendedAction(action="评估经质量校验的 INT4/AWQ 权重或原生 FP8 GPU，避免 3090 上的 Marlin FP8 fallback", risk="medium", impact="降低 GEMM 计算时间"),
        "gpu_p2p_unavailable": RecommendedAction(action="保持 disable_custom_all_reduce=true，并将跨卡 NCCL 作为硬件瓶颈记录", risk="low", impact="避免误判 custom all-reduce 收益"),
        "fp8_kv_uncalibrated": RecommendedAction(action="回滚 kv_cache_dtype=auto；只有生成并校验 KV scale 后才重测 FP8 KV", risk="low", impact="避免精度回归和运行不稳定"),
        "partial_prefill_unsupported": RecommendedAction(action="保持 max_num_partial_prefills=1", risk="low", impact="避免当前混合架构启动失败"),
        "prefix_cache_experimental": RecommendedAction(action="保留 prefix caching 时持续执行输出正确性门禁，并关注 vLLM 混合 Mamba 缓存修复", risk="low", impact="控制实验特性带来的正确性风险"),
        "possible_tensor_mismatch": RecommendedAction(action="升级 vLLM 前后复测 Qwen3.5 GDN/Mamba 输出哈希与质量门禁", risk="low", impact="确认 FLA shape warning 未造成输出回归"),
        "cuda_out_of_memory": RecommendedAction(action="恢复最近稳定的 gpu_memory_utilization/KV dtype，并重启 vLLM 清理异常状态", risk="low", impact="恢复请求处理"),
    }
    for signature, count in log_signatures.items():
        if _as_int(count) <= 0:
            continue
        evidence_items.append(EvidenceItem(label="vLLM 日志签名", source="logs", detail=f"{signature} x{count}"))
        action = signature_actions.get(signature)
        if action:
            actions.append(action)

    return DiagnoseResponse(
        status="completed", root_cause=root, confidence=0.9 if severity == "critical" else 0.86,
        impact=impact, evidence=evidence_items, recommended_actions=actions,
        related_resources=[
            RelatedResource(type="benchmark", id=_as_str(benchmark.get("run_id")), name=_as_str(benchmark.get("workload"))),
            RelatedResource(type="service", id=_as_str(benchmark.get("endpoint_id")), name=model_id),
        ],
        model_id=model_id, endpoint_id=_as_str(benchmark.get("endpoint_id")),
        category=category, severity=severity,
    )


def _inference_model_id(inference: Dict[str, Any], benchmark: Dict[str, Any]) -> str:
    """从压测/运行时证据推导模型，避免诊断结果把所有服务写成 27B。"""
    config = benchmark.get("config")
    vllm = config.get("vllm", {}) if isinstance(config, dict) else {}
    if not isinstance(vllm, dict):
        vllm = {}
    runtime = inference.get("runtime") or {}
    for value in (vllm.get("model_id"), vllm.get("model"), benchmark.get("model_id"), runtime.get("model")):
        if value:
            return str(value)
    endpoint = str(benchmark.get("endpoint_id") or "")
    return endpoint[:-5] if endpoint.endswith("-vllm") else endpoint or "unknown-model"


def _training_diagnose(evidence: Dict[str, Any]) -> DiagnoseResponse:
    training = evidence.get("training") or {}
    job = training.get("job") or {}
    live = training.get("pytorch_job") or {}
    pod = training.get("pod") or {}
    metadata = job.get("metadata") or {}
    status = _as_str(job.get("status")).lower()
    phase = _as_str(live.get("phase")) or _as_str(metadata.get("phase"))
    reason = _as_str(live.get("reason")) or _as_str(metadata.get("reason"))
    message = _as_str(live.get("message")) or _as_str(metadata.get("message"))
    logs = _as_str(pod.get("logs"))
    searchable = f"{reason}\n{message}\n{logs}".lower()

    if status == "failed" or phase.lower() == "failed":
        severity = "critical"
        if "out of memory" in searchable or "cuda oom" in searchable or "outofmemory" in searchable:
            category = "training_oom"
            root = "训练 Pod 出现 CUDA OOM，当前 batch、序列长度或激活显存超过可用显存。"
            actions = [
                RecommendedAction(action="降低 per_device_train_batch_size 并增加 gradient_accumulation_steps", risk="low", impact="降低峰值显存并保持有效 batch"),
                RecommendedAction(action="启用 gradient checkpointing、混合精度和 DeepSpeed ZeRO", risk="medium", impact="进一步减少训练显存"),
            ]
        elif any(token in searchable for token in ("nccl", "connection reset", "collective", "distributed", "torchrun")):
            category = "distributed_failure"
            root = "PyTorchJob 分布式通信失败，日志包含 NCCL/torchrun/collective 异常。"
            actions = [RecommendedAction(action="核对 WORLD_SIZE、RANK、NCCL 网卡和 Master/Worker 副本状态", risk="low", impact="恢复分布式 rendezvous 与通信")]
        elif any(token in searchable for token in ("jsondecode", "dataset", "column", "keyerror", "tokeniz")):
            category = "data_failure"
            root = "训练数据加载或 tokenization 失败，数据格式/字段与训练脚本契约不一致。"
            actions = [RecommendedAction(action="校验 DianJin 清洗集 messages 字段、编码、空样本与长度分布", risk="low", impact="修复数据入口")]
        elif any(token in searchable for token in ("checkpoint", "minio", "s3", "artifact", "permission denied")):
            category = "artifact_failure"
            root = "checkpoint 或 LoRA adapter 写入失败，产物存储路径/权限异常。"
            actions = [RecommendedAction(action="检查 OUTPUT_URI、MinIO 凭据、bucket 和挂载写权限", risk="low", impact="恢复产物归档")]
        else:
            category = "training_failure"
            root = f"PyTorchJob 训练失败：{message or reason or '未提供明确失败信息'}。"
            actions = [RecommendedAction(action="查看 Master Pod 尾部日志和 PyTorchJob conditions，补充原始异常", risk="low", impact="缩小根因范围")]
        impact = "本轮 LoRA adapter 无法产出，模型版本不能进入注册与推理验证。"
    elif status == "succeeded" and not job.get("output_artifact_uri"):
        category, severity = "artifact_failure", "warning"
        root = "训练任务已成功，但未记录 LoRA adapter 产物 URI，训练到模型注册的链路不完整。"
        impact = "模型版本无法追溯或部署。"
        actions = [RecommendedAction(action="核对 OUTPUT_URI 上传与成功回调中的模型注册逻辑", risk="low", impact="补齐产物血缘")]
    elif status in ("pending", "running"):
        category, severity = "training_in_progress", "info"
        root = f"训练任务处于 {status}（PyTorchJob phase={phase or 'Unknown'}），暂未发现明确失败信号。"
        impact = "任务仍在执行，当前结论仅代表实时快照。"
        actions = [RecommendedAction(action="持续观察 loss、GPU 显存、replica 状态和 checkpoint 事件", risk="low", impact="及时发现训练异常")]
    elif status == "cancelled":
        category, severity = "training_cancelled", "info"
        root = "训练任务由控制面取消，不判定为系统故障。"
        impact = "本轮训练未产出 adapter。"
        actions = [RecommendedAction(action="确认取消原因后按需重新提交任务", risk="low", impact="恢复实验")]
    else:
        category, severity = "training_healthy", "info"
        root = "训练任务已完成，且未发现控制面或产物异常。"
        impact = "训练链路状态正常。"
        actions = [RecommendedAction(action="进入 adapter 评测与推理回归", risk="low", impact="验证模型效果")]

    evidence_items = [
        EvidenceItem(label="训练任务", source="training", detail=f"job={job.get('id')}，status={status}，phase={phase or 'Unknown'}"),
        EvidenceItem(label="训练配置", source="training", detail=f"base_model={job.get('base_model')}，workers={job.get('workers')}，gpu/worker={job.get('gpus_per_worker')}"),
        EvidenceItem(label="副本状态", source="kubernetes", detail=json.dumps(live.get("replica_statuses") or {}, ensure_ascii=False)),
    ]
    if reason or message:
        evidence_items.append(EvidenceItem(label="失败条件", source="kubernetes", detail=f"reason={reason}，message={message}"))
    if logs:
        evidence_items.append(EvidenceItem(label="Pod 日志", source="logs", detail=logs[-500:]))
    return DiagnoseResponse(
        status="completed", root_cause=root, confidence=0.9 if severity == "critical" and logs else 0.75,
        impact=impact, evidence=evidence_items, recommended_actions=actions,
        related_resources=[
            RelatedResource(type="training_job", id=_as_str(job.get("id")), name=_as_str(job.get("name"))),
            RelatedResource(type="pytorch_job", id=_as_str(job.get("k8s_job_ref")), name=_as_str(job.get("name"))),
        ],
        model_id=_as_str(job.get("base_model")), endpoint_id=_as_str(job.get("k8s_job_ref")),
        category=category, severity=severity,
    )


def _scenario_memory(item: Dict[str, Any]) -> float:
    return _as_float((item.get("gpu_after") or {}).get("max_memory_utilization_percent"))


def _benchmark_param(benchmark: Dict[str, Any], key: str) -> int:
    return _as_int(((benchmark.get("config") or {}).get("vllm") or {}).get(key))


def _matching_baseline_scenario(baseline: Dict[str, Any], context_length: int, concurrency: int) -> Optional[Dict[str, Any]]:
    for item in (baseline.get("summary") or {}).get("scenarios", []):
        if _as_int(item.get("context_length")) == context_length and _as_int(item.get("concurrency")) == concurrency:
            return item
    return None


def _reason(error_rate: float, p95: float, active_incidents: List[Dict[str, Any]]):
    """Return (root_cause, impact, confidence, actions) by priority."""
    if error_rate >= ERROR_RATE_THRESHOLD:
        return (
            f"近窗口错误率 {error_rate:.1%} 超过 {ERROR_RATE_THRESHOLD:.0%} 阈值，推理链路存在失败请求。",
            "部分用户请求失败，影响可用性 SLA。",
            min(0.9, 0.5 + error_rate * 5),
            [
                RecommendedAction(
                    action="检查 vLLM/网关日志定位 5xx 来源（OOM / KV cache 满 / 上游超时）",
                    risk="low", impact="定位根因，无副作用",
                ),
                RecommendedAction(
                    action="必要时回滚最近一次部署或临时降低并发",
                    risk="medium", impact="恢复可用性，可能降低吞吐",
                ),
            ],
        )
    if p95 >= P95_LATENCY_THRESHOLD_MS:
        return (
            f"P95 延迟 {p95:.0f}ms 偏高，存在排队或 KV cache 压力。",
            "长尾延迟影响交互体验。",
            0.7,
            [
                RecommendedAction(
                    action="将路由策略切换为 least-latency / least-kv-cache",
                    risk="low", impact="缓解长尾延迟",
                ),
                RecommendedAction(
                    action="评估扩容副本或降低单请求 max_tokens",
                    risk="medium", impact="提升吞吐，增加成本",
                ),
            ],
        )
    if active_incidents:
        top = active_incidents[0]
        return (
            f"存在未解决故障事件「{top.get('title')}」（severity={top.get('severity')}），可能正在影响服务。",
            "依据事件严重度评估业务影响。",
            0.6,
            [RecommendedAction(
                action="跟进该故障事件的处理时间线并确认缓解措施",
                risk="low", impact="推进故障收敛",
            )],
        )
    return (
        "未发现显著异常：错误率与延迟均在阈值内，系统状态正常。",
        "无明显业务影响。",
        0.55,
        [RecommendedAction(
            action="维持当前配置，持续观测指标趋势",
            risk="low", impact="无",
        )],
    )


# ========================= live（LLM 出 JSON）=========================

SYSTEM_PROMPT = """你是云原生大模型训练与推理基础设施的运维诊断专家。基于给定证据（训练任务、\
推理压测、GPU、日志、Kubernetes 事件、配置和发布），定位根因并给出结构化诊断。

严格要求：只输出一个 JSON 对象，不要任何额外文字、解释或 markdown 代码块。证据最多 3 条，建议最多 3 条，每项文字简洁。键如下：
{
  "category": "general|request_failure|quality_regression|scheduler_saturation|memory_pressure|decode_bottleneck|prefill_bottleneck|training_oom|distributed_failure|data_failure|artifact_failure",
  "severity": "info|warning|critical",
  "root_cause": "一句话根因",
  "confidence": 0.0 到 1.0 之间的数字,
  "impact": "业务影响",
  "evidence": [{"label": "", "detail": "", "source": "benchmark|training|gpu|logs|metrics|incidents|deployments|kubernetes"}],
  "recommended_actions": [{"action": "", "risk": "low|medium|high", "impact": ""}],
  "related_resources": [{"type": "deployment|incident|config", "id": "", "name": ""}]
}
用简体中文填充文本字段，结论须引用证据中的具体数值。"""


def live_diagnose(req: DiagnoseRequest, cfg: Config) -> DiagnoseResponse:
    from aiservice import llm  # 延迟导入：单测 stub 时无需 requests

    base_url, model, _ = llm.resolve_upstream(cfg)
    messages = build_messages(req)
    content = llm.chat_completion(
        base_url, model, cfg.llm_api_key, messages,
        max_tokens=_opt_max_tokens(req, cfg),
        temperature=_opt_temperature(req, cfg),
        timeout=cfg.request_timeout,
        json_mode=True,
    )
    parsed = _parse_json_object(content)
    # 模型有时会把证据中的历史模型名抄进结论。推理专项诊断不能把
    # qwen36 的历史故障归到当前 qwen35 Run；检测到模型串线时回退到
    # 同一证据包的确定性规则结论，并在 error 中留下可解释原因。
    if parsed is not None and req.evidence.get("scope") == "inference":
        evidence_model = _inference_model_id(req.evidence.get("inference") or {}, (req.evidence.get("inference") or {}).get("benchmark") or {})
        if _parsed_mentions_other_model(parsed, evidence_model):
            fallback = stub_diagnose(req)
            fallback.mode = "rule_fallback"
            fallback.error = f"llm model mismatch: expected {evidence_model}"
            fallback.model_id = model
            fallback.endpoint_id = _endpoint_id_for_base(base_url)
            return fallback
    if parsed is None:
        # 绝不能把半截 JSON 当作“根因”投到前端。用相同证据生成可读的规则化结论，
        # 并保留 error 让控制台明确本次 LLM 结构化输出失败。
        fallback = stub_diagnose(req)
        fallback.mode = "rule_fallback"
        fallback.error = "llm returned incomplete or invalid JSON"
        fallback.model_id = model
        fallback.endpoint_id = _endpoint_id_for_base(base_url)
        return fallback
    return _response_from_parsed(parsed, cfg, model_id=model, endpoint_id=_endpoint_id_for_base(base_url))


def build_messages(req: DiagnoseRequest) -> List[Dict[str, str]]:
    summary = summarize_evidence(req.evidence or {})
    user = f"== 证据摘要 ==\n{summary}\n\n== 运维问题 ==\n{req.question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def summarize_evidence(evidence: Dict[str, Any], max_chars: int = 1800) -> str:
    """Compact the evidence bundle to fit qwen3-4b's context budget."""
    metrics = evidence.get("metrics") or {}
    incidents = evidence.get("incidents") or []
    deployments = evidence.get("deployments") or []
    config_items = evidence.get("config_items") or []
    k8s = evidence.get("kubernetes") or {}

    inference = evidence.get("inference") or {}
    benchmark = inference.get("benchmark") or {}
    current_model = _inference_model_id(inference, benchmark) if evidence.get("scope") == "inference" else ""

    def relevant(item: Dict[str, Any]) -> bool:
        if not current_model:
            return True
        text = json.dumps(item, ensure_ascii=False).lower()
        # 模型相关历史只保留当前模型；不含模型名的通用事件也保留，
        # 以免丢失 GPU/OOM 等跨服务证据。
        model_tokens = re.findall(r"qwen[0-9][a-z0-9_.-]*", text)
        return not model_tokens or current_model.lower() in text

    compact = {
        "scope": evidence.get("scope") or "general",
        "current_model": current_model or None,
        "metrics": {
            k: metrics.get(k) for k in (
                "request_count", "qps", "error_rate",
                "p95_latency_ms", "p99_latency_ms", "mean_latency_ms",
                "p95_ttft_ms", "total_tokens",
            ) if k in metrics
        },
        "incidents": [
            {"title": i.get("title"), "severity": i.get("severity"), "status": i.get("status")}
            for i in incidents if isinstance(i, dict) and relevant(i)
        ],
        "deployments": [
            {"name": d.get("name"), "status": d.get("status"), "version": d.get("version")}
            for d in deployments if isinstance(d, dict) and relevant(d)
        ],
        "config_items": [
            {"config_key": c.get("config_key"), "active_version": c.get("active_version")}
            for c in config_items[:5]
        ],
        "kubernetes_available": bool(k8s.get("available")),
    }
    if evidence.get("inference"):
        compact["inference"] = _compact_inference(evidence.get("inference") or {})
        compact["inference"]["model_id"] = current_model
    if evidence.get("training"):
        compact["training"] = _compact_training(evidence.get("training") or {})
    text = json.dumps(compact, ensure_ascii=False, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _parsed_mentions_other_model(parsed: Dict[str, Any], expected_model: str) -> bool:
    if not expected_model:
        return False
    text = json.dumps(parsed, ensure_ascii=False).lower()
    expected = expected_model.lower()
    for token in re.findall(r"qwen[0-9][a-z0-9_.-]*", text):
        token = token.rstrip(".,;:)]}，。；：）")
        if token and token != expected:
            return True
    return False


def _response_from_parsed(
    parsed: Dict[str, Any], cfg: Config, *, model_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
) -> DiagnoseResponse:
    evidence_items = [
        EvidenceItem(
            label=_as_str(e.get("label")),
            detail=_as_str(e.get("detail")),
            source=_as_str(e.get("source")) or "metrics",
        )
        for e in _as_list(parsed.get("evidence"))
    ]
    actions = [
        RecommendedAction(
            action=_as_str(a.get("action")),
            risk=_as_str(a.get("risk")) or "low",
            impact=_as_str(a.get("impact")),
        )
        for a in _as_list(parsed.get("recommended_actions"))
    ]
    related = [
        RelatedResource(
            type=_as_str(r.get("type")) or "service",
            id=_as_str(r.get("id")) or None,
            name=_as_str(r.get("name")) or None,
        )
        for r in _as_list(parsed.get("related_resources"))
    ]
    return DiagnoseResponse(
        status="completed",
        category=_as_str(parsed.get("category")) or "general",
        severity=_as_str(parsed.get("severity")) or "info",
        root_cause=_as_str(parsed.get("root_cause")),
        confidence=_as_optional_float(parsed.get("confidence")),
        impact=_as_str(parsed.get("impact")),
        evidence=evidence_items,
        recommended_actions=actions,
        related_resources=related,
        model_id=model_id or cfg.llm_model,
        endpoint_id=endpoint_id or _endpoint_id(cfg),
    )


def _compact_inference(inference: Dict[str, Any]) -> Dict[str, Any]:
    benchmark = inference.get("benchmark") or {}
    scenarios = (benchmark.get("summary") or {}).get("scenarios") or []
    return {
        "run_id": benchmark.get("run_id"),
        "endpoint_id": benchmark.get("endpoint_id"),
        "vllm": (benchmark.get("config") or {}).get("vllm") or {},
        "scenario_count": len(scenarios),
        "scenarios": [
            {
                key: item.get(key) for key in (
                    "context_length", "concurrency", "success_rate", "quality_gate_pass_rate",
                    "p95_ttft_ms", "p95_tpot_ms", "p95_ms", "output_throughput_tokens_per_second",
                )
            }
            for item in scenarios[:12]
        ],
        "runtime": inference.get("runtime") or {},
        "runtime_log_signatures": (inference.get("runtime_logs") or {}).get("signatures") or {},
        "gpu": inference.get("gpu") or {},
    }


def _compact_training(training: Dict[str, Any]) -> Dict[str, Any]:
    job = training.get("job") or {}
    live = training.get("pytorch_job") or {}
    pod = training.get("pod") or {}
    return {
        "job": {
            key: job.get(key) for key in (
                "id", "name", "status", "base_model", "dataset_uri", "workers",
                "gpus_per_worker", "k8s_job_ref", "output_artifact_uri", "hyperparams",
            )
        },
        "pytorch_job": {
            key: live.get(key) for key in (
                "available", "phase", "reason", "message", "replica_statuses",
            )
        },
        "pod": {
            "name": pod.get("pod"),
            "available": pod.get("available"),
            "logs_tail": _as_str(pod.get("logs"))[-2500:],
            "error": pod.get("error"),
        },
        "gpu": training.get("gpu") or {},
    }


def _parse_json_object(content: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from model output (handles prose/code-fence wrapping)."""
    if not content:
        return None
    text = content.strip()
    # 去掉 ```json ... ``` 围栏
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _endpoint_id(cfg: Config) -> str:
    """诊断真实走向：不能把直连 vLLM 标记为 AIBrix Gateway。"""
    base = (cfg.llm_base_url or "").lower()
    return "aibrix-gateway" if _is_aibrix_base(base) else "direct-vllm"


def _endpoint_id_for_base(base_url: str) -> str:
    base = (base_url or "").lower()
    return "aibrix-gateway" if _is_aibrix_base(base) else "direct-vllm"


def _is_aibrix_base(base_url: str) -> bool:
    return ":8010" in base_url or ":30080" in base_url or "aibrix" in base_url


# ---- coercion helpers（容错，证据来源可能字段缺失/类型不一）----


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def _as_list(v: Any) -> List[Dict[str, Any]]:
    return [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []


def _opt_max_tokens(req: DiagnoseRequest, cfg: Config) -> int:
    if req.options and req.options.max_tokens:
        return max(req.options.max_tokens, 1536)
    return max(cfg.default_max_tokens, 1536)


def _opt_temperature(req: DiagnoseRequest, cfg: Config) -> float:
    if req.options and req.options.temperature is not None:
        return req.options.temperature
    return cfg.default_temperature
