"""Unit tests for the deterministic stub diagnosis.

Pure logic —— 只依赖 pydantic，不需要 FastAPI / requests / GPU，可在任意环境跑：
    cd apps/ai-service && python -m pytest tests/ -q
"""

from aiservice.config import Config
from aiservice.diagnose import live_diagnose, stub_diagnose
from aiservice.schemas import DiagnoseRequest


def _req(evidence):
    return DiagnoseRequest(question="为什么延迟升高？", evidence=evidence)


def test_high_error_rate_dominates():
    resp = stub_diagnose(_req({"metrics": {
        "error_rate": 0.1, "p95_latency_ms": 100, "p99_latency_ms": 120,
        "request_count": 50, "qps": 0.08,
    }}))
    assert resp.status == "completed"
    assert "错误率" in resp.root_cause
    assert resp.confidence is not None and resp.confidence >= 0.5
    assert len(resp.recommended_actions) >= 1
    assert any(e.source == "metrics" for e in resp.evidence)


def test_high_latency_when_errors_ok():
    resp = stub_diagnose(_req({"metrics": {
        "error_rate": 0.0, "p95_latency_ms": 5000, "p99_latency_ms": 8000,
        "request_count": 30, "qps": 0.05,
    }}))
    assert "延迟" in resp.root_cause or "P95" in resp.root_cause
    assert resp.confidence == 0.7


def test_active_incident_when_metrics_clean():
    resp = stub_diagnose(_req({
        "metrics": {"error_rate": 0.0, "p95_latency_ms": 100, "p99_latency_ms": 150,
                    "request_count": 10, "qps": 0.01},
        "incidents": [{"id": "i1", "title": "GPU OOM", "severity": "high", "status": "open"}],
    }))
    assert "GPU OOM" in resp.root_cause
    assert any(r.type == "incident" for r in resp.related_resources)


def test_nominal_when_all_clean():
    resp = stub_diagnose(_req({"metrics": {
        "error_rate": 0.0, "p95_latency_ms": 100, "p99_latency_ms": 150,
        "request_count": 10, "qps": 0.01,
    }}))
    assert "正常" in resp.root_cause or "未发现" in resp.root_cause


def test_related_resources_from_deployments():
    resp = stub_diagnose(_req({
        "metrics": {},
        "deployments": [{"id": "d1", "name": "qwen3-4b", "status": "running", "version": "v2"}],
    }))
    assert any(r.type == "deployment" and r.name == "qwen3-4b" for r in resp.related_resources)


def test_deterministic():
    evidence = {"metrics": {
        "error_rate": 0.05, "p95_latency_ms": 200, "p99_latency_ms": 300,
        "request_count": 40, "qps": 0.06,
    }}
    first = stub_diagnose(_req(evidence)).model_dump()
    second = stub_diagnose(_req(evidence)).model_dump()
    assert first == second


def test_live_invalid_json_uses_readable_rule_fallback(monkeypatch):
    from aiservice import llm
    monkeypatch.setattr(llm, "chat_completion", lambda *args, **kwargs: '{"root_cause":"unfinished"')
    cfg = Config("http://127.0.0.1:8020/v1", "test-model", "EMPTY", 5, "auto", 1024, 0.2, "0.0.0.0", 8200, "embed", 4)
    resp = live_diagnose(_req({"metrics": {"error_rate": 0.1, "p95_latency_ms": 100}}), cfg)
    assert resp.mode == "rule_fallback"
    assert resp.root_cause != '{"root_cause":"unfinished"'
    assert resp.error and "invalid JSON" in resp.error
    assert resp.endpoint_id == "direct-vllm"


def _inference_evidence(success_rate=1.0, quality_rate=1.0):
    return {
        "scope": "inference",
        "inference": {
            "benchmark": {
                "run_id": "run-prefix",
                "endpoint_id": "vllm-local",
                "workload": "dianjin-csc",
                "config": {"vllm": {"max_num_seqs": 8, "max_num_batched_tokens": 4096}},
                "summary": {"scenarios": [
                    {
                        "context_length": 2048,
                        "concurrency": 16,
                        "success_rate": success_rate,
                        "quality_gate_pass_rate": quality_rate,
                        "p95_ttft_ms": 4200,
                        "p95_tpot_ms": 72,
                        "p95_ms": 8900,
                        "gpu_after": {"max_memory_utilization_percent": 90},
                        "bottleneck": {"labels": ["scheduler-saturation"]},
                    }
                ]},
            },
            "baseline": {"summary": {"scenarios": [
                {"context_length": 2048, "concurrency": 16, "p95_ttft_ms": 7200}
            ]}},
        },
    }


def test_inference_scheduler_saturation_uses_benchmark_evidence():
    resp = stub_diagnose(_req(_inference_evidence()))
    assert resp.category == "scheduler_saturation"
    assert resp.severity == "warning"
    assert "max_num_seqs=8" in resp.root_cause
    assert any(item.source == "benchmark" for item in resp.evidence)
    assert any("max_num_seqs=16" in action.action for action in resp.recommended_actions)


def test_inference_success_gate_dominates_performance():
    resp = stub_diagnose(_req(_inference_evidence(success_rate=0.9)))
    assert resp.category == "request_failure"
    assert resp.severity == "critical"
    assert "90.0%" in resp.root_cause


def test_inference_runtime_log_signatures_are_actionable_evidence():
    evidence = _inference_evidence()
    evidence["inference"]["runtime_logs"] = {"signatures": {
        "marlin_fp8_fallback": 2,
        "gpu_p2p_unavailable": 1,
        "cuda_out_of_memory": 0,
    }}

    resp = stub_diagnose(_req(evidence))

    log_details = [item.detail for item in resp.evidence if item.source == "logs"]
    actions = [item.action for item in resp.recommended_actions]
    assert "marlin_fp8_fallback x2" in log_details
    assert "gpu_p2p_unavailable x1" in log_details
    assert not any("cuda_out_of_memory" in detail for detail in log_details)
    assert any("INT4/AWQ" in action for action in actions)
    assert any("disable_custom_all_reduce=true" in action for action in actions)


def _training_evidence(status="failed", logs="", output_artifact_uri=None):
    return {
        "scope": "training",
        "training": {
            "job": {
                "id": "job-1",
                "name": "qwen35-lora",
                "status": status,
                "base_model": "/model/Qwen3.5-4B",
                "workers": 2,
                "gpus_per_worker": 1,
                "k8s_job_ref": "qwen35-lora",
                "output_artifact_uri": output_artifact_uri,
            },
            "pytorch_job": {"phase": status.title(), "replica_statuses": {"Master": {"failed": 1}}},
            "pod": {"pod": "qwen35-lora-master-0", "available": True, "logs": logs},
        },
    }


def test_training_cuda_oom_diagnosis():
    resp = stub_diagnose(_req(_training_evidence(logs="torch.cuda.OutOfMemoryError: CUDA out of memory")))
    assert resp.category == "training_oom"
    assert resp.severity == "critical"
    assert any("gradient_accumulation_steps" in action.action for action in resp.recommended_actions)
    assert any(item.source == "logs" for item in resp.evidence)


def test_training_nccl_failure_diagnosis():
    resp = stub_diagnose(_req(_training_evidence(logs="NCCL collective operation timeout")))
    assert resp.category == "distributed_failure"
    assert "NCCL" in resp.root_cause


def test_training_running_is_informational():
    resp = stub_diagnose(_req(_training_evidence(status="running")))
    assert resp.category == "training_in_progress"
    assert resp.severity == "info"


def test_training_success_requires_artifact():
    resp = stub_diagnose(_req(_training_evidence(status="succeeded")))
    assert resp.category == "artifact_failure"
    assert resp.severity == "warning"
