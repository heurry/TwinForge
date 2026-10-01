from aiservice.config import Config
from aiservice.llm import resolve_upstream


class _Response:
    def __init__(self, models):
        self._models = models

    def raise_for_status(self):
        return None

    def json(self):
        return {"data": [{"id": model} for model in self._models]}


def _cfg(**kwargs):
    values = dict(
        llm_base_url="http://first/v1", llm_model="auto", llm_api_key="EMPTY",
        request_timeout=5, stub_mode="auto", default_max_tokens=1024,
        default_temperature=0.2, host="0.0.0.0", port=8200,
        embed_model="embed", embed_dim=4,
        llm_base_urls=("http://first/v1", "http://second/v1"),
        llm_candidate_models=("qwen35-4b-customer", "qwen36-27b-fp8"),
    )
    values.update(kwargs)
    return Config(**values)


def test_auto_prefers_live_4b(monkeypatch):
    def get(url, timeout):
        assert url == "http://first/v1/models"
        return _Response(["qwen36-27b-fp8", "qwen35-4b-customer"])

    monkeypatch.setattr("aiservice.llm.requests.get", get)
    base, model, models = resolve_upstream(_cfg())
    assert base == "http://first/v1"
    assert model == "qwen35-4b-customer"
    assert models == ["qwen36-27b-fp8", "qwen35-4b-customer"]


def test_auto_falls_back_to_next_live_upstream(monkeypatch):
    def get(url, timeout):
        if url.startswith("http://first"):
            raise OSError("connection refused")
        return _Response(["qwen36-27b-awq"])

    monkeypatch.setattr("aiservice.llm.requests.get", get)
    base, model, _ = resolve_upstream(_cfg())
    assert base == "http://second/v1"
    assert model == "qwen36-27b-awq"


def test_explicit_model_is_strict(monkeypatch):
    monkeypatch.setattr("aiservice.llm.requests.get", lambda *args, **kwargs: _Response(["qwen35-4b-customer"]))
    base, model, _ = resolve_upstream(_cfg(llm_model="qwen35-4b-customer"))
    assert base == "http://first/v1"
    assert model == "qwen35-4b-customer"


def test_gateway_without_models_endpoint_uses_chat_probe(monkeypatch):
    def get(url, timeout):
        return type("Response", (), {"status_code": 404, "raise_for_status": lambda self: None, "json": lambda self: {}})()

    def post(url, headers, json, timeout):
        response = type("Response", (), {})()
        response.status_code = 200 if json["model"] == "qwen3-4b-customer" else 404
        return response

    monkeypatch.setattr("aiservice.llm.requests.get", get)
    monkeypatch.setattr("aiservice.llm.requests.post", post)
    base, model, models = resolve_upstream(_cfg(llm_base_urls=("http://gateway/v1",), llm_candidate_models=("qwen35-4b-customer", "qwen3-4b-customer")))
    assert base == "http://gateway/v1"
    assert model == "qwen3-4b-customer"
    assert models == ["qwen3-4b-customer"]


def test_inference_result_does_not_cross_contaminate_model(monkeypatch):
    from aiservice import llm
    from aiservice.diagnose import live_diagnose
    from aiservice.schemas import DiagnoseRequest

    monkeypatch.setattr(llm, "resolve_upstream", lambda cfg: ("http://first/v1", "qwen35-4b-customer", ["qwen35-4b-customer"]))
    monkeypatch.setattr(
        llm, "chat_completion",
        lambda *args, **kwargs: '{"category":"decode_bottleneck","severity":"critical","root_cause":"qwen36-27b-fp8 调度饱和"}',
    )
    req = DiagnoseRequest(
        question="检查推理服务",
        evidence={
            "scope": "inference",
            "inference": {
                "benchmark": {
                    "endpoint_id": "qwen35-4b-customer-vllm",
                    "summary": {"scenarios": [{"success_rate": 1, "quality_gate_pass_rate": 1, "p95_ttft_ms": 100, "p95_tpot_ms": 10, "p95_ms": 120, "context_length": 1024, "concurrency": 1}]},
                }
            },
        },
    )
    response = live_diagnose(req, _cfg())
    assert response.mode == "rule_fallback"
    assert response.model_id == "qwen35-4b-customer"
    assert response.error and "model mismatch" in response.error
