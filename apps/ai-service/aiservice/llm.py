"""OpenAI-compatible LLM access (AIBrix gateway / vLLM) + SSE helpers.

Self-contained re-implementation of the few helpers from the legacy monolith so
this service has no `src/` dependency.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Tuple

import requests

from aiservice.config import Config


def normalize_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    return trimmed if trimmed.endswith("/v1") else trimmed + "/v1"


def resolve_upstream(cfg: Config) -> Tuple[str, str, List[str]]:
    """探活 OpenAI-compatible 上游，返回 (base_url, model, available_models)。

    ``AI_LLM_MODEL=auto`` 时不再假定某个模型存在，而是读取每个上游的
    ``/v1/models``，按候选优先级选择实际存活的模型。显式模型仍会严格匹配，
    这样配置错误会在健康检查中清晰暴露，而不是悄悄伪装成 27B 或 stub。
    """
    # 兼容旧的 Python 单测/嵌入式调用：手工构造 Config 时没有探活列表，
    # 仍按旧行为直接使用显式模型；生产 load_config 总会填充候选列表。
    if not cfg.llm_base_urls and not cfg.llm_candidate_models and cfg.llm_model.lower() != "auto":
        return normalize_base_url(cfg.llm_base_url), cfg.llm_model, []
    urls = list(cfg.llm_base_urls or (cfg.llm_base_url,))
    preferred = list(cfg.llm_candidate_models or ())
    if cfg.llm_model and cfg.llm_model.lower() != "auto":
        preferred = [cfg.llm_model]
    errors: List[str] = []
    for base in urls:
        try:
            response = requests.get(normalize_base_url(base) + "/models", timeout=2)
            if getattr(response, "status_code", 200) in (404, 405):
                # AIBrix/Envoy 的数据面通常只暴露 chat/completions，不提供
                # /v1/models；逐个用 1 token 请求探活候选模型。
                for candidate in preferred:
                    if _probe_chat_model(base, candidate):
                        return normalize_base_url(base), candidate, [candidate]
            response.raise_for_status()
            payload = response.json()
            models = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
            if not models:
                errors.append(f"{base}: /models returned no models")
                continue
            if cfg.llm_model and cfg.llm_model.lower() != "auto":
                if cfg.llm_model in models:
                    return normalize_base_url(base), cfg.llm_model, models
                errors.append(f"{base}: model {cfg.llm_model} not found ({', '.join(models)})")
                continue
            for model in preferred:
                if model in models:
                    return normalize_base_url(base), model, models
            # 未知模型也可工作：选上游声明的第一个，而不是硬编码 27B。
            return normalize_base_url(base), models[0], models
        except Exception as exc:  # noqa: BLE001 - 探活必须继续尝试下一个上游
            errors.append(f"{base}: {type(exc).__name__}: {exc}")
    detail = "; ".join(errors) if errors else "no configured upstream"
    raise RuntimeError("no live LLM upstream: " + detail)


def _probe_chat_model(base_url: str, model: str) -> bool:
    try:
        response = requests.post(
            normalize_base_url(base_url) + "/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=5,
        )
        return 200 <= response.status_code < 300
    except Exception:  # noqa: BLE001 - 继续尝试其它候选模型
        return False


def sse_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _headers(api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
    json_mode: bool = False,
) -> str:
    """Non-streaming completion; returns the assistant message content."""
    url = normalize_base_url(base_url) + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_mode:
        # vLLM 的 OpenAI 兼容接口支持 response_format，可约束解码为 JSON 对象。
        payload["response_format"] = {"type": "json_object"}
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return str(data["choices"][0]["message"]["content"])


def chat_with_tools(
    base_url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    """E1：带 OpenAI 工具调用的非流式 completion；返回 assistant message（可能含 tool_calls）。"""
    url = normalize_base_url(base_url) + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return dict(data["choices"][0]["message"])


def stream_chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Iterator[str]:
    """Yield content deltas from the upstream SSE stream (token by token)."""
    url = normalize_base_url(base_url) + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    with requests.post(url, headers=_headers(api_key), json=payload, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            raw = line[6:].strip()
            if raw == "[DONE]":
                break
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            text = delta.get("content")
            if text:
                yield str(text)
