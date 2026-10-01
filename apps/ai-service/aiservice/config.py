"""Env-driven runtime config (stdlib only, importable without 3rd-party deps)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


def _env(key: str, default: str) -> str:
    value = os.environ.get(key)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Config:
    llm_base_url: str       # OpenAI 兼容上游（默认 AIBrix gateway）
    llm_model: str
    llm_api_key: str
    request_timeout: int    # 调 LLM 的超时秒数
    stub_mode: str          # auto（试 live 失败回退 stub）| on（强制 stub）| off（强制 live）
    default_max_tokens: int
    default_temperature: float
    host: str
    port: int
    embed_model: str        # 5B.4c：嵌入模型（默认 Qwen3-Embedding-0.6B）
    embed_dim: int          # 5B.4c：嵌入维度（Qwen3-Embedding-0.6B = 1024）
    # 可选的 OpenAI-compatible 上游列表。服务会逐个探活并选择可用模型；
    # 保留 llm_base_url 以兼容旧配置和单元测试。
    llm_base_urls: Tuple[str, ...] = ()
    llm_candidate_models: Tuple[str, ...] = ()
    # Embedding 可以绑定与聊天模型不同的 OpenAI-compatible 服务。
    embed_base_url: str = ""
    embed_api_key: str = ""


def load_config() -> Config:
    base_url = _env("AI_LLM_BASE_URL", "http://127.0.0.1:8020/v1")
    base_urls = tuple(
        item.strip() for item in _env("AI_LLM_BASE_URLS", base_url).split(",") if item.strip()
    )
    # auto 时按轻量模型优先，避免诊断默认占用双卡 27B；显式 AI_LLM_MODEL
    # 仍可固定到某个已注册模型。
    candidate_models = tuple(
        item.strip() for item in _env(
            "AI_LLM_CANDIDATE_MODELS",
            "qwen35-4b-customer,qwen3-4b-customer,qwen38-27b-fp8,qwen36-27b-awq,qwen36-27b-fp8",
        ).split(",") if item.strip()
    )
    return Config(
        llm_base_url=base_url,
        llm_model=_env("AI_LLM_MODEL", "auto"),
        llm_api_key=_env("AI_LLM_API_KEY", "EMPTY"),
        request_timeout=int(_env("AI_LLM_TIMEOUT_SECONDS", "60")),
        stub_mode=_env("AI_STUB_MODE", "auto").lower(),
        default_max_tokens=int(_env("AI_DEFAULT_MAX_TOKENS", "1024")),
        default_temperature=float(_env("AI_DEFAULT_TEMPERATURE", "0.2")),
        host=_env("AI_SERVICE_HOST", "0.0.0.0"),
        port=int(_env("AI_SERVICE_PORT", "8200")),
        embed_model=_env("AI_EMBED_MODEL", "qwen3-embedding-0.6b"),
        embed_dim=int(_env("AI_EMBED_DIM", "1024")),
        llm_base_urls=base_urls,
        llm_candidate_models=candidate_models,
        embed_base_url=_env("AI_EMBED_BASE_URL", base_url),
        embed_api_key=_env("AI_EMBED_API_KEY", _env("AI_LLM_API_KEY", "EMPTY")),
    )
