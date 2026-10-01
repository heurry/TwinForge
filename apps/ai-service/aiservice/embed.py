"""Text embeddings for RAG (5B.4c).

Live: calls the OpenAI-compatible `/v1/embeddings` of a cluster embedding model
(Qwen3-Embedding-0.6B, 1024-dim). Stub: deterministic unit pseudo-vectors so the
pipeline + CI run without a model/GPU. Query side gets the Qwen3-Embedding instruction
prefix; document side does not.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Tuple

import requests

from aiservice.config import Config
from aiservice.llm import normalize_base_url

# Qwen3-Embedding 推荐的检索查询指令（文档侧不加）。
_QUERY_INSTRUCT = "Instruct: Given a support question, retrieve relevant knowledge passages\nQuery: "


def _force_stub(cfg: Config) -> bool:
    base_url = cfg.embed_base_url or cfg.llm_base_url
    return cfg.stub_mode == "on" or (cfg.stub_mode != "off" and not base_url)


def _stub_vector(text: str, dim: int) -> List[float]:
    """确定性单位伪向量：由 text 的 sha256 播种的 LCG 生成，再 L2 归一化（余弦友好）。"""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") or 1
    x = seed
    vals: List[float] = []
    for _ in range(dim):
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        vals.append((x / 0x7FFFFFFF) * 2.0 - 1.0)
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _live_embed(texts: List[str], is_query: bool, cfg: Config) -> List[List[float]]:
    url = normalize_base_url(cfg.embed_base_url or cfg.llm_base_url) + "/embeddings"
    inputs = [(_QUERY_INSTRUCT + t) for t in texts] if is_query else list(texts)
    headers = {"Content-Type": "application/json"}
    embed_api_key = cfg.embed_api_key or cfg.llm_api_key
    if embed_api_key and embed_api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {embed_api_key}"
    resp = requests.post(
        url, headers=headers,
        json={"model": cfg.embed_model, "input": inputs},
        timeout=cfg.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    # OpenAI embeddings 契约：{"data": [{"embedding": [...]}, ...]}，保持入参顺序。
    return [list(item["embedding"]) for item in data["data"]]


def embed_texts(texts: List[str], is_query: bool, cfg: Config) -> Tuple[List[List[float]], str]:
    """返回 (向量列表, mode)。force_stub 或 live 失败 → 确定性 stub。"""
    if not texts:
        return [], "stub" if _force_stub(cfg) else "live"
    if _force_stub(cfg):
        return [_stub_vector(t, cfg.embed_dim) for t in texts], "stub"
    try:
        return _live_embed(texts, is_query, cfg), "live"
    except Exception:
        if cfg.stub_mode == "off":
            raise
        return [_stub_vector(t, cfg.embed_dim) for t in texts], "stub"
