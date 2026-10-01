"""Local OpenAI-compatible embedding server for Qwen3-Embedding.

This intentionally runs one process and serializes model inference: duplicating
the model for multiple Uvicorn workers wastes RAM, while concurrent PyTorch CPU
forwards oversubscribe cores and increase tail latency.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from transformers import AutoModel, AutoTokenizer


MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/models/Qwen3-Embedding-0.6B")
MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "qwen3-embedding-0.6b")
MAX_LENGTH = int(os.environ.get("EMBEDDING_MAX_LENGTH", "1024"))
MAX_BATCH_SIZE = int(os.environ.get("EMBEDDING_MAX_BATCH_SIZE", "16"))
CPU_THREADS = int(os.environ.get("EMBEDDING_CPU_THREADS", "8"))
DTYPE_NAME = os.environ.get("EMBEDDING_DTYPE", "float32").lower()

_DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
if DTYPE_NAME not in _DTYPES:
    raise RuntimeError(f"unsupported EMBEDDING_DTYPE={DTYPE_NAME!r}")

torch.set_num_threads(max(1, CPU_THREADS))


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: Literal["float"] = "float"


class ModelState:
    def __init__(self) -> None:
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.dimension = 0
        self.loaded_at = 0.0
        self.lock = threading.Lock()

    def load(self) -> None:
        if not os.path.isdir(MODEL_PATH):
            raise RuntimeError(f"embedding model directory does not exist: {MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            padding_side="left",
        )
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            dtype=_DTYPES[DTYPE_NAME],
        ).eval()
        self.dimension = int(self.model.config.hidden_size)
        self.loaded_at = time.time()

    def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("embedding model is not ready")
        with self.lock, torch.inference_mode():
            batch = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            output = self.model(**batch).last_hidden_state
            # padding_side=left guarantees the final token is the last real token.
            vectors = F.normalize(output[:, -1, :].float(), p=2, dim=1)
            return vectors.cpu().tolist(), int(batch["attention_mask"].sum().item())


state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.load()
    yield


app = FastAPI(title="TwinForge Local Embedding Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimension": state.dimension,
        "device": "cpu",
        "dtype": DTYPE_NAME,
        "max_length": MAX_LENGTH,
    }


@app.get("/v1/models")
def models() -> dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return {
        "object": "list",
        "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "local"}],
    }


@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    if request.model != MODEL_NAME:
        raise HTTPException(status_code=404, detail=f"unknown embedding model: {request.model}")
    texts = [request.input] if isinstance(request.input, str) else request.input
    if not texts:
        raise HTTPException(status_code=400, detail="input must contain at least one string")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(status_code=400, detail=f"batch size exceeds {MAX_BATCH_SIZE}")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="every input must be a non-empty string")

    try:
        vectors, prompt_tokens = await run_in_threadpool(state.embed, texts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"embedding inference failed: {exc}") from exc
    return {
        "object": "list",
        "model": MODEL_NAME,
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(vectors)
        ],
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }
