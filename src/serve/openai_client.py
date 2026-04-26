from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

import requests


def normalize_base_url(base_url: str) -> str:
    trimmed = base_url.rstrip("/")
    if trimmed.endswith("/v1"):
        return trimmed
    return trimmed + "/v1"


def _extract_error_detail(response: requests.Response) -> str | None:
    content_type = response.headers.get("Content-Type", "")
    if "application/json" in content_type:
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if isinstance(message, str) and message.strip():
                    return message.strip()
            for key in ("message", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    text = response.text.strip()
    if not text:
        return None
    return text if len(text) <= 500 else text[:497] + "..."


def _raise_for_status(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = _extract_error_detail(response)
        if detail:
            raise requests.HTTPError(f"{exc}. Response body: {detail}", response=response) from exc
        raise


def list_models(
    *,
    base_url: str,
    api_key: str = "EMPTY",
    timeout: int = 30,
) -> List[str]:
    url = normalize_base_url(base_url) + "/models"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    _raise_for_status(response)
    payload = response.json()
    data = payload.get("data", []) if isinstance(payload, dict) else []
    return [item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)]


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str = "EMPTY",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 300,
) -> Dict[str, Any]:
    url = normalize_base_url(base_url) + "/chat/completions"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    _raise_for_status(response)
    return response.json()


def stream_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    api_key: str = "EMPTY",
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = 300,
) -> Iterable[Dict[str, Any]]:
    url = normalize_base_url(base_url) + "/chat/completions"
    with requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        timeout=timeout,
        stream=True,
    ) as response:
        _raise_for_status(response)
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            payload = line[6:].strip()
            if payload == "[DONE]":
                break
            yield json.loads(payload)
