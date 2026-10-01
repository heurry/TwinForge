"""E1：agentic 诊断的单步推理（reasoner）。

Go 编排多轮循环，每步把「对话 + 可用工具」发来，本服务回：
  * tool_calls —— 模型还想取证（继续调只读工具）
  * final      —— 模型给出结构化最终诊断
两模式一契约（AgentStepResponse）：
  * live —— 调网关 LLM 的 OpenAI 工具调用；有 tool_calls 则回之，否则把 content 解析为 final
  * stub —— 确定性脚本：按提供的 tools 依次取证（每步一个），证据够了据其推导 final（复用 diagnose 规则）
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from aiservice.config import Config
from aiservice.diagnose import _as_float, _parse_json_object, _reason
from aiservice.schemas import (
    AgentFinal,
    AgentStepRequest,
    AgentStepResponse,
    AgentToolCall,
)

# stub 取证上限：最多调这么多工具就收口下结论（保证多轮但不无限）。
STUB_MAX_TOOLS = 5


def run_agent_step(req: AgentStepRequest, cfg: Config) -> AgentStepResponse:
    mode = cfg.stub_mode
    force_stub = mode == "on" or (mode != "off" and not cfg.llm_base_url)
    if force_stub:
        return _stub_step(req)
    try:
        return _live_step(req, cfg)
    except Exception as exc:  # noqa: BLE001 - 上游不稳定时优雅降级
        if mode == "off":
            return AgentStepResponse(
                mode="live",
                content=f"agent step failed: {type(exc).__name__}: {exc}",
                final=AgentFinal(root_cause=f"LLM 调用失败：{exc}", confidence=0.0, impact="无法完成取证"),
            )
        return _stub_step(req)  # auto：回退到确定性脚本，循环可继续


# ========================= live（LLM 工具调用）=========================


def _live_step(req: AgentStepRequest, cfg: Config) -> AgentStepResponse:
    from aiservice import llm  # 延迟导入：单测 stub 时无需 requests

    base_url, model, _ = llm.resolve_upstream(cfg)
    oa_tools = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in req.tools
    ]
    msg = llm.chat_with_tools(
        base_url, model, cfg.llm_api_key, req.messages, oa_tools,
        max_tokens=req.max_tokens, temperature=req.temperature, timeout=cfg.request_timeout,
    )
    content = str(msg.get("content") or "")

    calls: List[AgentToolCall] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = str(fn.get("name") or "")
        if not name:
            continue
        args: Dict[str, Any] = {}
        raw = fn.get("arguments")
        if isinstance(raw, dict):
            args = raw
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args = parsed
            except json.JSONDecodeError:
                args = {}
        calls.append(AgentToolCall(name=name, arguments=args))

    if calls:
        return AgentStepResponse(mode="live", tool_calls=calls, content=content)

    parsed = _parse_json_object(content)
    if parsed and parsed.get("root_cause"):
        return AgentStepResponse(mode="live", content=content, final=_final_from_parsed(parsed))
    # 有文本但非结构化 → 交回 Go 当结论（降级）。
    return AgentStepResponse(mode="live", content=content)


def _final_from_parsed(parsed: Dict[str, Any]) -> AgentFinal:
    actions = parsed.get("recommended_actions")
    related = parsed.get("related_resources")
    return AgentFinal(
        root_cause=str(parsed.get("root_cause") or ""),
        confidence=_as_optional_float(parsed.get("confidence")),
        impact=str(parsed.get("impact") or ""),
        recommended_actions=actions if isinstance(actions, list) else [],
        related_resources=related if isinstance(related, list) else [],
    )


# ========================= stub（确定性脚本）=========================


def _stub_step(req: AgentStepRequest) -> AgentStepResponse:
    called = {str(m.get("name")) for m in req.messages if m.get("role") == "tool"}
    uncalled = [t.name for t in req.tools if t.name not in called]
    if uncalled and len(called) < min(STUB_MAX_TOOLS, len(req.tools)):
        return AgentStepResponse(mode="stub", tool_calls=[AgentToolCall(name=uncalled[0])])
    return AgentStepResponse(mode="stub", final=_stub_final(req.messages))


def _stub_final(messages: List[Dict[str, Any]]) -> AgentFinal:
    metrics: Dict[str, Any] = {}
    incidents: List[Dict[str, Any]] = []
    not_running: List[str] = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m.get("content") or "")
        except (json.JSONDecodeError, TypeError):
            continue
        name = m.get("name")
        if name == "recent_metrics" and isinstance(data, dict):
            metrics = data
        elif name == "open_incidents" and isinstance(data, list):
            incidents = data
        elif name == "kubernetes_pods" and isinstance(data, dict):
            nr = data.get("not_running")
            not_running = nr if isinstance(nr, list) else []

    error_rate = _as_float(metrics.get("error_rate"))
    p95 = _as_float(metrics.get("p95_latency_ms"))
    active = [i for i in incidents if str(i.get("status")) != "resolved"]
    root_cause, impact, confidence, actions = _reason(error_rate, p95, active)
    if not_running:
        root_cause = f"{len(not_running)} 个 Pod 非 Running（{', '.join(str(x) for x in not_running[:3])}）；" + root_cause
        confidence = min(0.9, confidence + 0.1)

    related = [
        {"type": "incident", "id": str(i.get("id")), "name": str(i.get("title"))}
        for i in active[:3]
    ]
    return AgentFinal(
        root_cause=root_cause,
        confidence=round(confidence, 3),
        impact=impact,
        recommended_actions=[a.model_dump() for a in actions],
        related_resources=related,
    )


def _as_optional_float(v: Any):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
