"""Wire contract shared with the Go control plane (internal/aiclient).

Field names/types must stay in sync with aiclient.DiagnoseRequest/DiagnoseResult.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DiagnoseOptions(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class DiagnoseRequest(BaseModel):
    question: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    options: Optional[DiagnoseOptions] = None


class EvidenceItem(BaseModel):
    label: str
    detail: str
    source: str = "metrics"  # metrics | incidents | deployments | kubernetes | config


class RecommendedAction(BaseModel):
    action: str
    risk: str = "low"  # low | medium | high
    impact: str = ""


class RelatedResource(BaseModel):
    type: str  # deployment | incident | config | service
    id: Optional[str] = None
    name: Optional[str] = None


class DiagnoseResponse(BaseModel):
    status: str = "completed"  # completed | failed
    root_cause: str = ""
    confidence: Optional[float] = None
    impact: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
    related_resources: List[RelatedResource] = Field(default_factory=list)
    model_id: str = ""
    endpoint_id: str = ""
    latency_ms: Optional[float] = None
    mode: str = "stub"  # stub | live —— 排查无 GPU 时为何是假响应
    error: Optional[str] = None
    category: str = "general"
    severity: str = "info"  # info | warning | critical


class ChatRequest(BaseModel):
    question: Optional[str] = None
    messages: Optional[List[Dict[str, Any]]] = None
    max_tokens: int = 1024
    temperature: float = 0.2


# ===== E1：agentic 诊断单步（Go 编排循环，本服务当 reasoner）=====


class AgentTool(BaseModel):
    name: str
    description: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentStepRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tools: List[AgentTool] = Field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.2


class AgentToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AgentFinal(BaseModel):
    root_cause: str = ""
    confidence: Optional[float] = None
    impact: str = ""
    recommended_actions: List[Dict[str, Any]] = Field(default_factory=list)
    related_resources: List[Dict[str, Any]] = Field(default_factory=list)


class AgentStepResponse(BaseModel):
    mode: str = "stub"  # stub | live
    tool_calls: List[AgentToolCall] = Field(default_factory=list)
    final: Optional[AgentFinal] = None
    content: str = ""


class EmbedRequest(BaseModel):
    texts: List[str]
    is_query: bool = False  # 查询侧加 Qwen3-Embedding 指令前缀；文档侧不加


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dim: int
    mode: str = "stub"  # stub | live
