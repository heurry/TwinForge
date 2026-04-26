from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


ROLE_ALIASES = {
    "human": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "gpt": "assistant",
    "model": "assistant",
    "user": "user",
    "system": "system",
}


@dataclass(frozen=True)
class TokenizedSFTDocument:
    input_ids: List[int]
    labels: List[int]
    source: Optional[str] = None


@dataclass
class SFTPackingStats:
    conversations: int = 0
    tokens_before_packing: int = 0
    tokens_after_packing: int = 0
    packed_sequences: int = 0
    dropped_tail_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "conversations": self.conversations,
            "tokens_before_packing": self.tokens_before_packing,
            "tokens_after_packing": self.tokens_after_packing,
            "packed_sequences": self.packed_sequences,
            "dropped_tail_tokens": self.dropped_tail_tokens,
        }


def normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    role_text = str(role).strip().lower()
    return ROLE_ALIASES.get(role_text)


def normalize_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
        if parts:
            return "\n".join(parts)
    return None


def normalize_messages(raw_messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(raw_messages, list):
        return None

    merged_messages: List[Dict[str, str]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = normalize_role(raw_message.get("role"))
        content = normalize_content(raw_message.get("content"))
        if role is None or content is None:
            continue

        if merged_messages and merged_messages[-1]["role"] == role:
            merged_messages[-1]["content"] += "\n\n" + content
        else:
            merged_messages.append({"role": role, "content": content})

    if not merged_messages:
        return None

    system_messages: List[Dict[str, str]] = []
    conversation_messages = merged_messages
    if merged_messages[0]["role"] == "system":
        system_messages = [merged_messages[0]]
        conversation_messages = merged_messages[1:]

    while conversation_messages and conversation_messages[0]["role"] != "user":
        conversation_messages = conversation_messages[1:]

    if not conversation_messages:
        return None

    if not any(message["role"] == "assistant" for message in conversation_messages):
        return None

    normalized = system_messages + conversation_messages
    if len(normalized) < 2:
        return None
    return normalized


def assign_split(messages: List[Dict[str, str]], val_ratio: float) -> str:
    digest_input = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.md5(digest_input.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 10000
    return "val" if bucket < int(val_ratio * 10000) else "train"


def _flatten_token_sequence(values: Any) -> List[int]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, tuple):
        values = list(values)
    if isinstance(values, list) and values and isinstance(values[0], list):
        if len(values) != 1:
            raise ValueError("Expected a single token sequence, but received batched token ids.")
        values = values[0]
    return list(values)


def _extract_input_ids(encoded: Any) -> List[int]:
    if hasattr(encoded, "get"):
        return _flatten_token_sequence(encoded.get("input_ids"))
    return _flatten_token_sequence(encoded)


def _render_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not isinstance(rendered, str):
        raise TypeError(f"Expected rendered chat template to be a string, got {type(rendered)!r}")
    return rendered


def _build_assistant_spans(tokenizer, messages: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    rendered_prefixes = [""]
    for end in range(1, len(messages) + 1):
        rendered_prefixes.append(_render_chat_template(tokenizer, messages[:end]))

    assistant_spans: List[Tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start = len(rendered_prefixes[idx])
        end = len(rendered_prefixes[idx + 1])
        if end > start:
            assistant_spans.append((start, end))
    return assistant_spans


def tokenize_sft_messages(
    tokenizer,
    messages: List[Dict[str, str]],
    train_on_prompt: bool = False,
) -> TokenizedSFTDocument:
    rendered_text = _render_chat_template(tokenizer, messages)
    tokenized = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=not train_on_prompt,
    )
    input_ids = _extract_input_ids(tokenized)
    labels = list(input_ids) if train_on_prompt else [-100] * len(input_ids)
    if train_on_prompt:
        return TokenizedSFTDocument(input_ids=input_ids, labels=labels)

    assistant_spans = _build_assistant_spans(tokenizer, messages)
    offsets = tokenized.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer did not return offset_mapping for SFT assistant masking.")

    span_index = 0
    for position, (token_id, offset) in enumerate(zip(input_ids, offsets)):
        if not isinstance(offset, Sequence) or len(offset) != 2:
            continue
        start, end = int(offset[0]), int(offset[1])
        while span_index < len(assistant_spans) and start >= assistant_spans[span_index][1]:
            span_index += 1
        if span_index >= len(assistant_spans):
            break
        span_start, span_end = assistant_spans[span_index]
        if end > span_start and start < span_end:
            labels[position] = token_id

    return TokenizedSFTDocument(input_ids=input_ids, labels=labels)


def iter_packed_sft_examples(
    documents: Iterable[TokenizedSFTDocument],
    seq_length: int,
    drop_remainder: bool = True,
    stats: Optional[SFTPackingStats] = None,
) -> Iterator[Dict[str, List[int]]]:
    packing_stats = stats or SFTPackingStats()
    input_buffer: List[int] = []
    label_buffer: List[int] = []
    buffer_start = 0

    for document in documents:
        if not document.input_ids:
            continue
        packing_stats.conversations += 1
        packing_stats.tokens_before_packing += len(document.input_ids)
        input_buffer.extend(document.input_ids)
        label_buffer.extend(document.labels)

        while len(input_buffer) - buffer_start >= seq_length:
            end = buffer_start + seq_length
            chunk_input_ids = input_buffer[buffer_start:end]
            chunk_labels = label_buffer[buffer_start:end]
            packing_stats.packed_sequences += 1
            packing_stats.tokens_after_packing += len(chunk_input_ids)
            yield {
                "input_ids": chunk_input_ids,
                "attention_mask": [1] * len(chunk_input_ids),
                "labels": chunk_labels,
            }
            buffer_start = end

            if buffer_start >= seq_length * 8:
                input_buffer = input_buffer[buffer_start:]
                label_buffer = label_buffer[buffer_start:]
                buffer_start = 0

    tail_tokens = len(input_buffer) - buffer_start
    if tail_tokens > 0 and not drop_remainder:
        chunk_input_ids = input_buffer[buffer_start:]
        chunk_labels = label_buffer[buffer_start:]
        packing_stats.packed_sequences += 1
        packing_stats.tokens_after_packing += len(chunk_input_ids)
        yield {
            "input_ids": chunk_input_ids,
            "attention_mask": [1] * len(chunk_input_ids),
            "labels": chunk_labels,
        }
        tail_tokens = 0

    packing_stats.dropped_tail_tokens = tail_tokens
