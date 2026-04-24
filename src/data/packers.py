from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class TokenizedDocument:
    input_ids: List[int]
    source: Optional[str] = None
    language: Optional[str] = None


@dataclass(frozen=True)
class PackConfig:
    seq_length: int
    eos_token_id: Optional[int]
    add_eos: bool = True
    drop_remainder: bool = True


@dataclass
class PackingStats:
    documents: int = 0
    tokens_before_packing: int = 0
    tokens_after_packing: int = 0
    packed_sequences: int = 0
    dropped_tail_tokens: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "documents": self.documents,
            "tokens_before_packing": self.tokens_before_packing,
            "tokens_after_packing": self.tokens_after_packing,
            "packed_sequences": self.packed_sequences,
            "dropped_tail_tokens": self.dropped_tail_tokens,
        }


def iter_packed_examples(
    documents: Iterable[TokenizedDocument],
    config: PackConfig,
    stats: Optional[PackingStats] = None,
) -> Iterator[Dict[str, List[int]]]:
    packing_stats = stats or PackingStats()
    if config.add_eos and config.eos_token_id is None:
        raise ValueError("add_eos=True requires a tokenizer with eos_token_id.")

    buffer: List[int] = []
    buffer_start = 0

    for document in documents:
        token_ids = list(document.input_ids)
        if config.add_eos:
            token_ids.append(int(config.eos_token_id))
        if not token_ids:
            continue

        packing_stats.documents += 1
        packing_stats.tokens_before_packing += len(token_ids)
        buffer.extend(token_ids)

        while len(buffer) - buffer_start >= config.seq_length:
            chunk = buffer[buffer_start:buffer_start + config.seq_length]
            buffer_start += config.seq_length
            packing_stats.packed_sequences += 1
            packing_stats.tokens_after_packing += len(chunk)
            yield {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
            }

            if buffer_start >= config.seq_length * 8:
                buffer = buffer[buffer_start:]
                buffer_start = 0

    tail_tokens = len(buffer) - buffer_start
    if tail_tokens > 0 and not config.drop_remainder:
        chunk = buffer[buffer_start:]
        packing_stats.packed_sequences += 1
        packing_stats.tokens_after_packing += len(chunk)
        yield {
            "input_ids": chunk,
            "attention_mask": [1] * len(chunk),
        }
        tail_tokens = 0

    packing_stats.dropped_tail_tokens = tail_tokens
