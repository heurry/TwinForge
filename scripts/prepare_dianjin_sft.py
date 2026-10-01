#!/usr/bin/env python3
"""Convert DianJin-CSC parquet dialogues into assistant-response SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as parquet


ROLE_MAP = {"客户": "user", "客服": "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/dianjin_csc_train.parquet")
    parser.add_argument("--output", default="data/cleaned/dianjin_csc_sft_train.jsonl")
    parser.add_argument("--max-dialogues", type=int, default=0)
    parser.add_argument("--context-turns", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)

    table = parquet.read_table(source, columns=["dialogue", "source"])
    samples = 0
    dialogues = 0
    with destination.open("w", encoding="utf-8") as handle:
        for row in table.to_pylist():
            if args.max_dialogues > 0 and dialogues >= args.max_dialogues:
                break
            context: list[dict[str, str]] = []
            for turn in row["dialogue"] or []:
                role = ROLE_MAP.get((turn.get("speaker") or "").strip())
                text = (turn.get("text") or "").strip()
                if not role or not text:
                    continue
                if role == "assistant":
                    handle.write(
                        json.dumps(
                            {
                                "messages": context[-args.context_turns :] if args.context_turns > 0 else context,
                                "response": text,
                                "source": row.get("source") or "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    samples += 1
                context.append({"role": role, "content": text})
            dialogues += 1

    print(json.dumps({"dialogues": dialogues, "samples": samples, "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
