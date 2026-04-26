from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.serve.hf_generate import HFChatGenerator


ANSWER_PATTERN = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUMBER_PATTERN = re.compile(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model on GSM8K.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, default="data/raw/eval/gsm8k.jsonl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    return parser.parse_args()


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.replace(",", "").strip()
    return cleaned or None


def extract_gold_answer(answer: str) -> str | None:
    matched = ANSWER_PATTERN.search(answer)
    if matched:
        return normalize_number(matched.group(1))
    candidates = NUMBER_PATTERN.findall(answer)
    return normalize_number(candidates[-1]) if candidates else None


def extract_pred_answer(text: str) -> str | None:
    candidates = NUMBER_PATTERN.findall(text)
    return normalize_number(candidates[-1]) if candidates else None


def load_samples(path: Path, max_samples: int | None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def main() -> None:
    args = parse_args()
    samples = load_samples(Path(args.input_path), args.max_samples)
    generator = HFChatGenerator(model_path=args.model_path, max_new_tokens=args.max_new_tokens)

    correct = 0
    sample_outputs = []
    for index, sample in enumerate(samples):
        response = generator.generate(
            [
                {
                    "role": "system",
                    "content": "You are a careful math assistant. Solve the problem and end with 'Final Answer: <number>'.",
                },
                {"role": "user", "content": sample["question"]},
            ]
        )
        pred_answer = extract_pred_answer(response["text"])
        gold_answer = extract_gold_answer(sample["answer"])
        is_correct = pred_answer == gold_answer
        correct += int(is_correct)
        if index < 5:
            sample_outputs.append(
                {
                    "question": sample["question"],
                    "prediction": response["text"],
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "correct": is_correct,
                }
            )

    accuracy = correct / len(samples) if samples else 0.0
    result = {
        "task": "gsm8k",
        "model_path": args.model_path,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "exact_match": accuracy,
        "sample_outputs": sample_outputs,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote GSM8K results to {output_path}")


if __name__ == "__main__":
    main()
