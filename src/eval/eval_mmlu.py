from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from src.serve.hf_generate import HFChatGenerator


CHOICE_LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local model on MMLU mini.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, default="data/raw/eval/mmlu_mini.jsonl")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def load_samples(path: Path, max_samples: int | None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def build_prompt(sample: Dict[str, Any]) -> str:
    choice_lines = [f"{CHOICE_LETTERS[idx]}. {choice}" for idx, choice in enumerate(sample["choices"])]
    return "\n".join(
        [
            f"Subject: {sample['subject']}",
            f"Question: {sample['question']}",
            *choice_lines,
            "Answer with exactly one letter: A, B, C, or D.",
        ]
    )


def extract_choice(text: str) -> str | None:
    matched = re.search(r"\b([ABCD])\b", text.upper())
    return matched.group(1) if matched else None


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
                    "content": "You are a precise multiple-choice evaluator. Return only the final answer letter.",
                },
                {"role": "user", "content": build_prompt(sample)},
            ]
        )
        pred_choice = extract_choice(response["text"])
        gold_choice = CHOICE_LETTERS[int(sample["answer"])]
        is_correct = pred_choice == gold_choice
        correct += int(is_correct)
        if index < 5:
            sample_outputs.append(
                {
                    "question": sample["question"],
                    "prediction": response["text"],
                    "pred_choice": pred_choice,
                    "gold_choice": gold_choice,
                    "correct": is_correct,
                }
            )

    accuracy = correct / len(samples) if samples else 0.0
    result = {
        "task": "mmlu_mini",
        "model_path": args.model_path,
        "num_samples": len(samples),
        "accuracy": accuracy,
        "sample_outputs": sample_outputs,
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote MMLU results to {output_path}")


if __name__ == "__main__":
    main()
