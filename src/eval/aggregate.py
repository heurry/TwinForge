from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate task-level evaluation results.")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def load_result_files(input_dir: Path) -> List[Dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "summary.json":
            continue
        with path.open("r", encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    results = load_result_files(input_dir)
    task_summaries = []
    accuracy_values = []
    for result in results:
        accuracy = result.get("accuracy")
        if accuracy is not None:
            accuracy_values.append(float(accuracy))
        task_summaries.append(
            {
                "task": result.get("task"),
                "accuracy": result.get("accuracy"),
                "exact_match": result.get("exact_match"),
                "num_samples": result.get("num_samples"),
                "model_path": result.get("model_path"),
            }
        )

    summary = {
        "input_dir": str(input_dir),
        "num_tasks": len(task_summaries),
        "tasks": task_summaries,
        "mean_accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else None,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
