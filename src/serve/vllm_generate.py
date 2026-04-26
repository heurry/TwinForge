from __future__ import annotations

import argparse
import json

from src.serve.openai_client import chat_completion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call a vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--base_url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-1.7b-miniv2",
        help="OpenAI model id exposed by the server, not the local filesystem path.",
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=[
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(result["choices"][0]["message"]["content"].strip())


if __name__ == "__main__":
    main()
