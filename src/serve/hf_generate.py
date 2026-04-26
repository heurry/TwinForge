from __future__ import annotations

import argparse
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFChatGenerator:
    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.pad_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=False,
        )
        self.model.eval()

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated_ids.shape[0]),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one response with a local HF model.")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = HFChatGenerator(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    result = generator.generate(
        [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
    )
    print(result["text"])


if __name__ == "__main__":
    main()
