from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("model/Qwen3-1.7B", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "runs/sft/qwen3_1_7b_miniv2_sft_lora/checkpoint-final")
merged = model.merge_and_unload()
merged.save_pretrained("runs/sft/qwen3_1_7b_miniv2_sft_merged")
