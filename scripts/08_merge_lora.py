from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype="auto")
model = PeftModel.from_pretrained(base, "runs/sft/qwen25_3b_miniv2_sft_lora/checkpoint-final")
merged = model.merge_and_unload()
merged.save_pretrained("runs/sft/qwen25_3b_miniv2_sft_merged")
