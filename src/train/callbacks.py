from typing import Dict

import torch
from transformers import TrainerCallback


BYTES_PER_MIB = 1024**2


class CudaMemoryLoggingCallback(TrainerCallback):
    def __init__(self, logging_steps: int):
        self.logging_steps = max(int(logging_steps), 0)
        self._pending_logs: Dict[int, Dict[str, float]] = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self._pending_logs.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.logging_steps <= 0 or state.global_step <= 0:
            return control
        if state.global_step % self.logging_steps != 0:
            return control
        if not torch.cuda.is_available() or not state.is_world_process_zero:
            return control

        device = torch.cuda.current_device()
        self._pending_logs[state.global_step] = {
            "cuda_memory_allocated_mb": round(torch.cuda.memory_allocated(device) / BYTES_PER_MIB, 2),
            "cuda_memory_reserved_mb": round(torch.cuda.memory_reserved(device) / BYTES_PER_MIB, 2),
            "cuda_memory_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB, 2),
        }
        control.should_log = True
        return control

    def consume_pending_logs(self, step: int) -> Dict[str, float]:
        return self._pending_logs.pop(step, {})
