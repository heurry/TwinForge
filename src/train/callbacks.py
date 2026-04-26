import json
import os
from inspect import signature
from pathlib import Path
from typing import Any, Dict

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


class TorchProfilerCallback(TrainerCallback):
    def __init__(
        self,
        output_dir: str,
        *,
        enabled: bool = False,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        row_limit: int = 50,
        export_chrome_trace: bool = True,
    ):
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.wait = max(int(wait), 0)
        self.warmup = max(int(warmup), 0)
        self.active = max(int(active), 1)
        self.repeat = max(int(repeat), 1)
        self.record_shapes = bool(record_shapes)
        self.profile_memory = bool(profile_memory)
        self.with_stack = bool(with_stack)
        self.with_flops = bool(with_flops)
        self.row_limit = max(int(row_limit), 1)
        self.export_chrome_trace = bool(export_chrome_trace)
        self._profiler: Any = None
        self._trace_index = 0
        self._rank = int(os.environ.get("RANK", "0"))
        self._sort_by = "self_cpu_time_total"

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.enabled or not state.is_world_process_zero:
            return control

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
            self._sort_by = "self_cuda_time_total"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        def on_trace_ready(profiler):
            if not self.export_chrome_trace:
                return
            trace_path = self.output_dir / f"torch_trace_rank{self._rank}_{self._trace_index}.json"
            profiler.export_chrome_trace(str(trace_path))
            self._trace_index += 1

        profiler_kwargs = dict(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=on_trace_ready if self.export_chrome_trace else None,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
        )
        if "acc_events" in signature(torch.profiler.profile).parameters:
            profiler_kwargs["acc_events"] = True

        self._profiler = torch.profiler.profile(**profiler_kwargs)
        self._profiler.start()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self._profiler is not None:
            self._profiler.step()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self._profiler is None:
            return control

        self._profiler.stop()
        if not state.is_world_process_zero:
            self._profiler = None
            return control

        key_averages = self._profiler.key_averages()
        summary_path = self.output_dir / f"torch_profile_summary_rank{self._rank}.json"
        table_path = self.output_dir / f"torch_profile_table_rank{self._rank}.txt"

        summary_path.write_text(
            json.dumps(
                {
                    "sort_by": self._sort_by,
                    "events": [self._serialize_event(event) for event in key_averages],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        table_path.write_text(
            key_averages.table(sort_by=self._sort_by, row_limit=self.row_limit),
            encoding="utf-8",
        )

        self._profiler = None
        return control

    @staticmethod
    def _serialize_event(event: Any) -> Dict[str, Any]:
        return {
            "key": event.key,
            "count": int(event.count),
            "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
            "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
            "self_cuda_time_total_us": float(getattr(event, "self_cuda_time_total", 0.0)),
            "cuda_time_total_us": float(getattr(event, "cuda_time_total", 0.0)),
            "self_cpu_memory_usage_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
            "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0)),
            "self_cuda_memory_usage_bytes": int(getattr(event, "self_cuda_memory_usage", 0)),
            "cuda_memory_usage_bytes": int(getattr(event, "cuda_memory_usage", 0)),
            "flops": float(getattr(event, "flops", 0.0)),
        }
