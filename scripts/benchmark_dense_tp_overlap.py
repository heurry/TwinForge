#!/usr/bin/env python3
"""Benchmark dense TP communication/computation overlap on two CUDA ranks."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class Buffers:
    input_a: torch.Tensor
    input_b: torch.Tensor
    input_full: torch.Tensor
    weight: torch.Tensor
    output_a: torch.Tensor
    output_b: torch.Tensor
    output_full: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch-tokens", default="1,2,4,8,16,64,256,512")
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--local-inner-size", type=int, default=8704)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def allocate(tokens: int, hidden_size: int, local_inner_size: int) -> Buffers:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260810)
    input_a = torch.randn(
        tokens, local_inner_size, device="cuda", dtype=torch.float16, generator=generator
    )
    input_b = torch.randn(
        tokens, local_inner_size, device="cuda", dtype=torch.float16, generator=generator
    )
    weight = torch.randn(
        local_inner_size,
        hidden_size,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    input_full = torch.cat((input_a, input_b), dim=0)
    return Buffers(
        input_a=input_a,
        input_b=input_b,
        input_full=input_full,
        weight=weight,
        output_a=torch.empty(tokens, hidden_size, device="cuda", dtype=torch.float16),
        output_b=torch.empty(tokens, hidden_size, device="cuda", dtype=torch.float16),
        output_full=torch.empty(
            tokens * 2, hidden_size, device="cuda", dtype=torch.float16
        ),
    )


def timed(fn, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        local_ms = (time.perf_counter() - started) * 1000
        duration = torch.tensor([local_ms], device="cuda", dtype=torch.float64)
        dist.all_reduce(duration, op=dist.ReduceOp.MAX)
        samples.append(duration.item())
    return samples


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * quantile))
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
    }


def benchmark(tokens: int, args: argparse.Namespace) -> dict[str, object]:
    buffers = allocate(tokens, args.hidden_size, args.local_inner_size)
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()
    ready_a = torch.cuda.Event()
    ready_b = torch.cuda.Event()
    comm_done = torch.cuda.Event()

    def sequential() -> None:
        torch.mm(buffers.input_a, buffers.weight, out=buffers.output_a)
        dist.all_reduce(buffers.output_a)
        torch.mm(buffers.input_b, buffers.weight, out=buffers.output_b)
        dist.all_reduce(buffers.output_b)

    def monolithic() -> None:
        torch.mm(buffers.input_full, buffers.weight, out=buffers.output_full)
        dist.all_reduce(buffers.output_full)

    def compute_only() -> None:
        torch.mm(buffers.input_a, buffers.weight, out=buffers.output_a)
        torch.mm(buffers.input_b, buffers.weight, out=buffers.output_b)

    def comm_only() -> None:
        buffers.output_a.fill_(1)
        buffers.output_b.fill_(2)
        dist.all_reduce(buffers.output_a)
        dist.all_reduce(buffers.output_b)

    def overlapped() -> None:
        with torch.cuda.stream(compute_stream):
            torch.mm(buffers.input_a, buffers.weight, out=buffers.output_a)
            ready_a.record(compute_stream)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_a)
            work_a = dist.all_reduce(buffers.output_a, async_op=True)
        with torch.cuda.stream(compute_stream):
            torch.mm(buffers.input_b, buffers.weight, out=buffers.output_b)
            ready_b.record(compute_stream)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_event(ready_b)
            work_b = dist.all_reduce(buffers.output_b, async_op=True)
            comm_done.record(comm_stream)
        work_a.wait()
        work_b.wait()
        torch.cuda.current_stream().wait_event(comm_done)

    iterations = args.iterations
    if tokens >= 1024:
        iterations = min(iterations, 15)
    elif tokens >= 256:
        iterations = min(iterations, 20)

    monolithic()
    torch.cuda.synchronize()
    reference = buffers.output_full.clone()
    overlapped()
    torch.cuda.synchronize()
    candidate = torch.cat((buffers.output_a, buffers.output_b), dim=0)
    abs_error = (candidate - reference).abs()
    max_abs_error = abs_error.max()
    mean_abs_error = abs_error.mean()
    max_reference_abs = reference.abs().max()
    dist.all_reduce(max_abs_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(mean_abs_error, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_reference_abs, op=dist.ReduceOp.MAX)
    correct = torch.tensor(
        int(torch.allclose(candidate, reference, rtol=5e-3, atol=1.0)),
        device="cuda",
    )
    dist.all_reduce(correct, op=dist.ReduceOp.MIN)

    compute = summarize(timed(compute_only, args.warmup, iterations))
    comm = summarize(timed(comm_only, args.warmup, iterations))
    whole_batch = summarize(timed(monolithic, args.warmup, iterations))
    baseline = summarize(timed(sequential, args.warmup, iterations))
    overlap = summarize(timed(overlapped, args.warmup, iterations))

    split_gain = (baseline["median_ms"] - overlap["median_ms"]) / baseline[
        "median_ms"
    ]
    end_to_end_gain = (
        whole_batch["median_ms"] - overlap["median_ms"]
    ) / whole_batch["median_ms"]
    serial_components = compute["median_ms"] + comm["median_ms"]
    hidden_ms = max(0.0, serial_components - overlap["median_ms"])
    return {
        "microbatch_tokens": tokens,
        "total_tokens": tokens * 2,
        "message_bytes_per_collective": tokens * args.hidden_size * 2,
        "iterations": iterations,
        "compute_only": compute,
        "comm_only": comm,
        "monolithic": whole_batch,
        "sequential": baseline,
        "overlap": overlap,
        "split_serial_gain_percent": split_gain * 100,
        "end_to_end_gain_percent": end_to_end_gain * 100,
        "hidden_time_ms": hidden_ms,
        "max_abs_error": max_abs_error.item(),
        "mean_abs_error": mean_abs_error.item(),
        "relative_max_error": max_abs_error.item()
        / max(max_reference_abs.item(), 1e-12),
        "correct": bool(correct.item()),
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    results = []
    for raw in args.microbatch_tokens.split(","):
        tokens = int(raw.strip())
        result = benchmark(tokens, args)
        results.append(result)
        if rank == 0:
            print(json.dumps(result, separators=(",", ":")), flush=True)

    if rank == 0:
        report = {
            "host": socket.gethostname(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": ".".join(map(str, torch.cuda.nccl.version())),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "peer_access": torch.cuda.can_device_access_peer(0, 1),
            "hidden_size": args.hidden_size,
            "local_inner_size": args.local_inner_size,
            "results": results,
        }
        print("SUMMARY=" + json.dumps(report, separators=(",", ":")), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
