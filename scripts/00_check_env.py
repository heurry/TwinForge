#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REQUIRED_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "deepspeed",
    "peft",
    "trl",
    "jsonlines",
]


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        return 127, f"{exc.__class__.__name__}: {exc}"

    return proc.returncode, proc.stdout.strip()


def check_modules():
    rows = []
    missing = []
    for name in REQUIRED_MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            rows.append((name, "ok", str(version)))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, "missing", f"{exc.__class__.__name__}: {exc}"))
            missing.append(name)
    return rows, missing


def check_torch_runtime():
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {
            "importable": False,
            "message": f"{exc.__class__.__name__}: {exc}",
        }

    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        return {
            "importable": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_available": cuda_available,
            "device_count": device_count,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "importable": True,
            "version": getattr(torch, "__version__", "unknown"),
            "cuda_runtime_error": f"{exc.__class__.__name__}: {exc}",
        }


def build_markdown(
    python_path: str,
    project_root: Path,
    download_check: tuple[int, str],
    train_check: tuple[int, str],
    module_rows,
    missing_modules,
    torch_runtime,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    nvidia_code, nvidia_output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    lines = [
        "# Baseline Report",
        "",
        f"- Date: {timestamp}",
        f"- Project root: `{project_root}`",
        f"- Python executable: `{python_path}`",
        f"- Python version: `{platform.python_version()}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Summary",
        "",
    ]

    if not missing_modules and download_check[0] == 0 and train_check[0] == 0:
        lines.append("- Status: ready for smoke validation.")
    else:
        lines.append("- Status: blocked by missing dependencies or startup failures.")

    if missing_modules:
        lines.append(f"- Missing critical modules: `{', '.join(missing_modules)}`")
    else:
        lines.append("- All critical Python modules are importable.")

    if torch_runtime.get("importable"):
        lines.append(
            f"- Torch runtime: version `{torch_runtime.get('version', 'unknown')}`, "
            f"cuda_available=`{torch_runtime.get('cuda_available', False)}`, "
            f"device_count=`{torch_runtime.get('device_count', 'unknown')}`"
        )
        if "cuda_runtime_error" in torch_runtime:
            lines.append(f"- Torch CUDA runtime warning: `{torch_runtime['cuda_runtime_error']}`")
    else:
        lines.append(f"- Torch runtime: import failed: `{torch_runtime['message']}`")

    lines.extend([
        "",
        "## Module Check",
        "",
        "| Module | Status | Detail |",
        "| --- | --- | --- |",
    ])

    for name, status, detail in module_rows:
        lines.append(f"| `{name}` | `{status}` | `{detail}` |")

    lines.extend([
        "",
        "## Command Startup Check",
        "",
        f"- `python scripts/01_download_data.py --help`: exit_code=`{download_check[0]}`",
    ])
    if download_check[1]:
        lines.append("```text")
        lines.append(download_check[1])
        lines.append("```")

    lines.append(f"- `python src/train/train_cpt.py --help`: exit_code=`{train_check[0]}`")
    if train_check[1]:
        lines.append("```text")
        lines.append(train_check[1])
        lines.append("```")

    lines.extend([
        "",
        "## GPU Check",
        "",
        f"- `nvidia-smi`: exit_code=`{nvidia_code}`",
    ])
    if nvidia_output:
        lines.append("```text")
        lines.append(nvidia_output)
        lines.append("```")

    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Check environment readiness for TwinForge.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional markdown report output path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    module_rows, missing_modules = check_modules()
    torch_runtime = check_torch_runtime()
    download_check = run_cmd([sys.executable, "scripts/01_download_data.py", "--help"])
    train_check = run_cmd([sys.executable, "src/train/train_cpt.py", "--help"])

    report = build_markdown(
        python_path=sys.executable,
        project_root=project_root,
        download_check=download_check,
        train_check=train_check,
        module_rows=module_rows,
        missing_modules=missing_modules,
        torch_runtime=torch_runtime,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
        print(f"[DONE] wrote baseline report to {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
