from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "exportBabyMambaEdgeModels.py"
EXPORT_ROOT = REPO_ROOT / "ESP32Models"
PROJECT_DIR = REPO_ROOT / "embedded" / "esp32BabyMambaNative"
PROJECT_MAIN = PROJECT_DIR / "main"
PROJECT_HEADER = PROJECT_MAIN / "babyMambaWeights.h"
DEPLOY_ROOT = REPO_ROOT / "ESP32Models" / "deviceRuns"
METRICS_JSON = REPO_ROOT / "ESP32Models" / "babyMambaEsp32Metrics.json"
METRICS_MD = REPO_ROOT / "ESP32Models" / "babyMambaEsp32Metrics.md"

IDF_ROOT = Path(r"C:\esp\v5.4.4\esp-idf")
IDF_PYTHON = Path(r"C:\Users\Xeron\.espressif\python_env\idf5.4_py3.13_env\Scripts\python.exe")

VARIANT_CHOICES = ("crossoverBiDirBabyMambaHar", "ciBabyMambaHar")
DATASET_CHOICES = (
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BabyMamba ESP32 native deployment sweep.")
    parser.add_argument("--variants", default="all", help="Comma-separated variants or 'all'.")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--port", default="COM9", help="ESP32 serial port.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial monitor baud rate.")
    parser.add_argument("--timeout", type=int, default=120, help="Serial capture timeout in seconds.")
    parser.add_argument("--flash-timeout", type=int, default=600, help="Build/flash timeout in seconds.")
    parser.add_argument("--output-root", type=Path, default=DEPLOY_ROOT, help="Deployment output directory.")
    return parser.parse_args()


def parse_csv_or_all(raw: str, choices: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    items = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [item for item in items if item not in choices]
    if invalid:
        raise ValueError(f"Invalid values {invalid}. Choices: {choices}")
    return items


def run_command(command: Sequence[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        check=False,
        text=True,
        capture_output=True,
    )


def export_bundle(variant: str, dataset: str) -> Path:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    completed = run_command(
        [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--variant",
            variant,
            "--datasets",
            dataset,
            "--output-root",
            str(EXPORT_ROOT),
            "--projection-format",
            "int8",
            "--strict",
        ],
        cwd=REPO_ROOT,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Export failed for {variant}/{dataset}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    bundle_dir = EXPORT_ROOT / variant / dataset
    if not (bundle_dir / "babyMambaWeights.h").exists():
        raise FileNotFoundError(f"Missing exported header: {bundle_dir / 'babyMambaWeights.h'}")
    return bundle_dir


def activate_bundle(bundle_dir: Path) -> Dict[str, object]:
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shutil.copyfile(bundle_dir / "babyMambaWeights.h", PROJECT_HEADER)
    return manifest


def run_idf(args: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
    command = (
        f"call {IDF_ROOT / 'export.bat'} >nul && "
        f"idf.py {' '.join(args)}"
    )
    return run_command(["cmd", "/c", command], cwd=PROJECT_DIR, timeout=timeout)


def parse_flash_bytes(build_dir: Path) -> int | None:
    bin_path = build_dir / "babyMambaEsp32Native.bin"
    if not bin_path.exists():
        return None
    return bin_path.stat().st_size


def capture_serial(port: str, baud: int, timeout_seconds: int, log_path: Path) -> Dict[str, object]:
    script = f"""
import json
import sys
import time
from pathlib import Path
import serial

port = {port!r}
baud = {baud}
timeout_seconds = {timeout_seconds}
log_path = Path({str(log_path)!r})
log_path.parent.mkdir(parents=True, exist_ok=True)
fields = {{}}
lines = []
deadline = time.time() + timeout_seconds

with serial.Serial(port, baudrate=baud, timeout=0.25) as ser:
    ser.dtr = False
    ser.rts = False
    time.sleep(0.2)
    ser.reset_input_buffer()
    while time.time() < deadline:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode('utf-8', errors='replace').rstrip()
        lines.append(line)
        if '=' in line:
            key, value = line.split('=', 1)
            fields[key.strip()] = value.strip()
        if line.strip() == '=== DONE ===':
            break

log_path.write_text('\\n'.join(lines), encoding='utf-8')
print(json.dumps({{'fields': fields, 'done': '=== DONE ===' in lines, 'line_count': len(lines)}}))
"""
    completed = run_command([str(IDF_PYTHON), "-c", script], timeout=timeout_seconds + 30)
    if completed.returncode != 0:
        raise RuntimeError(f"Serial capture failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    payload = json.loads(completed.stdout.strip() or "{}")
    if not payload.get("done"):
        raise RuntimeError(f"Serial capture timed out without completion marker. Log: {log_path}")
    return payload["fields"]


def coerce_float(fields: Dict[str, str], key: str) -> float | None:
    value = fields.get(key)
    if value is None or value == "":
        return None
    return float(value)


def coerce_int(fields: Dict[str, str], key: str) -> int | None:
    value = fields.get(key)
    if value is None or value == "":
        return None
    return int(float(value))


def build_row(variant: str, dataset: str, manifest: Dict[str, object], fields: Dict[str, str], flash_bytes: int | None) -> Dict[str, object]:
    latencies = [
        int(value)
        for key, value in sorted(fields.items())
        if re.fullmatch(r"iter_\d+_latency_us", key)
    ]
    return {
        "variant": variant,
        "dataset": dataset,
        "status": "ok",
        "seed": manifest.get("seed"),
        "flash_bytes": flash_bytes,
        "scratch_bytes": coerce_int(fields, "scratch_bytes"),
        "heap_total_before": coerce_int(fields, "heap_total_before"),
        "heap_free_before": coerce_int(fields, "heap_free_before"),
        "heap_free_after": coerce_int(fields, "heap_free_after"),
        "heap_used_after": coerce_int(fields, "heap_used_after"),
        "largest_block_before": coerce_int(fields, "largest_block_before"),
        "largest_block_after": coerce_int(fields, "largest_block_after"),
        "avg_latency_ms": coerce_float(fields, "avg_latency_ms"),
        "avg_latency_us": None if not latencies else statistics.mean(latencies),
        "latency_min_us": None if not latencies else min(latencies),
        "latency_max_us": None if not latencies else max(latencies),
        "parity_vs_pytorch_pct": coerce_float(fields, "parity_vs_pytorch_pct"),
        "parity_vs_engine_pct": coerce_float(fields, "parity_vs_engine_pct"),
        "max_abs_err_vs_pytorch": coerce_float(fields, "max_abs_err_vs_pytorch"),
        "predicted_class": coerce_int(fields, "predicted_class"),
        "predicted_label": fields.get("predicted_label"),
        "expected_label": fields.get("expected_label"),
        "port": fields.get("port", "COM9"),
        "manifest": manifest,
    }


def write_markdown(summary_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# BabyMamba ESP32 Metrics",
        "",
        "| Variant | Dataset | Status | Flash (B) | Scratch (B) | Heap Free Before (B) | Heap Used After (B) | Avg Latency (ms) | Parity vs PyTorch (%) | Predicted | Expected |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(rows, key=lambda item: (str(item["variant"]), str(item["dataset"]))):
        lines.append(
            f"| {row['variant']} | {row['dataset']} | {row['status']} | {row.get('flash_bytes','')} | "
            f"{row.get('scratch_bytes','')} | {row.get('heap_free_before','')} | {row.get('heap_used_after','')} | "
            f"{row.get('avg_latency_ms','')} | {row.get('parity_vs_pytorch_pct','')} | "
            f"{row.get('predicted_label','')} | {row.get('expected_label','')} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    variants = parse_csv_or_all(args.variants, VARIANT_CHOICES)
    datasets = parse_csv_or_all(args.datasets, DATASET_CHOICES)

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    for variant in variants:
        for dataset in datasets:
            print(f"[bundle] {variant} {dataset}")
            bundle_dir = export_bundle(variant, dataset)
            manifest = activate_bundle(bundle_dir)

            target_result_dir = args.output_root / variant / dataset
            target_result_dir.mkdir(parents=True, exist_ok=True)

            build = run_idf(["-DIDF_TARGET=esp32", "build"], timeout=args.flash_timeout)
            (target_result_dir / "build.log").write_text(build.stdout + "\n" + build.stderr, encoding="utf-8")
            if build.returncode != 0:
                raise RuntimeError(f"Build failed for {variant}/{dataset}. See {target_result_dir / 'build.log'}")

            flash_bytes = parse_flash_bytes(PROJECT_DIR / "build")

            flash = run_idf(["-DIDF_TARGET=esp32", "-p", args.port, "flash"], timeout=args.flash_timeout)
            (target_result_dir / "flash.log").write_text(flash.stdout + "\n" + flash.stderr, encoding="utf-8")
            if flash.returncode != 0:
                raise RuntimeError(f"Flash failed for {variant}/{dataset}. See {target_result_dir / 'flash.log'}")

            time.sleep(2.0)
            serial_log = target_result_dir / "serial.log"
            fields = capture_serial(args.port, args.baud, args.timeout, serial_log)
            row = build_row(variant, dataset, manifest, fields, flash_bytes)
            rows.append(row)
            print(
                f"[measured] {variant} {dataset} latency={row.get('avg_latency_ms')}ms "
                f"parity={row.get('parity_vs_pytorch_pct')}%"
            )

            METRICS_JSON.write_text(
                json.dumps({"generated_on": time.strftime("%Y-%m-%d"), "results": rows}, indent=2),
                encoding="utf-8",
            )
            write_markdown(METRICS_MD, rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
