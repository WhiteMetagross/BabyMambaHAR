from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
PROJECT_DIR = REPO_ROOT / "embedded" / "pico2BabyMambaRuntime"
EXPORT_ROOT = REPO_ROOT / "Pico2Models"
REPORT_FILE = REPO_ROOT / "docs" / "Pico2DeploymentResultsReport.md"
DEPLOY_ROOT = REPO_ROOT / "Pico2Models"

ARDUINO_CLI = Path(r"C:\Program Files\Arduino CLI\arduino-cli.exe")
ARDUINO_PYTHON = Path(
    r"C:\Users\Xeron\AppData\Local\Arduino15\packages\rp2040\tools\pqt-python3\1.0.1-base-3a57aed-1\python.exe"
)
PYSERIAL_DIR = Path(
    r"C:\Users\Xeron\AppData\Local\Arduino15\packages\rp2040\hardware\rp2040\5.6.0\tools\pyserial"
)

sys.path.insert(0, str(PYSERIAL_DIR))
import serial  # type: ignore
from serial.tools import list_ports  # type: ignore


VARIANT_CHOICES = ("crossoverBiDirBabyMambaHar", "ciBabyMambaHar")
DEFAULT_DATASETS = (
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
)
COMPILE_FLASH_RE = re.compile(r"Sketch uses (\d+) bytes")
COMPILE_RAM_RE = re.compile(r"Global variables use (\d+) bytes .* leaving (\d+) bytes")


@dataclass
class Bundle:
    variant: str
    dataset: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile, flash, and benchmark BabyMamba variants on Raspberry Pi Pico 2."
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variants or 'all'.",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated datasets or 'all'.",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=EXPORT_ROOT,
        help="Root directory containing exported BabyMamba bundles.",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_DIR,
        help="Arduino-Pico BabyMamba sketch directory.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=60,
        help="Maximum serial capture time after flashing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory for serial logs and summaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print discovered bundles without compiling or flashing.",
    )
    return parser.parse_args()


def parse_csv_or_all(raw: str, choices: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    lookup = {choice.lower(): choice for choice in choices}
    values = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item.lower() not in lookup]
    if invalid:
        raise ValueError(f"Invalid values: {invalid}. Choices: {choices}")
    return [lookup[item.lower()] for item in values]


def run(cmd: Sequence[str], *, cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=check,
    )


def run_powershell(script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["powershell", "-NoProfile", "-Command", script], check=check)


def discover_bundles(export_root: Path, variants: Sequence[str], datasets: Sequence[str]) -> List[Bundle]:
    bundles: List[Bundle] = []
    for variant in variants:
        for dataset in datasets:
            candidate = export_root / variant / dataset
            if (candidate / "babyMambaWeights.h").exists():
                bundles.append(Bundle(variant=variant, dataset=dataset, path=candidate))
    return bundles


def find_bootsel_drive() -> Optional[str]:
    script = (
        "$vol = Get-CimInstance Win32_Volume | "
        "Where-Object { $_.Label -match 'RP2350|RPI|PICO' -and $_.DriveLetter } | "
        "Select-Object -First 1 -ExpandProperty DriveLetter; "
        "if ($vol) { Write-Output $vol }"
    )
    completed = run_powershell(script, check=False)
    drive = (completed.stdout or "").strip()
    return drive or None


def enter_bootsel(timeout_seconds: int) -> str:
    drive = find_bootsel_drive()
    if drive:
        return drive

    for port in list_rp_ports():
        try:
            with serial.Serial(port, 1200, timeout=0.2):
                pass
        except serial.SerialException:
            continue

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        drive = find_bootsel_drive()
        if drive:
            return drive
        time.sleep(0.5)
    raise TimeoutError("Pico 2 BOOTSEL drive did not appear after the 1200-baud reset.")


def list_rp_ports() -> List[str]:
    ports: List[str] = []
    for port in list_ports.comports():
        description = (port.description or "").lower()
        manufacturer = (port.manufacturer or "").lower()
        if port.vid == 0x2E8A or "pico" in description or "rp2350" in description or "raspberry pi" in manufacturer:
            ports.append(port.device)
    return sorted(set(ports))


def can_open_serial(port: str) -> bool:
    try:
        with serial.Serial(port, 115200, timeout=0.2):
            return True
    except serial.SerialException:
        return False


def wait_for_runtime_port(timeout_seconds: int, previous_ports: Sequence[str]) -> str:
    deadline = time.time() + timeout_seconds
    previous = set(previous_ports)
    while time.time() < deadline:
        current = list_rp_ports()
        new_ports = [port for port in current if port not in previous]
        candidates = new_ports + [port for port in current if port not in new_ports]
        for port in candidates:
            if can_open_serial(port):
                return port
        time.sleep(0.5)
    raise TimeoutError("No Pico 2 runtime serial port appeared after flashing.")


def prepare_project(project_dir: Path, bundle: Bundle) -> None:
    weight_src = bundle.path / "babyMambaWeights.h"
    weight_dst = project_dir / "babyMambaWeights.h"
    shutil.copyfile(weight_src, weight_dst)


def compile_project(project_dir: Path, build_dir: Path) -> Tuple[Path, Dict[str, int], str]:
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        str(ARDUINO_CLI),
        "compile",
        "--fqbn",
        "rp2040:rp2040:rpipico2:flash=4194304_0,arch=arm,uploadmethod=default,freq=150,opt=Optimize2,rtti=Disabled,exceptions=Disabled,dbgport=Disabled,dbglvl=None,usbstack=picosdk,ipbtstack=ipv4only",
        "--build-path",
        str(build_dir),
        "--build-property",
        "compiler.cpp.extra_flags=-O3 -ffast-math -funroll-loops -DPICO_RP2350=1",
        "--build-property",
        "compiler.c.extra_flags=-O3 -ffast-math -funroll-loops -DPICO_RP2350=1",
        str(project_dir),
    ]
    completed = run(compile_cmd, cwd=REPO_ROOT, check=False)
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    flash_match = COMPILE_FLASH_RE.search(combined)
    ram_match = COMPILE_RAM_RE.search(combined)
    metrics = {
        "flash_bytes": int(flash_match.group(1)) if flash_match else -1,
        "global_ram_bytes": int(ram_match.group(1)) if ram_match else -1,
        "global_ram_free_bytes": int(ram_match.group(2)) if ram_match else -1,
    }
    uf2_candidates = list(build_dir.rglob("*.uf2"))
    if not uf2_candidates:
        raise FileNotFoundError(f"No UF2 file produced under {build_dir}")
    return uf2_candidates[0], metrics, combined


def flash_uf2(uf2_path: Path, bootsel_drive: str) -> None:
    destination = f"{bootsel_drive}\\{uf2_path.name}"
    run_powershell(
        f"Copy-Item -LiteralPath '{uf2_path}' -Destination '{destination}' -Force",
        check=True,
    )


def capture_serial(port: str, timeout_seconds: int) -> Tuple[str, Dict[str, object]]:
    deadline = time.time() + timeout_seconds
    lines: List[str] = []
    metrics: Dict[str, object] = {"run_latency_us": []}
    with serial.Serial(port, 115200, timeout=0.5) as handle:
        time.sleep(2.0)
        while time.time() < deadline:
            raw = handle.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            lines.append(line)
            if line.startswith("==="):
                if line == "=== BABYMAMBA_PICO2_DONE ===":
                    break
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "run_latency_us":
                    metrics["run_latency_us"].append(int(float(value)))
                elif key in {
                    "dataset_key",
                    "dataset_display",
                    "variant_name",
                    "predicted_label",
                    "expected_label_name",
                    "final_logits",
                }:
                    metrics[key] = value
                elif key in {"model_seed", "scratch_bytes", "input_bytes", "reference_bytes", "predicted_class", "expected_label_index"}:
                    metrics[key] = int(float(value))
                else:
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
    if "avg_latency_us" not in metrics:
        raise RuntimeError("Serial capture did not reach the BabyMamba completion marker.")
    return "\n".join(lines) + "\n", metrics


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_rows(existing: Sequence[Dict[str, object]], updates: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str], Dict[str, object]] = {}
    for row in existing:
        key = (str(row.get("variant", "")), str(row.get("dataset", "")))
        merged[key] = dict(row)
    for row in updates:
        key = (str(row.get("variant", "")), str(row.get("dataset", "")))
        merged[key] = dict(row)
    return [
        merged[key]
        for key in sorted(merged.keys(), key=lambda item: (item[0], item[1]))
    ]


def write_markdown_summary(path: Path, rows: Sequence[Dict[str, object]], missing: Sequence[Dict[str, object]]) -> None:
    lines = [
        "# BabyMamba Pico 2 Results",
        "",
        "| Variant | Dataset | Status | Flash (B) | Global RAM (B) | Scratch (B) | Avg Latency (ms) | Parity vs PyTorch (%) | Port |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        status = str(row.get("status", "unknown"))
        lines.append(
            f"| {row.get('variant','')} | {row.get('dataset','')} | {status} | "
            f"{row.get('flash_bytes','')} | {row.get('global_ram_bytes','')} | "
            f"{row.get('scratch_bytes','')} | {row.get('avg_latency_ms','')} | "
            f"{row.get('parity_vs_pytorch_pct','')} | {row.get('port','')} |"
        )
    for row in missing:
        lines.append(
            f"| {row.get('variant','')} | {row.get('dataset','')} | {row.get('status','missing')} |  |  |  |  |  |  |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_paper_results(report_path: Path, rows: Sequence[Dict[str, object]], missing: Sequence[Dict[str, object]]) -> None:
    baby_lines = [
        "## BabyMamba Variants",
        "",
        "- Metrics JSON: "
        f"[{(DEPLOY_ROOT / 'babyMambaPico2_metrics.json')}]({(DEPLOY_ROOT / 'babyMambaPico2_metrics.json')})",
        "- Metrics Markdown: "
        f"[{(DEPLOY_ROOT / 'babyMambaPico2_metrics.md')}]({(DEPLOY_ROOT / 'babyMambaPico2_metrics.md')})",
        "",
        "| Variant | Dataset | Status | Flash (B) | Global RAM (B) | Scratch (B) | Avg Latency (ms) | Parity vs PyTorch (%) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        baby_lines.append(
            f"| {row.get('variant','')} | {row.get('dataset','')} | {row.get('status','')} | "
            f"{row.get('flash_bytes','')} | {row.get('global_ram_bytes','')} | {row.get('scratch_bytes','')} | "
            f"{row.get('avg_latency_ms','')} | {row.get('parity_vs_pytorch_pct','')} |"
        )
    for row in missing:
        baby_lines.append(
            f"| {row.get('variant','')} | {row.get('dataset','')} | {row.get('status','missing')} |  |  |  |  |  |"
        )
    baby_lines.append("")
    baby_lines.append("- All rows above are measured on-device Pico 2 runs using the handcrafted recurrent C++ engine.")
    if missing:
        baby_lines.append("- Any `missing_export_or_checkpoint` rows indicate datasets that were not yet exported or did not have a ready checkpoint at sweep time.")
    baby_section = "\n".join(baby_lines).strip() + "\n"

    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Pico 2 Deployment Results\n\n"
    marker = "## BabyMamba Variants"
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n\n"
    report_path.write_text(existing + baby_section, encoding="utf-8")


def main() -> int:
    args = parse_args()
    variants = parse_csv_or_all(args.variants, VARIANT_CHOICES)
    datasets = parse_csv_or_all(args.datasets, DEFAULT_DATASETS)

    bundles = discover_bundles(args.export_root, variants, datasets)
    missing: List[Dict[str, object]] = []
    for variant in variants:
        for dataset in datasets:
            candidate = args.export_root / variant / dataset / "babyMambaWeights.h"
            if not candidate.exists():
                missing.append(
                    {
                        "variant": variant,
                        "dataset": dataset,
                        "status": "missing_export_or_checkpoint",
                    }
                )

    if args.dry_run:
        for bundle in bundles:
            print(f"{bundle.variant} {bundle.dataset} -> {bundle.path}")
        for row in missing:
            print(f"{row['variant']} {row['dataset']} -> {row['status']}")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (DEPLOY_ROOT / f"babyMambaPico2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for bundle in bundles:
        print(f"[deploy] {bundle.variant} {bundle.dataset}")
        prepare_project(args.project_dir, bundle)
        build_dir = output_dir / "build" / f"{bundle.variant}_{bundle.dataset}"
        run_dir = output_dir / bundle.variant / bundle.dataset
        run_dir.mkdir(parents=True, exist_ok=True)

        uf2_path, compile_metrics, compile_log = compile_project(args.project_dir, build_dir)
        (run_dir / "compile.log").write_text(compile_log, encoding="utf-8")

        bootsel_drive = enter_bootsel(args.timeout_seconds)

        previous_ports = list_rp_ports()
        flash_uf2(uf2_path, bootsel_drive)
        port = wait_for_runtime_port(args.timeout_seconds, previous_ports)
        serial_log, serial_metrics = capture_serial(port, args.timeout_seconds)
        (run_dir / "serial.log").write_text(serial_log, encoding="utf-8")

        row: Dict[str, object] = {
            "variant": bundle.variant,
            "dataset": bundle.dataset,
            "status": "ok",
            "port": port,
            **compile_metrics,
            **serial_metrics,
        }
        avg_latency_us = float(serial_metrics["avg_latency_us"])
        row["avg_latency_ms"] = round(avg_latency_us / 1000.0, 3)
        rows.append(row)
        write_json(run_dir / "metrics.json", row)
        print(
            f"[done] {bundle.variant} {bundle.dataset} "
            f"latency={row['avg_latency_ms']}ms parity={row.get('parity_vs_pytorch_pct')}%"
        )

    aggregate_json = DEPLOY_ROOT / "babyMambaPico2_metrics.json"
    aggregate_md = DEPLOY_ROOT / "babyMambaPico2_metrics.md"
    existing_rows: List[Dict[str, object]] = []
    existing_missing: List[Dict[str, object]] = []
    if aggregate_json.exists():
        try:
            existing_payload = json.loads(aggregate_json.read_text(encoding="utf-8"))
            existing_rows = list(existing_payload.get("runs", []))
            existing_missing = list(existing_payload.get("missing", []))
        except Exception:
            existing_rows = []
            existing_missing = []

    merged_rows = merge_rows(existing_rows, rows)
    present_keys = {(str(row.get("variant", "")), str(row.get("dataset", ""))) for row in merged_rows}
    merged_missing = [
        row for row in (existing_missing + missing)
        if (str(row.get("variant", "")), str(row.get("dataset", ""))) not in present_keys
    ]

    write_json(aggregate_json, {"runs": merged_rows, "missing": merged_missing, "output_dir": str(output_dir)})
    write_markdown_summary(aggregate_md, merged_rows, merged_missing)
    update_paper_results(REPORT_FILE, merged_rows, merged_missing)
    print(f"[summary] json={aggregate_json}")
    print(f"[summary] markdown={aggregate_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

