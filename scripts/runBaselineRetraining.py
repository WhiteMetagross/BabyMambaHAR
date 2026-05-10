from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

MODEL_CHOICES = (
    "tinierhar",
    "tinyhar",
    "lightdeepconvlstm",
    "deepconvlstm",
)
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
    parser = argparse.ArgumentParser(
        description="Run paper-aligned baseline retraining sequentially across datasets."
    )
    parser.add_argument("--models", default="all", help="Comma-separated baseline models or 'all'.")
    parser.add_argument("--datasets", default="all", help="Comma-separated datasets or 'all'.")
    parser.add_argument("--seed-list", default="29", help="Comma-separated explicit seeds. Default is 29.")
    parser.add_argument("--epochs", type=int, default=200, help="Maximum epochs per run.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument(
        "--out-dir",
        default="results/paperRetraining/baselines",
        help="Base output directory for saved baseline artifacts.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    return parser.parse_args()


def parse_csv_or_all(raw: str, choices: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in values if item not in choices]
    if invalid:
        raise ValueError(f"Invalid values: {invalid}. Choices: {choices}")
    return values


def main() -> int:
    args = parse_args()
    models = parse_csv_or_all(args.models, MODEL_CHOICES)
    datasets = parse_csv_or_all(args.datasets, DATASET_CHOICES)

    commands: List[List[str]] = []
    for model_name in models:
        for dataset_name in datasets:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts" / "trainBaselines.py"),
                "--model",
                model_name,
                "--dataset",
                dataset_name,
                "--seed-list",
                args.seed_list,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--outDir",
                args.out_dir,
            ]
            commands.append(cmd)

    for cmd in commands:
        printable = " ".join(f'"{part}"' if " " in part else part for part in cmd)
        print(printable)
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
