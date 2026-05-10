from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

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
        description="Run paper-aligned CI-BabyMamba-HAR retraining sequentially across datasets."
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Comma-separated dataset keys or 'all'.",
    )
    parser.add_argument(
        "--seed-list",
        default="29",
        help="Comma-separated explicit seeds. Default is the paper rerun seed 29.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum epochs per dataset.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/training",
        help="Base output directory.",
    )
    parser.add_argument(
        "--tag",
        default="fixed_seed29",
        help="Run tag stored in the result path.",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile in the underlying training runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    return parser.parse_args()


def parse_csv_or_all(raw: str, choices: Sequence[str]) -> List[str]:
    if raw.strip().lower() == "all":
        return list(choices)
    values = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [value for value in values if value not in choices]
    if invalid:
        raise ValueError(f"Invalid datasets: {invalid}. Choices: {choices}")
    return values


def main() -> int:
    args = parse_args()
    datasets = parse_csv_or_all(args.datasets, DATASET_CHOICES)

    commands: List[List[str]] = []
    for dataset in datasets:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "trainCiBabyMambaHar.py"),
            "--dataset",
            dataset,
            "--seed-list",
            args.seed_list,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--outDir",
            args.out_dir,
            "--tag",
            args.tag,
            "--accumulation_steps",
            str(args.accumulation_steps),
        ]
        if args.compile:
            cmd.append("--compile")
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

