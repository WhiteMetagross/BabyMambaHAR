from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "exportBabyMambaEdgeModels.py"),
        "--output-root",
        str(REPO_ROOT / "ESP32Models"),
    ]
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
