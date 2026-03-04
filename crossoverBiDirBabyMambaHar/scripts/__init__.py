"""
BabyMamba-Crossover-BiDir Scripts Package

This package contains training and HPO scripts for the
BabyMamba-Crossover-BiDir architecture.

Scripts:
    - trainCrossoverBiDirBabyMambaHar.py: Multi-seed training script
    - hpoCrossoverBiDirBabyMambaHar.py: Hyperparameter optimization script
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

__all__ = ['SCRIPTS_DIR']
