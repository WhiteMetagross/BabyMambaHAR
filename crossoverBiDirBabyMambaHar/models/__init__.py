"""
BabyMamba-Crossover-BiDir Models Package

This package contains the BabyMamba-Crossover-BiDir architecture for Human Activity Recognition.
"""

from .crossoverBiDirBabyMamba import (
    CrossoverBiDirBabyMambaHar,
    createCrossoverBiDirBabyMambaHar,
    CROSSOVER_BIDIR_BABYMAMBA_CONFIG,
)

from .crossoverBiDirBlock import (
    WeightTiedBiDirMambaBlock,
    PureSelectiveScan,
    DropPath,
)

from .ablations import (
    CrossoverBiDirBabyMambaHarFull,
    CrossoverBiDirBabyMambaHarUnidirectional,
    CrossoverBiDirBabyMambaHar2Layer,
    CrossoverBiDirBabyMambaHarNoPatching,
    CrossoverBiDirBabyMambaHarCnnOnly,
    createAblationModel,
    ABLATION_MODELS,
)

__all__ = [
    # Main Model
    'CrossoverBiDirBabyMambaHar',
    'createCrossoverBiDirBabyMambaHar',
    'CROSSOVER_BIDIR_BABYMAMBA_CONFIG',
    # Block Components
    'WeightTiedBiDirMambaBlock',
    'PureSelectiveScan',
    'DropPath',
    # Ablation Models
    'CrossoverBiDirBabyMambaHarFull',
    'CrossoverBiDirBabyMambaHarUnidirectional',
    'CrossoverBiDirBabyMambaHar2Layer',
    'CrossoverBiDirBabyMambaHarNoPatching',
    'CrossoverBiDirBabyMambaHarCnnOnly',
    'createAblationModel',
    'ABLATION_MODELS',
]
