# Installation And Setup:

## Scope:

This document describes the environment expected by the BabyMamba-HAR training, export, and embedded deployment workflow. A Python training environment is required for retraining and export generation. Arduino-based board support is required for the Pico 2 runtime. The ESP32 export path is also included for portability of the handcrafted recurrent engine.

## Repository Preparation:

The repository should first be cloned and entered.

```bash
git clone https://github.com/WhiteMetagross/BabyMambaHAR.git
cd BabyMambaHAR
```

## Python Environment:

A dedicated virtual environment is recommended. The following commands may be used with either `venv`, Conda, or Mamba.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If GPU-backed Mamba kernels are desired, `mamba-ssm` and `causal-conv1d` should be installed in a CUDA-ready environment. The pure PyTorch fallback remains supported, but slower execution should be expected during CI-BabyMamba-HAR retraining.

## Dataset Layout:

The training and export programs expect the datasets to be available under a repository-local `datasets/` directory. The expected top-level layout is shown below.

```text
datasets/
├── UCI HAR Dataset/
├── motion-sense-master/
├── WISDM_ar_v1.1/
├── PAMAP2_Dataset/
├── OpportunityUCIDataset/
├── UniMiB-SHAR/
├── Skoda/
└── Daphnet/
```

If the datasets are stored elsewhere in a larger research workspace, symbolic links may be used.

## Device Toolchains:

For Raspberry Pi Pico 2 deployment, the following components are expected.

- Arduino CLI with the `rp2040:rp2040` core installed.
- A serial-capable Python environment with `pyserial`.
- Access to the Pico 2 USB bootloader path.

For ESP32 export and runtime preparation, the following components are recommended.

- Arduino CLI with an ESP32 board core, or ESP-IDF if a native runtime is preferred.
- A serial upload utility appropriate to the connected board.
- Sufficient internal SRAM for the selected deployment target.

## Committed Artifact Layout:

This repository snapshot already contains the main artifacts needed for inspection and reuse.

- `models/` contains the committed dataset-specific checkpoints and training summaries.
- `Pico2Models/` contains the committed embedded weight headers and Pico 2 benchmark JSON files.
- `ESP32Models/` contains the committed embedded weight headers for ESP32-class deployments.
- `embedded/` contains the runtime scaffolds that consume the generated `babyMambaWeights.h` files.

The comparison baselines are preserved inside the same layout.

- `models/baselines/` contains the seed-29 baseline checkpoint zoo.
- `Pico2Models/baselines/` contains the baseline Pico 2 bundles and summary files.
- `ESP32Models/baselines/` contains the baseline ESP32 bundles and summary files.

Because these artifacts are versioned, the repository may be used directly for deployment study and code review, even when the original training host is unavailable.

## Optional GPU Acceleration:

The channel-independent BabyMamba retraining path benefits substantially from fused selective state space kernels. If a CUDA-capable system is available, the environment should be validated with a short import check.

```bash
python -c "import torch; import mamba_ssm; print(torch.cuda.is_available())"
```

If these kernels are unavailable, training will still run through the fallback selective scan implementation.

## Verification:

After installation, the scripts below may be executed as a smoke test.

```bash
python scripts/runCiBabyMambaHarRetraining.py --datasets ucihar --seed-list 29 --epochs 1 --patience 1
python scripts/exportBabyMambaPico2Models.py
```

The first command verifies the training stack. The second verifies that the export tooling is importable and that the committed model zoo is discoverable.
