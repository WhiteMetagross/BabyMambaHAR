# InstallationAndSetup.md.

## Environment Configuration and Dataset Preparation.

**Author:** Mridankan Mandal

This document provides step-by-step instructions for setting up the development environment, installing dependencies, preparing datasets, and verifying that the installation is ready for training.

---

### Table of Contents.

1. [System Requirements.](#system-requirements)
2. [Environment Setup.](#environment-setup)
3. [Installing Dependencies.](#installing-dependencies)
4. [Installing Mamba CUDA Kernels (Optional).](#installing-mamba-cuda-kernels-optional)
5. [Dataset Preparation.](#dataset-preparation)
6. [Verifying the Installation.](#verifying-the-installation)
7. [Directory Structure.](#directory-structure)

---

### System Requirements.

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.8 | 3.10 or later. |
| GPU | Any CUDA-capable GPU. | NVIDIA GPU with 4GB or more VRAM. |
| CUDA | 11.7 | 12.1 or later. |
| RAM | 8 GB. | 16 GB or more. |
| OS | Linux, Windows, macOS. | Linux or Windows with CUDA. |

A GPU is not strictly required. All models can be trained on CPU, though training will be significantly slower. The CI-BabyMamba-HAR model has only approximately 27,000 parameters and trains within minutes on a modern GPU for most datasets.

---

### Environment Setup.

#### Option 1: Using Conda (Recommended).

Create a new conda environment to isolate the project dependencies.

```bash
conda create -n babymambahar python=3.10 -y
conda activate babymambahar
```

#### Option 2: Using venv.

```bash
python -m venv babymambahar_env
```

On Linux or macOS:

```bash
source babymambahar_env/bin/activate
```

On Windows:

```powershell
.\babymambahar_env\Scripts\Activate.ps1
```

---

### Installing Dependencies.

#### Step 1: Install PyTorch.

Install PyTorch with CUDA support matching your system. Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for the correct command for your platform.

Example for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Example for CPU only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 2: Install Remaining Dependencies.

```bash
pip install -r requirements.txt
```

This installs the following packages.

| Package | Purpose |
|---|---|
| numpy | Numerical array operations. |
| scipy | Signal processing (Butterworth filters for Signal Rescue). |
| pandas | Data loading and CSV parsing. |
| pyyaml | YAML configuration parsing. |
| tqdm | Progress bars during training. |
| matplotlib | Plotting training curves and results. |
| seaborn | Statistical visualization. |
| fvcore | FLOPs and parameter counting. |
| thop | MACs computation for profiling. |
| scikit-learn | Classification metrics (F1, precision, recall, confusion matrix). |
| einops | Tensor reshape and rearrange operations. |
| wandb | Weights and Biases experiment tracking (optional). |

#### Step 3: Install the Package in Development Mode (Optional).

To make the `ciBabyMambaHar` and `crossoverBiDirBabyMambaHar` packages importable from any directory, install in editable mode.

```bash
pip install -e .
```

---

### Installing Mamba CUDA Kernels (Optional).

The `mamba-ssm` package provides optimized CUDA kernels for the Selective State Space scan. These kernels significantly speed up training and inference but are not required. Without them, the model automatically falls back to a pure PyTorch implementation that produces identical results.

#### Requirements for mamba-ssm.

- Linux operating system (CUDA compilation is not supported on Windows natively).
- CUDA 11.7 or later.
- A C++ compiler compatible with your CUDA version (GCC on Linux).
- PyTorch compiled with CUDA support.

#### Installation.

```bash
pip install causal-conv1d>=1.0.0
pip install mamba-ssm>=1.0.0
```

#### Verifying mamba-ssm Installation.

```python
python -c "from mamba_ssm import Mamba; print('mamba-ssm installed successfully.')"
```

If this command fails, the pure PyTorch fallback will be used automatically. No code changes are needed.

#### Windows Users.

The `mamba-ssm` package does not compile natively on Windows. Windows users have two options.

1. **Use WSL (Windows Subsystem for Linux):** Install WSL2, set up a Linux environment, and install `mamba-ssm` inside WSL. This is the recommended approach for Windows users who want CUDA kernel acceleration.
2. **Use the pure PyTorch fallback:** Simply skip installing `mamba-ssm`. The model works identically with the built-in `PureSelectiveScan` implementation. Training will be slower but all results will be reproduced exactly.

---

### Dataset Preparation.

Datasets should be placed in a `datasets/` directory at the project root. Each dataset has its own expected directory structure.

#### Dataset Download Locations.

| Dataset | Source | Expected Directory |
|---|---|---|
| UCI-HAR | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) | `datasets/UCI HAR Dataset/` |
| MotionSense | [GitHub: mmalekzadeh/motion-sense](https://github.com/mmalekzadeh/motion-sense) | `datasets/motion-sense-master/` |
| WISDM | [WISDM Lab](https://www.cis.fordham.edu/wisdm/dataset.php) | `datasets/WISDM_ar_v1.1/` |
| PAMAP2 | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) | `datasets/PAMAP2_Dataset/` |
| Opportunity | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/opportunity+activity+recognition) | `datasets/Opportunity/` |
| UniMiB-SHAR | [UniMiB](http://www.sal.disco.unimib.it/technologies/unimib-shar/) | `datasets/UniMiB-SHAR/` |
| Skoda | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+a+Single+Body-Worn+Accelerometer) | `datasets/Skoda/` |
| Daphnet | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Daphnet+Freezing+of+Gait) | `datasets/Daphnet/` |

#### Creating the Dataset Directory.

```bash
mkdir datasets
```

Download each dataset and extract it into the `datasets/` directory so that the expected subdirectory names match the table above. The dataset loader classes handle all preprocessing, windowing, and normalization automatically.

---

### Verifying the Installation.

Run the following commands to verify that the environment is correctly configured.

#### Check PyTorch and CUDA.

```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Check Model Import.

```python
python -c "from ciBabyMambaHar.models import CiBabyMambaHar; print('CI-BabyMamba-HAR imported successfully.')"
```

#### Check Mamba Backend.

```python
python -c "from ciBabyMambaHar.models.ciBabyMambaBlock import MAMBA_AVAILABLE; print(f'CUDA Mamba available: {MAMBA_AVAILABLE}')"
```

#### Quick Smoke Test.

Run a minimal training job to verify everything works end-to-end. This trains for 2 epochs on 2 HPO trials.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 2 --epochs 2
```

If this completes without errors, the environment is ready for full training and experimentation.

---

### Directory Structure.

After setup, the project root should have the following layout.

```
BabyMamba-HAR/
    ciBabyMambaHar/
        models/
        data/
        utils/
    crossoverBiDirBabyMambaHar/
        models/
        scripts/
    baselines/
    scripts/
    configs/
    datasets/            <-- You create this.
        UCI HAR Dataset/
        motion-sense-master/
        WISDM_ar_v1.1/
        PAMAP2_Dataset/
        Opportunity/
        UniMiB-SHAR/
        Skoda/
        Daphnet/
    docs/
        img/
    results/             <-- Created automatically during training.
    requirements.txt
    setup.py
    README.md
    CodeBaseIndex.md
    Usage.md
    InstallationAndSetup.md
```

The `results/` directory and its subdirectories (`results/hpo/`, `results/training/`, `results/ablations/`, `results/latency/`) are created automatically by the training, HPO, and benchmarking scripts.
