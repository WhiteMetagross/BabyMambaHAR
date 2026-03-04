# BabyMamba-HAR.

## Ultra-Lightweight State Space Models for Human Activity Recognition.

**Author:** Mridankan Mandal

**Paper:** [BabyMamba-HAR: Ultra-Lightweight State Space Models for Human Activity Recognition](https://arxiv.org/abs/2602.09872v1)

**Repository:** [https://github.com/WhiteMetagross/BabyMambaHAR](https://github.com/WhiteMetagross/BabyMambaHAR)

---

### Overview.

BabyMamba-HAR is a family of ultra-lightweight State Space Model (SSM) architectures designed specifically for Human Activity Recognition (HAR) on resource-constrained edge devices. The primary model, **CI-BabyMamba-HAR**, achieves competitive accuracy with only approximately 27,000 parameters and O(N) linear inference complexity, making it suitable for deployment on microcontrollers and wearable devices.

This repository contains the complete source code for the two model variants, baseline implementations, training scripts, hyperparameter optimization (HPO) pipelines, and ablation studies described in the research paper.

![CI-BabyMamba-HAR Architecture](docs/img/nanoharmamba_architecture.png)

*Figure 1. The CI-BabyMamba-HAR architecture. Input sensor data passes through a Channel-Independent Stem, Discrete Patch Embedding, four Weight-Tied Bidirectional SSM Blocks, Context-Gated Temporal Attention, and a Classification Head.*

---

### Key Contributions.

1. **CI-BabyMamba-HAR:** A Channel-Independent SSM with Context-Gated Temporal Attention, expanded state memory (d_state=16), and weight-tied bidirectional processing. This model achieves approximately 27,000 parameters with O(N) complexity.

2. **Crossover-BiDir-BabyMamba-HAR:** A Weight-Tied Bidirectional SSM that processes sensor sequences with shared SSM weights in both temporal directions, achieving approximately 25,000 parameters.

3. **Signal Rescue Recipes:** Dataset-specific preprocessing pipelines (Butterworth low-pass filters, robust scaling, class weighting) that stabilize training on challenging datasets such as PAMAP2, Skoda, and Daphnet.

4. **Comprehensive Ablation Studies:** Systematic evaluation of each architectural component, including bidirectionality, layer depth, discrete patching, and SSM versus CNN comparisons.

---

### Model Specifications.

| Specification | CI-BabyMamba-HAR | Crossover-BiDir-BabyMamba-HAR |
|---|---|---|
| Parameters | ~27,000-29,000 | ~25,000 |
| d_model | 24 | 24 |
| d_state | 16 | 8 |
| n_layers | 4 | 4 |
| Expand Factor | 2 | 2 |
| Bidirectional | Yes (Weight-Tied) | Yes (Weight-Tied) |
| Gated Attention | Yes | No |
| Channel Independent | Yes | No |
| Inference Complexity | O(N) | O(N) |

---

### Benchmark Results (5-Seed Average).

Results are reported as mean plus/minus standard deviation over 5 random seeds.

| Dataset | CI-BabyMamba-HAR Acc. | CI-BabyMamba-HAR F1 | Best Baseline Acc. | Best Baseline |
|---|---|---|---|---|
| UCI-HAR | 95.55 +/- 0.88% | 95.60 +/- 0.83% | 96.46% | TinyHAR |
| **MotionSense** | **94.78 +/- 0.26%** | **93.24 +/- 0.34%** | 94.12% | DeepConvLSTM |
| WISDM | 81.37 +/- 0.69% | 72.54 +/- 8.13% | 86.35% | TinierHAR |
| PAMAP2 | 67.87 +/- 2.40% | 66.96 +/- 3.12% | 77.45% | TinierHAR |
| **UniMiB** | **91.68 +/- 1.25%** | **85.15 +/- 0.89%** | 92.71% | DeepConvLSTM |
| Opportunity | 86.26 +/- 0.67% | 88.13 +/- 0.64% | 87.45% | TinyHAR |

**Bold** indicates datasets where CI-BabyMamba-HAR achieves the best or most competitive results at its parameter budget.

![Benchmark Results Grid](docs/img/babymamba_results_grid.png)

*Figure 2. Benchmark results comparison across six HAR datasets. CI-BabyMamba-HAR achieves competitive or superior accuracy with significantly fewer parameters than attention-based baselines.*

---

### Architectural Highlights.

#### Channel-Independent Stem.

Instead of mixing sensor channels early, the CI-Stem treats each channel as an independent sample. A shared single-channel 1D convolution kernel processes each sensor channel independently. This prevents noise in one sensor from corrupting features extracted from other sensors, which is particularly beneficial for datasets such as Skoda (machine vibration noise), PAMAP2 (heart rate sensor noise), and Daphnet (sensor artifacts).

#### Weight-Tied Bidirectional SSM.

The core SSM block processes the input sequence in both forward and backward directions using the same weights. This doubles the receptive field without doubling the parameter count. The mathematical formulation is as follows:

- Forward: h_fwd = SSM(x; theta)
- Backward: h_bwd = SSM(flip(x); theta) -- same weights theta
- Fusion: h_out = h_fwd + flip(h_bwd)

#### Context-Gated Temporal Attention.

This replaces simple global mean pooling at the output of the SSM backbone. It uses a tanh gating mechanism with a learnable context vector and softmax attention to selectively weight temporal patches. This prevents the dilution of transient events such as gait freezing (Daphnet) or impact dynamics (PAMAP2).

![SSM Block Detail](docs/img/ssm_block_detail.png)

*Figure 3. Detailed view of the Weight-Tied Bidirectional SSM Block used in both CI-BabyMamba-HAR and Crossover-BiDir-BabyMamba-HAR. The same SSM weights are shared for both forward and backward scans.*

---

### Ablation Studies.

The following ablation variants are included to validate each architectural choice.

| Variant | Description | Effect |
|---|---|---|
| Full Model (A) | Complete CI-BabyMamba-HAR with all components. | Baseline reference. |
| Unidirectional (B) | Forward-only SSM, no backward pass. | Accuracy decreases. |
| 2-Layer (C) | Only 2 SSM layers instead of 4. | Reduced capacity for complex patterns. |
| No Patching (D) | SSM operates on full sequence without patching. | Less efficient, longer sequences. |
| CNN Only (E) | SSM blocks replaced with CNN blocks. | Lower accuracy on longer sequences. |

![Ablation Studies Combined](docs/img/babymamba_ablation_combined.png)

*Figure 4. Ablation study results. Each variant removes or modifies a single architectural component to measure its contribution to overall performance.*

---

### Repository Structure.

```
BabyMamba-HAR/
    README.md                          -- This file.
    CodeBaseIndex.md                   -- Detailed file-by-file index.
    Usage.md                           -- Training, HPO, and evaluation guide.
    InstallationAndSetup.md            -- Setup and installation instructions.
    requirements.txt                   -- Python dependencies.
    setup.py                           -- Package setup file.
    ciBabyMambaHar/                    -- CI-BabyMamba-HAR model package.
        models/                        -- Model definitions.
        data/                          -- Dataset loaders.
        utils/                         -- Utilities (metrics, profiling, etc.).
    crossoverBiDirBabyMambaHar/        -- Crossover-BiDir-BabyMamba-HAR package.
        models/                        -- Model definitions.
        scripts/                       -- Training and HPO scripts.
    baselines/                         -- Baseline model implementations.
    scripts/                           -- Training, HPO, ablation, evaluation scripts.
    configs/                           -- Dataset-specific YAML configurations.
    docs/
        img/                           -- Architecture diagrams and result figures.
    results/                           -- Output directory for experiment results.
```

---

### Quick Start.

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train on UCI-HAR.

```bash
python scripts/trainCiBabyMambaHar.py --dataset ucihar
```

3. Run hyperparameter optimization.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 50
```

4. Run ablation studies.

```bash
python scripts/runCiBabyMambaHarAblations.py --dataset ucihar
```

For detailed instructions, see [Usage.md](Usage.md) and [InstallationAndSetup.md](InstallationAndSetup.md).

---

### Baseline Models.

The following baseline models are included for comparison. MicroBiConvLSTM is not included in this repository.

| Model | Source | Parameters | Description |
|---|---|---|---|
| DeepConvLSTM | Ordonez et al. (2016) | ~130K-154K | 4-layer CNN + 2-layer LSTM. |
| TinierHAR | Adaptation of TinyHAR | ~7K-124K | 4-block ResNet with self-attention. |
| TinyHAR | Zhou et al. (ISWC 2022) | ~36K-123K | Self-attention with temporal weighted aggregation. |
| LightDeepConvLSTM | Lightweight variant | ~15K-21K | Reduced-width DeepConvLSTM. |
| HARMamba | Mamba-based baseline | ~77K+ | Large Mamba model with channel mixer. |

---

### Datasets.

The models are evaluated on the following benchmark HAR datasets.

| Dataset | Classes | Channels | Sequence Length | Type |
|---|---|---|---|---|
| UCI-HAR | 6 | 9 | 128 | Smartphone IMU. |
| MotionSense | 6 | 6 | 128 | Smartphone accelerometer and gyroscope. |
| WISDM | 6 | 3 | 128 | Single-axis accelerometer. |
| PAMAP2 | 12 | 19 | 128 | Multi-IMU body-worn sensors. |
| Opportunity | 5 | 79 | 128 | Full-body sensor network. |
| UniMiB-SHAR | 9 | 3 | 128 | Smartphone accelerometer (ADL and falls). |

---

### Citation.

If you use this code in your research, please cite the following paper.

```bibtex
@article{mandal2026babymambahar,
  title={BabyMamba-HAR: Ultra-Lightweight State Space Models for Human Activity Recognition},
  author={Mandal, Mridankan},
  journal={arXiv preprint arXiv:2602.09872},
  year={2026}
}
```

---

### License.

This project is licensed under the MIT License.

---

### Acknowledgments.

This work builds upon the Mamba architecture by Gu and Dao (2023) and extends it for the Human Activity Recognition domain with novel contributions including channel-independent processing, context-gated temporal attention, and weight-tied bidirectional scanning.
