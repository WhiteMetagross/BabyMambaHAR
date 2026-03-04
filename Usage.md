# Usage.md.

## Training, Evaluation, and Experimentation Guide.

**Author:** Mridankan Mandal

This document provides detailed instructions for training models, running hyperparameter optimization (HPO), executing ablation studies, evaluating trained models, and benchmarking inference performance.

---

### Table of Contents.

1. [Prerequisites.](#prerequisites)
2. [Training CI-BabyMamba-HAR.](#training-ci-babymamba-har)
3. [Hyperparameter Optimization (HPO).](#hyperparameter-optimization-hpo)
4. [Training Baseline Models.](#training-baseline-models)
5. [Ablation Studies.](#ablation-studies)
6. [Evaluation.](#evaluation)
7. [Benchmarking.](#benchmarking)
8. [Training Crossover-BiDir-BabyMamba-HAR.](#training-crossover-bidir-babymamba-har)
9. [Configuration Files.](#configuration-files)
10. [Signal Rescue Recipes.](#signal-rescue-recipes)
11. [Pure PyTorch Fallback.](#pure-pytorch-fallback)
12. [Troubleshooting.](#troubleshooting)

---

### Prerequisites.

Before training, ensure the following steps are completed.

1. Python 3.8 or later is installed.
2. All dependencies from `requirements.txt` are installed.
3. At least one dataset is downloaded and placed in the `datasets/` directory at the project root.
4. A CUDA-capable GPU is recommended. CPU training is supported but significantly slower.

See [InstallationAndSetup.md](InstallationAndSetup.md) for detailed environment setup instructions.

---

### Training CI-BabyMamba-HAR.

The main training script is `scripts/trainCiBabyMambaHar.py`. It trains the CI-BabyMamba-HAR model with the frozen architecture configuration and tunable training hyperparameters.

#### Basic Training on a Single Dataset.

```bash
python scripts/trainCiBabyMambaHar.py --dataset ucihar
```

#### Training with Custom Settings.

```bash
python scripts/trainCiBabyMambaHar.py --dataset ucihar --seeds 3 --epochs 50
```

#### Training on All Datasets.

```bash
python scripts/trainCiBabyMambaHar.py --dataset all --seeds 5
```

#### Training Parameters.

| Parameter | Default | Description |
|---|---|---|
| `--dataset` | Required. | Dataset name: `ucihar`, `motionsense`, `wisdm`, `pamap2`, `opportunity`, `unimib`, `skoda`, `daphnet`, or `all`. |
| `--seeds` | 5 | Number of random seeds for statistical significance. |
| `--epochs` | 200 | Maximum training epochs. |

#### Training Protocol Details.

The training script uses the following protocol.

- **Optimizer:** AdamW with beta values (0.9, 0.999).
- **Scheduler:** Cosine Annealing with linear warmup.
- **Warmup:** 10 epochs of linear learning rate warmup.
- **Early Stopping:** Patience of 10 epochs, matching the baseline protocol for fair comparison.
- **Mixed Precision:** FP16 AMP training is enabled by default on CUDA devices.
- **Seeds:** 5 random seeds are used by default. Results are reported as mean plus/minus standard deviation.
- **HPO Loading:** The script automatically loads the best hyperparameters from `results/hpo/` if available.

#### Frozen Architecture Configuration.

The following architecture parameters are frozen and must not be changed for reproducibility with the research paper results.

| Parameter | Value | Description |
|---|---|---|
| d_model | 24 | Model dimension (width). |
| d_state | 16 | SSM state dimension. Increased from 8 for nuanced activities. |
| n_layers | 4 | Number of SSM blocks. |
| expand | 2 | Inner dimension expansion factor (inner = 48). |
| dt_rank | 2 | Delta discretization rank. |
| d_conv | 4 | Local convolution kernel size inside SSM. |
| bidirectional | True | Weight-tied bidirectional SSM. |
| gated_attention | True | Context-gated temporal attention pooling. |
| channel_independent | True | Channel-independent stem processing. |

#### Tunable Training Hyperparameters.

These are the only parameters tuned via HPO. The architecture remains frozen.

| Parameter | Range | Description |
|---|---|---|
| learning_rate | 0.0003 to 0.003 (log scale) | Learning rate for AdamW optimizer. |
| weight_decay | 0.005 to 0.05 (log scale) | L2 regularization strength. |
| dropout | 0.0 to 0.3 | Dropout rate in the classification head. |
| drop_path | 0.0 to 0.2 | Stochastic depth rate. |
| label_smoothing | 0.0 to 0.2 | Label smoothing for cross-entropy loss. |
| batch_size | 64 to 512 | Batch size (dataset-specific). |

#### Output.

Training results are saved to `results/training/` as JSON files containing per-seed accuracy, F1 score, precision, recall, confusion matrix, parameter count, MACs, latency, and training time.

---

### Hyperparameter Optimization (HPO).

The HPO script tunes only the training hyperparameters while keeping the architecture frozen. The search uses Optuna with the Tree-structured Parzen Estimator (TPE) sampler.

#### Running HPO for a Single Dataset.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset ucihar
```

#### Running HPO with Custom Trial Count.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 50
```

#### Quick Smoke Test.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset ucihar --n-trials 2 --epochs 2
```

#### Running HPO for All Datasets.

```bash
python scripts/hpoCiBabyMambaHar.py --dataset all
```

#### HPO Protocol.

| Parameter | Value |
|---|---|
| Sampler | TPE (Tree-structured Parzen Estimator). |
| Trials | 50 per dataset. |
| Epochs per Trial | 10. |
| Early Stopping Patience | 5 epochs. |
| Optimization Metric | Validation macro F1 score. |
| Seed | 17 (fixed for reproducibility). |
| Data Augmentation | Disabled during HPO for speed. |

#### HPO Search Space.

| Parameter | Distribution | Range |
|---|---|---|
| learning_rate | Log-uniform | 1e-4 to 1e-2. |
| weight_decay | Log-uniform | 0.005 to 0.05. |
| dropout | Uniform | 0.0 to 0.5. |

#### Output.

HPO results are saved to `results/hpo/hpo_ciBabyMambaHar_{dataset}.json` containing the best hyperparameters, best validation F1, and trial history.

---

### Training Baseline Models.

Baseline models are trained with the same protocol as CI-BabyMamba-HAR for fair comparison.

#### Training All Baselines on a Dataset.

```bash
python scripts/trainBaselines.py --dataset ucihar
```

#### Training All Baselines on All Datasets.

```bash
python scripts/runBaselines.py
```

#### HPO for Baselines.

```bash
python scripts/hpoBaselines.py --dataset ucihar --n-trials 50
```

#### Available Baseline Models.

| Model | Description | Approximate Parameters |
|---|---|---|
| DeepConvLSTM | 4-layer CNN + 2-layer LSTM (Ordonez et al., 2016). | 130K-154K. |
| TinierHAR | 4-block ResNet with self-attention. | 7K-124K. |
| TinyHAR | Self-attention with temporal weighted aggregation (Zhou et al., ISWC 2022). | 36K-123K. |
| LightDeepConvLSTM | Reduced-width DeepConvLSTM variant. | 15K-21K. |

---

### Ablation Studies.

The ablation studies systematically evaluate the contribution of each architectural component by removing or modifying one component at a time.

#### Running CI-BabyMamba-HAR Ablations.

```bash
python scripts/runCiBabyMambaHarAblations.py --dataset ucihar
```

#### Running Ablations on All Datasets.

```bash
python scripts/runCiBabyMambaHarAblations.py --dataset all --epochs 100 --patience 10 --seed 17 --use-hpo
```

#### Ablation Variants.

| Variant ID | Name | Modification | Expected Effect |
|---|---|---|---|
| A (Full) | CiBabyMambaHarFull | Complete model. All components included. | Baseline reference performance. |
| B (Unidirectional) | CiBabyMambaHarUnidirectional | Forward-only SSM. No backward pass. | Accuracy decrease due to missing backward temporal context. |
| C (2-Layer) | CiBabyMambaHar2Layer | Only 2 SSM layers instead of 4. | Reduced capacity for hierarchical pattern recognition. |
| D (No Patching) | CiBabyMambaHarNoPatching | No discrete patching. SSM operates on full sequence. | Lower efficiency, longer input to SSM. |
| E (CNN Only) | CiBabyMambaHarCnnOnly | SSM blocks replaced with CNN blocks. | Lower accuracy on sequences requiring long-range dependencies. |

#### Collecting Ablation Results.

After running ablation experiments, aggregate results across seeds.

```bash
python scripts/collectAblationResults.py
```

This computes mean and standard deviation for each ablation variant.

#### Output.

Ablation results are saved to `results/ablations/` as JSON files with per-seed and aggregated metrics.

![Ablation Studies Combined](docs/img/babymamba_ablation_combined.png)

*Figure 1. Ablation study results showing the contribution of each architectural component to overall model performance.*

---

### Evaluation.

The evaluation script loads a trained model checkpoint and computes comprehensive metrics.

#### Evaluating a Trained Model.

```bash
python scripts/evaluate.py --checkpoint results/training/best_model.pth --dataset ucihar
```

#### Evaluation Metrics.

The evaluation script reports the following metrics.

- **Accuracy:** Overall classification accuracy.
- **Macro F1 Score:** Unweighted mean F1 across all classes.
- **Precision:** Macro-averaged precision.
- **Recall:** Macro-averaged recall.
- **Confusion Matrix:** Per-class prediction breakdown.
- **Parameter Count:** Total trainable parameters.
- **MACs:** Multiply-Accumulate Operations per forward pass.
- **Inference Latency:** Average inference time per sample.

---

### Benchmarking.

#### Parameter and FLOP Benchmarks.

```bash
python scripts/benchmarkModels.py
```

This computes parameter counts, MACs, and FLOPs for all models across all dataset configurations.

#### Inference Latency Benchmarks.

```bash
python scripts/benchmarkLatency.py
```

This measures inference latency on the available hardware (CPU or GPU) with warm-up runs for accurate timing.

---

### Training Crossover-BiDir-BabyMamba-HAR.

The Crossover-BiDir-BabyMamba-HAR variant can be trained using its dedicated scripts.

#### Training.

```bash
python crossoverBiDirBabyMambaHar/scripts/trainCrossoverBiDirBabyMambaHar.py --dataset ucihar
```

#### HPO.

```bash
python crossoverBiDirBabyMambaHar/scripts/hpoCrossoverBiDirBabyMambaHar.py --dataset ucihar
```

This variant uses d_state=8 (instead of 16) and does not include Channel-Independent processing or Gated Attention.

---

### Configuration Files.

Dataset-specific configurations are stored in YAML files in the `configs/` directory. Each file contains the following settings.

- **Dataset Parameters:** Number of classes, channels, sequence length, and data root path.
- **Model Parameters:** Frozen architecture configuration values.
- **Training Parameters:** Default learning rate, weight decay, dropout, batch size, and scheduler settings.
- **Signal Rescue Parameters:** Dataset-specific filter cutoff frequencies, label smoothing, and class weighting (where applicable).

#### Example Configuration (uciHar.yaml).

```yaml
dataset:
  name: UCI-HAR
  numClasses: 6
  inChannels: 9
  seqLen: 128
  root: ./datasets/UCI HAR Dataset

model:
  dModel: 24
  dState: 16
  nLayers: 4
  expand: 2
  bidirectional: true
  gatedAttention: true
  channelIndependent: true

training:
  lr: 0.001
  weightDecay: 0.01
  dropout: 0.1
  batchSize: 64
  epochs: 200
  patience: 10
```

---

### Signal Rescue Recipes.

For challenging datasets with sensor noise or class imbalance, the following preprocessing pipelines are applied automatically by the training scripts.

#### Skoda (5Hz Butterworth Filter).

- **Problem:** Machine vibration noise corrupts sensor readings.
- **Solution:** Apply a 5Hz Butterworth low-pass filter before windowing. Use label smoothing of 0.1 for fuzzy gesture boundaries. Apply class weights to handle the dominant Null class.

#### PAMAP2 (10Hz Butterworth Filter + Robust Scaling).

- **Problem:** Heart rate sensor noise and extreme multi-modal dynamics across 19 channels.
- **Solution:** Apply a 10Hz Butterworth low-pass filter. Use robust scaling (median and IQR normalization) instead of standard scaling. Apply gradient clipping at 1.0. Use class weights for imbalanced activity distribution.

#### Daphnet (12Hz Butterworth Filter + Class Weights).

- **Problem:** Highly imbalanced binary classification (Walk >> Freeze). Sensor jitter in ankle/thigh/trunk sensors.
- **Solution:** Apply a 12Hz Butterworth low-pass filter. Use data-computed class weights for the severe Walk/Freeze imbalance. Use a 2-epoch warmup for training stability.

---

### Pure PyTorch Fallback.

The CI-BabyMamba-HAR model includes a pure PyTorch implementation of the Selective State Space scan algorithm (`PureSelectiveScan`). This is used automatically when the `mamba-ssm` CUDA kernels are not installed. The pure PyTorch fallback produces identical results but runs slower because it does not use the optimized CUDA parallel scan.

To check which backend is being used, run the following.

```python
from ciBabyMambaHar.models.ciBabyMambaBlock import MAMBA_AVAILABLE
print(f"CUDA Mamba available: {MAMBA_AVAILABLE}")
```

---

### Troubleshooting.

#### Common Issues.

**Out of Memory (OOM) on GPU.**

The Channel-Independent Stem multiplies memory usage by the number of input channels (since each channel is processed independently). For high-channel datasets such as Opportunity (79 channels) or Skoda (30 channels), reduce the batch size or use the training script's built-in CPU data loading with batched GPU transfers.

```bash
python scripts/trainCiBabyMambaHar.py --dataset opportunity
```

The training script automatically keeps data on CPU and transfers batches to GPU during training to manage VRAM.

**mamba-ssm Installation Fails.**

The `mamba-ssm` package requires CUDA and a compatible compiler. If installation fails, the model will automatically use the `PureSelectiveScan` fallback. Training will be slower but produce the same results.

**Dataset Not Found.**

Ensure datasets are placed in the `datasets/` directory at the project root. The expected directory names are listed in the dataset loader files. For example, UCI-HAR expects the directory `datasets/UCI HAR Dataset/`.

**Low Accuracy on PAMAP2 or Skoda.**

Ensure the Signal Rescue recipes are enabled. The training script enables them by default when the dataset name matches. If running custom training code, make sure to set `applyFilter=True` and the corresponding `filterCutoff` value in the dataset loader constructor.
