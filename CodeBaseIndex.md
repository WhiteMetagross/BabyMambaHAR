# CodeBaseIndex.md.

## Detailed File-by-File Index of the BabyMamba-HAR Repository.

**Author:** Mridankan Mandal

This document provides a comprehensive index of every file in the repository, organized by directory. Each entry includes the file path, a description of its purpose, and the key classes or functions it contains.

---

### Table of Contents.

1. [Root Files.](#root-files)
2. [ciBabyMambaHar/ -- CI-BabyMamba-HAR Model Package.](#cibabymambahar----ci-babymamba-har-model-package)
3. [crossoverBiDirBabyMambaHar/ -- Crossover-BiDir-BabyMamba-HAR Package.](#crossoverbidirbabymambahar----crossover-bidir-babymamba-har-package)
4. [baselines/ -- Baseline Model Implementations.](#baselines----baseline-model-implementations)
5. [scripts/ -- Training, HPO, and Evaluation Scripts.](#scripts----training-hpo-and-evaluation-scripts)
6. [configs/ -- Dataset Configuration Files.](#configs----dataset-configuration-files)
7. [docs/ -- Documentation and Figures.](#docs----documentation-and-figures)
8. [results/ -- Experiment Output Directory.](#results----experiment-output-directory)

---

### Root Files.

| File | Description |
|---|---|
| `README.md` | Project overview, model specifications, benchmark results, and quick start guide. |
| `CodeBaseIndex.md` | This file. Detailed file-by-file index of the repository. |
| `Usage.md` | Comprehensive guide for training, HPO, evaluation, and ablation studies. |
| `InstallationAndSetup.md` | Installation instructions, environment setup, and dataset preparation. |
| `requirements.txt` | Python package dependencies. Core dependencies include PyTorch, NumPy, SciPy, and Pandas. Optional dependencies include mamba-ssm CUDA kernels and Weights and Biases. |
| `setup.py` | Python package setup file for installing the project as a pip package. |

---

### ciBabyMambaHar/ -- CI-BabyMamba-HAR Model Package.

This is the primary model package containing the CI-BabyMamba-HAR architecture, dataset loaders, and utility modules.

#### ciBabyMambaHar/models/ -- Model Definitions.

| File | Description | Key Classes and Functions |
|---|---|---|
| `__init__.py` | Package initialization. Exports all model classes and frozen configuration constants. | `CiBabyMambaHar`, `createCiBabyMambaHar`, `CI_BABYMAMBA_HAR_CONFIG`, `WeightTiedBiDirMambaBlock`, `PureSelectiveScan`, `SimpleStem`, `ClassificationHead`. |
| `ciBabyMamba.py` | Main CI-BabyMamba-HAR model definition. Contains the complete architecture with Channel-Independent Stem, Patch Embedding, Weight-Tied Bidirectional SSM Backbone, Context-Gated Temporal Attention, and Classification Head. The frozen configuration dictionary `CI_BABYMAMBA_HAR_CONFIG` is defined here. | `CiBabyMambaHar` (main model class), `createCiBabyMambaHar` (factory function), `GatedTemporalAttention`, `ChannelIndependentStem`, `CI_BABYMAMBA_HAR_CONFIG`. |
| `ciBabyMambaBlock.py` | Core SSM building blocks. Contains the Weight-Tied Bidirectional Mamba Block and a pure PyTorch fallback for the Selective State Space scan algorithm. Also includes stochastic depth (DropPath) regularization. | `WeightTiedBiDirMambaBlock`, `PureSelectiveScan`, `DropPath`, `dropPath`. Legacy classes: `BabyMambaBlock`, `BiDirectionalMambaBlock`, `RecursiveBiDirectionalBlock`, `SEBlock`. |
| `stems.py` | Input stem modules for initial feature extraction from raw sensor data. The `SimpleStem` is the primary stem used for the Crossover-BiDir variant. Legacy stems are kept for ablation studies. | `SimpleStem`, `WideEyeStem`, `SpectralTemporalStem`, `TimeOnlyStem`, `HollowStem`, `SensorStem`, `DepthwiseSeparableConv1d`. |
| `heads.py` | Classification head modules that map model features to class predictions. | `ClassificationHead`, `MultiHeadClassificationHead`. |
| `ciBabyMambaAblations.py` | Ablation model variants used in the ablation studies section of the research paper. Each variant removes or modifies a single architectural component. | `CiBabyMambaHarFull` (Ablation A, full model baseline), `CiBabyMambaHarUnidirectional` (Ablation B, forward-only SSM), `CiBabyMambaHar2Layer` (Ablation C, 2 layers instead of 4), `CiBabyMambaHarNoPatching` (Ablation D, no discrete patching), `CiBabyMambaHarCnnOnly` (Ablation E, CNN instead of SSM), `getAblationModel` (factory function). |

#### ciBabyMambaHar/data/ -- Dataset Loaders.

Each dataset loader handles downloading (where applicable), preprocessing, windowing, and splitting of the corresponding HAR dataset. All loaders return PyTorch Dataset objects with `(data, label)` pairs where data has shape `[T, C]` (time steps, channels).

| File | Description | Key Classes and Functions |
|---|---|---|
| `__init__.py` | Package initialization. Exports all dataset classes and loader functions. | All dataset classes and `get*Loaders` functions. |
| `augmentations.py` | Data augmentation strategies for HAR training. Includes time warping, random scaling, random noise injection, and random rotation. | `HARaugment`, `RandomScaling`, `RandomNoise`, `RandomRotation`, `TimeWarping`, `Compose`, `getTrainAugmentation`. |
| `uciHar.py` | UCI-HAR dataset loader. Uses the official train/test split with 7,352 training and 2,947 test samples. 6 activity classes from smartphone accelerometer and gyroscope data. | `UciHarDataset`, `getUciHarLoaders`. |
| `motionSense.py` | MotionSense dataset loader. Loads smartphone accelerometer and gyroscope data with 6 activity classes. Subject-based train/test split. | `MotionSenseDataset`, `getMotionSenseLoaders`. |
| `wisdm.py` | WISDM dataset loader. Single-axis accelerometer data with 6 activity classes. Sliding window segmentation. | `WisdmDataset`, `getWisdmLoaders`. |
| `pamap2.py` | PAMAP2 dataset loader. Multi-IMU body-worn sensor data with 12 activity classes. Supports the Signal Rescue recipe with 10Hz Butterworth low-pass filtering and robust scaling. | `Pamap2Dataset`, `getPamap2Loaders`, `computeClassWeights`. |
| `skoda.py` | Skoda Mini Checkpoint dataset loader. 30-channel sensor data with 11 gesture classes. Supports the Signal Rescue recipe with 5Hz Butterworth low-pass filtering. | `SkodaDataset`, `getSkodaLoaders`, `computeSkodaClassWeights`. |
| `daphnet.py` | Daphnet Freezing of Gait dataset loader. Binary classification (Walk versus Freeze) with 9 channels. Supports the Signal Rescue recipe with 12Hz Butterworth low-pass filtering and class-weighted sampling. | `DaphnetDataset`, `getDaphnetLoaders`, `computeDaphnetClassWeights`. |
| `opportunity.py` | Opportunity dataset loader. Full-body sensor network with 79 channels. Supports locomotion and gesture classification tasks. | `OpportunityDataset`, `getOpportunityLoaders`. |
| `unimib.py` | UniMiB-SHAR dataset loader. Smartphone accelerometer data for Activities of Daily Living (ADL) and fall detection with 9 ADL classes. | `UniMiBSHARDataset`, `getUniMiBLoaders`. |

#### ciBabyMambaHar/utils/ -- Utility Modules.

| File | Description | Key Classes and Functions |
|---|---|---|
| `__init__.py` | Package initialization. Exports all utility classes and functions. | All utility exports. |
| `metrics.py` | Evaluation metrics for HAR. Includes accuracy, macro F1 score, confusion matrix computation, and running average tracking. | `Accuracy`, `F1Score`, `ConfusionMatrix`, `AverageMeter`, `MetricsTracker`. |
| `profiling.py` | Model profiling utilities. Counts parameters, computes MACs (Multiply-Accumulate Operations) using thop or fvcore, and benchmarks inference latency. | `countParameters`, `computeMacs`, `profileModel`, `benchmarkLatency`. |
| `optim.py` | Optimizer and learning rate scheduler utilities. Supports AdamW, Adam, and SGD optimizers with Cosine Annealing, Step, Linear, and OneCycleLR schedulers. Includes warmup scheduling. | `getOptimizer`, `getScheduler`, `WarmupScheduler`. |
| `checkpoint.py` | Model checkpoint utilities for saving and loading model weights and training state. | `saveCheckpoint`, `loadCheckpoint`, `saveModelOnly`, `loadModelOnly`. |

---

### crossoverBiDirBabyMambaHar/ -- Crossover-BiDir-BabyMamba-HAR Package.

This package contains the Crossover-BiDir-BabyMamba-HAR model, which is the predecessor to CI-BabyMamba-HAR with reduced d_state (8 instead of 16) and without Channel-Independent processing or Gated Attention.

#### crossoverBiDirBabyMambaHar/models/ -- Model Definitions.

| File | Description | Key Classes and Functions |
|---|---|---|
| `__init__.py` | Package initialization. Exports crossover model classes and configuration. | `CrossoverBiDirBabyMambaHar`, `CROSSOVER_BIDIR_BABYMAMBA_CONFIG`. |
| `crossoverBiDirBabyMamba.py` | Main Crossover-BiDir-BabyMamba-HAR model definition. Standalone architecture with 4 layers, d_state=8, and approximately 25,000 parameters. | `CrossoverBiDirBabyMambaHar`, factory functions. |
| `crossoverBiDirBlock.py` | Bidirectional SSM block implementation for the Crossover variant. | SSM block classes. |
| `ablations.py` | Ablation model variants for the Crossover-BiDir-BabyMamba-HAR architecture. | Ablation variant classes matching the CI-BabyMamba-HAR ablation structure. |

#### crossoverBiDirBabyMambaHar/scripts/ -- Crossover Training Scripts.

| File | Description |
|---|---|
| `__init__.py` | Package initialization. |
| `trainCrossoverBiDirBabyMambaHar.py` | Training script for Crossover-BiDir-BabyMamba-HAR with multiple random seeds, early stopping, and mixed-precision training. |
| `hpoCrossoverBiDirBabyMambaHar.py` | Hyperparameter optimization script for Crossover-BiDir-BabyMamba-HAR using Optuna with TPE sampler. |

---

### baselines/ -- Baseline Model Implementations.

Re-implementations of the comparison models used for fair BabyMamba-HAR benchmarking.

| File | Description | Key Classes |
|---|---|---|
| `__init__.py` | Package initialization. Exports all baseline classes. | `TinyHAR`, `DeepConvLSTM`, `LightDeepConvLSTM`, `TinierHAR`, `HARMamba`, `HARMambaLite`, `createHARMamba`. |
| `deepConvLstm.py` | DeepConvLSTM baseline (Ordonez et al., 2016) and variants. Includes the original 4-layer CNN + 2-layer LSTM architecture, TinierHAR (4-block ResNet with self-attention), and LightDeepConvLSTM (reduced-width variant). | `DeepConvLSTM` (~130K-154K params), `TinierHAR` (~7K-124K params), `LightDeepConvLSTM` (~15K-21K params). |
| `harMamba.py` | HARMamba baseline. A larger Mamba-based model with BiMambaBlock and ChannelMixer. Serves as a comparison point to show the efficiency of BabyMamba-HAR at a much smaller parameter budget. | `HARMamba`, `HARMambaLite`, `BiMambaBlock`, `createHARMamba`. |
| `tinyHar.py` | TinyHAR baseline (Zhou et al., ISWC 2022). Self-Attention based model with Temporal Weighted Aggregation. | `TinyHAR`, `SelfAttention`, `TemporalWeightedAggregation`. |

---

### scripts/ -- Training, HPO, and Evaluation Scripts.

| File | Description |
|---|---|
| `trainCiBabyMambaHar.py` | Main training script for CI-BabyMamba-HAR. Supports training on any dataset with 5 random seeds, 200 epochs, early stopping (patience=10), linear warmup (10 epochs), mixed-precision (AMP) training, and optional loading of HPO-optimized hyperparameters. Includes Signal Rescue recipes for Skoda, PAMAP2, and Daphnet datasets. |
| `hpoCiBabyMambaHar.py` | Hyperparameter optimization for CI-BabyMamba-HAR using Optuna. Searches over learning rate, weight decay, and dropout while keeping the architecture frozen. Uses 50 trials with 10 epochs per trial and TPE sampler. |
| `trainBaselines.py` | Training script for all baseline models (DeepConvLSTM, TinierHAR, TinyHAR, LightDeepConvLSTM) with the same training protocol as CI-BabyMamba-HAR for fair comparison. |
| `exportBabyMambaEdgeModels.py` | Core handcrafted-export generator for Pico 2 and ESP32. Supports float and row-wise `INT8` projection storage, fixture export, and parity metadata generation. |
| `exportBabyMambaPico2Models.py` | Convenience wrapper that writes the BabyMamba Pico 2 bundles into `Pico2Models/`. |
| `exportBabyMambaEsp32Models.py` | Convenience wrapper that writes the BabyMamba ESP32 bundles into `ESP32Models/` with the native `INT8` projection configuration. |
| `runBabyMambaPico2Sweep.py` | Serial deployment and measurement harness for the Raspberry Pi Pico 2 runtime. |
| `runBabyMambaEsp32Sweep.py` | Native ESP-IDF deployment and serial measurement harness for the classic ESP32 BabyMamba study. |
| `hpoBaselines.py` | HPO script for baseline models using the same Optuna protocol. |
| `runBaselines.py` | Convenience script to run training for all baseline models across all datasets. |
| `runBaselineRetraining.py` | Sequential seed-29 baseline retraining launcher used for the committed paper-aligned checkpoint sweep. |
| `runAblations.py` | Runs ablation studies for the legacy BabyMamba architecture variants. |
| `runCiBabyMambaHarAblations.py` | Runs the CI-BabyMamba-HAR ablation studies (Full, Unidirectional, 2-Layer, No Patching, CNN Only) across specified datasets. |
| `evaluate.py` | Comprehensive evaluation script. Loads a trained model checkpoint and computes accuracy, F1 score, confusion matrix, parameter count, MACs, and inference latency. |
| `benchmarkModels.py` | Benchmarks all models for parameter counts, MACs, and FLOPs. |
| `benchmarkLatency.py` | Benchmarks inference latency for all models across different hardware configurations. |
| `collectAblationResults.py` | Aggregates ablation study results from multiple runs and computes mean and standard deviation across seeds. |

---

### configs/ -- Dataset Configuration Files.

Each YAML file contains dataset-specific configuration parameters including input dimensions, number of classes, sequence length, HPO search ranges, and Signal Rescue recipe parameters where applicable.

| File | Description |
|---|---|
| `uciHar.yaml` | UCI-HAR configuration. 6 classes, 9 channels, 128 time steps. |
| `motionSense.yaml` | MotionSense configuration. 6 classes, 6 channels, 128 time steps. |
| `wisdm.yaml` | WISDM configuration. 6 classes, 3 channels, 128 time steps. |
| `pamap2.yaml` | PAMAP2 configuration. 12 classes, 19 channels, 128 time steps. Includes 10Hz Butterworth filter and robust scaling parameters. |
| `skoda.yaml` | Skoda configuration. 11 classes, 30 channels, 98 time steps. Includes 5Hz Butterworth filter and label smoothing. |
| `daphnet.yaml` | Daphnet configuration. 2 classes, 9 channels, 64 time steps. Includes 12Hz Butterworth filter and class weighting. |
| `ablation.yaml` | Ablation study configuration. Defines the five ablation variants (A0-A4), training parameters, and seed settings. |

---

### docs/ -- Documentation and Figures.

| File | Description |
|---|---|
| `docs/img/nanoharmamba_architecture.png` | CI-BabyMamba-HAR architecture diagram showing the complete data flow from input to classification. |
| `docs/img/nanomamba_crossover_bidir_architecture.png` | Crossover-BiDir-BabyMamba-HAR architecture diagram. |
| `docs/img/ssm_block_detail.png` | Detailed diagram of the Weight-Tied Bidirectional SSM Block internals, including the selective scan mechanism. |
| `docs/img/babymamba_results_grid.png` | Grid visualization of benchmark results across all datasets and models. |
| `docs/img/babymamba_ablation_combined.png` | Combined ablation study results visualization. |
| `docs/BabyMambaEdgeDeployment.md` | Canonical BabyMamba edge deployment methodology, including the MambaLite-Micro style export path, mixed-precision projection quantization, and hardware runtime adaptations. |
| `docs/EdgeDeployment.md` | Compatibility pointer that redirects older documentation references to `docs/BabyMambaEdgeDeployment.md`. |
| `docs/ESP32DeploymentResultsReport.md` | Native ESP32 deployment report for both BabyMamba families, including the `INT8` projection optimization path. |
| `docs/BaselineDeploymentResultsReport.md` | Consolidated baseline checkpoint and hardware deployment report for Pico 2 and ESP32. |

---

### results/ -- Experiment Output Directory.

This directory is initially empty. Running the training, HPO, or ablation scripts will populate it with JSON result files, model checkpoints, and summary outputs.

Expected subdirectories after running experiments:

| Subdirectory | Contents |
|---|---|
| `results/training/` | Per-seed training results with accuracy, F1, and confusion matrices. |
| `results/hpo/` | HPO trial results with best hyperparameters per dataset. |
| `results/ablations/` | Ablation study results for each variant and dataset. |
| `results/latency/` | Inference latency benchmarks. |
