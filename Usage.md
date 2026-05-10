# Usage:

## Overview:

The practical workflow in this repository is divided into three stages. First, checkpoints are trained or retrained. Second, the trained weights are exported into device-friendly C headers. Third, the embedded runtimes are compiled and benchmarked on the target board. The committed artifacts allow any of these stages to be resumed independently.

## Training The CI-BabyMamba-HAR Models:

The paper-aligned CI-BabyMamba-HAR retraining path has been preserved in `scripts/trainCiBabyMambaHar.py`. The following command runs the full seed-29 retraining sweep sequentially across all datasets.

```bash
python scripts/runCiBabyMambaHarRetraining.py --datasets all --seed-list 29 --epochs 200 --patience 10
```

A single dataset run may also be launched directly.

```bash
python scripts/trainCiBabyMambaHar.py --dataset ucihar --seed-list 29 --epochs 200 --patience 10 --outDir results/training
```

The saved outputs are written under `results/training/ciBabyMambaHar/` and include the following files.

- `best_model_seed29.pt`.
- `model_state_seed29.pt`.
- `run_config_seed29.json`.
- `train_result_seed29.json`.
- `summary.json`.

## Reusing The Committed Checkpoints:

The validated checkpoint zoo is already committed under `models/`. This layout allows export generation without retraining.

- `models/ciBabyMambaHar/<dataset>/`.
- `models/crossoverBiDirBabyMambaHar/<dataset>/`.

Each dataset folder contains the deployable checkpoint and its associated run metadata.

## Exporting Pico 2 Model Bundles:

The handheld deployment path is driven by `scripts/exportBabyMambaEdgeModels.py`. A convenience wrapper is also included.

```bash
python scripts/exportBabyMambaPico2Models.py
```

If a single family should be exported, the core script may be called directly.

```bash
python scripts/exportBabyMambaEdgeModels.py --variant ciBabyMambaHar --datasets all --output-root Pico2Models
python scripts/exportBabyMambaEdgeModels.py --variant crossoverBiDirBabyMambaHar --datasets all --output-root Pico2Models
```

Each export bundle contains the following files.

- `babyMambaWeights.h`.
- `manifest.json`.

The generated header contains the weights, fixture sample, class names, and parity references required by the embedded runtime.

## Exporting ESP32 Model Bundles:

The ESP32 bundle generation path uses the same recurrent C representation. A convenience wrapper is included for symmetry with the Pico 2 workflow.

```bash
python scripts/exportBabyMambaEsp32Models.py
```

The generated output is written under `ESP32Models/`. The bundles are portable because the handcrafted recurrence engine is not tied to a graph compiler backend.

## Running The Pico 2 Benchmark Sweep:

The Arduino-based Pico 2 runtime is stored in `embedded/pico2BabyMambaRuntime/`. The measured sweep helper is provided in `scripts/runBabyMambaPico2Sweep.py`.

```bash
python scripts/runBabyMambaPico2Sweep.py --variants all --datasets all
```

When a full sweep is not desired, a narrower invocation may be used.

```bash
python scripts/runBabyMambaPico2Sweep.py --variants ciBabyMambaHar --datasets ucihar,motionsense
```

The output directory will contain serial logs and a merged JSON summary. The committed reference results are already available in `Pico2Models/babymamba_pico2_metrics.json`.

## Runtime Directories:

The main embedded runtime folders are listed below.

- `embedded/pico2BabyMambaRuntime/`. Raspberry Pi Pico 2 runtime using the handcrafted recurrent engine.
- `embedded/esp32BabyMambaRuntime/`. ESP32-oriented runtime scaffold that consumes the same generated header format.

In both cases, the active dataset bundle is selected by copying the desired `babyMambaWeights.h` file into the runtime project directory before compilation.

## Result Interpretation:

The committed Pico 2 study should be read with the two BabyMamba families in mind.

- `CrossoverBiDirBabyMambaHar` offers the lower-latency deployment profile.
- `CiBabyMambaHar` preserves very high parity while carrying a much larger recurrent cost.

Detailed measured tables are provided in `docs/Pico2DeploymentResultsReport.md`.

## Notes:

- The export tools rely on repository-local datasets when a fresh export is generated from checkpoints.
- The committed `Pico2Models/` and `ESP32Models/` folders already contain validated generated bundles for direct inspection.
- The training programs remain compatible with both fused `mamba-ssm` kernels and the pure PyTorch fallback scan path.
