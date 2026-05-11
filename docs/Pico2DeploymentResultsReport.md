# Raspberry Pi Pico 2 Deployment Results Report:

## Scope:

This report summarizes the committed Raspberry Pi Pico 2 deployment study for the BabyMamba-HAR model families. The measurements were obtained from the handcrafted recurrent C++ runtime and were stored in `Pico2Models/babymamba_pico2_metrics.json`. Two model families were evaluated.

- `CiBabyMambaHar`.
- `CrossoverBiDirBabyMambaHar`.

## Experimental Setting:

The deployment study was executed with dataset-specific exported headers and a fixed serial benchmark harness. For each bundle, ten timed inference iterations were recorded after a warm-up pass. Flash footprint, global RAM usage, scratch memory usage, and parity against the PyTorch reference were preserved.

No graph-compiled `FP32` versus `INT8` split applies to this report. The Pico 2 runtime is a handcrafted recurrent implementation of the BabyMamba selective scan, and the committed exports should therefore be interpreted as native recurrent headers rather than as TFLite-style variants.

## Executive Summary:

The crossover bidirectional family was found to be substantially faster on the Pico 2 while preserving essentially perfect parity. The channel-independent family also ran successfully across all committed datasets, but much larger latency was observed because the recurrent scan is executed independently across channels and layers.

The family-level summary is given below.

| Model Family | Successful Runs | Average Latency (ms) | Average Parity vs PyTorch (%) | Flash Range (B) | Global RAM Range (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CI-BabyMamba-HAR | 8 | 11762.049 | 99.937286 | 182476-222612 | 29180-49212 |
| Crossover-BiDir-BabyMamba-HAR | 8 | 481.898 | 99.974827 | 174572-252492 | 28848-43824 |

## Interpretation:

Several conclusions may be drawn from the measured study.

- Full Pico 2 coverage was obtained for both BabyMamba families across the committed dataset bundles.
- Parity remained extremely high after the corrected export path was applied to the channel-independent selective scan.
- The crossover family should be preferred when latency is the primary deployment constraint.
- The channel-independent family remains attractive when a more expressive recurrent formulation is desired and the latency budget is more relaxed.

## CI-BabyMamba-HAR Results:

| Dataset | Status | Flash (B) | Global RAM (B) | Scratch (B) | Average Latency (ms) | Parity vs PyTorch (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UCI-HAR | ok | 186772 | 49212 | 38948 | 7115.815 | 99.822037 |
| MotionSense | ok | 185228 | 49212 | 38948 | 4703.395 | 99.897537 |
| WISDM | ok | 183684 | 49212 | 38948 | 2336.074 | 99.989006 |
| PAMAP2 | ok | 193580 | 43068 | 32804 | 11886.876 | 99.946793 |
| Opportunity | ok | 222612 | 43068 | 32804 | 49644.104 | 99.974693 |
| UniMiB-SHAR | ok | 184580 | 43068 | 32804 | 1860.691 | 99.901375 |
| Skoda | ok | 194452 | 36316 | 26052 | 13873.470 | 99.985771 |
| Daphnet | ok | 182476 | 29180 | 18916 | 2675.968 | 99.981079 |

The channel-independent family was observed to be stable and highly faithful on-device after the export recurrence was aligned with the fallback chunked selective scan used during desktop inference. The long latency measured on Opportunity should therefore be interpreted as a computational property of the model family rather than as an export artifact.

## Crossover-BiDir-BabyMamba-HAR Results:

| Dataset | Status | Flash (B) | Global RAM (B) | Scratch (B) | Average Latency (ms) | Parity vs PyTorch (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| UCI-HAR | ok | 180364 | 43824 | 33560 | 506.995 | 99.987808 |
| MotionSense | ok | 177244 | 43824 | 33560 | 500.558 | 99.905952 |
| WISDM | ok | 174572 | 43824 | 33560 | 493.979 | 99.987518 |
| PAMAP2 | ok | 192900 | 43824 | 33560 | 519.886 | 99.970535 |
| Opportunity | ok | 252492 | 43824 | 33560 | 694.997 | 99.990433 |
| UniMiB-SHAR | ok | 175284 | 43824 | 33560 | 496.756 | 99.972000 |
| Skoda | ok | 199172 | 36544 | 26280 | 396.776 | 99.985718 |
| Daphnet | ok | 176060 | 28848 | 18584 | 245.236 | 99.998650 |

The crossover bidirectional family exhibited a much tighter latency band and a simpler memory profile. This behavior is consistent with the more deployment-friendly recurrent structure used by the model.

## Comparative Discussion:

The Pico 2 study suggests that BabyMamba-HAR should be viewed as a family of deployment tradeoffs rather than as a single operating point.

- `CiBabyMambaHar` offered broader channel-wise modeling at the cost of substantial latency.
- `CrossoverBiDirBabyMambaHar` offered a much sharper latency profile while preserving near-perfect agreement with the PyTorch reference.
- Both families remained comfortably within the measured flash and RAM envelope of the Pico 2 runtime used in this study.

## Stored Artifacts:

The main result artifacts are committed in the following locations.

- `Pico2Models/babymamba_pico2_metrics.json`.
- `Pico2Models/babymamba_pico2_metrics.md`.
- `Pico2Models/ciBabyMambaHar/`.
- `Pico2Models/crossoverBiDirBabyMambaHar/`.

These files should be treated as the canonical deployment record for the BabyMamba Pico 2 study contained in this repository snapshot.
