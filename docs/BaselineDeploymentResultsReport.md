# Baseline Deployment Results Report:

## Scope:

This document summarizes the baseline deployment artifacts that accompany the BabyMamba-HAR repository. The preserved families are `TinyHAR`, `TinierHAR`, and `DeepConvLSTM`. The baseline record should be read as a comparative deployment appendix to the BabyMamba study, rather than as a separate primary model report.

The corrected ESP32 baseline summary is stored in `ESP32Models/baselines/baselineEsp32Metrics.json` and `ESP32Models/baselines/baselineEsp32Metrics.md`. The affected TinyHAR and TinierHAR artifact folders were also refreshed with the repaired quantized bundles used in the updated ESP32 analysis.

## Main Update:

The previous catastrophic ESP32 quantized parity collapses on selected baseline rows were revisited. A repaired quantized export path with `INT16` activations and `INT8` weights was promoted for the rows where it materially improved parity.

The corrected baseline ESP32 picture is as follows.

- `TinyHAR / motionsense` improved to `88.106%`.
- `TinyHAR / daphnet` improved from `0%` to `69.743%`.
- `TinierHAR / motionsense` improved to `91.593%`.
- `TinierHAR / daphnet` improved from `0%` to `64.089%`.
- `TinyHAR / ucihar` recovered strongly on desktop parity, but the repaired quantized bundle did not fit the classic ESP32 internal-SRAM limit and therefore became a real deployment failure.

## ESP32 Family Summary:

| Model Family | Successful Bundles | Total Bundles | Average Latency (ms) | Average Parity vs PyTorch (%) | Peak Arena Used (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyHAR | 5 | 8 | 455.584 | 69.958 | 148108 |
| TinierHAR | 6 | 8 | 192.204 | 80.336 | 176400 |
| DeepConvLSTM | 0 | 8 | N/A | N/A | N/A |

## Corrected ESP32 Rows:

| Model | Dataset | Status | Quant Recipe | Latency (ms) | Arena Used (B) | Parity vs PyTorch (%) | Comment |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| TinyHAR | UCI-HAR | Fails. | `INT16 activations + INT8 weights`. | N/A | N/A | N/A | The repaired quantized graph exceeded the classic ESP32 SRAM limit. |
| TinyHAR | MotionSense | Runs. | `INT16 activations + INT8 weights`. | 735.682 | 148108 | 88.106 | A real parity recovery was obtained. |
| TinyHAR | Daphnet | Runs. | `INT16 activations + INT8 weights`. | 518.130 | 100684 | 69.743 | The collapse was repaired, but fidelity remained limited. |
| TinierHAR | MotionSense | Runs. | `INT16 activations + INT8 weights`. | 231.314 | 173996 | 91.593 | This became a strong deployable quantized row. |
| TinierHAR | Daphnet | Runs. | `INT16 activations + INT8 weights`. | 154.479 | 97644 | 64.089 | The degenerate result was repaired, but the row remained weak. |

## Pico 2 Note:

The baseline Pico 2 record was not remeasured in this pass. The refreshed baseline bundle folders were synchronized so that the canonical quantized artifacts match the corrected exporter state, but the published Pico 2 latency tables remain the previously validated board measurements.

## Conclusion:

The corrected baseline record now separates two issues that had previously been conflated. Part of the low-parity ESP32 baseline behavior was caused by an unstable quantized export recipe. That issue was materially improved on several rows. However, the corrected study also shows that better quantized fidelity does not remove the board limits. On the classic no-PSRAM ESP32, some repaired baseline bundles remain weak, and some become memory-limited once the more faithful quantized graph is used.
