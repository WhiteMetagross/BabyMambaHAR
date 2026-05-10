# Baseline ESP32 Metrics:

## Overview:

This file summarizes the committed baseline ESP32 results preserved in `baselineEsp32Metrics.json`. The detailed comparative discussion is provided in [`docs/BaselineDeploymentResultsReport.md`](../../docs/BaselineDeploymentResultsReport.md).

## Family Summary:

| Model Family | Successful Bundles | Total Bundles | Average Latency (ms) | Average Parity vs PyTorch (%) | Peak Arena Used (B) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TinyHAR | 5 | 8 | 442.089 | 60.237 | 120760 |
| TinierHAR | 6 | 8 | 188.391 | 76.263 | 176400 |
| DeepConvLSTM | 0 | 8 | N/A | N/A | N/A |

## Interpretation:

On the committed classic ESP32 path, `TinierHAR` produced the strongest baseline coverage, while `DeepConvLSTM` remained blocked by internal-SRAM limits across all datasets. These measurements should be interpreted together with the richer discussion in the baseline deployment report.
