# Baseline Pico 2 Metrics:

## Overview:

This file summarizes the committed Raspberry Pi Pico 2 baseline results preserved in `baselinePico2Metrics.json`. The detailed comparative discussion is provided in [`docs/BaselineDeploymentResultsReport.md`](../../docs/BaselineDeploymentResultsReport.md).

## Family Summary:

| Model Family | Successful Bundles Preserved | Average Latency (ms) | Average Parity vs PyTorch (%) | Peak Arena Used (B) |
| --- | ---: | ---: | ---: | ---: |
| TinyHAR | 7 | 354.053 | 60.447 | 213380 |
| TinierHAR | 8 | 229.326 | 54.230 | 351668 |
| DeepConvLSTM | 1 | 768.780 | 25.482 | 244084 |

## Interpretation:

The Pico 2 baseline record should be interpreted carefully. Multiple baseline bundles were shown to execute on-device, but parity varied sharply across datasets and the preserved DeepConvLSTM coverage remained sparse. The BabyMamba deployment tables should therefore be treated as the primary edge result, with this file serving as the comparison-model appendix.
