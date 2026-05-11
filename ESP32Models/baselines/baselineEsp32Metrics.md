# ESP32 Baseline Metrics:

This file summarizes the committed ESP32 baseline deployment results carried in the BabyMamba-HAR repository.

| Model | Success / Total | Avg Latency ms | Avg Parity vs PyTorch (%) | Peak Arena Used (B) |
| --- | ---: | ---: | ---: | ---: |
| TinyHAR | 4 / 8 | 487.026 | 88.577 | 148108 |
| TinierHAR | 6 / 8 | 204.503 | 90.759 | 176400 |
| DeepConvLSTM | 0 / 8 |  |  |  |

| Model | Dataset | Success | Quant Recipe | Latency ms | Parity % | Failure |
| --- | --- | --- | --- | ---: | ---: | --- |
| TinyHAR | ucihar | False | int16_activations_int8_weights |  |  | Failed to allocate tensor arena from internal SRAM. |
| TinyHAR | motionsense | True | int16_activations_int8_weights | 735.682 | 88.106 | None |
| TinyHAR | wisdm | True | full_int8 | 347.674 | 99.206 |  |
| TinyHAR | pamap2 | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| TinyHAR | opportunity | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| TinyHAR | unimib | True | full_int8 | 346.616 | 97.252 |  |
| TinyHAR | skoda | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| TinyHAR | daphnet | True | int16_activations_int8_weights | 518.130 | 69.743 | None |
| TinierHAR | ucihar | True | full_int8 | 209.216 | 92.012 |  |
| TinierHAR | motionsense | True | int16_activations_int8_weights | 231.314 | 91.593 | None |
| TinierHAR | wisdm | True | full_int8 | 132.246 | 98.506 |  |
| TinierHAR | pamap2 | False | full_int8 |  |  | Failed to resize buffer. Requested: 48640, available 38536, missing: 10104 |
| TinierHAR | opportunity | False | full_int8 |  |  | Failed to resize buffer. Requested: 202240, available 38536, missing: 163704 |
| TinierHAR | unimib | True | full_int8 | 130.707 | 98.953 |  |
| TinierHAR | skoda | True | full_int8 | 369.053 | 99.400 |  |
| TinierHAR | daphnet | True | int16_activations_int8_weights | 154.479 | 64.089 | None |
| DeepConvLSTM | ucihar | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | motionsense | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | wisdm | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | pamap2 | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | opportunity | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | unimib | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | skoda | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
| DeepConvLSTM | daphnet | False | full_int8 |  |  | Failed to allocate tensor arena from internal SRAM. |
