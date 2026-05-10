# Baseline Deployment Results Report:

## Scope:

This document consolidates the comparison-model artifacts committed alongside the BabyMamba-HAR study. The preserved baseline families are `TinyHAR`, `TinierHAR`, and `DeepConvLSTM`. Seed-29 checkpoints, device bundles, and sanitized hardware summaries are now versioned directly inside the repository so that the comparative edge study can be inspected without reconstructing the original research workspace.

The baseline study should be interpreted as a deployment companion to the BabyMamba report, rather than as a replacement for it. The present document is intended to clarify which baseline bundles were reproduced, which board-level results were measured, and which operating points remained unstable or memory-limited on the embedded targets.

## Artifact Layout:

The committed baseline assets are organized as follows.

- `models/baselines/`. Seed-29 checkpoints, run summaries, and per-dataset training metadata.
- `Pico2Models/baselines/`. Pico 2 TFLite Micro bundles and the preserved Pico 2 summary JSON.
- `ESP32Models/baselines/`. ESP32-oriented bundles and the preserved native ESP32 summary JSON.
- `scripts/trainBaselines.py`. Baseline training program with explicit checkpoint and JSON artifact saving.
- `scripts/runBaselineRetraining.py`. Sequential seed-29 launcher for the paper-aligned baseline retraining sweep.

`LightDeepConvLSTM` support remains present in the training code, but a validated edge-artifact bundle was not available in the committed study snapshot. That model family is therefore documented as training-supported but not deployment-preserved here.

## Experimental Notes:

The baseline measurements were obtained from the already validated embedded export bundles. The Pico 2 records correspond to the TFLite Micro deployment path that was used during the baseline study, while the ESP32 records correspond to the native classic-ESP32 deployment path preserved in the artifact summaries. Because these measurements were drawn from the committed study outputs, the tables below distinguish between three cases.

- A dataset that ran successfully on the target and produced a measured latency and parity value.
- A dataset that reached the target but yielded weak parity, which should be treated as a cautionary deployment outcome rather than as a silent success.
- A dataset that failed because the tensor arena could not be satisfied from the target memory budget.

## Family-Level Summary:

### Raspberry Pi Pico 2:

| Model Family | Successful Bundles Preserved | Average Latency (ms) | Average Parity vs PyTorch (%) | Peak Arena Used (B) | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| TinyHAR | 7 | 354.053 | 60.447 | 213380 | Broad coverage was achieved, but parity varied strongly across datasets. |
| TinierHAR | 8 | 229.326 | 54.230 | 351668 | All committed bundles ran, yet several datasets showed severe logit distortion. |
| DeepConvLSTM | 1 | 768.780 | 25.482 | 244084 | Only the preserved Daphnet run remained available in the committed record. |

### ESP32:

The ESP32 study was carried out on the classic no-PSRAM board used in the native runtime experiments.

| Model Family | Successful Bundles | Total Bundles | Average Latency (ms) | Average Parity vs PyTorch (%) | Peak Arena Used (B) | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TinyHAR | 5 | 8 | 442.089 | 60.237 | 120760 | A usable but selective deployment profile was observed. |
| TinierHAR | 6 | 8 | 188.391 | 76.263 | 176400 | Better ESP32 coverage was obtained after the runtime fixes. |
| DeepConvLSTM | 0 | 8 | N/A | N/A | N/A | The family remained blocked by internal-SRAM limits on this board. |

## Raspberry Pi Pico 2 Results:

### TinyHAR:

| Dataset | Status | Latency (ms) | Arena Used (B) | Flash (B) | Parity vs PyTorch (%) | Top-1 Match | Comment |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| UCI-HAR | Runs. | 328.974 | 126916 | 597924 | 6.124 | No. | Severe parity collapse was observed despite a completed run. |
| MotionSense | Runs. | 249.543 | N/A | N/A | 66.585 | Yes. | Functional but only moderately faithful. |
| PAMAP2 | Runs. | 641.389 | N/A | N/A | 80.223 | No. | Long latency and a label flip were observed. |
| WISDM | Runs. | 171.717 | N/A | N/A | 96.965 | Yes. | This was the cleanest Pico TinyHAR result. |
| UniMiB-SHAR | Runs. | 170.132 | 84868 | 588180 | 88.440 | Yes. | Strong practical agreement was retained. |
| Skoda | Runs. | 742.740 | 213380 | 592052 | 84.792 | Yes. | The heaviest TinyHAR Pico runtime was observed here. |
| Daphnet | Runs. | 173.879 | 67588 | 512420 | 0.000 | Yes. | The preserved logits were degenerate and should not be treated as a reliable deployment result. |

### TinierHAR:

| Dataset | Status | Latency (ms) | Arena Used (B) | Flash (B) | Parity vs PyTorch (%) | Top-1 Match | Comment |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| MotionSense | Runs. | 100.396 | 164788 | 685252 | 66.702 | Yes. | Faster than TinyHAR, but parity remained only moderate. |
| UCI-HAR | Runs. | 121.846 | 172468 | 691444 | 77.517 | Yes. | Functional deployment was obtained with noticeable logit drift. |
| PAMAP2 | Runs. | 205.091 | 198068 | 721908 | 16.485 | No. | The run completed, but the preserved output was not deployment-grade. |
| WISDM | Runs. | 76.412 | 157108 | 688620 | 79.044 | Yes. | A compact runtime profile was obtained. |
| UniMiB-SHAR | Runs. | 77.248 | 157108 | 688836 | 81.975 | Yes. | Moderate parity loss was still present. |
| Skoda | Runs. | 257.794 | 182244 | 664164 | 84.647 | Yes. | The larger activity set remained tractable on-device. |
| Daphnet | Runs. | 65.231 | 88340 | 547916 | 0.000 | Yes. | The preserved logits were again degenerate in the committed study record. |
| Opportunity | Runs. | 930.591 | 351668 | 844172 | 27.469 | No. | This was the most expensive TinierHAR Pico bundle and did not preserve parity well. |

### DeepConvLSTM:

| Dataset | Status | Latency (ms) | Arena Used (B) | Flash (B) | Parity vs PyTorch (%) | Top-1 Match | Comment |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Daphnet | Runs. | 768.780 | 244084 | 1239620 | 25.482 | Yes. | Only one preserved Pico 2 run was available, and fidelity remained weak. |

The Pico baseline record should therefore be read cautiously. The preserved bundles show that multiple baseline families can execute on the RP2350-based path, but the BabyMamba study remained much cleaner in parity retention and deployment consistency.

## ESP32 Results:

### TinyHAR:

| Dataset | Status | Latency (ms) | Arena Used (B) | Parity vs PyTorch (%) | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| UCI-HAR | Runs. | 673.217 | 120760 | 19.600 | Strong distortion remained in the committed result. |
| MotionSense | Runs. | 501.759 | 99424 | 85.127 | A practical run was achieved with moderate accuracy drift. |
| WISDM | Runs. | 347.674 | 78400 | 99.206 | This was the most faithful TinyHAR ESP32 result. |
| PAMAP2 | Fails. | N/A | N/A | N/A | Internal SRAM could not satisfy the tensor arena request. |
| Opportunity | Fails. | N/A | N/A | N/A | Internal SRAM could not satisfy the tensor arena request. |
| UniMiB-SHAR | Runs. | 346.616 | 78704 | 97.252 | Strong parity was retained. |
| Skoda | Fails. | N/A | N/A | N/A | Internal SRAM could not satisfy the tensor arena request. |
| Daphnet | Runs. | 341.178 | 63608 | 0.000 | The run completed, but the preserved logits were degenerate. |

### TinierHAR:

| Dataset | Status | Latency (ms) | Arena Used (B) | Parity vs PyTorch (%) | Comment |
| --- | --- | ---: | ---: | ---: | --- |
| UCI-HAR | Runs. | 209.216 | 164848 | 92.012 | Good parity was retained on the classic ESP32. |
| MotionSense | Runs. | 176.314 | 157168 | 68.708 | A successful run was obtained, but parity remained moderate. |
| WISDM | Runs. | 132.246 | 149488 | 98.506 | One of the strongest TinierHAR ESP32 deployments. |
| PAMAP2 | Fails. | N/A | N/A | N/A | Buffer expansion exceeded the remaining arena by `10104` bytes. |
| Opportunity | Fails. | N/A | N/A | N/A | Buffer expansion exceeded the remaining arena by `163704` bytes. |
| UniMiB-SHAR | Runs. | 130.707 | 149488 | 98.953 | Strong parity was preserved. |
| Skoda | Runs. | 369.053 | 176400 | 99.400 | The largest successful TinierHAR ESP32 arena was observed here. |
| Daphnet | Runs. | 112.807 | 84272 | 0.000 | The run completed, but the committed logits remained degenerate. |

### DeepConvLSTM:

| Dataset | Status | Comment |
| --- | --- | --- |
| UCI-HAR | Fails. | Tensor arena allocation from internal SRAM failed. |
| MotionSense | Fails. | Tensor arena allocation from internal SRAM failed. |
| WISDM | Fails. | Tensor arena allocation from internal SRAM failed. |
| PAMAP2 | Fails. | Tensor arena allocation from internal SRAM failed. |
| Opportunity | Fails. | Tensor arena allocation from internal SRAM failed. |
| UniMiB-SHAR | Fails. | Tensor arena allocation from internal SRAM failed. |
| Skoda | Fails. | Tensor arena allocation from internal SRAM failed. |
| Daphnet | Fails. | Tensor arena allocation from internal SRAM failed. |

## Comparative Interpretation:

Three broad conclusions may be drawn from the preserved baseline record.

- `TinierHAR` was the most competitive classical baseline on ESP32, where six dataset bundles were carried successfully and the latency profile remained materially lower than `TinyHAR`.
- `TinyHAR` was more broadly represented in the Pico 2 record, but the parity spread was wide enough that each dataset should be read individually rather than collapsed into a single reliable deployment claim.
- `DeepConvLSTM` remained the least microcontroller-friendly family in the committed edge study. Even when a run completed on Pico 2, its memory footprint and parity profile were not competitive with the BabyMamba families.

The baseline record therefore reinforces the main edge-computing narrative of the BabyMamba paper. The handcrafted BabyMamba recurrent path was not only smaller in the intended operating regime, but also more stable across datasets once the export recurrence had been aligned with the reference implementation.

## Reproducibility And Limitations:

The baseline checkpoint zoo and device bundles were committed so that the comparative study remains inspectable. At the same time, the following limitations should be kept explicit.

- `LightDeepConvLSTM` training support is present, but its validated edge bundles were not available in the committed study snapshot.
- The Pico 2 baseline summary preserves only the runs that were retained in the study artifacts, rather than a fully symmetric success and failure matrix for every family.
- Several baseline runs completed while still showing very weak parity, so successful execution should not automatically be interpreted as deployment readiness.

For these reasons, the baseline assets should be read as a transparent comparative record rather than as a curated best-case benchmark table.
