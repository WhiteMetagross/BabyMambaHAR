# ESP32 Deployment Results Report:

## Scope:

This document describes the ESP32-related assets committed in this repository release. The primary purpose of the present update was to preserve the BabyMamba training path, the checkpoint zoo, the handcrafted recurrent export representation, and the target-specific runtime scaffolds. Export-ready ESP32 model bundles were therefore committed under `ESP32Models/`.

## What Is Included:

The following ESP32-facing assets are present.

- `ESP32Models/ciBabyMambaHar/`.
- `ESP32Models/crossoverBiDirBabyMambaHar/`.
- `embedded/esp32BabyMambaRuntime/`.
- `scripts/exportBabyMambaEsp32Models.py`.

Each dataset bundle contains a `babyMambaWeights.h` file and a matching `manifest.json`. The same recurrent representation is used across Pico 2 and ESP32 targets, so the exported weight bundles remain portable.

## Interpretation:

The committed ESP32 content should be interpreted as an export and runtime preparation package rather than as a complete measured hardware benchmark study. A full BabyMamba ESP32 benchmark table is not claimed in this repository snapshot.

This distinction is important for two reasons.

- The export representation is target portable, so the device bundles are useful independently of a specific board benchmark.
- The present repository release was focused on preserving the reproducible deployment path and the committed model zoo in a clean publishable form.

## Recommended Use:

If an ESP32 hardware study is to be extended from this repository, the following workflow should be followed.

1. Select the desired dataset bundle from `ESP32Models/`.
2. Copy the corresponding `babyMambaWeights.h` file into `embedded/esp32BabyMambaRuntime/`.
3. Compile the runtime with the preferred ESP32 toolchain.
4. Record flash, RAM, latency, and parity metrics with a serial benchmark harness.

## Relationship To The Pico 2 Study:

The Pico 2 study remains the measured deployment reference committed in this repository. The ESP32 export bundles were prepared from the same validated model and recurrence path, so they are suitable as a starting point for a future ESP32 benchmark extension.
