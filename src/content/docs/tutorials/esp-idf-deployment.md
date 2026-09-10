---
title: "Deploy on ESP32-S3 (ESP-IDF)"
description: "Run a TiGrIS-compiled INT8 model on an ESP32-S3 with the ESP-NN kernels, using the ESP-IDF example app."
sidebar:
  order: 200
---

This guide runs a TiGrIS-compiled INT8 model on an ESP32-S3 under ESP-IDF, with
the accelerated ESP-NN kernels, from an ONNX file to an inference running on the
board. It uses the ESP-IDF example that ships in the
[tigris-runtime](https://github.com/raws-labs/tigris-runtime) repository. The
runtime is also published on the ESP Component Registry as
`raws-labs/tigris-runtime`, for adding to an existing project — covered in the
last section.

Compared with the [Cortex-M guide](/tutorials/cortex-m-deployment/),
the ESP32 flow is a little different: the example loads the `.tgrs` plan from a
dedicated flash **partition** at runtime rather than embedding it in the
firmware, and the runtime executes the plan through its generic loader — so
there is no per-model code generation step, and swapping models is just
re-flashing the plan.

## What you'll do

1. Get the ESP-IDF example from the runtime repository.
2. Compile your model to a plan and flash it to the board's plan partition.
3. Build and flash the app, then read the on-device output and check it against
   the host.

## Prerequisites

- ESP-IDF v5.x installed and its environment sourced
- Python 3.10+ with `tigris-ml` installed (`pip install tigris-ml`)
- An ESP32-S3 board with PSRAM — the example is set up for the
  ESP32-S3-DevKitC-1 N16R8 (16 MB flash, 8 MB PSRAM)

## Step 1: Get the example

```bash
git clone https://github.com/raws-labs/tigris-runtime.git
cd tigris-runtime
git checkout v0.6.0
cd examples/esp32
```

The example includes the runtime as a **local** ESP-IDF component (by relative
path), so it builds straight from this clone with nothing to fetch. To pull the
runtime into a *different* project instead, use the managed component — see
[the last section](#use-the-runtime-as-a-managed-component-in-your-own-project).

## Step 2: Set the target

```bash
idf.py set-target esp32s3
```

This applies the ESP32-S3 defaults the example ships: PSRAM enabled, 16 MB
flash, and a custom partition table with a dedicated `plan` partition that holds
the model.

## Step 3: Compile your model

```bash
tigris compile model.onnx -m 64K -m 8M --xip -o model.tgrs
```

| Flag | Meaning |
|------|---------|
| `-m 64K -m 8M` | Memory pools, fast to slow: SRAM then PSRAM. The compiler places tensors across them. |
| `--xip` | Execute-in-place — weights are read from flash at runtime instead of copied to SRAM. |
| `-o model.tgrs` | Output path for the binary plan. |

PSRAM (the second `-m` tier) is what lets multi-stage models fit; without it,
only models whose activations fit a single SRAM arena are supported. See the
[`compile`](/toolchain/compile/) reference and the
[Quickstart](/getting-started/quickstart/) for more on
budgets and XIP.

## Step 4: Build and flash the app

For the accelerated ESP-NN kernels:

```bash
idf.py build -DTIGRIS_ENABLE_ESP_NN=ON
idf.py flash
```

Enabling ESP-NN requires the `espressif/esp-nn` managed component to be
available to the project; see the example's `README.md` for the one-time
component step. Omit the flag to build with the portable INT8 kernels instead —
useful as a fallback or for a first bring-up.

## Step 5: Flash the plan

The app reads the model from the `plan` partition, so flash the plan there
separately from the firmware:

```bash
scripts/flash_plan.sh model.tgrs
```

Because the plan lives in its own partition, deploying a different model later is
just recompiling and re-running this step — no rebuild.

## Step 6: Run and read the output

```bash
idf.py monitor
```

The app finds the `plan` partition, loads the model, prepares the backend, runs
one inference on a fixed input, and prints the INT8 output and timing. Exit the
monitor with `Ctrl-]`.

## Step 7: Check the output against the host

The runtime produces the same INT8 output on the device as on the host for the
same plan and input. Run the same model and input through the host — the
`reference` backend, or ONNX Runtime on the original float model — and compare
the INT8 vectors. They should match exactly; a mismatch usually means the
on-device input differs from the one you compared against.

## Swapping models and other targets

- **A different model:** recompile to a new `.tgrs` and re-run
  `scripts/flash_plan.sh` — the firmware does not need rebuilding.
- **A plain ESP32** (no `-S3`): the example ships `partitions_esp32.csv` as well;
  set the target accordingly. Boards without PSRAM are limited to single-arena
  models.

## Troubleshooting

- **`partition 'plan' not found`** — the plan hasn't been flashed, or the
  partition table isn't the example's. Run `scripts/flash_plan.sh` after
  `idf.py set-target`.
- **Multi-stage model won't fit** — confirm PSRAM is enabled (it is in the
  example's `esp32s3` defaults) and that you passed the second `-m` PSRAM tier at
  compile time.
- **ESP-NN not active** — a build without `-DTIGRIS_ENABLE_ESP_NN=ON`, or without
  the `espressif/esp-nn` component, falls back to the portable kernels; the
  output is identical but slower.

## Use the runtime as a managed component in your own project

The steps above build the bundled example. To add the runtime to an existing
ESP-IDF project instead of cloning the repository, install it from the ESP
Component Registry:

```bash
idf.py add-dependency "raws-labs/tigris-runtime"
```

This adds `raws-labs/tigris-runtime` — and its `espressif/esp-nn` dependency — to
your project's `main/idf_component.yml`. You then supply the parts the example
provides here: a `plan` flash partition for the `.tgrs` (copy
`partitions_esp32s3.csv` and `scripts/flash_plan.sh` from the example), and the
load-and-run code (use the example's `main/main.c` as the template — it finds the
`plan` partition, loads the model, prepares the backend, and runs one inference).

## Next steps

- [`compile`](/toolchain/compile/) — the full CLI
  reference.
- The [Cortex-M deployment guide](/tutorials/cortex-m-deployment/)
  — the same idea on Arm Cortex-M, with the plan embedded in the firmware.
- [Operator and Backend Support](/runtime/operator-support/)
  — what the ESP-NN backend accelerates.
