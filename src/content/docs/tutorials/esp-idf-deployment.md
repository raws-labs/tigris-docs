---
title: "Deploy on ESP32-S3 (ESP-IDF)"
description: "Run a TiGrIS-compiled INT8 model on an ESP32-S3 with the ESP-NN kernels, from the bundled example to your own model."
sidebar:
  order: 200
---

TiGrIS ships as a published ESP-IDF component,
[`raws-labs/tigris-runtime`](https://components.espressif.com/components/raws-labs/tigris-runtime),
on the ESP Component Registry. This guide first runs the **bundled example** on
an ESP32-S3, then shows how to deploy **your own model** by adding the runtime
to a project.

Everything runs through the runtime's generic plan loader, so there is no
per-model code generation. You compile an ONNX model to a `.tgrs` plan and the
runtime executes it.

## Prerequisites

- ESP-IDF **v5.x** installed and its environment sourced.
- An **ESP32-S3 with PSRAM** (the example targets an ESP32-S3-DevKitC-1 N16R8:
  16 MB flash, 8 MB PSRAM).
- For your own models: Python 3.10+ with `tigris-ml` (`pip install tigris-ml`).

## Run the bundled example

The `getting-started` example runs a real 256x256 INT8 U-Net. Its largest
activation is about 1.19 MiB and its naive peak is about 2.38 MiB, which makes an
arena-based runtime such as TFLite Micro run out of memory. TiGrIS 2D-tiles the
model into a **232 KiB** fast arena and spills the skip tensors to PSRAM. The
plan and a golden output are embedded in the app, so there is nothing to flash
separately.

```bash
idf.py create-project-from-example "raws-labs/tigris-runtime:getting-started"
cd getting-started
idf.py set-target esp32s3
idf.py flash monitor
```

The serial output prints the model, the fit numbers, the inference latency, and
a self-check against the embedded reference, ending in:

```
SELF_CHECK: PASS
TIGRIS_DONE
```

The encoder convolutions run on the **ESP-NN** accelerated kernels. The decoder
(ConvTranspose and Concat) runs on the portable INT8 reference kernels, so the
run is dominated by the decoder and takes about 30 seconds. Exit the monitor
with `Ctrl-]`.

## Deploy your own model

### Step 1: Add the runtime to your project

```bash
idf.py add-dependency "raws-labs/tigris-runtime"
```

This adds `raws-labs/tigris-runtime` and its `espressif/esp-nn` dependency to
your project's `main/idf_component.yml`, and defines `TIGRIS_HAS_ESP_NN`, so the
accelerated kernels are available with no extra flags.

### Step 2: Compile your model

```bash
tigris compile model.onnx -m 232K -m 8M -o model.tgrs
```

| Flag | Meaning |
|------|---------|
| `-m 232K -m 8M` | Memory pools, fast to slow: internal SRAM then PSRAM. The compiler places and tiles tensors across them. |
| `--xip` | Optional. Execute-in-place: weights are read from flash at runtime instead of held in the arena. |
| `-o model.tgrs` | Output path for the binary plan. |

PSRAM (the second `-m` tier) is what lets multi-stage models fit. Without it,
only models whose activations fit a single SRAM arena are supported. See the
[`compile`](/toolchain/compile/) reference and the
[Quickstart](/getting-started/quickstart/) for more on budgets and XIP.

### Step 3: Embed the plan and run it

The simplest integration embeds the plan in the firmware, the way the
`getting-started` example does. Use its `main/` as a working template:

- **`main/CMakeLists.txt`**: `EMBED_FILES "model.tgrs"` so the plan is linked
  into the app.
- **`main/main.c`**:
  1. Load the embedded bytes with `tigris_plan_load()`.
  2. Allocate a two-tier arena. Take the fast tier from internal SRAM
     (`MALLOC_CAP_INTERNAL`) and the slow tier from PSRAM (`MALLOC_CAP_SPIRAM`),
     plus the executor workspace from `tigris_executor_workspace_required()`,
     and initialize `tigris_mem_t` with `tigris_mem_init()`.
  3. Call `tigris_esp_nn_prepare()` (guarded by `TIGRIS_HAS_ESP_NN`) to reserve
     the ESP-NN scratch.
  4. Fill the model input, then run with `tigris_run_with_workspace_buffer()`
     using `tigris_dispatch_kernel_esp_nn` (or `tigris_dispatch_kernel_s8` for
     the portable INT8 path).

Enable PSRAM in `sdkconfig.defaults.esp32s3` (`CONFIG_SPIRAM=y`, plus the octal
or quad mode for your board) and size the app partition to hold the embedded
plan.

### Step 4 (alternative): load the plan from a flash partition

To swap models without rebuilding the firmware, store the `.tgrs` in a dedicated
data partition instead of embedding it, and memory-map it at runtime with
`esp_partition_mmap()` before `tigris_plan_load()`. Recompiling and re-flashing
just the partition then deploys a new model with no rebuild.

## Check the output against the host

The runtime produces the same INT8 output on the device as the reference kernels
do on the host, within the INT8 requant tolerance. Run the same plan and input
through the host (the `reference` backend, or ONNX Runtime on the original float
model) and compare. A large divergence usually means the on-device input differs
from the one you compared against.

## Troubleshooting

- **`create-project-from-example` cannot find the example**: update the component
  manager (`pip install -U idf-component-manager`) and confirm the name is
  `raws-labs/tigris-runtime:getting-started`.
- **Allocation fails or the model will not fit**: confirm PSRAM is enabled and
  that you passed the second `-m` PSRAM tier at compile time.
- **ESP-NN not active**: a project without the `espressif/esp-nn` component falls
  back to the portable kernels. The output is identical but slower.

## Next steps

- [`compile`](/toolchain/compile/): the full CLI reference.
- The [Cortex-M deployment guide](/tutorials/cortex-m-deployment/): the same idea
  on Arm Cortex-M, with the plan embedded in the firmware.
- [Operator and Backend Support](/runtime/operator-support/): what the ESP-NN
  backend accelerates.
