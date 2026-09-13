---
title: "Quickstart"
description: "End-to-end walkthrough: ONNX model to embedded deployment."
sidebar:
  order: 105
---

Take a supported ONNX model, compile it for a bounded SRAM budget,
and generate the target integration code. This walkthrough uses the matched
INT8 MobileNetV1 model from the benchmark repository and a 64 KiB fast-memory
budget. Exact stage counts and tiling decisions are compiler results, so they
may improve between releases. Independently captured measurements remain in
[Introducing TiGrIS](/blog/introducing-tigris/).

## Prerequisites

- Python 3.10+ with `tigris-ml` installed
- A supported float32 or fully quantized INT8 ONNX model. A dimension the
  model leaves free, such as the batch dimension of a stock export, is bound
  to 1; `--input-shape input:4x3x224x224` compiles for a different one

```bash
pip install tigris-ml
```

To reproduce this walkthrough, run `python models/prepare.py` from the root of
[tigris-bench](https://github.com/raws-labs/tigris-bench), then copy
`models/output/mobilenet_v1_matched.onnx` into your working directory. That
model is reconstructed from the benchmark's TFLite model so its weights and
quantization match the hardware comparison. For supported operators and
backend qualifications, see
[Operator and Backend Support](/runtime/operator-support/).

## Step 1: Analyze

Check whether the model fits within a 64 KB SRAM + 8 MB PSRAM budget (typical for an ESP32-S3):

```bash
tigris analyze mobilenet_v1_matched.onnx -m 64K -m 8M -f 16M
```

The SRAM panel reports the current scheduled peak, stages, spill/reload volume,
and tiling decision. Continue only when it ends in a `PASS` verdict. The flash
panel independently confirms whether the estimated plan fits the 16 MiB flash
budget. No hardware is required for analysis.

## Step 2: Compile

Generate a binary execution plan:

```bash
tigris compile mobilenet_v1_matched.onnx -m 64K -m 8M -f 16M --xip -o mobilenet.tgrs
```

| Flag | Meaning |
|------|---------|
| `-m 64K -m 8M` | Memory pools, fast to slow. First is SRAM budget, second is PSRAM. The compiler decides what goes where. |
| `-f 16M` | Flash budget. Warns if the plan doesn't fit. |
| `--xip` | Execute-in-place. Weights are read from flash at runtime, not copied to SRAM. |
| `-o mobilenet.tgrs` | Output path for the binary plan. |

PSRAM is required for multi-stage models. Without it, only single-stage models (where the full model fits in one SRAM arena) are supported.

## Step 3: Generate C code

Use `codegen` to produce a backend-specific C harness:

```bash
tigris codegen mobilenet.tgrs --backend esp-nn -o mobilenet.c
```

| Backend | Target |
|---------|--------|
| `reference` | Portable C99 (any platform) |
| `esp-nn` | Espressif optimized kernels (ESP32 family) |
| `cmsis-nn` | Arm optimized kernels (Cortex-M family) |

Same plan, different kernels. `codegen` writes target-specific runtime glue;
the `.tgrs` plan remains a separate deployment artifact loaded from a file,
flash partition, or linker-provided flash symbols depending on the backend.
See [Runtime Integration](/runtime/integration/) for
manual loading details.

## Step 4: Simulate (optional)

Inspect the execution trace before deploying:

```bash
tigris simulate mobilenet_v1_matched.onnx -m 64K -m 8M
```

This prints the current per-stage operator order, tensor shapes, live-memory
estimate, tile geometry, and spill/reload actions. It does not run inference.

## Step 5: Deploy

The `.tgrs` plan contains the operator schedule, memory map, tiling parameters, and weights. On your target:

1. Add the generated source to the firmware project.
2. Store the plan at the location expected by the selected generated harness.
3. Supply the generated integration with the required fast and slow arenas.
4. Initialize the selected backend, fill every model input, and check the
   generated run function's result before reading model outputs.

The generated source is the maintained starting point: compiler CI syntax-checks
a fresh portable C99 harness against the matching runtime headers. Applications
that need custom ownership, RTOS task-local workspaces, or their own entry point
should use `codegen --format core`. The complete checked manual sequence,
including plan-sized executor workspace and backend preparation, is in
[Runtime Integration](/runtime/integration/).

## What's next

- [Core Concepts](/getting-started/concepts/): tiling strategies, memory pools, execute-in-place
- [CLI Reference](/toolchain/compile/): supported commands and flags
- [Runtime Integration](/runtime/integration/): full C API and firmware integration
