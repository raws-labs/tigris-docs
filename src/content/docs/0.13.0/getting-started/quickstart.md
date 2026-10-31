---
title: "Quickstart"
description: "From a TFLite model to C source for a Cortex-M or ESP32 target, checked on your machine along the way."
sidebar:
  order: 105
slug: 0.13.0/getting-started/quickstart
---

Take a TFLite model, see whether it fits a memory budget, compile it, run it on
your machine and generate the C code for a target. This walkthrough uses the MLPerf
Tiny person-detection model, an int8 MobileNetV1 for 96x96 images, and a target
with 32 KiB of on-chip SRAM for activations next to 1 MiB of slower RAM.

Every step before the last runs on your laptop; no board is needed.

## Install

Python 3.10 or later:

```bash
pip install tigris-ml
```

On Linux with glibc 2.28 or newer, macOS 11 or newer and Windows x86-64, the wheel
bundles the runtime that steps 3 and 5 use. See
[Installation](/0.13.0/getting-started/installation/) for other platforms.

## Get a model

```bash
curl -LO https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
```

Any int8 or float32 `.tflite` or ONNX file works the same way. This one is also in
the compiler's test suite, where its outputs are checked against TFLite Micro
byte for byte.

## Step 1: Inspect

```bash
tigris inspect vww_96_int8.tflite
```

```text
vww_96_int8.tflite   TFLite, model main, schema 3, 31 operators, 325.48 KiB
  input    input_1_int8    1x96x96x3 int8
  output   Identity_int8   1x2 int8

operators
                      count      weights             MACs
  CONV_2D                14   197.68 KiB   92%     6.69 M   89%
  DEPTHWISE_CONV_2D      13    15.74 KiB    7%   798.34 K   11%
  FULLY_CONNECTED         1        520 B   <1%        512   <1%
  RESHAPE                 1          8 B   <1%
  AVERAGE_POOL_2D         1
  SOFTMAX                 1
  total                  31   213.94 KiB           7.49 M

largest activations
  1x48x48x16 int8   36.00 KiB   step 2, CONV_2D
  1x96x96x3 int8    27.00 KiB   model input
  1x48x48x8 int8    18.00 KiB   step 0, CONV_2D

quantization   int8 activations; int8 weights 27 of 28 per channel, symmetric
converts       yes
```

`inspect` reads the file without compiling it: the interface, what each operator
type costs, the largest activations, and whether the model converts. When it does
not, the `converts` line names every reason.

## Step 2: Analyze

```bash
tigris analyze vww_96_int8.tflite -m 32K -m 1M
```

```text
vww_96_int8.tflite   int8, 31 operators
  input    input_1_int8    1x96x96x3 int8
  output   Identity_int8   1x2 int8
fits 32.00 KiB fast memory, 8 stages, 7 tiled

memory
  TFLite Micro arena     72.00 KiB   its planner and reduction scratch; persistent allocations and other kernel scratch excluded
  TFLite Micro export   -24.69 KiB   head 72.00 KiB -> 45.00 KiB, persistent +2.31 KiB, +0.25% MACs, reference kernels on arm-none-eabi
  unscheduled            54.00 KiB
  largest tensor         36.00 KiB   1x48x48x16
  this plan              28.22 KiB   3.78 KiB headroom
  slow memory            45.00 KiB   budget 1.00 MiB
  slow traffic          109.13 KiB   per inference: 18.00 KiB written, 91.12 KiB read
  also fits at           16.00 KiB   12 stages, 11 tiled
  also fits at            8.00 KiB   24 stages, 23 tiled
  also fits at            4.00 KiB   28 stages, 27 tiled

flash
  plan   239.32 KiB   weights 213.93 KiB, overhead 25.39 KiB
```

The first `-m` is fast memory, the second slow memory. The verdict line says the
model fits: the compiler split it into 8 stages and tiled 7 of them, so the plan
needs 28.22 KiB of fast memory where TFLite Micro's planner places 72 KiB of tensors
in one arena. The tensors that pass between stages wait in slow memory, 45 KiB of it,
and moving them costs 109.13 KiB of traffic per inference. The `also fits at` rows
show how far the budget can go down: to 4 KiB, with more stages and more traffic.
The `TFLite Micro export` row is for a model that stays on TFLite Micro:
[`tigris export`](/0.13.0/toolchain/export/) rewrites the file so that TFLite Micro's own
arena needs 24.69 KiB less.

When a model does not fit, the verdict names the stages that are too large and why.
[When a Model Does Not Fit](/0.13.0/guides/model-does-not-fit/) goes through what to change.

## Step 3: Trace one run (optional)

```bash
tigris analyze vww_96_int8.tflite -m 32K -m 1M --trace
```

```text
vww_96_int8.tflite   traced on runtime 0.13.0, host reference backend, zero input
  moved       18.00 KiB written, 91.12 KiB read
  fast peak   28.22 KiB of 28.22 KiB

stage   kind      tiles        read     written   fast peak   slow used
    0   chain 7      12   73.12 KiB   18.00 KiB   28.22 KiB   45.00 KiB
    7   untiled           18.00 KiB         2 B   22.50 KiB   18.03 KiB

272 events; their bytes equal the runtime's own counters. Host alignment is 32 bytes; a target with less uses at most these bytes.
```

`--trace` runs the plan once on the runtime bundled with the package, in a fast
arena of the size the plan states, and reports what it measured: the first seven
stages run as one chain in 12 tiles whose intermediate results never leave fast
memory, and the rest runs untiled. Add `-v` for every allocation, load, spill and
kernel call.

## Step 4: Compile

```bash
tigris compile vww_96_int8.tflite -m 32K -m 1M -o vww.tgrs
```

`compile` prints the same report without the rows that take extra compiles or a
run (`TFLite Micro export`, `slow traffic`, `also fits at`) and ends with:

```text
wrote vww.tgrs   239.32 KiB
```

The `.tgrs` plan holds the schedule, the tiling, the quantization and the weights.
It is data the runtime reads, not code, and the same file runs on every kernel
backend.

## Step 5: Run it on your machine

The input is int8 `1x96x96x3`, 27648 bytes. An all-zero image:

```bash
head -c 27648 /dev/zero > input.bin
tigris run vww.tgrs --input input.bin --output prediction.bin
```

```text
vww.tgrs   runtime 0.13.0, bundled, host reference backend
  fast arena peak   28.22 KiB   measured; plan and workspace excluded
  slow arena peak   45.00 KiB

wrote prediction.bin   1x2 int8
```

This runs the portable reference kernels and shows the plan loads and executes; it
does not test CMSIS-NN or ESP-NN numerics or device latency. See
[`tigris run`](/0.13.0/toolchain/run/) for real inputs and the Python API.

## Step 6: Generate C code

```bash
tigris codegen vww.tgrs --backend cmsis-nn -o vww.c
```

```text
wrote vww.c   app, cmsis-nn backend, 5.43 KiB, kernels vww_kernels.h
```

| Backend | Target |
|---------|--------|
| `reference` | Portable C99, any platform, float32 and int8 |
| `cmsis-nn` | Arm Cortex-M, Arm's optimized int8 kernels |
| `esp-nn` | ESP32 family, Espressif's optimized int8 kernels |

The generated C sets up the runtime and the backend; the plan stays a separate
file that it loads from a file, a flash partition or linker-provided flash symbols,
depending on the target. `--format core` generates glue for an application that
already owns its entry point and memory.

## Step 7: Deploy

On the target:

1. Add the generated source and the [runtime](https://github.com/raws-labs/tigris-runtime)
   to the firmware project.
2. Store the plan where the generated harness expects it.
3. Give it a fast arena of at least the `this plan` figure, which the runtime checks
   before it runs anything, and a slow arena for the tensors between stages; with
   external PSRAM, that is where the slow arena goes. A plan for which `compile`
   prints no `slow memory` row needs no slow arena.
4. Fill the input, run, and check the result before reading the output.

The tutorials take this through to a board:
[Cortex-M](/0.13.0/tutorials/cortex-m-deployment/) and
[ESP-IDF](/0.13.0/tutorials/esp-idf-deployment/). [Runtime Integration](/0.13.0/runtime/integration/)
has the full C sequence.

To start from a model that is already compiled and validated, see
[`tigris zoo`](/0.13.0/toolchain/zoo/).

## What's next

- [CLI Overview](/0.13.0/toolchain/overview/): every command and the flags that add depth
- [Core Concepts](/0.13.0/getting-started/concepts/): stages, tiling, memory pools, execute-in-place
- [Compiling a TFLite Model](/0.13.0/guides/tflite-models/): which operators convert, with which limits
- [Introducing TiGrIS](/blog/introducing-tigris/): measurements on hardware
