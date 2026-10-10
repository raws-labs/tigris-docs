---
title: "tigris analyze"
description: "Use tigris analyze to check whether an ONNX or TFLite model fits your target's memory constraints before you compile it."
sidebar:
  order: 210
---

Check if an ONNX or TFLite model fits your target's memory constraints before compiling.

## Usage

```bash
tigris analyze MODEL [OPTIONS]
```

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | ONNX (`.onnx`) or TFLite (`.tflite`) model file |
| `-m`, `--mem` | size (multiple) | no | Memory pools, fast to slow (e.g. `-m 256K` or `-m 256K -m 8M`) |
| `-f`, `--flash` | size | no | Flash budget for plan fit check (e.g. `4M`) |
| `-v`, `--verbose` | flag | no | Add the per-stage table |
| `--json` | flag | no | Emit the analysis as versioned JSON |
| `--trace` | flag | no | Print the step-by-step execution trace instead of the summary |
| `--input-shape` | `NAME:1x3x224x224` (multiple) | no | Shape to compile an input for. Overrides what the model declares; a dimension the model leaves free is otherwise bound to 1 |

## Size Syntax

All size arguments accept the following formats (case-insensitive):

| Format | Example | Bytes |
|--------|---------|-------|
| Plain bytes | `262144` | 262144 |
| Kilobytes | `256K` or `256KB` | 262144 |
| Megabytes | `4M` or `4MB` | 4194304 |
| Fractional | `2.5M` | 2621440 |

## Output

```bash
tigris analyze mobilenet_v1_matched.onnx -m 64K -m 8M -f 16M
```

```text
mobilenet_v1_matched.onnx   int8, 31 operators
  input    input    1x3x128x128 float32, stored as int8 at scale 0.0392704, zero point -1
  output   l33_dq   1x10 float32, stored as int8 at scale 0.00390625, zero point -128
fits 64.00 KiB fast memory, 13 stages, 12 tiled

memory
  unscheduled       384.00 KiB
  largest tensor    256.00 KiB   1x64x64x64
  this plan          64.00 KiB   0 B headroom
  slow memory       192.00 KiB   budget 8.00 MiB
  also fits at       32.00 KiB   25 stages, 24 tiled
  also fits at       16.00 KiB   29 stages, 28 tiled
  does not fit at     8.00 KiB

flash
  plan            3.35 MiB   weights 3.09 MiB, overhead 269.93 KiB
  flash budget   16.00 MiB   fits
```

The first block names the model, its inputs and outputs in the caller's axis order,
and the verdict. The verdict is one of:

- `fits <budget> fast memory, N stages, K tiled`: a plan can be compiled. A stage
  counts as tiled when it runs in tiles or as part of a chain.
- `does not fit <budget> fast memory: N stages exceed it`, followed by one line per
  stage with the fast memory it needs and why it cannot shrink, largest first.
- `cannot compile: N unsupported operators`, followed by the operators.
- `fits <budget> fast memory; slow memory X needed, Y given`.
- `no budget given; memory figures only`, when `-m` is absent.

`memory` rows:

- `TFLite Micro tensors`, for a single-subgraph TFLite model: the tensor arena TFLite
  Micro's memory planner places for the same model. Kernel scratch and TFLite Micro's
  persistent allocations come on top.
- `unscheduled`: the activation peak if the whole model ran as one stage.
- `largest tensor`: the largest activation and its shape.
- `this plan`: the fast memory the plan uses, and the headroom left in the budget.
- `slow memory`: what the tensors that cross stage boundaries need in the slow pool.
- `also fits at`, `fits at`, `does not fit at`: up to three more budgets, halving from
  a budget that fits or doubling from one that does not. Each is a full compile run.

`flash` rows: the plan size as `compile` writes it, split into weights and overhead;
`with -c lz4` and `as int8` estimates where they apply; and the flash verdict when `-f`
is given.

With `-v`, a `stages` table follows, one row per stage: its operators, its peak
before tiling, its input and output tensor counts, and how it tiles.

## Execution trace

`--trace` prints the schedule step by step instead of the summary: per stage, the
tensors it reloads from slow memory, each operator with its input and output shapes
and the live fast memory after it, and the tensors it spills. It does not run
inference.

```bash
tigris analyze mobilenet_v1_matched.onnx -m 64K -m 8M --trace
```

## Examples

Basic feasibility check:

```bash
tigris analyze ds_cnn.onnx -m 256K
```

Check against both SRAM and flash budgets:

```bash
tigris analyze mobilenetv2.onnx -m 256K -f 4M
```

Two-pool memory (SRAM + PSRAM):

```bash
tigris analyze yolov5n.onnx -m 232K -m 6M -f 4M
```

Verbose output with the per-stage table:

```bash
tigris analyze ds_cnn.onnx -m 64K -v
```

## JSON

`--json` prints one object with `report: "tigris-analysis"` and a `version`, then the
sections `model`, `tflite_micro`, `fast`, `slow`, `flash`, `budgets_tried` and `stages`. Sizes are in
bytes. Interface shapes are in the caller's axis order. A field that does not apply is
`null`; for example `tflite_micro` is `null` for an ONNX model.

```bash
tigris analyze model.tflite -m 64K --json > analysis.json
```
