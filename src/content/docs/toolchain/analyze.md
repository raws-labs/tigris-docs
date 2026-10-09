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
| `-v`, `--verbose` | flag | no | Show per-stage breakdown, tiling analysis, and budget sweep tables |
| `--json` | flag | no | Emit the analysis as versioned JSON instead of panels |
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

When run with a memory budget (`-m`), the analysis produces three panels:

### Model Panel

Summary of the loaded model:
- Operator and tensor counts
- Peak activation memory (the minimum SRAM needed if the entire model ran in a single stage)
- Largest tensor shape and size
- Dtype (int8 or float32)
- For a single-subgraph TFLite model, the tensor arena TFLite Micro's memory planner
  places for the same model. It covers tensors only; kernel scratch buffers and TFLite
  Micro's persistent allocations come on top.

### SRAM Panel

Memory feasibility analysis against the fast-memory budget:
- Budget and stage count
- Spill/reload I/O between stages (bytes transferred between fast and slow memory)
- Tiling breakdown: how many stages need spatial tiling, how many are tileable vs untileable
- Verdict: **PASS** (fits, or resolved by tiling) or **FAIL** (untileable stages remain or slow memory overflow)

Verdicts:
- **ok**: all stages fit within the SRAM budget
- **partitioned**: temporal partitioning splits the graph into stages, no tiling needed
- **tiled**: spatial tiling resolves all oversized stages
- **needs_work**: some stages cannot be tiled and exceed the budget

### Flash Panel

Plan size and flash fit check:
- Weight data size
- Plan overhead (headers, section directory, tensor/op descriptors)
- Plan size, as the compiled `.tgrs` will be
- LZ4-compressed plan size estimate (shown if compression saves >5%)
- INT8 plan size estimate (shown for float32 models)
- Flash fit verdict when `-f` is provided

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

Verbose output with per-stage and tiling tables:

```bash
tigris analyze ds_cnn.onnx -m 64K -v
```

With `-v`, additional tables are shown:
- **Budget Comparison**: how stage counts and tiling requirements change across a range of SRAM budgets
- **Stages**: per-stage op count, peak memory, input/output tensor counts, and warnings
- **Tiling Analysis**: per-stage tile height, tile count, halo, receptive field, and tiled peak memory

## JSON

`--json` prints one object with `report: "tigris-analysis"` and a `version`, then the
sections `model`, `tflite_micro`, `fast`, `slow`, `flash` and `stages`. Sizes are in
bytes. Interface shapes are in the caller's axis order. A field that does not apply is
`null`; for example `tflite_micro` is `null` for an ONNX model.

```bash
tigris analyze model.tflite -m 64K --json > analysis.json
```
