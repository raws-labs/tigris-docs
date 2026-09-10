---
title: tigris simulate
description: Run inference on the host for validation.
sidebar:
  order: 240
slug: 0.6.0/toolchain/simulate
---

Print a step-by-step execution trace for an ONNX model on the host, without target hardware. Use this to validate operator ordering, tensor lifetimes, and memory usage before flashing a device.

## Usage

```bash
tigris simulate MODEL [OPTIONS]
```

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | ONNX model file (.onnx) |
| `-m`, `--mem` | size (multiple) | no | Memory pools, fast to slow (e.g. `-m 256K` or `-m 256K -m 8M`) |

Size arguments accept plain bytes (`262144`), kilobytes (`256K`), megabytes (`4M`), or fractional values (`2.5M`). See [tigris analyze](/0.6.0/toolchain/analyze/) for the full size syntax table.

## Output

The trace is organized into panels and tables:

### Header Panel

A summary line showing:

* Total operator count
* Stage count (if the model is partitioned into multiple stages)
* Memory budget (if `-m` is provided)
* Peak activation memory

### Per-Stage Trace

For each stage the following is printed:

* **Stage header** (multi-stage only) with the op index range, peak memory, and tiling status (tile count, tile height, halo, receptive field).
* **Reload inputs** listing tensors loaded from slow memory at stage entry.
* **Op table** with columns: Step, Op name, Op type, Input shape, Output shape, and Live memory at that step.
* **Spill outputs** listing tensors written to slow memory at stage exit.

## When to Use

Run `tigris simulate` after `tigris analyze` reports a passing verdict but before `tigris compile`. It lets you inspect the exact execution order and confirm that live-memory peaks, stage boundaries, and spill/reload points match your expectations. This is especially useful when debugging tiling or multi-stage partitioning behavior.

## Examples

Trace a single-stage model:

```bash
tigris simulate ds_cnn.onnx
```

Trace with a memory budget to see stage partitioning and spill/reload:

```bash
tigris simulate mobilenetv2.onnx -m 256K
```

Two-pool memory trace:

```bash
tigris simulate yolov5n.onnx -m 232K -m 6M
```
