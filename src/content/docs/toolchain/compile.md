---
title: "tigris compile"
description: "Use tigris compile to turn an ONNX model into a binary .tgrs execution plan for deployment on embedded devices."
sidebar:
  order: 220
---

Compile an ONNX model into a binary `.tgrs` execution plan for deployment on embedded devices.

## Usage

```bash
tigris compile MODEL [OPTIONS]
```

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | ONNX model file (.onnx) |
| `-m`, `--mem` | size (multiple) | yes | Memory pools, fast to slow (e.g. `-m 256K` or `-m 256K -m 8M`) |
| `-o`, `--output` | path | no | Output .tgrs path (default: `MODEL.tgrs`) |
| `-f`, `--flash` | size | no | Flash budget. Warns if plan exceeds this size. |
| `-c`, `--compress` | `none` / `lz4` | no | Weight compression (default: `none`) |
| `--xip` | flag | no | Execute-in-place: weights read directly from flash at runtime |

## Compilation Pipeline

The compiler runs a 7-stage pipeline:

1. **Load.** Import the ONNX model, fold constants, canonicalize the graph.
2. **Normalize.** Fold QDQ patterns, extract quantization parameters, lower ops to TiGrIS op types.
3. **Lifetimes.** Compute tensor lifetimes from the execution order.
4. **Memory Timeline.** Build the activation memory timeline, compute peak memory.
5. **Temporal Partition.** Split the graph into stages that each fit within the SRAM budget, inserting spill/reload ops at stage boundaries.
6. **Spatial Partition.** For stages that still exceed the budget, compute spatial tiling plans (tile height, halo, receptive field) and detect chain-tileable stage sequences.
7. **Binary Emit.** Serialize to the `.tgrs` binary format.

The output is a single `.tgrs` file designed for zero-copy, zero-alloc loading on the target device.

## XIP (Execute In Place)

When `--xip` is enabled on an uncompressed plan, the runtime reads weights directly from flash via memory-mapped I/O instead of copying the full weight blob to RAM. The plan binary format is designed for memory-mapped access, so the C loader returns pointers directly into the mapped buffer. If weight compression is also enabled, weights are still stored compactly in flash but each stage's compressed block is decompressed into the fast arena before execution.

## Weight Compression

LZ4 compression reduces plan size on flash at the cost of a small SRAM overhead for decompression at runtime.

```bash
tigris compile model.onnx -m 256K -c lz4 -o model.tgrs
```

When compression is enabled:
- Weights are compressed per stage into individual blocks
- At runtime, the executor decompresses one stage's weights at a time into a reserved prefix of the SRAM arena
- Size the complete core fast arena with `tigris_fast_arena_required(&plan)`; it includes the simultaneous decompression reservation without duplicating the runtime's arithmetic
- The plan reports both compressed and uncompressed sizes

## Examples

Compile with a 256K SRAM budget:

```bash
tigris compile ds_cnn.onnx -m 256K -o ds_cnn.tgrs
```

Output:
```
Binary plan written to ds_cnn.tgrs
  28 ops, 3 stages @ 256.00 KiB budget
  plan size: 87.42 KiB
```

Compile with LZ4 compression and flash budget check:

```bash
tigris compile mobilenetv2.onnx -m 256K -c lz4 -f 4M -o mobilenetv2.tgrs
```

Output:
```
Binary plan written to mobilenetv2.tgrs (LZ4 compressed)
  53 ops, 12 stages @ 256.00 KiB budget
  plan size: 2.85 MiB (uncompressed: 3.41 MiB, ratio: 0.84x)
  flash 4.00 MiB: fits
```

Two-pool memory (SRAM + PSRAM):

```bash
tigris compile yolov5n.onnx -m 232K -m 6M -o yolov5n.tgrs
```
