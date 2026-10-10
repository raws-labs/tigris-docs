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
| `--trace` | flag | no | Run the plan on the host runtime and print what it did instead of the summary |
| `--input` | path (multiple) | no | Input for `--trace`, in the formats `tigris run` reads; zeros when omitted |
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

`--trace` compiles the plan in memory, runs it once on the host runtime bundled with
the package, and prints what the runtime did. The numbers are measured, not
predicted: the runtime reports every load, spill, allocation and tile as it happens.

```bash
tigris analyze mobilenet_v1_matched.onnx -m 64K -m 8M --trace
```

```text
mobilenet_v1_matched.onnx   traced on runtime 0.11.4, host reference backend, zero input
  moved       208.01 KiB written, 559.38 KiB read
  fast peak   64.00 KiB of 64.00 KiB

stage   kind      tiles         read      written   fast peak    slow used
    0   chain 5      16    87.38 KiB   128.00 KiB   63.62 KiB   176.00 KiB
    5   chain 4      16   308.00 KiB    64.00 KiB   54.00 KiB   192.00 KiB
    9   chain 3       8   148.00 KiB    16.00 KiB   50.00 KiB    80.00 KiB
   12   untiled            16.00 KiB         10 B   64.00 KiB    16.03 KiB

541 events; their bytes equal the runtime's own counters. Host alignment is 32 bytes; a target with less uses at most these bytes.
```

`moved` is the traffic between fast and slow memory for one inference: bytes written
to slow memory and bytes read back. On a target whose slow pool is external PSRAM,
this traffic costs time on every inference. Each row of the table is one stage or one
chain of stages: how it ran, how many tiles, what it read and wrote, the highest fast
memory use and the slow memory in use. When weights are compressed, a `weights` row
gives the bytes decompressed into fast memory.

The last line checks the trace against the runtime's own counters. If the two
disagree, a `mismatch:` line names the difference. The host aligns tensors to 32
bytes; a target with smaller alignment needs at most the memory shown.

With `-v`, each stage lists its events in order: `alloc`, `load` and `spill` with
the tensor, the source and destination offsets (`slow+N`, `fast+N`) and the bytes,
`move` for compaction and line-buffer rolls, `op` for each kernel call, `reset` when
a tile frees its fast memory, and `weights` and `copy` for decompressed weights and
state or control-flow copies. Tiled stages group events under `tile N` with the rows
and columns the tile covers.

The trace runs on zeros unless `--input` gives real data, in the same formats as
[`tigris run`](/toolchain/run/). The schedule is fixed at compile time, so the input
changes the trace only where the plan branches on data (`If`, `While`).

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
