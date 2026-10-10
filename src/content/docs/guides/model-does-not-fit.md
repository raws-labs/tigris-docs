---
title: "When a Model Does Not Fit"
description: "Read the analyze report, find the stage that sets the floor on SRAM, and work through the levers that actually move it."
sidebar:
  order: 150
---

`tigris analyze` either says the model fits or names the stages that are too big.
This page is about the second case: what the report is telling you, and which levers
move the number.

Every figure below is from MobileNetV2 as published in the ONNX model zoo, at its
224x224 input, measured with the CLI.

## Start from the report

```bash
tigris analyze mobilenetv2.onnx -m 256K
```

```text
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
mobilenetv2.onnx   float32, 65 operators
  input    input    1x3x224x224 float32
  output   output   1x1000 float32
fits 256.00 KiB fast memory, 58 stages, 53 tiled

memory
  unscheduled         5.74 MiB
  largest tensor      4.59 MiB   1x96x112x112
  this plan         252.00 KiB   4.00 KiB headroom
  slow memory         5.17 MiB
  slow traffic       20.58 MiB   per inference: 7.33 MiB written, 13.25 MiB read
  also fits at      128.00 KiB   64 stages, 62 tiled
  also fits at       64.00 KiB   64 stages, 63 tiled
  does not fit at    32.00 KiB

flash
  plan      13.31 MiB   weights 13.30 MiB, overhead 9.41 KiB
  as int8    3.34 MiB   estimate
```

**Unscheduled** is the high-water mark of simultaneously live activations with no
partitioning, so it is the arena a single-stage execution would need. It is a property
of the model.

**This plan** is what the plan uses after temporal partitioning and tiling, and it is a
property of the budget you gave it. The solver takes the largest tile that fits,
because fewer tiles mean less halo recompute and fewer stage boundaries, so raising the
budget raises the figure and lowers the stage count:

| `-m` | This plan | Stages | Tiled |
|---|---|---|---|
| 64K | 64.00 KiB | 64 | 63 |
| 256K | 252.00 KiB | 58 | 53 |
| 512K | 511.00 KiB | 43 | 38 |
| 1M | 1008.00 KiB | 17 | 15 |

So this figure does not tell you whether a smaller part would work. The `also fits at`
rows do, and the floor appears once the budget is too small.

## The three ways a compile fails

### A stage does not fit the fast arena

Same model at 32 KiB:

```text
does not fit 32.00 KiB fast memory: 4 stages exceed it
  stage 62   45.00 KiB   smallest tile
  stage 51   37.50 KiB   smallest 2D tile
  stage 55   37.50 KiB   smallest 2D tile
  stage 59   37.50 KiB   smallest 2D tile
```

Each line is a stage the solver has already shrunk to its smallest tile, with the fast
memory that tile still needs. The first line is the floor: no budget below 45.00 KiB
compiles, and the `fits at` row below the verdict names the next budget that does.
`compile` prints the same summary and ends with `Error: no plan written`.

Two kinds of stage set this floor. Stage 62 is the `GlobalAveragePool`, which tiles by
rows of its 1x1280x7x7 input; one row and the accumulated output need 45.00 KiB in
float32. Stages 51, 55 and 59 are 960-channel depthwise convolutions, which tile in
both height and width; their smallest tile reads a 3x3x960 window and writes 960
outputs, 37.50 KiB in float32.

### Spills do not fit the slow pool

```text
fits 256.00 KiB fast memory, 58 stages, 53 tiled; slow memory 5.17 MiB needed, 4.00 MiB given
```

Tensors that cross a stage boundary are written to the slow pool and read back. A
second `-m` sizes that pool: `-m 256K+8M` gives 256 KiB of fast arena and 8 MiB of
slow. Without a second `-m`, `slow memory` reports what the plan needs and the compile
does not check it; the application has to provide that much.

### The plan does not fit flash

```text
Error: no plan written: the plan is 13.31 MiB, the flash budget is 4.00 MiB
```

`-f` is a check, not a constraint the compiler can solve. Weights dominate the plan,
so this is a quantization question rather than a scheduling one.

## Levers that move the floor

Quantization moves the floor a lot; resolution moves it less than you might expect.
Everything below is measured on the same model.

| Variant | Unscheduled | Floor |
|---|---|---|
| float32, 224x224 | 5.74 MiB | 45.00 KiB |
| float32, 160x160 | 2.93 MiB | 37.50 KiB |
| float32, 128x128 | 1.88 MiB | 37.50 KiB |
| int8, 224x224 | 1.44 MiB | 15.00 KiB |
| int8, 128x128 | 480.00 KiB | 11.25 KiB |

Each variant compiles at its floor and fails 1 KiB below it.

**Quantize to int8.** Every activation becomes a quarter of its size, so the floor
falls by about that much, and the plan goes from 13.31 MiB to 3.39 MiB. This is the
lever with the best ratio of effort to result, and it is usually the only one that
does not change what the model computes. TiGrIS reads ONNX QDQ models directly.

**Lower the input resolution.** This shrinks every activation and the unscheduled
peak with it, but the floor is set by the smallest tile of the widest layers, and a
2D tile does not get smaller when the image does. At 160x160 the pooling stage drops
below the depthwise stages, which then set the floor at 37.50 KiB, and 128x128 does
not lower it further. `analyze` and `compile` take `--input-shape input:1x3x160x160`,
so you can measure before deciding whether to retrain or re-export.

Both together give 11.25 KiB for the int8 model at 128x128, down from 45.00 KiB for
the float model at its native resolution.

## Levers that do not move it

**`--xip`** does not change the plan's size or the floor. It sets a flag that
tells the runtime to read weights in place from flash instead of copying them
into RAM, which matters for the arena the application owns, not for what the
compiler schedules.

**`-c lz4`** rarely pays on a model of this shape. Compressed weights are
decompressed one stage at a time into a reservation carved out of the fast
arena, so the largest stage's weight block has to fit there alongside the
activations. For MobileNetV2 that reservation is 4.89 MiB, far past any arena the
model is aimed at, and the compile fails:

```text
Error: no plan written: decompressed weights need 4.89 MiB of fast memory, the budget is 256.00 KiB
```

On a model whose per-stage weights are small it compiles, but the saving is
thin. DS-CNN at `-m 128K` goes from 89.94 KiB to 88.73 KiB of plan, about one
percent, in exchange for 66.50 KiB of the fast memory. Reach for it when weights are
compressible and stages are small, not as a general fix.

## What it looks like when it fits

The int8 model at a 64 KiB budget:

```bash
tigris analyze mobilenetv2-int8.onnx -m 64K
```

```text
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
mobilenetv2-int8.onnx   int8, 65 operators
  input    input    1x3x224x224 float32, stored as int8 at scale 0.0186584, zero point -14
  output   output   1x1000 float32, stored as int8 at scale 0.160799, zero point -52
fits 64.00 KiB fast memory, 58 stages, 53 tiled

memory
  unscheduled        1.44 MiB
  largest tensor     1.15 MiB   1x96x112x112
  this plan         63.00 KiB   1.00 KiB headroom
  slow memory        1.29 MiB
  slow traffic       5.14 MiB   per inference: 1.83 MiB written, 3.31 MiB read
  also fits at      32.00 KiB   64 stages, 62 tiled
  also fits at      16.00 KiB   64 stages, 63 tiled
  does not fit at    8.00 KiB

flash
  plan   3.39 MiB   weights 3.38 MiB, overhead 15.85 KiB
```

The input and output rows are the plan's contract. The model declares float32 at both
ends while the plan executes on int8, so the report names the declared dtype and the
encoding it is stored in. The runtime converts across that boundary, so an application
hands over and reads back float32.

Read the trace before calling it done. `tigris analyze mobilenetv2-int8.onnx -m 64K
--trace` runs the plan on the host runtime and reports, per stage, the bytes it reads
from the slow pool and writes back. All of that is traffic per inference, and on a part where the slow pool is external
PSRAM it is what sets the latency. A larger fast arena buys fewer stages and less of it.

## When nothing works

If the floor is still above the arena after quantizing and reducing resolution, the
stages the report names have to change. A 2D tile's floor follows the channel count of
its layer, so fewer channels in the widest layers lower it; a row tile's floor follows
the width of one input row. That is a change to the model, so it belongs with training
rather than with the compiler.

For the report's fields, see [`analyze`](/toolchain/analyze/). For how the
compiler arrives at those numbers, see
[Tiling](/architecture/tiling/).
