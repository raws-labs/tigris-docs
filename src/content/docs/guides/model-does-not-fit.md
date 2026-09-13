---
title: "When a Model Does Not Fit"
description: "Read the analyze report, find the stage that sets the floor on SRAM, and work through the levers that actually move it."
sidebar:
  order: 150
---

`tigris analyze` ends in PASS or names the stage that is too big. This page is
about the second case: what the report is telling you, and which levers move
the number.

Every figure below is from MobileNetV2 as published in the ONNX model zoo, at
its 224x224 input, measured with the CLI.

## Start from the report

```bash
tigris analyze mobilenetv2.onnx -m 256K
```

```
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
╭─────────────────────────── TiGrIS - mobilenetv2 ───────────────────────────╮
│ Operators            65                                                    │
│ Tensors              244 (66 activations)                                  │
│ Peak memory (naive)  5.74 MiB                                              │
│ Largest tensor       1x96x112x112 (4.59 MiB)                               │
│ Dtype                float32                                               │
│ Input                input 1x3x224x224 float32                             │
│ Output               output 1x1000 float32                                 │
╰────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────── SRAM ───────────────────────────────────╮
│ Budget              256.00 KiB                                             │
│ Scheduled peak      252.00 KiB (4.3% of naive peak)                        │
│ Stages              58                                                     │
│ Spill / reload I/O  26.21 MiB / 27.55 MiB                                  │
│                                                                            │
│ Need tiling         47 of 58 stages                                        │
│   tileable          11 (138 tiles, max halo 2)                             │
╰───────────────────  PASS - tiling resolves all stages  ────────────────────╯
```

**Naive peak** is the high-water mark of simultaneously live activations with
no partitioning, so it is the arena a single-stage execution would need. It is
a property of the model.

**Scheduled peak** is what the plan needs after temporal partitioning and
tiling, and it is a property of the budget you gave it. The solver takes the
largest tile that fits, because fewer tiles mean less halo recompute and fewer
stage boundaries to spill across, so raising the budget raises the peak and
lowers the traffic:

| `-m` | Scheduled peak | Stages | Tiles | Spill |
|---|---|---|---|---|
| 250K | 250.00 KiB | 58 | 142 | 26.21 MiB |
| 256K | 252.00 KiB | 58 | 138 | 26.21 MiB |
| 512K | 511.00 KiB | 43 | 6 | 24.59 MiB |
| 1M | 1001.00 KiB | 17 | 4 | 18.52 MiB |

So the scheduled peak is not the number that tells you whether a smaller part
would work. That number appears only when the budget is too small.

## The three ways a compile fails

### A stage does not fit the fast arena

Same model, same command, at half the budget:

```
╭─────────────────────────────────── SRAM ───────────────────────────────────╮
│ Budget                   128.00 KiB                                        │
│ Scheduled peak           250.00 KiB (4.3% of naive peak)                   │
│ Stages                   64                                                │
│ Spill / reload I/O       26.31 MiB / 27.70 MiB                             │
│                                                                            │
│ Need tiling              61 of 64 stages                                   │
│   tileable               41 (620 tiles, max halo 2)                        │
│   untileable             1 - blocked by: GlobalAveragePool                 │
│   min SRAM (hard floor)  250.00 KiB                                        │
│     stage 62             250.00 KiB (GlobalAveragePool)                    │
│                                                                            │
│ Infeasible               stage 62 requires 256,000 bytes but the           │
│                          fast-memory budget is 131,072 bytes (untileable   │
│                          operators: GlobalAveragePool_97                   │
│                          (GlobalAveragePool))                              │
╰──────────────  FAIL - minimum execution unit exceeds budget  ──────────────╯
```

`min SRAM (hard floor)` is the number to attack. It is the smallest fast arena
a single execution unit can be squeezed into, tiling included, and it does not
move with the budget: the solver has already taken the smallest tile it has for
every stage it can tile, and one stage still does not fit. `compile` reports
the same thing as an error:

```
Error: Cannot compile an infeasible memory plan: stage 62 requires 256,000 bytes
but the fast-memory budget is 131,072 bytes (untileable operators:
GlobalAveragePool_97 (GlobalAveragePool))
```

`GlobalAveragePool` reduces over the whole spatial extent, so it has to see
every row before it can produce a single output value. No tile of it yields a
complete result, so its stage holds input and output together: 1x1280x7x7 plus
1x1280x1x1 in float32 is 64,000 values, or 256,000 bytes. That is the floor.

### Spills do not fit the slow pool

```
Error: Cannot compile this plan: 2 stage(s) overflow slow memory
(5.74 MiB needed, 4.00 MiB available)
```

Tensors that cross a stage boundary are written to the slow pool and read back.
A second `-m` sizes that pool: `-m 256K+8M` gives 256 KiB of fast arena and
8 MiB of slow. Without a second `-m` there is no slow pool at all, and only
models whose activations fit one arena compile.

### The plan does not fit flash

```
Error: Cannot compile a plan that exceeds the flash budget: plan is 13.31 MiB
but the flash budget is 4.00 MiB
```

`-f` is a check, not a constraint the compiler can solve. Weights dominate the
plan, so this is a quantization question rather than a scheduling one.

## Levers that move the floor

Two things change the floor, and both change it a lot. Everything below is
measured on the same model.

| Variant | Naive peak | Hard floor |
|---|---|---|
| float32, 224x224 | 5.74 MiB | 250.00 KiB |
| float32, 160x160 | 2.93 MiB | 130.00 KiB |
| float32, 128x128 | 1.88 MiB | 85.00 KiB |
| int8, 224x224 | 1.44 MiB | 62.50 KiB |

A budget above the floor compiles and a budget below it does not, with nothing
in between: 224x224 float32 passes at `-m 250K`, which is the floor exactly, and
fails at `-m 249K`; 160x160 passes at `-m 132K`; 128x128 passes at `-m 88K`;
int8 passes at `-m 64K` and fails at `-m 32K`.

**Quantize to int8.** Every activation becomes a quarter of its size, so the
floor does too, and the plan goes from 13.31 MiB to 3.39 MiB. This
is the lever with the best ratio of effort to result, and it is usually the
only one that does not change what the model computes. TiGrIS reads ONNX QDQ
models directly.

**Lower the input resolution.** The floor tracks the spatial extent of the
tensor feeding the blocking operator, so it falls roughly with area. `analyze`
and `compile` take `--input-shape input:1x3x160x160`, so you can measure the
new floor before deciding whether to retrain or re-export at that resolution.

Both together compound. The int8 model at 128x128 has a floor of 21.25 KiB and
compiles at `-m 24K`, down from 250.00 KiB for the float model at its native
resolution.

## Levers that do not move it

**`--xip`** does not change the plan's size or the floor. It sets a flag that
tells the runtime to read weights in place from flash instead of copying them
into RAM, which matters for the arena the application owns, not for what the
compiler schedules.

**`-c lz4`** rarely pays on a model of this shape. Compressed weights are
decompressed one stage at a time into a reservation carved out of the fast
arena, so the largest stage's weight block has to fit there alongside the
activations. For MobileNetV2 that reservation is 1,284,000 bytes at `-m 64K`
and 3,489,248 bytes at `-m 256K`, both far past any arena the model is aimed
at, and the compile fails:

```
Error: Fast-memory reservation (3,489,248 bytes) exceeds the total budget
(262,144 bytes)
```

On a model whose per-stage weights are small it compiles, but the saving is
thin. DS-CNN goes from 92,096 to 91,054 bytes of plan, about one percent, in
exchange for a 90,432-byte reservation. Reach for it when weights are
compressible and stages are small, not as a general fix.

## What it looks like when it fits

The int8 model at a 64 KiB budget, which is the 250 KiB float floor divided by
four:

```bash
tigris analyze mobilenetv2-int8.onnx -m 64K
```

```
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
╭──────────────────────── TiGrIS - mobilenetv2-int8 ─────────────────────────╮
│ Operators            65                                                    │
│ Tensors              175 (66 activations)                                  │
│ Peak memory (naive)  1.44 MiB                                              │
│ Largest tensor       1x96x112x112 (1.15 MiB)                               │
│ Quantization         INT8 (QDQ)                                            │
│ Input                input 1x3x224x224 float32, stored as int8 at scale    │
│                      0.0186584, zero point -14                             │
│ Output               output 1x1000 float32, stored as int8 at scale        │
│                      0.160799, zero point -52                              │
╰────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────── SRAM ───────────────────────────────────╮
│ Budget              64.00 KiB                                              │
│ Scheduled peak      63.00 KiB (4.3% of naive peak)                         │
│ Stages              58                                                     │
│ Spill / reload I/O  6.55 MiB / 6.89 MiB                                    │
│                                                                            │
│ Need tiling         47 of 58 stages                                        │
│   tileable          11 (138 tiles, max halo 2)                             │
╰───────────────────  PASS - tiling resolves all stages  ────────────────────╯
```

The Input and Output rows are the plan's contract. The model declares float32
at both ends while the plan executes on int8, so the report names the declared
dtype and the encoding it is stored in. The runtime converts across that
boundary, so an application hands over and reads back float32.

Read the spill and reload figures before calling it done. 6.55 MiB written and
6.89 MiB read back, per inference, is traffic the slow pool has to carry, and
on a part where that pool is external PSRAM it is what sets the latency. A
larger fast arena buys fewer stages and less of it.

## When nothing works

If the floor is still above the arena after quantizing and reducing resolution,
the blocking operator has to change. The report names it, and the usual answer
is to make the tensor it reduces over smaller: an extra stride or pool earlier
in the network shrinks everything downstream of it. That is a change to the
model, so it belongs with training rather than with the compiler.

For the report's fields, see [`analyze`](/toolchain/analyze/). For how the
compiler arrives at those numbers, see
[Tiling](/architecture/tiling/).
