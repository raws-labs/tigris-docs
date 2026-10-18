---
title: "tigris compile"
description: "Use tigris compile to turn an ONNX or TFLite model into a binary .tgrs execution plan for a memory budget, with optional flash check, weight compression and execute-in-place."
sidebar:
  order: 220
---

Compile an ONNX or TFLite model into a `.tgrs` execution plan for a memory budget.
`compile` prints the same report as [`tigris analyze`](/toolchain/analyze/), then
writes the plan only if the model fits every budget given.

## Usage

```bash
tigris compile MODEL -m FAST [-m SLOW] [-f FLASH] [-c lz4] [--xip] [-o PLAN]
```

## Options

| Flag | Type | Description |
|------|------|-------------|
| `MODEL` | path | ONNX or TFLite model |
| `-m`, `--mem` | size, repeatable, required | Fast memory, then slow memory: `-m 32K -m 1M` or `-m 32K+1M` |
| `-o`, `--output` | path | Plan to write; defaults to `MODEL.tgrs` |
| `-f`, `--flash` | size | Flash for the plan; no plan is written when it is larger |
| `-c`, `--compress` | `none`, `lz4` | Compress weights per stage (default `none`) |
| `--xip` | flag | Read weights in place from flash at run time instead of copying them to RAM |
| `--input-shape` | `NAME:1x3x224x224`, repeatable | Shape for an input the model leaves open; an open dimension is otherwise bound to 1 |
| `--keep-names` | flag | Keep every operator and tensor name in the plan for `inspect`; by default the plan names only the model, its inputs and its outputs |

Sizes and the two memory pools are explained in the
[CLI overview](/toolchain/overview/#memory-budgets).

## Output

```bash
tigris compile vww_96_int8.tflite -m 32K -m 1M -o vww.tgrs
```

```text
vww_96_int8.tflite   int8, 31 operators
  input    input_1_int8    1x96x96x3 int8
  output   Identity_int8   1x2 int8
fits 32.00 KiB fast memory, 8 stages, 7 tiled

memory
  TFLite Micro tensors   72.00 KiB   its planner; kernel scratch excluded
  unscheduled            54.00 KiB
  largest tensor         36.00 KiB   1x48x48x16
  this plan              28.22 KiB   3.78 KiB headroom
  slow memory            45.00 KiB   budget 1.00 MiB

flash
  plan   239.31 KiB   weights 213.93 KiB, overhead 25.38 KiB

wrote vww.tgrs   239.31 KiB
```

The report is `analyze`'s without the rows that need extra work: no other budgets
are tried, the slow traffic is not measured and LZ4 compression is not tried. Use
`analyze` for those, and [`inspect`](/toolchain/inspect/) to read the plan back.

## When no plan is written

`compile` exits with status 1 and leaves no file behind when a budget is not met.
The verdict line says which one:

```text
does not fit 1.00 KiB fast memory: 8 stages exceed it
```

```text
fits 32.00 KiB fast memory, 8 stages, 7 tiled; slow memory 45.00 KiB needed, 16.00 KiB given
```

Each ends with `Error: no plan written`. Two more checks come after the schedule:

```text
Error: no plan written: the plan is 3.36 MiB, the flash budget is 1.00 MiB
Error: no plan written: decompressed weights need 88.31 KiB of fast memory, the budget is 64.00 KiB
```

[When a Model Does Not Fit](/guides/model-does-not-fit/) covers what to change.

## What a plan holds

A `.tgrs` file holds the operator schedule, the stages and how each one tiles, the
tensor shapes and quantization, the memory the plan needs, and the weights. It
names the model and its inputs and outputs, which is all the runtime and generated
code read; `--keep-names` also keeps the operator and intermediate tensor names, at
about 14 KiB for the MLPerf Tiny VWW model, so that
[`inspect -v`](/toolchain/inspect/) can show them. It is data, not code: the runtime reads it in place and executes it in the fast and
slow buffers the application supplies, without a general-purpose allocator. The
same plan runs on every kernel backend; [`codegen`](/toolchain/codegen/) picks one.
[Compilation Pipeline](/architecture/pipeline/) describes how the compiler gets
there.

## Execute in place

`--xip` marks an execute-in-place deployment. The runtime reads uncompressed
weights directly from the loaded plan buffer; the application must place that
buffer in memory-mapped flash to keep the weights out of RAM. The flag does not
map flash or select a weight-copying path. Flash reads are slower than SRAM reads; how much that costs depends
on the device's flash cache and the model's access pattern. `--xip` does not
change the schedule or the plan's size.

## Weight compression

`-c lz4` compresses each stage's weights into its own block. At run time the
executor decompresses one stage's block at a time into a reserved part of the
fast arena, so the largest block has to fit there next to the stage's
activations. The last line says how much of the fast memory that takes:

```bash
tigris compile ds_cnn.onnx -m 128K -c lz4 -f 4M -o ds_cnn.tgrs
```

```text
wrote ds_cnn.tgrs   88.44 KiB, LZ4, 66.50 KiB of the 128.00 KiB fast memory holds decompressed weights
```

Size the application's fast arena with `tigris_fast_arena_required(&plan)`,
which includes that reservation. Compression pays when weights compress well and
each stage's weights are small; on a model with large stages the reservation
outgrows the budget and no plan is written.

## Examples

On-chip SRAM next to external PSRAM, weights read in place from a 4 MiB flash:

```bash
tigris compile model.tflite -m 232K -m 6M -f 4M --xip -o model.tgrs
```

A single fast pool, for a model that fits in one stage:

```bash
tigris compile ds_cnn.onnx -m 256K -o ds_cnn.tgrs
```

A stock ONNX export with a free batch dimension, at batch 4:

```bash
tigris compile model.onnx -m 256K --input-shape input:4x3x224x224
```
