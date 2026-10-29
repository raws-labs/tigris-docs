---
title: "tigris analyze"
description: "Use tigris analyze to check whether an ONNX or TFLite model fits a memory budget, see what stops it when it does not, and trace one run of the plan on the host runtime."
sidebar:
  order: 210
---

Check whether a model fits a memory budget before you compile it, and see what
stops it when it does not. With `--trace`, run the plan once on the host runtime
and see what it did.

## Usage

```bash
tigris analyze MODEL -m FAST [-m SLOW] [-f FLASH] [-v | --trace | --json]
```

## Options

| Flag | Type | Description |
|------|------|-------------|
| `MODEL` | path | ONNX or TFLite model |
| `-m`, `--mem` | size, repeatable | Fast memory, then slow memory: `-m 32K -m 1M` or `-m 32K+1M`. Without `-m`, only the model's own figures are reported. |
| `-f`, `--flash` | size | Flash for the plan; adds a flash verdict |
| `-v`, `--verbose` | flag | Adds the stages table, or every event of a trace |
| `--json` | flag | The analysis as a versioned JSON document |
| `--trace` | flag | Runs the plan on the host runtime and reports what it did instead |
| `--input` | path, repeatable | Input for `--trace`: `.bin` or `.npy`, `NAME=FILE` for several; zeros when omitted |
| `--input-shape` | `NAME:1x3x224x224`, repeatable | Shape for an input the model leaves open; an open dimension is otherwise bound to 1 |

Sizes and the two memory pools are explained in the
[CLI overview](/toolchain/overview/#memory-budgets).

## The report

```bash
tigris analyze vww_96_int8.tflite -m 32K -m 1M -f 1M
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
  plan           239.32 KiB   weights 213.93 KiB, overhead 25.39 KiB
  flash budget     1.00 MiB   fits
```

### Verdict

The first block names the model, its inputs and outputs in the axis order the
caller hands over, and the verdict:

| Verdict | Meaning |
|---|---|
| `fits <budget> fast memory, N stages, K tiled` | A plan can be compiled. A stage counts as tiled when it runs in tiles or as part of a chain. |
| `fits <budget> fast memory; slow memory X needed, Y given` | The fast memory is enough, the second `-m` is not. |
| `does not fit <budget> fast memory: N stages exceed it` | One line per stage follows, largest first, with the fast memory it still needs and why it cannot shrink. |
| `cannot compile: N unsupported operators` | The operators and reasons follow; `--json` lists all of them. |
| `cannot compile: REASON` | An output no operator computes, or an activation dtype the runtime does not run. |
| `no budget given; memory figures only` | No `-m` was given. |

A stage that does not fit names its reason:

| Reason | What sets the floor |
|---|---|
| `smallest tile` | The stage is already tiled as finely as its operators allow. |
| `smallest 2D tile` | The same, for a stage tiled in height and width. |
| `smallest tile of a chain of N stages` | A chain of stages whose tiles flow through fast memory together. |
| `runs untiled: NAME, ...` | The named operators cannot tile, so the stage needs its whole input and output at once. |

[When a Model Does Not Fit](/guides/model-does-not-fit/) works through what to
change in each case.

### Memory

| Row | Meaning |
|---|---|
| `TFLite Micro arena` | For a single-graph TFLite model that [`tigris export`](/toolchain/export/) can read, with its buffers inside the FlatBuffer, and whose tensor types TFLite Micro's arena model knows: the arena head TFLite Micro's own memory planner places for the same file, its tensors plus the scratch buffers of `MEAN`, `SUM`, `REDUCE_MAX`, `REDUCE_MIN` and `REDUCE_ALL`. Other kernels' scratch and TFLite Micro's persistent allocations come on top. |
| `TFLite Micro export` | How much TFLite Micro's arena goes down when [`tigris export`](/toolchain/export/) rewrites the model: the head before and after, the growth of the persistent allocations, the multiply-accumulates the rewrite adds, and last the kernels and target these figures are for, TFLite Micro's reference kernels on `arm-none-eabi`. Shown with the `TFLite Micro arena` row, as `none` and the reason when export would write nothing, followed by the kernels and target when the reason is that no rewrite lowers the arena. |
| `unscheduled` | The activation peak if the whole model ran as one stage. |
| `largest tensor` | The largest activation and its shape. |
| `this plan` | The fast memory the plan uses, and the headroom left in the budget. |
| `slow memory` | What the tensors that cross stage boundaries need in the slow pool. |
| `slow traffic` | Bytes written to and read back from slow memory in one inference, measured by running the plan once on the host runtime. It reads `not measured` with the reason when there is no host runtime or when a `While` loop's iteration count depends on the input values. It is absent when the plan does not fit the slow budget. |
| `also fits at`, `fits at`, `does not fit at` | Up to three more budgets, halving from a budget that fits or doubling from one that does not. Each is a full compile. |

### Flash

`plan` is the size `compile` writes, split into weights and overhead. `with -c lz4`
estimates the plan with compressed weights; it appears only where `compile -c lz4`
writes a plan at the given budgets, which also have to hold each stage's
decompressed weights, and where it is at least 5% smaller than `plan`. `as int8`
estimates a float32 model's plan after quantization. With `-f`, `flash budget` says whether the plan fits.

### Runtime build

A plan that needs more than a default runtime build holds, for example more tensors
or more outputs from one stage than the runtime's limits allow by default, ends the
report with the `-D` options its runtime needs, as [`inspect`](/toolchain/inspect/)
states them for a compiled plan:

```text
runtime build   -DTIGRIS_MAX_STAGE_OUTPUTS=19
```

A runtime built without them refuses the plan. The row is absent when the defaults
suffice. [Runtime Integration](/runtime/integration/) lists the limits.

## Stages

`-v` adds one row per stage:

```text
stages
  stage    ops   untiled peak   in   out   tiling
      0      0      45.00 KiB    1     1   chain of 7, tile height 2
      1      1      36.00 KiB    1     1   in chain 0
      2      2      54.00 KiB    1     1   in chain 0
      3      3      45.00 KiB    1     1   in chain 0
      4      4      27.00 KiB    1     1   in chain 0
      5      5      36.00 KiB    1     1   in chain 0
      6      6      36.00 KiB    1     1   in chain 0
      7   7-30      22.50 KiB    1     1   untiled
```

`ops` are the operator indices the stage runs, `untiled peak` is what the stage
would need without tiling, and `in` and `out` count the tensors that enter and
leave the stage. `tiling` says how the stage runs: `untiled`, `N tiles, axis A,
halo K` for tiles along one axis, `N tiles of H x W, halo K` for tiles in height
and width, `chain of N, tile height H` for the first stage of a chain, or
`in chain C` for the others.

## Execution trace

`--trace` compiles the plan in memory, runs it once on the host runtime bundled
with the package, and prints what the runtime did. The numbers are measured, not
predicted: the runtime reports every allocation, load, spill and tile as it
happens. It needs `-m` and a plan that fits the fast and the slow budget;
otherwise it stops with `no plan to trace`, and `analyze` without `--trace` says
why.

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

`moved` is the traffic between fast and slow memory for one inference: bytes
written to slow memory and bytes read back. When the slow pool is external PSRAM,
this traffic costs time on every inference. `fast peak` is the highest fast memory
use, out of the arena the host runtime gives the plan: the size
`tigris_fast_arena_required()` returns. Each row of the table is one stage or
one chain of stages: how it ran, how many tiles, what it read and wrote, the
highest fast memory use and the slow memory in use. When weights are compressed,
a `weights` row gives the bytes decompressed into fast memory.

The last line checks the trace against the runtime's own counters. If the two
disagree, a `mismatch:` line names the difference. The host aligns tensors to 32
bytes; a target with smaller alignment needs at most the memory shown.

With `-v`, every event follows, grouped by stage and by tile:

```text
stage 0   chain 7
    alloc   model/activation_6/Relu;model...                slow+27648   18.00 KiB
    alloc   model/activation/Relu;model/b...                fast+0        4.12 KiB
    ...
  tile 1    rows 2-24
    alloc   input_1_int8                                    fast+22272    6.47 KiB
    load    input_1_int8                       slow+576     fast+22272    6.47 KiB   rows 2-24
    move    model/activation/Relu;model/b...   fast+384     fast+0        2.62 KiB
    op      Conv_3
    ...
    spill   model/activation_6/Relu;model...   fast+20736   slow+29184    1.50 KiB   rows 2-3
    reset                                                   fast+22272
```

| Event | Meaning |
|---|---|
| `alloc` | A tensor gets its place in fast or slow memory. |
| `load`, `spill` | A tensor, or the rows of it a tile needs, moves from slow to fast memory or back, with source and destination offsets. |
| `move` | Fast memory compacts, or a chain keeps the rows the next tile shares with this one. |
| `op` | One kernel call. |
| `reset` | A tile frees its fast memory. |
| `weights`, `copy` | Decompressed weights, and state or control-flow operands copied between buffers. |

The trace runs on zeros unless `--input` gives real data, in the formats
[`tigris run`](/toolchain/run/) reads. The schedule is fixed at compile time, so
the input changes the trace only where the plan branches on data (`If`, `While`).
A `While` loop whose iteration count depends on the input values does not run on
zeros: `--trace` then needs `--input`.
`--trace` needs the host runtime, which platform wheels bundle; see
[Installation](/getting-started/installation/).

## JSON

`--json` prints one document with `report: "tigris-analysis"` and a `version`,
then these sections:

| Section | Holds |
|---|---|
| `model` | Name, format, dtype, interface, operator and tensor counts, unscheduled peak, largest tensor, unsupported operators |
| `tflite_micro` | TFLite Micro's arena head, what it excludes, and `export` with what [`tigris export`](/toolchain/export/) finds for `arm-none-eabi`; `null` for an ONNX model and for a TFLite model without the `TFLite Micro arena` row |
| `fast` | Budget, verdict, scheduled peak, stage counts, every blocking stage with its reason |
| `slow` | Budget, peak, whether it fits, measured `traffic` as `written` and `read` bytes or `null` |
| `flash` | Budget, weight and plan bytes, whether it fits, estimates |
| `runtime_build` | Each build limit the plan needs above its default, with `name`, `needed` and `default`; empty when the defaults suffice |
| `budgets_tried` | The other budgets and their outcome |
| `stages` | One entry per stage |

Sizes are in bytes. A field that does not apply is `null`.

```bash
tigris analyze vww_96_int8.tflite -m 32K -m 1M --json
```

```json
  "slow": {
    "budget_bytes": 1048576,
    "peak_bytes": 46080,
    "fits": true,
    "overflow_stages": [],
    "traffic": {
      "written": 18434,
      "read": 93312
    }
  },
```

`--json` and `--trace` are two different outputs; a command takes one of them.

## Examples

Will it fit, and which smaller budgets also work:

```bash
tigris analyze model.tflite -m 64K
```

On-chip SRAM next to external PSRAM, with a flash check:

```bash
tigris analyze model.onnx -m 232K -m 6M -f 4M
```

A stock ONNX export with a free batch dimension, at batch 4:

```bash
tigris analyze model.onnx -m 256K --input-shape input:4x3x224x224
```

The per-stage table and a trace on a real input:

```bash
tigris analyze model.tflite -m 32K -v
tigris analyze model.tflite -m 32K --trace --input sample.npy
```
