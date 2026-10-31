---
title: "tigris export"
description: "Use tigris export to rewrite an int8 TFLite model so that TFLite Micro runs it in a smaller arena, with the same outputs."
sidebar:
  order: 225
slug: 0.13.0/toolchain/export
---

Rewrite a TFLite model so that TFLite Micro runs it in a smaller arena. The
result is an ordinary TFLite file for the TFLite Micro build you already have; it
does not use the TiGrIS runtime.

Export expects a model that TFLite Micro already runs. It does not check that
TFLite Micro's kernels accept the model's tensor types, so run the original on
TFLite Micro first. The figures it prints are for TFLite Micro's reference
kernels built for the chosen target.

## Usage

```bash
tigris export MODEL [-o OUTPUT] [--target arm-none-eabi|ilp32|lp64]
```

## Options

| Flag | Type | Description |
|------|------|-------------|
| `MODEL` | path | TFLite model |
| `-o`, `--output` | path | Where to write the rewritten model; default `<model>_export.tflite` next to the input |
| `--target` | `arm-none-eabi`, `ilp32`, `lp64` | C ABI of the TFLite Micro build the arena is computed for; default `arm-none-eabi` |

## Output

```bash
tigris export vww_96_int8.tflite
```

```text
vww_96_int8.tflite   31 operators
operators 2-3 computed in 4 row bands, +0.25% MACs; 46 operators after export

TFLite Micro arena, reference kernels on arm-none-eabi
  head            45.00 KiB   72.00 KiB before
  persistent      +2.31 KiB
  total change   -24.69 KiB

wrote vww_96_int8_export.tflite   328.78 KiB
```

The second line says what changed: operators 2 and 3 of the input now run in 4
row bands, which adds 0.25% to the model's multiply-accumulates, and the written
model has 46 operators. When PAD operators were folded, the line says how many.

| Row | Meaning |
|---|---|
| `head` | The arena TFLite Micro's memory planner places for the written model's tensors and the scratch buffers of its reduction kernels, then the same figure for the input |
| `persistent` | How much TFLite Micro's persistent allocations grow: the records it keeps per tensor and operator, kernel data and quantization arrays. The added operators and tensors cost bytes here. |
| `total change` | The head change plus the persistent growth: how much the arena TFLite Micro needs goes down |

The arena is the one a `MicroInterpreter` uses on a TFLite Micro build without
`USE_TFLM_COMPRESSION`. Optimized kernels such as CMSIS-NN or ESP-NN keep their
own kernel data and scratch buffers, which these figures do not include.

## What export changes

Export looks for a run of int8 operators that contains a strided or pooling
operator and computes it in 2 to 10 horizontal bands of rows. Early in a vision
model such a run holds the largest activations. Each band:

1. takes the input rows it reads with `SLICE`, including the overlap its kernels
   need;
2. adds the padding the original operator applied at that band's edges with
   `PAD`, or `PADV2` in front of a max pool;
3. runs copies of the operators with `VALID` padding, which share the original
   weights.

A `CONCATENATION` joins the band outputs into the original output tensor. The
run's intermediate tensors now exist one band at a time, so the arena's peak goes
down; the file grows only by the added operators, from 325.48 KiB to 328.78 KiB
in the example. The written model computes the same outputs as the input, bit for
bit: each band runs the same int8 kernels with the same quantization on the rows
it needs.

A run can hold `CONV_2D`, `DEPTHWISE_CONV_2D`, `MAX_POOL_2D`, `AVERAGE_POOL_2D`
where it adds no padding, `PAD`, `PADV2`, and the elementwise operators `RELU`,
`RELU6`, `LOGISTIC`, `TANH`, `HARD_SWISH`, `LEAKY_RELU`, `QUANTIZE`, `ABS`,
`ADD`, `SUB`, `MUL`, `MAXIMUM` and `MINIMUM`. Every tensor the run produces,
except its last output, is read only inside the run.

A `PAD` in front of a convolution that equals the convolution's `SAME` padding is
folded into it. Export tries every run and band count, with and without the
folds, and writes the one with the lowest total. One run is banded per model.

## When nothing is written

Export writes a file only when the total goes down. Otherwise it exits with
status 1, writes no file and names the reason:

```text
kws_ref_model.tflite   13 operators
Error: no model written: no rewrite lowers the arena
```

| Reason | Meaning |
|---|---|
| `no rewrite lowers the arena` | No candidate has a smaller total, for example because the largest tensors are not in a run that can be banded |
| `models with several subgraphs are not exported` | The model has more than one subgraph, as models with `IF` or `WHILE` do |
| `metadata NAME is not supported` | The model carries metadata that may refer to tensors by index, such as an offline memory plan. Only `min_runtime_version` and `CONVERSION_METADATA` are kept. |
| `a tensor type TFLite Micro's arena model does not know` | A tensor has a storage type such as int4 |
| `TFLite Micro's scratch buffers for KIND are not modeled` | A kernel requests scratch buffers the arena figure does not size: `CUSTOM` (for example detection post-processing), `MIRROR_PAD`, `UNIDIRECTIONAL_SEQUENCE_LSTM`, `SVDF`, `TRANSPOSE_CONV`, `ADD_N` |
| `TFLite Micro's persistent allocations for KIND are not modeled` | An operator whose persistent allocations are not modeled |
| `OWNER: ITEM index I is out of range for N ITEMs` | An operator, the model's inputs or outputs, a signature or the metadata refers to a tensor, buffer, operator code or subgraph the file does not have |
| `the TABLE table sets field N, which is not supported`, and the other messages about tables, options, quantization details or buffers | The file uses a part of the TFLite schema that export does not carry; it refuses the file rather than drop that part |

## Targets

The persistent allocations depend on how TFLite Micro was compiled: the size of
a pointer and the size of an enum. `--target` names the C ABI:

| Target | Pointer | Enum | Builds |
|---|---|---|---|
| `arm-none-eabi` | 4 bytes | the smallest integer that holds its values | GCC for Arm Cortex-M with its default short enums |
| `ilp32` | 4 bytes | 4 bytes | 32-bit builds with `-fno-short-enums` |
| `lp64` | 8 bytes | 4 bytes | 64-bit host builds |

The head is the same on every target. The persistent growth, and so the total,
differs: on `lp64` the example grows by 3.17 KiB, for a total change of
-23.83 KiB. Export can pick a different run or band count for another target.

## In analyze

For a TFLite model that gets the `TFLite Micro arena` row,
[`tigris analyze`](/0.13.0/toolchain/analyze/) runs the same search for `arm-none-eabi`
and reports it as a memory row, without writing a file:

```text
  TFLite Micro arena     72.00 KiB   its planner and reduction scratch; persistent allocations and other kernel scratch excluded
  TFLite Micro export   -24.69 KiB   head 72.00 KiB -> 45.00 KiB, persistent +2.31 KiB, +0.25% MACs, reference kernels on arm-none-eabi
```

When export would write nothing, the row reads `none` and the reason, followed by
the kernels and target when the reason is that no rewrite lowers the arena. A
model with several subgraphs or a tensor type TFLite Micro's arena model does not
know gets neither row, and so does a file export does not read, such as one with
buffers stored outside the FlatBuffer.
`analyze --json` reports the same result under `tflite_micro.export`.

## Examples

Rewrite a model for a Cortex-M build of TFLite Micro:

```bash
tigris export model.tflite -o model_small.tflite
```

For a TFLite Micro build on a 64-bit host:

```bash
tigris export model.tflite --target lp64
```
