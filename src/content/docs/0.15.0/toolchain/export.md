---
title: "tigris export"
description: "Use tigris export to rewrite an int8 TFLite model or an ExecuTorch Cortex-M program in row bands, so that TFLite Micro or ExecuTorch runs it in less RAM with the same outputs."
sidebar:
  order: 225
slug: 0.15.0/toolchain/export
---

Rewrite a model so that the runtime you already deploy runs it in less RAM.
Export splits runs of operators into horizontal row bands, so fewer activations
are held at once, and writes a file of the same format that computes the same
outputs:

- a TFLite model keeps running on TFLite Micro, in a smaller arena;
- an ExecuTorch program lowered for Cortex-M keeps running on the same
  ExecuTorch build, with a smaller planned buffer.

The written file runs on that runtime, not on the TiGrIS runtime.
[`tigris analyze`](/0.15.0/toolchain/analyze/) prints the same figures without writing
a file, and [`tigris run`](/0.15.0/toolchain/run/) runs the input and the written file
on the host, so you can [compare their outputs](#check-the-outputs).

## Usage

```bash
tigris export MODEL [-o OUTPUT] [--require-gain] [--json PATH]
tigris export MODEL.tflite [--target arm-none-eabi|ilp32|lp64]
tigris export MODEL.pte [--cpu CORE] [--allow-input-overwrite] [--arena SIZE] [--method SIZE]
                        [--max-extra-macs PERCENT] [--header PATH]
```

The format is read from the file contents, so the extension does not matter.

## Options

| Flag | Applies to | Description |
|------|------------|-------------|
| `MODEL` | both | TFLite model or ExecuTorch program |
| `-o`, `--output` | both | Where to write; default `<model>_export.tflite` or `<model>_export.pte` next to the input |
| `--require-gain` | both | Exit 3 when the rewrite saves nothing |
| `--json PATH` | both | Write the report as JSON |
| `--target` | TFLite | C ABI of the TFLite Micro build the arena is computed for: `arm-none-eabi` (default), `ilp32`, `lp64`; see [Targets](#targets) |
| `--cpu` | ExecuTorch | The firmware's Cortex-M core, `M0` to `M85`; read from the program by default, see [Kernel class](#kernel-class) |
| `--allow-input-overwrite` | ExecuTorch | Let the rewrite place tensors over the model input once it has been read |
| `--arena SIZE` | ExecuTorch | Limit on the planned buffer, for example `64K` |
| `--method SIZE` | ExecuTorch | Limit on the method allocator, for example `32K` |
| `--max-extra-macs PERCENT` | ExecuTorch | Limit on recomputed multiply-accumulates, `5%` or `5` |
| `--header PATH` | ExecuTorch | Write the pool sizes as a C header |

Sizes take `K` and `M` suffixes as the [memory budgets](/0.15.0/toolchain/overview/#memory-budgets)
do. An option for the other format is a usage error, for example
`Error: --cpu applies to ExecuTorch programs`.

## Exit status

| Status | Meaning | File written |
|---|---|---|
| 0 | A rewrite saves memory, or nothing smaller was found | The rewrite, or an unchanged copy of the input |
| 1 | The file was refused; the error names the reason | None |
| 2 | Usage error | None |
| 3 | A stated limit was not met: `--arena`, `--method` or `--require-gain` | An unchanged copy of the input |

With status 0 and 3, `-o`, `--json` and `--header` are always written, so a
build can depend on the output path. An unchanged copy is the input byte for
byte.

## Check the outputs

[`tigris run`](/0.15.0/toolchain/run/) runs a TFLite model on TFLite Micro's reference
kernels and an ExecuTorch program on ExecuTorch with CMSIS-NN's portable C
kernels. Run the input and the written file on the same input and compare the
output files byte for byte:

```bash
tigris run model.pte --executorch ~/executorch --input x.bin --output a.bin
tigris run model_export.pte --executorch ~/executorch --input x.bin --output b.bin
cmp a.bin b.bin
```

`cmp` prints nothing and exits with 0 when the outputs are identical. A TFLite
model needs no `--executorch`:

```bash
tigris run model.tflite --input x.bin --output a.bin
tigris run model_export.tflite --input x.bin --output b.bin
cmp a.bin b.bin
```

`x.bin` holds the input in the axis order and dtype
[`tigris inspect`](/0.15.0/toolchain/inspect/) shows. The run page lists what each
runtime needs on the host, how `run` builds ExecuTorch from a checkout and the
input layouts an ExecuTorch program takes:
[TFLite models](/0.15.0/toolchain/run/#tflite-models) and
[ExecuTorch programs](/0.15.0/toolchain/run/#executorch-programs).

## ExecuTorch programs

Export takes an ExecuTorch 1.6 program lowered for Cortex-M by ExecuTorch's
Cortex-M backend, with int8 inputs and outputs. It rewrites runs of
`quantized_conv2d`, `quantized_depthwise_conv2d` and `quantized_add` into row
bands, with the program's own operators, and redoes the memory plan of the
whole method. [Compiling a PyTorch Model](/0.15.0/guides/pytorch-models/#band-an-executorch-program)
shows the lowering from a PyTorch model.

```bash
tigris export mbv2.pte --json mbv2.json --header mbv2_pools.h
```

```text
mbv2.pte   ExecuTorch program, 64 instructions, CMSIS-NN DSP kernels (Cortex-M4, M7, M33, M35P)

rewrite
  instructions 0-2     8 row bands, output in place
  instructions 3-5     8 row bands, output in place
  instructions 6-8     2 row bands
  extra MACs           +3.67%
  instructions after   109

ExecuTorch RAM on Cortex-M, from the program
  model memory       37.50 KiB   135.00 KiB before   3.60x less; activations and CMSIS-NN scratch
  method allocator   29.28 KiB    23.75 KiB before   ExecuTorch's records for the method
  temp allocator         448 B        448 B before   needed to load the method
  total              67.22 KiB   159.19 KiB before   2.37x less
  plus the runner's own input and output buffers, stack and statics

model input memory reused once read, as in the original

wrote mbv2_export.pte   1.70 MiB   +6.38 KiB
```

The program is MobileNetV2 0.35 at 96x96 with int8 input and output, lowered for
Cortex-M7. The first line names the kernels it was lowered for. `rewrite` lists
each run of the original program's instructions that now runs in row bands.
`output in place` means the run's output shares bytes with its largest input,
placed so that no band overwrites rows a later band still reads. `extra MACs` is
what the overlap between bands recomputes, as a share of the program's
multiply-accumulates, and `instructions after` counts the kernel calls of the
written program.

### Memory

Every figure in this section is ExecuTorch's, for the program as written and,
after `before`, for the input, computed from the program file for a 32-bit
Cortex-M target.

| Row | What it counts |
|---|---|
| `model memory` | The planned buffer: every activation and every CMSIS-NN scratch buffer, at the offsets the program assigns them. ExecuTorch reads this size from the program when it loads the method. |
| `method allocator` | What ExecuTorch takes from the method allocator to load the method: its values, tensor records and kernel argument lists. The bands add records, so this row grows. |
| `temp allocator` | The temp allocator size from which ExecuTorch resolves every kernel's argument metadata while it loads the method. Give it at least this much: with less, ExecuTorch takes that metadata from the method allocator, beyond the figure above. |
| `total` | The three pools together |

`model memory` and `total` carry the ratio when they go down, and the signed
change otherwise. The rest of the firmware, its own buffers, stack and
statics, keeps its size. ExecuTorch's runtime statics include its kernel
registry, which [On a device](#on-a-device) measures.

The input line says whether the method input's memory is reused once the input
has been read:

| Line | Meaning |
|---|---|
| `model input memory reused once read, as in the original` | The program as lowered already places tensors over the input, so the rewrite does too |
| `model input memory reused once read (--allow-input-overwrite)` | The rewrite places tensors over the input because the option allows it |
| `model input untouched` | The input keeps its bytes for the whole inference |

The last line names the written file, its size and how much it grew: the added
kernel calls and their arguments, 6.38 KiB in the example.

### Kernel class

ExecuTorch sizes each convolution's scratch buffer for the CMSIS-NN kernels of
one class of Cortex-M core, and export keeps every band on the same kernels.
Export reads the class from the scratch sizes and the linear layers:

| Class | Cores |
|---|---|
| `scalar` | Cortex-M0, M0+, M3, M23 |
| `DSP` | Cortex-M4, M7, M33, M35P |
| `MVE` | Cortex-M55, M85 |

When the sizes fit several classes, the line reads, for example,
`CMSIS-NN scalar or DSP kernels`. `--cpu` names the core of the firmware:
export checks it against the class, and the first line ends with
`checked against --cpu M7`. A core of another class stops export:

```text
Error: mbv2.pte was lowered for DSP kernels (Cortex-M4, M7, M33, M35P); --cpu M55 runs MVE kernels
```

A program lowered for MVE kernels gets a warning before the report, because
those kernels take a precomputed kernel sum in place of a linear layer's bias:

```text
warning: lowered for MVE kernels (Cortex-M55, M85); firmware without MVE runs its linear layers without their bias
```

Without a linear layer the warning ends `it runs on firmware built with MVE`.
Deploy such a program on a Cortex-M55 or M85 firmware built with MVE.

### Limits

Export searches over the runs it can band and the band count of each, plans every
candidate in full and writes the one with the lowest total. A limit narrows the
search to candidates within it, and export then writes the lowest total among
those:

```bash
tigris export mbv2.pte --arena 32K
```

```text
  model memory       30.47 KiB   135.00 KiB before   4.43x less; activations and CMSIS-NN scratch
  method allocator   49.23 KiB    23.75 KiB before   ExecuTorch's records for the method
  temp allocator         448 B        448 B before   needed to load the method
  total              80.14 KiB   159.19 KiB before   1.99x less
```

Here the planned buffer fits 32 KiB with 48 bands in the first run and
+20.27% MACs, and the method allocator grows with the bands. Without the limit,
the lowest total is the 67.22 KiB above. `--max-extra-macs 1%` keeps the
recomputation at +0.93% and gives 42.75 KiB of model memory, 71.48 KiB in
total.

When no candidate meets a limit, export writes an unchanged copy and exits with
status 3, one line per limit:

```bash
tigris export mbv2.pte --arena 8K
```

```text
mbv2.pte   ExecuTorch program, 64 instructions, CMSIS-NN DSP kernels (Cortex-M4, M7, M33, M35P)
smallest model memory found is 30.47 KiB, above --arena 8K (total 80.14 KiB)
mbv2_export.pte is an unchanged copy
```

When each limit is met by some candidate but none meets both, the line reads
`no program found meets --arena 40K and --method 24K together`.

### When nothing is smaller

When no candidate has a lower total than the input, export writes an unchanged
copy and exits with status 0, or 3 with `--require-gain`:

```text
model.pte   ExecuTorch program, 10 instructions, CMSIS-NN DSP kernels (Cortex-M4, M7, M33, M35P)
no smaller total found; model_export.pte is an unchanged copy
```

### Refused programs

Export exits with status 1 and writes nothing for a program it does not
rewrite. The error names the file and the reason:

```text
Error: mbv2_float_io.pte calls cortex_m::quantize_per_tensor; tigris export rewrites programs that take and return int8 tensors (ExecuTorch's QuantizeInputs and QuantizeOutputs passes)
```

| Reason starts with | Meaning |
|---|---|
| `calls cortex_m::quantize_per_tensor` or `dequantize_per_tensor` | Lower with ExecuTorch's `QuantizeInputs` and `QuantizeOutputs` passes, so that the program takes and returns int8 tensors |
| `delegates to BACKEND` | Export rewrites programs that run on cortex_m operators; a program delegated to a backend such as Ethos-U runs its layers there |
| `uses the explicit NHWC layout` | Lower with the Cortex-M backend's default layout; the `*_nhwc` operators are outside what export rewrites |
| `calls operator NAME` | The program calls an operator outside the cortex_m set export reads, which the message lists: `quantized_conv2d`, `quantized_depthwise_conv2d`, `quantized_transpose_conv2d`, `quantized_add`, `quantized_mul`, `quantized_avg_pool2d`, `quantized_max_pool2d`, `quantized_linear`, `quantized_activation`, `pad` and `softmax` |
| `was lowered for CLASS; --cpu CORE runs CLASS` | `--cpu` names a core of another kernel class; lower the program for that core, or give the core it was lowered for |
| `is laid out differently from the programs ExecuTorch 1.6 writes` | Export reads programs written by ExecuTorch 1.6 and writes them back byte for byte; lower with ExecuTorch 1.6 |
| `plans N buffers`, `has N methods with instructions`, `has a tensor with a dynamic shape` and the other messages about the program's structure | Export rewrites a program with one method, one chain of kernel calls, one planned buffer and static shapes |

### Pool sizes as a header

`--header` writes the sizes the firmware gives ExecuTorch, for the program at `-o`:

```c
/* Pool sizes for mbv2_export.pte, from tigris export <version> (sha256 1b08579d3bf14266). */
#define TIGRIS_ET_PLANNED_BUFFER_1_BYTES 38400u
#define TIGRIS_ET_METHOD_ALLOCATOR_MIN_BYTES 29984u
#define TIGRIS_ET_TEMP_ALLOCATOR_MIN_BYTES 448u
#define TIGRIS_ET_INPUT_0_OFFSET 0u
#define TIGRIS_ET_OUTPUT_0_OFFSET 0u
```

There is one `PLANNED_BUFFER` line per planned buffer, numbered as ExecuTorch
numbers them from 1. `METHOD_ALLOCATOR_MIN_BYTES` and `TEMP_ALLOCATOR_MIN_BYTES`
are the smallest pools the method loads from. The offsets locate each method
input and output inside its planned buffer, so the firmware can write the input
there and read the output in place, without buffers of its own. The comment
names the file the figures belong to by its hash. When export writes an
unchanged copy, the header holds that copy's figures.

```c
#include "mbv2_pools.h"

static uint8_t method_pool[TIGRIS_ET_METHOD_ALLOCATOR_MIN_BYTES] __attribute__((aligned(16)));
static uint8_t temp_pool[TIGRIS_ET_TEMP_ALLOCATOR_MIN_BYTES] __attribute__((aligned(16)));
static uint8_t planned[TIGRIS_ET_PLANNED_BUFFER_1_BYTES] __attribute__((aligned(16)));
```

The firmware measured below declares its pools this way and loads the method
in pools of exactly these sizes.

### JSON

`--json` writes the report with exact byte counts:

| Key | Holds |
|---|---|
| `runtime`, `source` | `executorch` and `program`: the figures are ExecuTorch's, computed from the program file |
| `tigris_version`, `input_sha256`, `output_sha256` | The compiler that wrote the report and the files it describes |
| `written`, `reason` | Whether a rewrite was written, and otherwise why not, as the command prints it |
| `warnings` | The warnings printed before the report |
| `cmsis_class`, `cpu` | The kernel classes the program fits, `SCALAR`, `DSP` or `MVE`, and the `--cpu` given or `null` |
| `model_memory_bytes`, `method_allocator_bytes`, `temp_allocator_bytes`, `total_bytes` | Each as `before` and `after`; `after` is the written file, the rewrite or the copy |
| `macs`, `extra_macs` | The program's multiply-accumulates and what the rewrite adds |
| `pte_bytes` | File size `before` and `after` |
| `regions` | Each banded run: `first` and `last` instruction of the original, `bands`, `in_place` |
| `input_overwritten` | Whether the input's memory is reused, `before` and `after` |
| `instruction_map` | For each instruction of the written program, the `original` instruction it runs and its `band`, `null` outside the banded runs |

```json
  "model_memory_bytes": {
    "before": 138240,
    "after": 38400
  },
  "method_allocator_bytes": {
    "before": 24324,
    "after": 29984
  },
```

### On a device

Three models, each lowered for Cortex-M7 with int8 input and output by
ExecuTorch 1.6.0, ran on a NUCLEO-H753ZI (Cortex-M7 at 480 MHz) as lowered and
after `tigris export` with default options, on ExecuTorch 1.6.0 with CMSIS-NN
v8.0.0. Both programs gave the same int8 outputs on every input: 0 of 240 bytes
differ for DS-CNN (20 inputs), 0 of 200 for ResNet-8 (20 inputs) and 0 of 1,000
for MobileNetV2 0.35 at 96x96 (one input).

ExecuTorch model memory, the planned buffer: the program figure, which the
device writes in full.

| Model | ExecuTorch as lowered | ExecuTorch after `tigris export` | Change |
|---|---|---|---|
| DS-CNN | 16,000 B | 12,160 B | 1.32x less |
| ResNet-8 | 49,728 B | 24,192 B | 2.06x less |
| MobileNetV2 0.35, 96x96 | 138,240 B | 38,400 B | 3.60x less |

ExecuTorch method allocator, measured on the device after loading the method,
equal to the program figure in every case:

| Model | ExecuTorch as lowered | ExecuTorch after `tigris export` |
|---|---|---|
| DS-CNN | 4,860 B | 7,700 B |
| ResNet-8 | 5,352 B | 9,800 B |
| MobileNetV2 0.35, 96x96 | 24,324 B | 29,984 B |

Firmware RAM total: `.data` and `.bss` from the linker map plus the heap used
and the stack high-water mark measured on the device. Each total includes about
3.7 KB of the measuring firmware's own C library, heap and stack. ExecuTorch's
kernel registry holds 2,000 kernel slots by default, 24,000 B on a 32-bit
target; ExecuTorch built with `MAX_KERNEL_NUM=51`, the 51 kernels these
firmwares register (28 prim ops and the 23 cortex_m operators), needs 612 B:

| Model | Kernel registry | ExecuTorch as lowered | ExecuTorch after `tigris export` | Change |
|---|---|---|---|---|
| DS-CNN | 2,000 slots | 49,800 B | 48,800 B | 1.02x less |
| DS-CNN | 51 slots | 26,412 B | 25,412 B | 1.04x less |
| ResNet-8 | 2,000 slots | 84,004 B | 62,916 B | 1.34x less |
| ResNet-8 | 51 slots | 60,616 B | 39,528 B | 1.53x less |
| MobileNetV2 0.35, 96x96 | 2,000 slots | 191,504 B | 97,324 B | 1.97x less |
| MobileNetV2 0.35, 96x96 | 51 slots | 168,116 B | 73,936 B | 2.27x less |

The model memory goes down by more than the firmware total, because the method
allocator grows with the bands and the rest of the firmware keeps its size. On
DS-CNN, 2,840 B of the 3,840 B saved go to the method allocator.

Time per inference, default registry build, median of the device's cycle counter
at 480 MHz: DS-CNN 12.09 ms as lowered and 12.39 ms after export (+2.5%, 27
kernel calls instead of 11), ResNet-8 45.60 and 50.17 ms (+10.0%, +10.62% MACs),
MobileNetV2 74.76 and 78.16 ms (+4.5%, +3.67% MACs). The program in flash grows
by 4,096, 5,376 and 6,528 B.

[Compiling a PyTorch Model](/0.15.0/guides/pytorch-models/#on-a-device) puts these
firmwares next to the TiGrIS runtime running the same models.

## TFLite models

For a TFLite model, export writes a TFLite file for the TFLite Micro build you
already have. Export expects a model that TFLite Micro already runs. It does not
check that TFLite Micro's kernels accept the model's tensor types, so run the
original on TFLite Micro first, for example with
[`tigris run`](/0.15.0/toolchain/run/#tflite-models). The figures it prints are for TFLite Micro's
reference kernels built for the chosen target.

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

wrote vww_96_int8_export.tflite   328.75 KiB
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

`--json` writes `runtime` `tflite-micro` and `source` `model`, then the fields
`analyze --json` reports under `tflite_micro.export`.

### What export changes

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
down; the file grows only by the added operators, from 325.48 KiB to 328.75 KiB
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

### When nothing is smaller

When no rewrite lowers the total, export writes an unchanged copy and exits
with status 0, or 3 with `--require-gain`:

```text
kws_ref_model.tflite   13 operators
no smaller arena found; kws_ref_model_export.tflite is an unchanged copy
```

### Refused models

A model export cannot plan is refused with status 1, and nothing is written:

```text
Error: no model written: REASON
```

| Reason | Meaning |
|---|---|
| `models with several subgraphs are not exported` | The model has more than one subgraph, as models with `IF` or `WHILE` do |
| `metadata NAME is not supported` | The model carries metadata that may refer to tensors by index, such as an offline memory plan. Only `min_runtime_version` and `CONVERSION_METADATA` are kept. |
| `a tensor type TFLite Micro's arena model does not know` | A tensor has a storage type such as int4 |
| `TFLite Micro's scratch buffers for KIND are not modeled` | A kernel requests scratch buffers the arena figure does not size: `CUSTOM` (for example detection post-processing), `MIRROR_PAD`, `UNIDIRECTIONAL_SEQUENCE_LSTM`, `SVDF`, `TRANSPOSE_CONV`, `ADD_N` |
| `TFLite Micro's persistent allocations for KIND are not modeled` | An operator whose persistent allocations are not modeled |
| `OWNER: ITEM index I is out of range for N ITEMs` | An operator, the model's inputs or outputs, a signature or the metadata refers to a tensor, buffer, operator code or subgraph the file does not have |
| `the TABLE table sets field N, which is not supported`, and the other messages about tables, options, quantization details or buffers | The file uses a part of the TFLite schema that export does not carry; it refuses the file rather than drop that part |

### Targets

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
[`tigris analyze`](/0.15.0/toolchain/analyze/) runs the same search for `arm-none-eabi`
and reports it as a memory row, without writing a file:

```text
  TFLite Micro arena     72.00 KiB   its planner and reduction scratch; persistent allocations and other kernel scratch excluded
  TFLite Micro export   -24.69 KiB   head 72.00 KiB -> 45.00 KiB, persistent +2.31 KiB, +0.25% MACs, reference kernels on arm-none-eabi
```

When export would write nothing, the row reads `none` and the reason, followed by
the kernels and target when the reason is that no smaller arena was found. A
model with several subgraphs or a tensor type TFLite Micro's arena model does not
know gets neither row, and so does a file export does not read, such as one with
buffers stored outside the FlatBuffer.
`analyze --json` reports the same result under `tflite_micro.export`.

For an ExecuTorch program, `tigris analyze model.pte` prints the program as
lowered and after export, with the same options; see
[ExecuTorch programs](/0.15.0/toolchain/analyze/#executorch-programs) on the analyze page.

## Examples

Rewrite a model for a Cortex-M build of TFLite Micro:

```bash
tigris export model.tflite -o model_small.tflite
```

For a TFLite Micro build on a 64-bit host:

```bash
tigris export model.tflite --target lp64
```

Rewrite an ExecuTorch program for a Cortex-M7 firmware, with its pool sizes as a
header, and fail the build when nothing is saved:

```bash
tigris export model.pte --cpu M7 --header model_pools.h --require-gain
```

Keep the planned buffer within 64 KiB and the recomputation within 5%:

```bash
tigris export model.pte --arena 64K --max-extra-macs 5%
```
