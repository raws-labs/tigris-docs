---
title: "tigris inspect"
description: "Use tigris inspect to read the interface, operator costs and memory needs of a TFLite, ONNX or PyTorch model, an ExecuTorch program or a compiled .tgrs plan without compiling or running it."
sidebar:
  order: 205
slug: 0.15.0/toolchain/inspect
---

Read what a TFLite, ONNX or PyTorch model contains, or what an ExecuTorch program
or a compiled `.tgrs` plan records, without compiling or executing anything.

## Usage

```bash
tigris inspect MODEL [OPTIONS]
```

The format is detected from the file contents, so the extension does not matter.

## Options

| Flag | Type | Description |
|------|------|-------------|
| `MODEL` | path | TFLite model, ONNX model, PyTorch program (`.pt2`), ExecuTorch program (`.pte`) or `.tgrs` plan |
| `-v`, `--verbose` | flag | Adds the per-step or per-stage tables and the full tensor listings |
| `--json` | flag | Complete, versioned metadata as JSON |

## Models

For a model, `inspect` reports the interface, what each operator type costs, the
largest activations and, for TFLite, how the model is quantized and whether it
converts to a plan.

```bash
tigris inspect ds_cnn_i8.tflite
```

```text
ds_cnn_i8.tflite   TFLite, model main, schema 3, 11 operators, 41.23 KiB
  input    serving_default_input:0       1x49x10x1 int8
  output   StatefulPartitionedCall_1:0   1x12 int8

operators
                      count     weights             MACs
  CONV_2D                 5   19.75 KiB   83%     2.37 M   89%
  DEPTHWISE_CONV_2D       4    3.25 KiB   14%   288.00 K   11%
  FULLY_CONNECTED         1       768 B    3%        768   <1%
  MEAN                    1         8 B   <1%
  total                  11   23.76 KiB           2.66 M

largest activations
  1x25x5x64 int8   7.81 KiB   step 0, CONV_2D
  1x25x5x64 int8   7.81 KiB   step 1, DEPTHWISE_CONV_2D
  1x25x5x64 int8   7.81 KiB   step 2, CONV_2D

quantization   int8 activations; int8 weights per channel, symmetric
converts       yes
```

- `weights` counts every constant an operator type reads, where it is read. A weight
  that reaches its operator through other operators on constants, such as an ONNX
  `QuantizeLinear` and `DequantizeLinear` pair, counts at the operator that uses it.
- `MACs` counts the multiply-accumulates of convolutions, fully connected layers and
  matrix products from their shapes. It is a measure of size, not of latency.
- `largest activations` lists the three largest tensors the model computes or takes as
  input, and the step that writes each.
- `converts` says whether `analyze` and `compile` accept a TFLite model, and if
  not, every reason. For the TFLite Micro Ethos-U example, whose single operator
  is a custom one:

  ```text
  converts       no, 1 reason
                 operator 0 CUSTOM:ethos-u: not supported
  ```

  For an ONNX model, `analyze` names the operators it cannot compile.

With `-v`, a `steps` table lists every operator with its output shape and size, the
weights it reads and its MACs, followed by the constants.

An ONNX model is read as it declares itself: shapes are not inferred and external
tensor data is not loaded. A figure that needs an undeclared shape shows as `?`.

A PyTorch program saved with `torch.export.save` is exported to ONNX with
`torch.onnx.export`, which needs `torch` and `onnxscript`, and reported as that
ONNX model. The ONNX keeps the quantize and dequantize operators of a converted
PT2E program and declares float32 at both ends; `analyze` and `compile` fold them
into int8 operators:

```text
mbv2.pt2   PyTorch program, exported to ONNX, model main_graph, IR 10, opsets ai.onnx 20, 340 operators, 3.14 MiB
  input    x                          1x3x96x96 float32
  output   dequantize_per_tensor_78   1x1000 float32

operators
                     count      weights            MACs
  Conv                  52   456.91 KiB   27%   10.65 M   89%
  Gemm                   1     1.22 MiB   73%    1.28 M   11%
  QuantizeLinear        67        274 B   <1%
  Reshape                1         16 B   <1%
  Clip                  35          8 B   <1%
  DequantizeLinear     173
  Add                   10
  AveragePool            1
  total                340     1.67 MiB         11.93 M

largest activations
  1x48x48x48 float32   432.00 KiB   step 21, Conv
  1x48x48x48 float32   432.00 KiB   step 22, Clip
  1x48x48x48 float32   432.00 KiB   step 24, DequantizeLinear

initializers   306 dense, 0 sparse, 0 external

Declared metadata: shapes are not inferred, external data is not loaded; ? marks a figure an undeclared shape hides.
```

## ExecuTorch programs

For an ExecuTorch program lowered for Cortex-M, `inspect` reports its method,
the kernels it was lowered for, its interface and the RAM the program records:

```bash
tigris inspect mbv2.pte
```

```text
mbv2.pte   ExecuTorch program, 64 instructions, CMSIS-NN DSP kernels (Cortex-M4, M7, M33, M35P)
  input    0   1x96x96x3 int8, dim order 0,2,3,1
  output   0   1x1000 int8

ExecuTorch RAM on Cortex-M, from the program
  model memory       135.00 KiB   activations and CMSIS-NN scratch
  method allocator    23.75 KiB   ExecuTorch's records for the method
```

- The first line counts the instructions of the method `forward` and names the
  [kernel class](/0.15.0/toolchain/export/#kernel-class) the program was lowered for.
- Each input and output shows its shape in memory order, the order in which a
  `.bin` file for [`tigris run`](/0.15.0/toolchain/run/#input-layout) holds the
  elements. `dim order` follows when PyTorch's order differs: this input is the
  1x3x96x96 tensor PyTorch takes, kept channels last.
- `model memory` is the planned buffer and `method allocator` what ExecuTorch
  takes from the method allocator to load the method, both for a 32-bit
  Cortex-M target. [`analyze`](/0.15.0/toolchain/analyze/#executorch-programs) adds the
  temp allocator and what [`export`](/0.15.0/toolchain/export/#executorch-programs)
  saves.

`--json` reports `format` `pte` and a `program` object with the `method`, the
`instructions`, the `cmsis_nn_classes`, the `inputs` and `outputs` with their
`shape` in PyTorch's sizes and their `dim_order`, the `memory` in bytes and the
`operators`.

## Compiled plans

For a `.tgrs` plan, `inspect` reports the plan schema, the interface in stored axis
order and declared dtype, the weights per operator type, the memory the plan needs and
whether a default runtime build can load it. This is the interface
[`tigris run`](/0.15.0/toolchain/run/) expects its inputs in.

```bash
tigris inspect mobilenet.tgrs
```

```text
mobilenet.tgrs   plan, model mobilenet_v1_matched, schema 12, 31 operators, 13 stages, 12 tiled, 3.18 MiB
  input    input    1x128x128x3 float32   stored int8, NHWC
  output   l33_dq   1x10 float32          stored int8, model order

operators
                      count     weights
  Conv                   15    3.03 MiB   98%
  DepthwiseConv          13   62.97 KiB    2%
  Flatten                 1
  GlobalAveragePool       1
  Softmax                 1
  total                  31    3.09 MiB

memory
  fast arena     64.00 KiB
  unscheduled   384.00 KiB
  weights         3.09 MiB   read in place (XIP), uncompressed

runtime build   default limits suffice
```

- `fast arena` is what the runtime requires for the plan's fast arena. For an LZ4 plan
  it includes the buffer that decompressed weights occupy, sized for the largest tensor
  alignment the runtime supports, so on a target with smaller alignment it is an upper
  bound.
- `unscheduled` is the activation peak the compiler recorded for the model run as one
  stage.
- `state`, shown for a stateful plan, is the memory kept between runs.
- `runtime build` names each `-D` limit the plan needs above the runtime's default, or
  says the defaults suffice.

With `-v`, a `stages` table lists each stage's operators, its peak before tiling, the
weights it reads and how it tiles, followed by every operator's parameters and the
stored tensors. A stage tiled on both spatial axes shows as `N tiles of H x W, halo K`,
one tiled along a single axis as `N tiles, axis A, halo K`, and the first stage of a
chain as `chain of N` with its tile height. A schema 10 tile record states neither the
tile count nor the halo, so such a stage shows as `tiled` with its axis. A plan compiled without
[`--keep-names`](/0.15.0/toolchain/compile/#options) names only the model, its inputs and
its outputs: its operators show as `(unnamed)` and its other tensors by their id,
for example `tensor 3`.

A plan's schema number says which plan format it uses; it does not by itself name a
runtime version.

## JSON output

`--json` prints everything `inspect` reads, with exact byte counts and without weight
values, including the cost, quantization and requirement figures above. The top-level
`inspection_version` field versions the document, so scripts can check it before reading
further. Version 2 lists each tile record by the fields its schema states: a schema 11
record adds `tile_extent`, `num_tiles`, `original_extent`, `halo`, `band_axis`,
`band_kind` and `band_mode` to the scheduling choices, and a schema 10 record leaves
those seven `null`.

```bash
tigris inspect model.tgrs --json
```

## What inspect does not do

- It does not compile, execute or validate kernel numerics.
- It does not infer ONNX shapes or load external ONNX tensor data.
- It needs no native library, so it works on every platform `tigris-ml` installs on.
  See [Installation](/0.15.0/getting-started/installation/).
