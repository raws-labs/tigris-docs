---
title: "tigris inspect"
description: "Use tigris inspect to read the interface, operator costs and memory needs of a TFLite or ONNX model or a compiled .tgrs plan without compiling or running it."
sidebar:
  order: 205
---

Read what a TFLite or ONNX model contains, or what a compiled `.tgrs` plan records,
without compiling or executing anything.

## Usage

```bash
tigris inspect MODEL [OPTIONS]
```

The format is detected from the file contents, so the extension does not matter.

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | TFLite model, ONNX model or `.tgrs` plan |
| `-v`, `--verbose` | flag | no | Add the per-step or per-stage tables and the full tensor listings |
| `--json` | flag | no | Print complete, versioned metadata as JSON |

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
- `converts` says whether `analyze` and `compile` accept the model, and if not, why.

With `-v`, a `steps` table lists every operator with its output shape and size, the
weights it reads and its MACs, followed by the constants.

An ONNX model is read as it declares itself: shapes are not inferred and external
tensor data is not loaded. A figure that needs an undeclared shape shows as `?`.

## Compiled plans

For a `.tgrs` plan, `inspect` reports the plan schema, the interface in stored axis
order and declared dtype, the weights per operator type, the memory the plan needs and
whether a default runtime build can load it. This is the interface
[`tigris run`](/toolchain/run/) expects its inputs in.

```bash
tigris inspect mobilenet.tgrs
```

```text
mobilenet.tgrs   plan, model mobilenet_v1_matched, schema 9, 31 operators, 13 stages, 12 tiled, 3.35 MiB
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
stored tensors.

A plan's schema number says which plan format it uses; it does not by itself name a
runtime version.

## JSON output

`--json` prints everything `inspect` reads, with exact byte counts and without weight
values, including the cost, quantization and requirement figures above. The top-level
`inspection_version` field versions the document, so scripts can check it before reading
further.

```bash
tigris inspect model.tgrs --json
```

## What inspect does not do

- It does not compile, execute or validate kernel numerics.
- It does not infer ONNX shapes or load external ONNX tensor data.
- It needs no native library, so it works on every platform `tigris-ml` installs on.
  See [Installation](/getting-started/installation/).
