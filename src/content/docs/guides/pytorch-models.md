---
title: "Compiling a PyTorch Model"
description: "Quantize a PyTorch model with PT2E and ExecuTorch's Cortex-M quantizer, export it to ONNX and compile it, with the two settings that decide whether the result is int8."
sidebar:
  order: 162
---

TiGrIS reads a PyTorch model through ONNX. There are two routes:

- Export the float model with `torch.onnx.export` and quantize the ONNX file
  with ONNX Runtime, as [Quantizing a Model to INT8](/guides/quantizing-to-int8/)
  describes.
- Quantize in PyTorch with PT2E, then export the quantized program to ONNX.
  This page covers that route with ExecuTorch's Cortex-M quantizer.

The exported ONNX carries int8 weights, per-channel for convolutions and
per-tensor for fully connected layers, int32 biases and
`QuantizeLinear`/`DequantizeLinear` pairs, which the compiler reads like any
other QDQ model. No conversion tool and no graph rewrite sits in between.

## Install

```bash
pip install torch==2.15.0 torchvision executorch==1.5.1 tosa-tools==2026.5.0 onnxscript tigris-ml
```

The `executorch` package does not declare `torch`, so it is listed here. Its
Cortex-M quantizer imports Arm's TOSA serializer, which the package installs
only with its Ethos-U and VGF extras; without `tosa-tools` the import fails with
`ModuleNotFoundError: No module named 'tosa_serializer'`. `onnxscript` is what
`torch.onnx.export(..., dynamo=True)` needs. These versions are the ones this
page was checked with.

## The recipe

```python
import torch
import torchvision
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer

model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()


def channels_last(x):
    return x.to(memory_format=torch.channels_last)


example = (channels_last(torch.randn(1, 3, 224, 224)),)

graph = export(model, example).module(check_guards=False)
graph = prepare_pt2e(graph, CortexMQuantizer())
for x in representative_samples:  # NCHW float tensors, preprocessed as deployed
    graph(channels_last(x))
graph = convert_pt2e(graph)

onnx_program = torch.onnx.export(export(graph, example), example, dynamo=True)
onnx_program.save("mobilenet_v2.int8.onnx")
```

Two settings decide whether this produces an int8 model:

- **Channels-last inputs.** The quantizer annotates the convolutions only when
  the example and calibration inputs are channels-last. With contiguous inputs
  it leaves them in float, and the compiler refuses the model. For the DS-CNN
  in the table below:

  ```text
  cannot compile: 2 unsupported operators
    node_conv2d (Conv: an int8 Conv needs an int32 bias)
    node_conv2d_8 (Conv: an int8 Conv needs an int32 bias)
  ```

- **`check_guards=False`.** By default `ExportedProgram.module()` adds a guard
  submodule, and the quantizer's preparation pass stops on it with
  `call_module is not supported`.

The ONNX keeps the model's NCHW input shape, and `analyze` prints the input in
that order, 1x3x224x224. The plan stores it channels-last, 1x224x224x3, as
[`tigris inspect`](/toolchain/inspect/) shows, and
[`tigris run`](/toolchain/run/) and the runtime take it in that order.

## Compile with `--int8-io`

```bash
tigris analyze mobilenet_v2.int8.onnx --int8-io -m 256K
tigris compile mobilenet_v2.int8.onnx --int8-io -m 256K -o mobilenet_v2.tgrs
```

```text
mobilenet_v2.int8.onnx   int8, 65 operators
  input    x                          1x3x224x224 int8 at scale 0.0311183, zero point -1
  output   dequantize_per_tensor_78   1x1000 int8 at scale 0.0586708, zero point 6
fits 256.00 KiB fast memory, 17 stages, 15 tiled

memory
  unscheduled         1.44 MiB
  largest tensor      1.15 MiB   1x96x112x112
  this plan         247.62 KiB   8.38 KiB headroom
  slow memory       343.00 KiB
  slow traffic        1.84 MiB   per inference: 784.98 KiB written, 1.08 MiB read
  smallest budget     9.38 KiB   9.34 KiB does not fit

flash
  plan   3.52 MiB   weights 3.38 MiB, overhead 141.69 KiB
```

The quantized program starts with a quantize and ends with a dequantize, so the
export declares float32 at both ends. Compiled without the option, the
application hands over float32 and the runtime quantizes it by ONNX's rule: a
division by the scale, rounded with ties to even. ExecuTorch's scalar Cortex-M
kernel multiplies by the reciprocal of the scale instead, and its MVE kernel
rounds that product differently again. Over 200 million values at 2,000 scales,
ONNX's rule and the scalar kernel gave a different int8 value 97 times. With
[`--int8-io`](/toolchain/compile/#model-interface) the application quantizes with the rule its own pipeline uses, and
the plan starts from the same int8 values. The output name comes from the
exporter, here the last dequantize node.

## Check the result

Compare on int8 inputs, so that every reference starts from the same values.
The reference is ExecuTorch's Cortex-M lowering of the same converted program,
run on the host implementations of its operators. These perform CMSIS-NN's
integer requantization, as the TiGrIS int8 kernels do. On 20 random int8 inputs
per model, each plan compiled untiled and at the smallest budget `analyze`
reports:

| Model | Weights | Smallest budget | Plan against the ExecuTorch program, CMSIS-NN pooling | Plan against the converted PT2E model |
|---|---|---|---|---|
| DS-CNN, 1x49x10 | random | 640 B | 0 of 240 values differ | 94 of 240, max 2 LSB |
| ResNet-8, 3x32x32 | random | 640 B | 0 of 200 | 73 of 200, max 2 LSB |
| MobileNetV2 0.35, 3x96x96 | random | 3.31 KiB | 0 of 20,000 | 18,694 of 20,000, max 37 LSB |
| MobileNetV2, 3x224x224 | ImageNet | 9.38 KiB | 0 of 20,000 | 12,794 of 20,000, max 5 LSB |

The untiled plan and the smallest-budget plan gave the same values on every
model. The reference computes each global average pool by CMSIS-NN's
`arm_avgpool_s8` rule, the integer window sum divided by the count and rounded
half away from zero. ExecuTorch's host implementation pools in float instead;
that changes the reference only for ResNet-8, where 11 of 200 values then
differ by 1 LSB.

The converted PT2E model computes the requantization in float, so it differs
from both the plan and the ExecuTorch program. That difference belongs to the
reference, not to the compiler. What quantization costs against the float
model is a separate measurement, as in
[Quantizing a Model to INT8](/guides/quantizing-to-int8/#check-the-result).

These are host runs. They do not cover ExecuTorch's runtime on a device.
