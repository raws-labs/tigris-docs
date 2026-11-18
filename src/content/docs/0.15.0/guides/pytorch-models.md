---
title: "Compiling a PyTorch Model"
description: "Quantize a PyTorch model with PT2E and ExecuTorch's Cortex-M quantizer, then compile it for the TiGrIS runtime or band ExecuTorch's own Cortex-M program with tigris export, with RAM measured on a Cortex-M7."
sidebar:
  order: 162
slug: 0.15.0/guides/pytorch-models
---

This page quantizes a PyTorch model with PT2E and ExecuTorch's Cortex-M
quantizer. The quantized program then goes one of two ways:

- Compile it for the TiGrIS runtime, from the ONNX file `torch.onnx.export`
  writes or from the `.pt2` archive `torch.export.save` writes.
- Keep ExecuTorch: lower the program for Cortex-M with ExecuTorch and let
  [`tigris export`](/0.15.0/toolchain/export/#executorch-programs) rewrite the `.pte`
  into row bands, so the same ExecuTorch build runs it in less RAM.

[On a device](#on-a-device) compares the two on a Cortex-M7. A float model can
also be exported with `torch.onnx.export` and quantized as an ONNX file with
ONNX Runtime, as [Quantizing a Model to INT8](/0.15.0/guides/quantizing-to-int8/)
describes.

The exported ONNX carries int8 weights, per-channel for convolutions and
per-tensor for fully connected layers, int32 biases and
`QuantizeLinear`/`DequantizeLinear` pairs, which the compiler reads like any
other QDQ model. No conversion tool and no graph rewrite sits in between.

## Install

```bash
pip install executorch==1.6.0 tosa-tools==2026.5.0 onnxscript tigris-ml --extra-index-url https://download.pytorch.org/whl/cpu
```

PyTorch's index, added as an extra index, gives the CPU-only builds of
`executorch` 1.6.0, the `torch` 2.15.0 it pins and `torchao` 0.18.0, without the
CUDA runtime the PyPI wheels install on Linux; the other packages come from
PyPI. ExecuTorch's Cortex-M quantizer imports Arm's TOSA serializer, which the
package installs only with its Ethos-U and VGF extras; without `tosa-tools` the
import fails with `ModuleNotFoundError: No module named 'tosa_serializer'`.
`onnxscript` is what `torch.onnx.export(..., dynamo=True)` needs. These versions
are the ones this page was checked with. The recipe below takes MobileNetV2 from
`torchvision`; add it to the same line to run the recipe as written.

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
[`tigris inspect`](/0.15.0/toolchain/inspect/) shows, and
[`tigris run`](/0.15.0/toolchain/run/) and the runtime take it in that order.

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
[`--int8-io`](/0.15.0/toolchain/compile/#model-interface) the application quantizes with the rule its own pipeline uses, and
the plan starts from the same int8 values. The output name comes from the
exporter, here the last dequantize node.

### From the `.pt2`

The converted program can also go to the compiler as it is. Save it, and
`tigris` exports it to ONNX itself:

```python
torch.export.save(export(graph, example), "mobilenet_v2.pt2")
```

```bash
tigris compile mobilenet_v2.pt2 --int8-io -m 256K -o mobilenet_v2.tgrs
```

For a `.pt2`, `analyze` and `compile` add the
[`ExecuTorch model memory`](/0.15.0/toolchain/analyze/#pytorch-programs) row: the
planned buffer of ExecuTorch's Cortex-M7 lowering of the same program, measured
when `executorch` is installed.

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
[Quantizing a Model to INT8](/0.15.0/guides/quantizing-to-int8/#check-the-result).

These are host runs; [On a device](#on-a-device) runs ExecuTorch and the
TiGrIS runtime on a Cortex-M7.

## Band an ExecuTorch program

To stay on ExecuTorch, lower the converted program for the core of your
firmware, write the `.pte`, and rewrite it with `tigris export`. Continuing the
recipe:

```python
from executorch.backends.cortex_m.edge_compile_config import cortex_m_edge_compile_config
from executorch.backends.cortex_m.passes.cortex_m_pass_manager import CortexMPassManager
from executorch.backends.cortex_m.target_config import CortexM, CortexMTargetConfig
from executorch.exir import ExecutorchBackendConfig, to_edge
from executorch.exir.passes.quantize_io_pass import QuantizeInputs, QuantizeOutputs

edge = to_edge(export(graph, example), compile_config=cortex_m_edge_compile_config())
edge = edge.transform(passes=[QuantizeInputs(edge, [0]), QuantizeOutputs(edge, [0])])
edge = edge.transform(CortexMPassManager(target_config=CortexMTargetConfig(cpu=CortexM.M7)))
with open("mobilenet_v2.pte", "wb") as f:
    f.write(edge.to_executorch(ExecutorchBackendConfig()).buffer)
```

`QuantizeInputs` and `QuantizeOutputs` make the program take and return int8
tensors, as `--int8-io` does for a plan; `tigris export` rewrites programs with
int8 inputs and outputs. `CortexMTargetConfig` names the core, and ExecuTorch
sizes each CMSIS-NN scratch buffer for that core's kernels. Export keeps every
band on those kernels.

For the MobileNetV2 0.35 at 96x96 of the table above, this lowering writes a
program of 64 kernel calls, here `mbv2.pte`:

```bash
tigris export mbv2.pte --header mbv2_pools.h
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

The firmware loads `mbv2_export.pte` as it loaded the original, with its
planned buffer, method allocator and temp allocator sized from `mbv2_pools.h`.
[`tigris export`](/0.15.0/toolchain/export/#executorch-programs) explains each row, the
limits it takes and the header.

[`tigris run`](/0.15.0/toolchain/run/#executorch-programs) runs a program on the host,
on ExecuTorch built from a 1.6 checkout. Run both programs on the same input and
compare the outputs byte for byte:

```bash
tigris run mbv2.pte --executorch ~/executorch --input x.bin --output a.bin
tigris run mbv2_export.pte --executorch ~/executorch --input x.bin --output b.bin
cmp a.bin b.bin
```

`x.bin` holds the int8 input in memory order, 1x96x96x3, as
`tigris inspect mbv2.pte` shows it; a `.npy` of the 1x3x96x96 tensor the
program takes in PyTorch works too. For DS-CNN, ResNet-8, MobileNetV2 0.35 and
MobileNetV2 at 224x224 of the table above, lowered for Cortex-M7, both programs
write the same bytes on each of 5 random inputs.

## On a device

DS-CNN, ResNet-8 and MobileNetV2 0.35 at 96x96 from the table above ran on a
NUCLEO-H753ZI (Cortex-M7 at 480 MHz) three ways: on ExecuTorch 1.6.0, the
program as lowered and the program `tigris export` wrote, each lowered for
Cortex-M7 with int8 input and output; and on the TiGrIS runtime, a plan compiled
from the same `.pt2` with `--int8-io` and generated with
`codegen --backend cmsis-nn`. All link CMSIS-NN v8.0.0. All three gave the same
int8 outputs on every input, 0 of 240, 0 of 200 and 0 of 1,000 bytes differing,
and the same outputs as ExecuTorch's runtime on the host.

Model memory, the activations and kernel scratch of one inference. For
ExecuTorch, the planned buffer: the program figure, which the device writes in
full. For TiGrIS, the fast arena the CMSIS-NN backend requires, the plan's
activations plus CMSIS-NN scratch, and the slow arena: the plan figures the
firmware's arenas were sized to.

| Model | ExecuTorch as lowered | ExecuTorch after `tigris export` | TiGrIS runtime |
|---|---|---|---|
| DS-CNN | 16,000 B | 12,160 B, 1.32x less | 16,256 B at `-m 16K` |
| ResNet-8 | 49,728 B | 24,192 B, 2.06x less | 51,456 B at `-m 48K` |
| MobileNetV2 0.35, 96x96 | 138,240 B | 38,400 B, 3.60x less | 56,192 B at `-m 20K`: 23,936 B fast, 32,256 B slow |

Runtime bookkeeping, what each runtime keeps besides the model memory. For
ExecuTorch, the method allocator (measured on the device, equal to the program
figure), the temp allocator pool (the program figure, the capacity the runtime
needs to resolve kernel arguments from it) and the runtime's statics (linker
map), which include its kernel registry: 2,000 slots, 24,000 B, by default, or
612 B with ExecuTorch built with `MAX_KERNEL_NUM=51`, the 51 kernels these
firmwares register (28 prim ops and the 23 cortex_m operators). For TiGrIS,
the tensor table, the executor workspace and the runtime's statics (linker
map).

| Model | ExecuTorch registry | ExecuTorch as lowered | ExecuTorch after `tigris export` | TiGrIS runtime |
|---|---|---|---|---|
| DS-CNN | 2,000 slots | 30,112 B | 32,952 B | 279 B |
| DS-CNN | 51 slots | 6,724 B | 9,564 B | 279 B |
| ResNet-8 | 2,000 slots | 30,576 B | 35,024 B | 327 B |
| ResNet-8 | 51 slots | 7,188 B | 11,636 B | 327 B |
| MobileNetV2 0.35, 96x96 | 2,000 slots | 49,576 B | 55,236 B | 1,867 B |
| MobileNetV2 0.35, 96x96 | 51 slots | 26,188 B | 31,848 B | 1,867 B |

Firmware RAM total: `.data` and `.bss` from the linker map plus the heap used
and the stack high-water mark measured on the device. It is the model memory,
the runtime bookkeeping and the measuring firmware's own C library, heap and
stack, 3,688 to 3,700 B with ExecuTorch and 3,921 to 4,317 B with TiGrIS.

| Model | ExecuTorch registry | ExecuTorch as lowered | ExecuTorch after `tigris export` | TiGrIS runtime |
|---|---|---|---|---|
| DS-CNN | 2,000 slots | 49,800 B | 48,800 B | 20,456 B |
| DS-CNN | 51 slots | 26,412 B | 25,412 B | 20,456 B |
| ResNet-8 | 2,000 slots | 84,004 B | 62,916 B | 55,720 B |
| ResNet-8 | 51 slots | 60,616 B | 39,528 B | 55,720 B |
| MobileNetV2 0.35, 96x96 | 2,000 slots | 191,504 B | 97,324 B | 62,376 B |
| MobileNetV2 0.35, 96x96 | 51 slots | 168,116 B | 73,936 B | 62,376 B |

ExecuTorch after `tigris export` holds the least model memory on all three
models. The firmware total adds each runtime's bookkeeping, which is where the
two runtimes differ most. With ExecuTorch's registry sized to its kernels,
ExecuTorch after `tigris export` needs the least RAM in total for ResNet-8,
39,528 B against 55,720 B for TiGrIS, and TiGrIS needs the least for DS-CNN,
20,456 B against 25,412 B, and for MobileNetV2, 62,376 B against 73,936 B. With
the default registry, TiGrIS needs the least for all three.

For each model the TiGrIS row is the plan with the smallest firmware total among
the budgets measured: DS-CNN and ResNet-8 run as one stage, MobileNetV2 runs
tiled. The H753's SRAM holds both TiGrIS arenas; on a part with a small fast
SRAM next to a larger memory, the tiled plan keeps 23,936 B in the fast one.
[`tigris export`](/0.15.0/toolchain/export/#on-a-device) gives the method allocator and
the time per inference for the two ExecuTorch programs.
