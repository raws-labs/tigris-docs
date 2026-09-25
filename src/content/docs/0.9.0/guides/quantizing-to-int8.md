---
title: Quantizing a Model to INT8
description: Produce an ONNX model TiGrIS compiles to an int8 plan, the two
  quantizer settings that decide whether it works, and how to check the result.
sidebar:
  order: 160
slug: 0.9.0/guides/quantizing-to-int8
---

Quantizing is the lever with the best ratio of effort to result: activations
become a quarter of their size, so the SRAM a model needs falls with them, and
so does the flash its weights occupy. On MobileNetV2 at 224x224 it is the
difference between a model that needs 250 KiB of SRAM and one that needs
62.50 KiB.

TiGrIS reads ONNX QDQ models directly, so quantizing is something you do to the
model before the compiler sees it. This page covers what the compiler requires,
the settings that decide whether a quantized model compiles at all, and how to
check the result.

## What TiGrIS requires

**QDQ format.** The quantizer must write `QuantizeLinear` and
`DequantizeLinear` nodes around float operators. That is `QuantFormat.QDQ` in
ONNX Runtime.

**Static quantization.** Scales and zero points have to be in the file, because
the plan is built ahead of time and the arena is sized from them.

**A fully quantized graph.** A plan executes on one dtype, so an operator the
quantizer skipped leaves the graph mixed and the compiler refuses it.

Nothing else. Activations stated as uint8, which the ONNX Runtime quantizer
emits by default, are handled: uint8 value `v` and int8 value `v - 128` denote
the same real number under zero points that differ by the same 128, so the
compiler restates them in the signed domain the kernels work in.

## The recipe

```python
import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationMethod, QuantFormat, QuantType, quantize_static,
)

model = onnx.load("model.onnx")
graph_input = model.graph.input[0]
shape = [d.dim_value or 1 for d in graph_input.type.tensor_type.shape.dim]


class Calibration:
    """Feeds representative samples. Random data is fine for a mechanical
    check and wrong for anything you intend to deploy."""

    def __init__(self, samples):
        self.samples = iter(samples)

    def get_next(self):
        return next(self.samples, None)


quantize_static(
    "model.onnx",
    "model.int8.onnx",
    Calibration(representative_samples),
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    calibrate_method=CalibrationMethod.MinMax,
)
```

Then compile as usual:

```bash
tigris analyze model.int8.onnx -m 64K
```

```
warning: input axis 0 (batch_size) is unset; using 1 (--input-shape overrides)
╭────────────────────── TiGrIS - mobilenetv2.qdq-int8 ───────────────────────╮
│ Operators            65                                                    │
│ Tensors              174 (66 activations)                                  │
│ Peak memory (naive)  1.44 MiB                                              │
│ Largest tensor       1x96x112x112 (1.15 MiB)                               │
│ Quantization         INT8 (QDQ)                                            │
│ Input                input 1x3x224x224 float32, stored as int8 at scale    │
│                      0.0413059, zero point -2                              │
│ Output               output_QuantizeLinear_Input 1x1000 float32, stored as │
│                      int8 at scale 0.0587218, zero point 6                 │
╰────────────────────────────────────────────────────────────────────────────╯
```

Note the Input and Output rows. The model declares float32 at both ends and the
plan executes on int8, so the runtime converts at the boundary: an application
hands over and reads back float32 either way. Quantizing does not change the
interface your code talks to.

## Per-channel or per-tensor

`per_channel=True` gives each output channel its own weight scale. It costs
plan size and buys accuracy. Measured on MobileNetV2, calibrated identically
both ways:

| Weights | Plan | Quantization error against the float model |
|---|---|---|
| per-tensor | 3.39 MiB | max 49.1 LSB, mean 45.9 |
| per-channel | 3.81 MiB | max 17.5 LSB, mean 13.8 |

Roughly a third of the error for 0.42 MiB. Take per-channel unless flash is the
binding constraint. The activation memory does not change either way: both have
a 1.44 MiB naive peak and a 62.50 KiB floor, because only the weights are
affected.

One catch: `axis` on `DequantizeLinear` arrived in opset 13, so per-channel
quantization of an opset-12 model produces a file ONNX Runtime refuses to load,
with `Unrecognized attribute: axis for operator DequantizeLinear`. Convert
first:

```python
from onnx import version_converter
onnx.save(version_converter.convert_version(model, 13), "model.opset13.onnx")
```

## Settings that do not work, and how they fail

**`QuantFormat.QOperator`** writes `QLinearConv`, `QLinearAdd`, `QGemm` and
friends instead of QDQ pairs. TiGrIS has none of those operators, and shape
inference cannot see through them, so the failure arrives as a shape error
rather than a format one:

```
Error: Tensor 'output' has unknown rank; TiGrIS requires concrete deployment shapes
```

If you see that on a model you just quantized, check the format before
anything else.

**`quantize_dynamic`** computes activation scales at run time, so the file
carries none. The compiler reports the weight dtype it cannot place:

```
Error: Cannot compile a plan with unsupported tensor dtypes: unsupported
activation tensor dtype(s): ONNX dtype 2 (317_quantized, 317_zero_point, ...)
```

**A partially quantized graph** fails with both dtypes named:

```
Error: Cannot compile a plan with unsupported tensor dtypes: mixed activation
dtypes cannot use a graph-wide runtime dispatcher: float32 (...), int8 (...)
```

Read the float32 list. It names the operators the quantizer skipped, which is
usually an operator it does not support, and the fix is upstream in the
quantizer rather than in the compiler.

## Check the result

Quantization changes what the model computes, so measure rather than assume.
Compare three things on the same inputs:

1. The float model under ONNX Runtime, which is the reference.
2. The quantized model under ONNX Runtime, which tells you what quantization
   cost you. This is the number that decides whether the model is still good
   enough, and it has nothing to do with TiGrIS.
3. The compiled plan, which tells you whether TiGrIS reproduces the quantized
   model.

On MobileNetV2 the third comparison lands well inside the second:

```
TiGrIS vs ONNX Runtime int8    : max  6.0 LSB, mean  5.0 LSB
ONNX Runtime int8 vs float     : max 17.5 LSB, mean 13.8 LSB
```

A few LSB between TiGrIS and ONNX Runtime is expected and is not drift. Both
requantize with integer arithmetic but break rounding ties differently, and a
one-LSB difference early in a network is amplified by any later layer whose
effective scale is large.

If the third comparison is large while the second is small, that is a compiler
problem and worth reporting. If the second is large, the model needs better
calibration data or per-channel weights, and no amount of compiler work will
recover it.

## Calibration data

Everything above is mechanical. The one judgement call is the calibration set,
which decides the scales and therefore the accuracy. It has to look like what
the model will see when deployed: the same preprocessing, the same value range,
the same kind of content. A few dozen representative samples is usually enough,
and random noise is not representative of anything, however convenient it is
for checking that the pipeline runs.
