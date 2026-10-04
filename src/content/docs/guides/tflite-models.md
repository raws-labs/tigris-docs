---
title: "Compiling a TFLite Model"
description: "Compile a .tflite file directly, what the compiled plan keeps from the model, and which TFLite operators convert with which limits."
sidebar:
  order: 165
---

TiGrIS compiles a `.tflite` file as it is. `tigris analyze` and `tigris compile`
accept it in place of an ONNX model; there is no separate conversion step.

```bash
tigris analyze model.tflite -m 256K
tigris compile model.tflite -m 256K -o model.tgrs
```

The model is read into the same pipeline an ONNX model goes through, so
tiling, memory budgets and code generation behave the same way. A model that
uses an operator or a form the compiler cannot run is refused before
compiling, with the operator's index, its TFLite name and the reason:

```text
operator 12 STRIDED_SLICE: a zero stride
```

## What the plan keeps from the model

- **Interface dtypes.** An int8 model takes int8 inputs and returns int8
  outputs, as TFLite hands them over; a float32 model stays float32. ARG_MAX
  and ARG_MIN outputs are int32, and comparison results are bool.
- **Interface shapes.** Inputs and outputs keep TFLite's channels-last axis
  order, for example `1x96x96x3` for an image.
- **Results.** Every operator below is checked against TFLite Micro with a
  committed one-operator corpus, int8 and float32, compared byte for byte. The
  few cases where TFLite Micro and TFLite's reference kernels disagree are
  listed with the corpus in the compiler repository. The MLPerf Tiny reference
  models (keyword spotting, visual wake words, image classification and
  anomaly detection) produce TFLite Micro's int8 outputs byte for byte, tiled
  and untiled.

## Model requirements

- One subgraph. Control flow (`WHILE`, `IF`) and custom operators do not
  convert.
- All activations int8, or all float32. Bool and int32 tensors appear only
  where the operators below produce or take them. int16 activations and hybrid
  models (int8 weights with float32 activations) do not convert.
- int8 activations quantized per tensor; weights constant, int8 weights
  symmetric, per channel or per tensor.
- Fused activations `NONE`, `RELU` or `RELU6`.
- Shapes fixed in the file. Operators that compute shapes at run time
  (`SHAPE`, `FILL` and similar) do not convert; the TFLite converter folds
  them away in models with static shapes.

## Supported operators

int8 and float32 unless the table says otherwise. Operators that take
indices, axes, padding or bounds need them as constants in the file.

### Convolution, pooling and matrix products

| Operator | Limits |
|---|---|
| `CONV_2D`, `DEPTHWISE_CONV_2D`, `TRANSPOSE_CONV` | |
| `FULLY_CONNECTED` | shuffled weight format refused |
| `BATCH_MATMUL` | operands of equal rank |
| `AVERAGE_POOL_2D`, `MAX_POOL_2D` | |
| `L2_POOL_2D` | float32 only, as in TFLite Micro |

### Elementwise

| Operator | Limits |
|---|---|
| `ADD`, `SUB`, `MUL`, `DIV`, `MAXIMUM`, `MINIMUM`, `SQUARED_DIFFERENCE` | operands broadcast from the right, as in TFLite |
| `ABS`, `RSQRT` | |
| `NEG`, `EXP`, `LOG`, `SQRT`, `SQUARE`, `FLOOR`, `CEIL`, `ROUND`, `SIN`, `COS`, `FLOOR_DIV`, `FLOOR_MOD` | float32 only, as in TFLite Micro |
| `ADD_N` | int8 inputs share one quantization; refused where the int8 accumulator could overflow, from 16 inputs up |
| `RELU`, `RELU6`, `LOGISTIC`, `TANH`, `HARD_SWISH`, `LEAKY_RELU`, `ELU`, `PRELU` | `PRELU` slope constant |

### Normalization

| Operator | Limits |
|---|---|
| `SOFTMAX` | over the last axis; beta 1 |
| `LOG_SOFTMAX` | over the last axis |
| `L2_NORMALIZATION` | over the last axis; no fused activation |

### Reductions and scans

| Operator | Limits |
|---|---|
| `MEAN` | constant axes |
| `SUM`, `REDUCE_MAX`, `REDUCE_MIN` | constant axes, adjacent to each other |
| `REDUCE_ALL` | bool input; constant adjacent axes |
| `CUMSUM` | one constant axis |
| `ARG_MAX`, `ARG_MIN` | int32 output that is a model output and feeds no other operator |

### Comparison, logic and selection

| Operator | Limits |
|---|---|
| `EQUAL`, `NOT_EQUAL`, `LESS`, `LESS_EQUAL`, `GREATER`, `GREATER_EQUAL` | bool output; int8 input scales below 1 |
| `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT` | bool |
| `SELECT_V2` | int8 values and output share one quantization |
| `CAST` | bool to float32, or bool to int8 with scale 1 and zero point 0 |

### Shape and data movement

| Operator | Limits |
|---|---|
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS`, `TRANSPOSE` | |
| `CONCATENATION`, `SPLIT`, `SPLIT_V`, `PACK`, `UNPACK` | int8 inputs share the output's quantization |
| `PAD`, `PADV2`, `MIRROR_PAD` | |
| `SLICE`, `STRIDED_SLICE` | rank 4 at most; no ellipsis, new axes or offset; no zero stride; a dropped axis needs a positive stride |
| `GATHER`, `GATHER_ND`, `EMBEDDING_LOOKUP` | constant indices |
| `REVERSE_V2` | adjacent axes |
| `BROADCAST_TO`, `DYNAMIC_UPDATE_SLICE` | `DYNAMIC_UPDATE_SLICE` with constant start indices |
| `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND` | rank-4 input |
| `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR` | not `align_corners` and `half_pixel_centers` together |
| `QUANTIZE`, `DEQUANTIZE` | float32 or int8 to int8, int8 to float32 |

## Memory

Reductions, scans, `ARG_MAX`/`ARG_MIN` and the index-based data movement
operators (`GATHER`, `GATHER_ND`, `EMBEDDING_LOOKUP`, strided slices,
`MIRROR_PAD`, `REVERSE_V2`, `DYNAMIC_UPDATE_SLICE`) run on whole tensors, so
their stage needs room for its input and output at once, as does a broadcast
that is not a simple repetition, such as `[1, 6, 6, 4]` against `[6, 1, 4]`.
Contiguous gathers,
stride-1 slices and the elementwise, convolution and pooling operators tile.
`tigris analyze` reports which stages tile and how much fast memory the plan
needs; see [When a Model Does Not Fit](/guides/model-does-not-fit/).

## Not yet supported

`SVDF` and `UNIDIRECTIONAL_SEQUENCE_LSTM` keep state between invocations,
which plans do not hold yet. Models that use them are refused by name.
