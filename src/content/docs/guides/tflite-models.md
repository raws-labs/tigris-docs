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
  and ARG_MIN outputs and index inputs are int32, and comparison results are
  bool.
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

- Custom operators do not convert, except `TFLite_Detection_PostProcess`.
- All activations int8, or all float32. Bool and int32 tensors appear only
  where the operators below produce or take them, and the detection results
  are float32 in an int8 model too. int16 activations and hybrid models (int8
  weights with float32 activations) do not convert.
- int8 activations quantized per tensor; weights constant, int8 weights
  symmetric, per channel or per tensor.
- Fused activations `NONE`, `RELU` or `RELU6`.
- Shapes fixed in the file. `SHAPE`, `ZEROS_LIKE`, `FILL` and
  `BROADCAST_ARGS` convert when everything they read is fixed in the file:
  they become constants. A `FILL` of a value computed at run time does not
  convert.

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
| `ADD`, `SUB`, `MUL`, `DIV`, `MAXIMUM`, `MINIMUM`, `SQUARED_DIFFERENCE` | operands broadcast from the right, as in TFLite; `ADD`, `SUB` and `MUL` also on int32, wrapping on overflow, without a fused activation |
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
| `MEAN` | constant axes, adjacent to each other |
| `SUM`, `REDUCE_MAX`, `REDUCE_MIN` | constant axes, adjacent to each other |
| `REDUCE_ALL` | bool input; constant adjacent axes |
| `CUMSUM` | one constant axis |
| `ARG_MAX`, `ARG_MIN` | int32 output, used as a model output or as the indices of the operators below |

### Recurrent

| Operator | Limits |
|---|---|
| `SVDF` | a bias, which TFLite Micro requires; float32 with Relu or Relu6, or int8 with int16 state and time weights |
| `UNIDIRECTIONAL_SEQUENCE_LSTM` | every gate; no peepholes, projection or layer normalization, which TFLite Micro does not run; tanh cell activation; float32, or int8 with a symmetric int16 cell and per-tensor weights |

### Control flow

| Operator | Limits |
|---|---|
| `IF` | a one-element bool condition; float32 or int32 operands; no variables or recurrent operators inside a branch |
| `WHILE` | float32 or int32 loop variables, from inputs or constants; a condition subgraph giving one bool; no variables or recurrent operators inside; a loop starting from a constant does not compress with `-c lz4` |

### Detection

| Operator | Limits |
|---|---|
| `TFLite_Detection_PostProcess` (custom) | float32 anchors in the file; box encodings `[1, boxes, 4]`, float32 or dequantized from int8; fast or regular non-max suppression; one class per detection in the fast form; its four float32 results are model outputs no operator reads |

The detections match TFLite Micro's bit for bit, including its order among
equal scores. In the fast form, rows past the detection count are zero, where
TFLite Micro leaves them unwritten. The operator runs untiled and holds working
memory of about 41 bytes per box during its stage.

### Comparison, logic and selection

| Operator | Limits |
|---|---|
| `EQUAL`, `NOT_EQUAL`, `LESS`, `LESS_EQUAL`, `GREATER`, `GREATER_EQUAL` | bool output; int8 input scales below 1; int32 inputs |
| `LOGICAL_AND`, `LOGICAL_OR`, `LOGICAL_NOT` | bool |
| `SELECT_V2` | int8 values and output share one quantization |
| `CAST` | bool to float32, or bool to int8 with scale 1 and zero point 0; int32 to float32 |

### Shape and data movement

| Operator | Limits |
|---|---|
| `RESHAPE`, `SQUEEZE`, `EXPAND_DIMS`, `TRANSPOSE` | |
| `CONCATENATION`, `SPLIT`, `SPLIT_V`, `PACK`, `UNPACK` | int8 inputs share the output's quantization |
| `PAD`, `PADV2`, `MIRROR_PAD` | |
| `SLICE`, `STRIDED_SLICE` | rank 4 at most; no ellipsis, new axes or offset; no zero stride; a dropped axis needs a positive stride |
| `GATHER`, `GATHER_ND`, `EMBEDDING_LOOKUP` | indices constant, or int32 from a model input, `ARG_MAX`/`ARG_MIN` or int32 `ADD`/`SUB`/`MUL`; an index outside its axis stops the run with an error |
| `REVERSE_V2` | adjacent axes |
| `BROADCAST_TO`, `DYNAMIC_UPDATE_SLICE` | start indices constant, or int32 from a model input or `ARG_MAX`/`ARG_MIN`; starts are clamped so the update fits, as in TFLite |
| `SPACE_TO_DEPTH`, `DEPTH_TO_SPACE`, `SPACE_TO_BATCH_ND`, `BATCH_TO_SPACE_ND` | rank-4 input |
| `RESIZE_BILINEAR`, `RESIZE_NEAREST_NEIGHBOR` | not `align_corners` and `half_pixel_centers` together |
| `QUANTIZE`, `DEQUANTIZE` | float32 or int8 to int8, int8 to float32 |

## Memory

Reductions, scans, `ARG_MAX`/`ARG_MIN` and the index-based data movement
operators (`GATHER`, `GATHER_ND`, `EMBEDDING_LOOKUP`, strided slices,
`MIRROR_PAD`, `REVERSE_V2`, `DYNAMIC_UPDATE_SLICE`) tile in bands along an axis
they do not reduce, index or reverse. Where no such axis exists, and for indices
computed at run time, they run on whole tensors, so their stage needs room for
its input and output at once. The same holds for a broadcast that is not a simple
repetition, such as `[1, 6, 6, 4]` against `[6, 1, 4]`, and for `SVDF` and
`UNIDIRECTIONAL_SEQUENCE_LSTM`. A `RESHAPE` tiles when each band of its input
rows maps onto whole rows of its output. Contiguous gathers, stride-1 slices and
the elementwise, convolution and pooling operators tile.
`tigris analyze` reports which stages tile and how much fast memory the plan
needs; see [When a Model Does Not Fit](/guides/model-does-not-fit/).

## State

Resource variables (`VAR_HANDLE`, `READ_VARIABLE`, `ASSIGN_VARIABLE`) in float32
models keep their values from one invocation to the next, as in TFLite Micro.
A `CALL_ONCE` subgraph that assigns constants sets their initial values. The
variable tensors of `SVDF` and `UNIDIRECTIONAL_SEQUENCE_LSTM` are kept the same
way, starting at zero, or at the zero point for an int8 hidden state. The plan
holds all of them in a state buffer the caller supplies next to the arenas:
`tigris_state_required()` gives its size, `tigris_state_init()` writes the
initial values, and `tigris_run_with_state()` runs on it. In Python,
`Session.reset_state()` starts the variables over. A `tigris codegen`
application allocates the buffer itself; the generated core states its size as
`<CORE>_STATE_BYTES` and runs through `<core>_run_with_state()`.

## Control flow

An `IF` runs the subgraph its condition selects, and a `WHILE` runs its body for
as long as its condition subgraph gives true, as in TFLite Micro. Each subgraph is
planned as a model of its own under the same memory budget and runs as its own
stages; the operands are copied in and the results copied out. A `WHILE` has no
iteration limit, as in TFLite Micro. Control flow may nest inside a branch or loop
body, up to the runtime's `TIGRIS_MAX_SUBGRAPH_DEPTH` (4 by default) operators
deep and `TIGRIS_MAX_SUBGRAPHS` (16 by default) graphs in all.

The converter keeps the boundary of an int8 `IF` or `WHILE` in float32, which a
plan cannot run alongside int8 operators, so control flow converts in float32
models.

## Not yet supported

int8 resource variables are refused by name.
