---
title: "Compilation Pipeline"
description: "How TiGrIS compiles an ONNX model into a .tgrs execution plan through seven stages, ending in a binary the C99 runtime runs directly on the target device."
sidebar:
  order: 310
slug: 0.8.0/architecture/pipeline
---

TiGrIS compiles an ONNX model into a `.tgrs` execution plan through seven stages. Each stage transforms the model representation, culminating in a binary that the C99 runtime executes directly on the target device.

## 1. Model Loading

Parses the ONNX protobuf and builds an `AnalyzedGraph`, the compiler's internal IR. This stage:

- Gives every graph input a concrete shape. An input named by `--input-shape` takes the shape given there; any other free dimension is bound to 1 and reported, since the plan is then sized for a value the compiler chose.
- Folds the shape subgraph. Exporters compute the classifier reshape from `Shape` at run time, which leaves everything downstream at unknown rank; evaluating that arithmetic to constants is what lets inference resolve the rest. A dimension that still cannot be resolved is an error.
- Runs ONNX shape inference to resolve the remaining tensor dimensions.
- Extracts weight tensors from ONNX initializers into `weight_data` entries.
- Topologically sorts the graph using DFS post-order with reversed child order. Reversing child order biases the schedule toward early tensor consumption, which reduces peak live memory.

```bash
tigris analyze model.onnx -m 256K -f 4M
```

The `analyze` command runs stages 1 through 4 and prints the results without producing a binary.

## 2. Normalization

Seventeen passes run in fixed order. Each pass simplifies the graph toward the runtime's operator set. Order matters because some passes depend on earlier ones having run.

### Pass order

1. **Constant op fold.** Converts ONNX `Constant` nodes into `weight_data` entries. Must run first so that downstream passes see weights, not op references.

2. **QDQ fold.** Extracts `QuantizeLinear` / `DequantizeLinear` pairs into `QuantParam` annotations on tensors. Keeps int8 weight values. Uses deferred cleanup to handle shared scale/zero-point constants. An ONNX quantizer states activations as uint8 by default; uint8 value `v` and int8 value `v - 128` denote the same real number under zero points that differ by the same 128, so the tensor is restated in the signed domain the kernels work in, stored data included. The dtype the model declares at each boundary is recorded in the plan, so folding the quantization into a boundary tensor does not change the interface the runtime presents.

3. **Matrix product relabel.** The fully-connected kernels index the weight as `W[oc * IC + ic]`, which is `Gemm` with `transB=1`. A `MatMul` states the same product with the weight the other way round, and so does a `Gemm` that leaves `transB` at its default, so a constant weight is transposed and the operator relabeled, with a per-channel quantization axis moving along with it. A product whose second operand is not a constant matrix is left alone, since there is no kernel for it.

4. **BatchNorm fold.** Absorbs BatchNorm parameters (gamma, beta, mean, variance) into the preceding Conv's weight and bias tensors. The BatchNorm node is removed.

5. **Constant bias fold.** A constant `Add` reading a quantized operator's output is requantized into that operator's int32 accumulator domain and becomes its bias. A quantizer that leaves a classifier bias unfused writes it as a float `Add` on the dequantized product, which is the one form the runtime cannot execute, since a constant operand carries no scale in the plan. The plan then stores the result in the product's int8 encoding; the model's declared float output is unaffected, since the boundary conversion reads it back in the dtype the model states. A float graph executes the `Add` as written and is left alone.

6. **SiLU decompose.** Rewrites `Silu` into `Sigmoid` + `Mul`. The runtime implements Sigmoid via an int8 lookup table.

7. **DepthwiseConv relabel.** Conv nodes where `group == C_in` are relabeled to `DepthwiseConv`. This selects the correct runtime kernel.

8. **Conv1D relabel.** Conv nodes with 1D kernels are relabeled to `Conv1D`.

9. **Clip to Relu6.** Replaces `Clip(min=0, max=6)` with `Relu6`.

10. **ReduceMean to GAP.** Replaces `ReduceMean(axes=[2,3])` with `GlobalAveragePool`.

11. **Shape op fold.** Removes any shape-computation chain (Shape, Gather, Unsqueeze, Concat) that reaches the IR still feeding a Reshape. The target shape is resolved statically and stored as an attribute. Model loading folds most of this arithmetic before the IR exists, so this pass catches what inference left behind.

12. **Resize scale extract.** Pulls integer scale factors from Resize's constant inputs and stores them as op attributes.

13. **Metadata operand strip.** Drops constant shape and bound operands from Clip, Pad, ReduceMean, Reshape, Resize, Squeeze and Unsqueeze. The passes above have lifted them into attributes and into the resolved output shape, and leaving them on the operator makes the emitter bind an index vector as its weight. An operand no pass resolved is left in place so validation reports the operator.

14. **Concat axis normalize.** Rewrites Concat's `axis` attribute from NCHW to NHWC layout convention (the runtime's native layout).

15. **Output transpose trim.** Removes trailing `Transpose` ops before model outputs. Host-side post-processing handles layout conversion.

16. **Activation fuse.** Absorbs `Relu` or `Relu6` following Conv, DepthwiseConv, Gemm, Conv1D, or Add into a `fused_activation` attribute on the preceding op, which then takes over the activation's output tensor and its quantization. Quantize is monotonic, so the plan encodes the clamp as a lower bound at the output zero point. An intermediate carrying its own scale is a quantization step of its own and is left in place, since folding past it would drop that rounding. Runs after all relabeling and rewiring is done.

17. **Unreferenced constant drop.** Forgets constants no remaining operator consumes. Folded subgraphs and absorbed activations leave their operands behind, and every `weight_data` entry is written into the plan blob. Runs last so it sees the final operator set.

## 3. Lifetime Analysis

For each activation tensor, the compiler computes:

- **birth_step**: the execution step of the op that produces the tensor.
- **death_step**: the execution step of the last op that consumes the tensor.

Constants and weights are excluded because they reside in flash and have no SRAM footprint. A tensor is live during steps `[birth_step, death_step]`.

## 4. Memory Timeline

An event-sweep algorithm walks execution steps in order. At each step:

1. Free tensors whose `death_step` equals this step.
2. Allocate tensors whose `birth_step` equals this step.

Freeing before allocating at the same step reduces measured peak. The sweep produces `peak_memory_bytes`, the maximum total size of simultaneously live tensors. This is the lower bound on SRAM required for single-stage execution.

## 5. Temporal Partitioning

When `peak_memory_bytes` exceeds the SRAM budget, the graph must be split into stages. Each stage executes independently, resetting the SRAM arena between stages.

The algorithm is a greedy forward pass:

1. Start a new stage.
2. Add ops sequentially. After each addition, run the memory timeline for the current stage.
3. If peak exceeds budget, cut before this op and start a new stage.
4. Repeat until all ops are assigned.

A stage's **inputs** are tensors consumed by ops in the stage but produced by an earlier stage. These must be spilled to PSRAM (slow buffer) at the producing stage's end and reloaded at the consuming stage's start.

A stage's **outputs** are tensors produced by ops in the stage but consumed by a later stage. These are spilled after the stage completes.

## 6. Spatial Partitioning

Stages that still exceed budget after temporal partitioning are candidates for spatial tiling. The compiler classifies each op:

| Category | Ops | Tileable? |
|---|---|---|
| Spatial | Conv, DepthwiseConv, MaxPool, AvgPool, GlobalAveragePool | Yes |
| Pointwise | Add, Mul, Relu, Relu6, Sigmoid, Concat, Resize, Pad, BatchNorm | Yes |
| Structural | Flatten, Reshape, Gemm, MatMul, Softmax, Transpose | No |

A stage is tileable only if it contains zero untileable ops and at most one spatial op.

### Receptive field calculation

Starting from the stage's output, walk spatial ops in reverse:

```
RF = 1
for each spatial op (reverse order):
    eff_kernel = kernel + (kernel - 1) * (dilation - 1)
    RF = RF + (eff_kernel - 1) * cumulative_stride
```

The **halo** is `RF - 1`, the number of overlap rows needed at each tile boundary.

### Tile height selection

```
tile_h = floor(budget * H / peak) - halo
```

The tiled peak memory is approximately:

```
tiled_peak ~= peak * (tile_h + halo) / H
```

Overhead from recomputing halo rows: `halo_bytes * (num_tiles - 1)`.

### Chain tiling

When two or more consecutive tileable stages each have exactly one output feeding the next stage's input, they form a **chain**. Chain tiling processes the entire chain per tile iteration, so intermediate activations between stages stay in SRAM and never touch PSRAM.

Chain conditions:
- Both stages are tileable (no untileable ops, at most one spatial op each).
- Each stage has exactly one output tensor consumed by exactly one next stage.
- No fan-out from intermediate tensors.

The compiler back-propagates tile heights from the last stage's output through each stage's spatial parameters (kernel, stride, dilation). A binary search finds the maximum tile height where all intermediate tile buffers plus aligned allocations fit within the SRAM budget.

Chain tiling gives the lowest overhead: only the first stage's input and the last stage's output touch slow memory.

## 7. Binary Emission

The final stage serializes the execution plan to the `.tgrs` binary format:

1. **Layout transpose.** All weight tensors are transposed from NCHW (ONNX convention) to NHWC (runtime convention). Depthwise weights go from `[C, KH, KW]` to `[KH, KW, C]`.

2. **Quantization scale computation** (int8 models only). For each quantized op, the effective scale is computed:
   ```
   effective_scale = (input_scale * weight_scale) / output_scale
   ```
   This is decomposed into a Q0.31 fixed-point multiplier and a right-shift value for integer-only arithmetic on the target.

3. **Weight compression.** Optionally, weights are compressed with LZ4, grouped by stage. The runtime decompresses into the `fast_reserved` prefix of the SRAM arena before executing the stage.

4. **Binary write.** Op descriptors, tensor metadata, tiling parameters, and weight blobs are written to the `.tgrs` file.

```bash
tigris compile model.onnx -m 256K -f 4M --xip -o model.tgrs
```
