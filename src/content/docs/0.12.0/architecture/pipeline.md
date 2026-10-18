---
title: "Compilation Pipeline"
description: "How TiGrIS compiles an ONNX or TFLite model into a .tgrs execution plan through seven stages, ending in a binary the C99 runtime runs directly on the target device."
sidebar:
  order: 310
slug: 0.12.0/architecture/pipeline
---

TiGrIS compiles an ONNX model into a `.tgrs` execution plan through seven stages. Each stage transforms the model representation, culminating in a binary that the C99 runtime executes directly on the target device.

A TFLite model enters the same pipeline: the frontend reads the FlatBuffer and states each operator as an ONNX operator or a pattern of them, or as a compiler operator that carries TFLite's exact semantics where ONNX has no equivalent. Constants and static shape arithmetic are folded first, and an operator that cannot convert is refused with its index, name and reason.

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

`tigris analyze` runs the whole pipeline in memory and reports what it found without writing a binary.

## 2. Normalization

Normalization passes run in fixed order. Each pass simplifies the graph toward the runtime's operator set. Order matters because some passes depend on earlier ones having run.

### Selected passes

1. **Constant op fold.** Converts ONNX `Constant` nodes into `weight_data` entries. Runs before passes that need constant values rather than op references.

2. **QDQ fold.** Extracts `QuantizeLinear` / `DequantizeLinear` pairs into `QuantParam` annotations on tensors. Keeps int8 weight values. Uses deferred cleanup to handle shared scale/zero-point constants. An ONNX quantizer states activations as uint8 by default; uint8 value `v` and int8 value `v - 128` denote the same real number under zero points that differ by the same 128, so the tensor is restated in the signed domain the kernels work in, stored data included. Float sections retain explicit Dequantize and Quantize operators, with ordinary float32 tensors between them. The dtype the model declares at each boundary is recorded in the plan, so folding the quantization into a boundary tensor does not change the interface the runtime presents.

3. **Matrix product relabel.** The fully-connected kernels index the weight as `W[oc * IC + ic]`, which is `Gemm` with `transB=1`. A `MatMul` states the same product with the weight the other way round, and so does a `Gemm` that leaves `transB` at its default, so a constant weight is transposed and the operator relabeled, with a per-channel quantization axis moving along with it. A product with a dynamic second operand remains MatMul and is validated against that kernel's shape limits.

4. **BatchNorm fold.** Absorbs BatchNorm parameters (gamma, beta, mean, variance) into the preceding Conv's weight and bias tensors. The BatchNorm node is removed.

5. **Constant operands and bias.** Uniform float constants in encoded Add and Mul operators are restated as int8 constants. A float channel-bias Add can fold into a preceding float product. An Add after Dequantize instead remains a float operation, after the quantized product has rounded and clipped.

6. **SiLU decompose.** Rewrites `Silu` into `Sigmoid` + `Mul`. The int8 Sigmoid kernel uses a lookup table; the float32 kernel uses float arithmetic.

7. **DepthwiseConv relabel.** Conv nodes where `group == C_in` are relabeled to `DepthwiseConv`. This selects the correct runtime kernel.

8. **Conv1D relabel.** Conv nodes with 1D kernels are relabeled to `Conv1D`.

9. **Clip to Relu6.** Replaces `Clip(min=0, max=6)` with `Relu6`.

10. **ReduceMean to GAP.** Replaces `ReduceMean(axes=[2,3])` with `GlobalAveragePool`.

11. **Shape op fold.** Removes any shape-computation chain (Shape, Gather, Unsqueeze, Concat) that reaches the IR still feeding a Reshape. The target shape is resolved statically and stored as an attribute. Model loading folds most of this arithmetic before the IR exists, so this pass catches what inference left behind.

12. **Resize scale extract.** Pulls integer scale factors from Resize's constant inputs and stores them as op attributes.

13. **Metadata operand strip.** Drops constant shape and bound operands from Clip, Pad, ReduceMean, Reshape, Resize, Squeeze and Unsqueeze. The passes above have lifted them into attributes and into the resolved output shape, and leaving them on the operator makes the emitter bind an index vector as its weight. An operand no pass resolved is left in place so validation reports the operator.

14. **Transpose validation.** Validates explicit and default permutations. The plan retains them as operator attributes, including before model outputs.

15. **Activation fuse.** Absorbs `Relu` or `Relu6` following Conv, DepthwiseConv, Gemm, Conv1D, or Add into a `fused_activation` attribute on the preceding op, which then takes over the activation's output tensor and its quantization. Quantize is monotonic, so the plan encodes the clamp as a lower bound at the output zero point. An intermediate carrying its own scale is a quantization step of its own and is left in place, since folding past it would drop that rounding. Runs after all relabeling and rewiring is done.

16. **Concat axis normalize.** Rewrites Concat's `axis` attribute into the tensor's stored axis order, after layouts are assigned.

17. **Unreferenced constant drop.** Forgets constants no remaining operator consumes. Folded subgraphs and absorbed activations leave their operands behind, and every `weight_data` entry is written into the plan blob. Runs after operator rewrites so it sees the final operator set.

## 3. Lifetime Analysis

For each activation tensor, the compiler computes:

- **birth_step**: the execution step of the op that produces the tensor.
- **death_step**: the execution step of the last op that consumes the tensor.

Constants and weights are excluded because they reside in flash and have no SRAM footprint. A tensor is live during steps `[birth_step, death_step]`.

## 4. Memory Timeline

An event-sweep algorithm walks execution steps in order. At each step:

1. Free tensors whose last consumer finished at the previous step.
2. Allocate tensors whose `birth_step` equals this step.

An operator's inputs and outputs are live together during its execution. The sweep produces `peak_memory_bytes`, the maximum total size of simultaneously live tensors, including alignment and shared reshape storage.

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

| Category | Examples | How it tiles |
|---|---|---|
| Spatial | Conv, DepthwiseConv, MaxPool, AveragePool | Height bands or height and width, with halo |
| Pointwise | Add, Mul, Relu, Gelu, Quantize, Dequantize | Shape-preserving tiles, with each tensor's own dtype |
| Rows and independent axes | Gemm, MatMul, Softmax, Reshape, Flatten, Transpose, reductions and gathers | Where a band preserves the operator's semantics |
| Separate solves | ConvTranspose, Resize, GlobalAveragePool | Expanded output tiles, resampled height bands, or accumulated input bands |

A stage tiles only when every operator supports its selected axis and band geometry. The [tiling table](/0.12.0/architecture/tiling/#op-tileability) lists the conditions.

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
- Both stages support tiling along the chain's height axis.
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
