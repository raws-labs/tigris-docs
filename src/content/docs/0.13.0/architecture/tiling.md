---
title: "Tiling"
description: "Tiling is how TiGrIS fits large models into small SRAM: processing feature maps in horizontal strips cuts peak activation memory in proportion to strip height."
sidebar:
  order: 320
slug: 0.13.0/architecture/tiling
---

Tiling is TiGrIS's core technique for fitting large models into small SRAM. A Conv2D at position (x, y) reads only a local neighborhood of the input. By processing feature maps in horizontal strips, peak activation memory drops in proportion to the strip height.

## Why tiling

A 96x96 feature map with 64 channels occupies 576 KiB in int8, already larger than the total SRAM on most Cortex-M and ESP32 targets. Without tiling, the entire feature map must be live in SRAM simultaneously. With tiling, only a strip of height `tile_h + halo` needs to be resident.

The tradeoff: halo rows at tile boundaries are recomputed, adding a small amount of redundant work. In practice the overhead is low, typically under 5% for models with moderate receptive fields.

## Execution strategies

TiGrIS applies three strategies. A stage that fits the budget runs normally and a stage that does not is tiled. Runs of consecutive tileable stages then become chains, which can also take in stages that would fit on their own.

### 1. Normal execution

The stage fits in the SRAM budget without tiling. Load inputs from slow memory, execute all ops, spill outputs to slow memory, reset the arena. A plan that is a single such stage reads its inputs in place in the fast arena and leaves its outputs there, under the condition given for [`tigris_input_alloc`](/0.13.0/runtime/api-reference/#tigris_input_alloc).

This is the fastest path with no tile iteration overhead and no halo recomputation.

### 2. Spatial tiling

The stage exceeds the budget, but all ops in the stage are tileable. The output feature map is split into horizontal strips. Each strip is computed independently.

```text
Input feature map             Output feature map
+------------------+          +------------------+
|   halo overlap   |--+       |                  |
+------------------+  +------>|   Tile 0 out     |
|   Tile 0 rows    |--+       |                  |
+------------------+          +------------------+
|   halo overlap   |--+       |                  |
+------------------+  +------>|   Tile 1 out     |
|   Tile 1 rows    |--+       |                  |
+------------------+          +------------------+
|   halo overlap   |--+       |                  |
+------------------+  +------>|   Tile 2 out     |
|   Tile 2 rows    |--+       |                  |
+------------------+          +------------------+
```
Each input tile includes halo rows from the neighboring tile. The halo provides the spatial context that Conv/Pool kernels need at boundaries. Halo regions overlap between adjacent tiles.

A rank-4 stage whose operators also cut along the width can run as a grid of 2D tiles, each with its halo in both directions. The compiler takes the 2D tile when no strip fits the budget, or when the grid loads fewer input bytes than the strips. `tigris analyze -v` prints such a stage as, for example, `48 tiles of 7 x 52, halo 2` (height x width).

The tile loop for a single stage:

```text
for tile_idx in 0..num_tiles:
    1. Compute input row range: [start - halo_top, end + halo_bottom]
    2. Load input strip from slow memory into arena
    3. Execute all ops in stage on the strip
    4. Write output strip to slow memory
    5. Reset arena
```

### 3. Chain tiling

Two or more consecutive tileable stages form a chain when each stage has exactly one output and the next stage reads it. Chain tiling fuses the tile loops so all stages in the chain execute per tile iteration, and intermediate activations never leave SRAM.

```text
                     SRAM arena
                +--------------------+
                |  Input tile        | <-- from slow
                |  (stage 0 in)      |
                +--------------------+
 Stage 0 ops -->|  Intermediate tile | <-- stays in SRAM
                |                    |
                +--------------------+
 Stage 1 ops -->|  Output tile       | --> to slow
                |  (stage 1 out)     |
                +--------------------+
```

Only the first input and last output touch slow memory. Intermediates stay in the arena per tile.

Chain tiling gives the lowest spill overhead. For deep networks where many stages are individually tileable, chains can span many stages, eliminating most PSRAM traffic.

## Op tileability

A stage tiles when every operator in it supports the same way of cutting its
tensors. Operators the compiler has not audited for a tiling contract run whole,
so their stage needs its inputs and outputs in fast memory at once. These rules
also apply to float32 operators inside int8 plans, with each tensor's own element
size included in the memory bound.

| How it tiles | Operators | Notes |
|---|---|---|
| Height, or height and width, with a halo | Conv, Conv1D, DepthwiseConv, MaxPool, AveragePool, L2Pool | Contribute to the receptive field; a wide rank-4 layer can tile in both height and width |
| Height and width of the expanded output | ConvTranspose | One ConvTranspose with optional unary pointwise operators; excluded from chains |
| With the tile, no halo | Add, Sub, Mul, Div, Max, Min, activations (Relu, Relu6, Sigmoid, Tanh, HardSwish, Gelu, LeakyRelu, PRelu, Elu), Quantize, Dequantize, elementwise math (Exp, Log, Sqrt, Abs, Floor and the rest), comparisons, logic, Where, Cast, Concat along the channel axis | Shape-preserving; int32 arithmetic, comparisons and casts run whole |
| With the tile, whole rows | Softmax, LogSoftmax, L2Normalization, LayerNormalization | They normalize along the last stored axis, which no tile cuts |
| Height bands of the output | Resize, ResizeLinear | One resize with compatible pointwise operators; the source band follows the resampling coordinates |
| Bands of input rows, accumulated | GlobalAveragePool | One partial sum per channel is carried between bands; below one row, groups of channels over runs of positions |
| Bands of matrix rows | Gemm, MatMul, Reshape, Flatten, with elementwise operators | The second operand of a matrix product is read whole |
| Bands along an independent axis | ReduceMean, ReduceMax, ReduceMin, ReduceSum, ReduceAll, CumSum, ArgMax, ArgMin, Gather, GatherND, EmbeddingLookup, StridedSlice, Pad, MirrorPad, ReverseV2, DynamicUpdateSlice, Split, Concat, Transpose, broadcasting binary operators | Along an axis the operator does not reduce, index, reverse or move; a constant table can tile independent batches of runtime indices |
| Bands of a reshape | Reshape | When each band of input rows maps onto whole output rows |
| Whole | TFLite_Detection_PostProcess, SVDF and LSTM beyond their batch axis | The stage needs its inputs and outputs at once |

Every operator in a tiled or chained stage also has to read the band the stage
holds. A branch off the input of a strided convolution, for example, reads rows
the tile no longer holds, so its stage runs untiled and stays out of chains. The
runtime checks the same rule when it loads a plan and refuses a chain or tile
record that breaks it with `TIGRIS_ERR_CHAIN_BAND` or `TIGRIS_ERR_TILE_BAND`;
such a plan from an earlier release has to be recompiled.

If and While hold nothing in fast memory themselves: each subgraph is planned and
tiled as a model of its own under the same budget.

`tigris analyze -v` shows how each stage of a given model tiles.

## Receptive field and halo

The receptive field determines how many input rows are needed to produce one output row across the entire stage.

### Calculation

Walk spatial ops in execution order, accumulating the receptive field:

```
RF = 1
cumulative_stride = 1

for each spatial op in execution order:
    eff_kernel = kernel_h + (kernel_h - 1) * (dilation_h - 1)
    RF = RF + (eff_kernel - 1) * cumulative_stride
    cumulative_stride = cumulative_stride * stride_h
```

The halo is:

```
halo = RF - 1
```

### Examples

| Op sequence | Kernel | Stride | Dilation | RF | Halo |
|---|---|---|---|---|---|
| 3x3 Conv | 3 | 1 | 1 | 3 | 2 |
| 5x5 Conv | 5 | 1 | 1 | 5 | 4 |
| 3x3 Conv, stride 2 | 3 | 2 | 1 | 3 | 2 |
| 3x3 DWConv + 3x3 Conv | 3+3 | 1+1 | 1+1 | 5 | 4 |
| 3x3 dilated Conv (d=2) | 3 | 1 | 2 | 5 | 4 |

## Tile height selection

The tile height counts rows of the stage output, of height `H`. For a band of
`rows` output rows, `band_bytes(rows)` is the stage's working set as the runtime
allocates it: every stage input over the input rows the band reads, plus every
operator output over its rows, each rounded up to the allocation alignment.

```
tile_h     = largest rows <= H with band_bytes(rows) <= budget   (binary search)
num_tiles  = ceil(H / tile_h)
tile_h     = smallest rows >= ceil(H / num_tiles) that reads no more input rows
tiled_peak = band_bytes(tile_h)
```

The second step keeps the tile count and evens out the bands, so the last tile
is not a short remainder. Redundant input reads:

```
overhead_bytes = halo * input_row_bytes * (num_tiles - 1)
```

When even a one-row band exceeds the budget and no 2D tile fits either, the
stage does not fit, and `tigris analyze` names it.

## Chain tiling details

### Chain detection

The compiler scans consecutive stages looking for chains:

1. Stage N is tileable along the height, and so is Stage N+1: convolutions, pools and shape-preserving operators that read the band the stage holds, with no resize.
2. Every input and output of both stages is a rank-4 feature map.
3. Stage N has exactly one output tensor, Stage N+1 reads it, and Stage N+1 reads nothing from outside the chain.
4. No tensor a stage before the last produces is read outside the chain or is a model output.

Chains are extended greedily: if Stage N+2 also qualifies, it joins the chain. A later stage may also read a tensor an earlier member produced, as a residual Add does, when no convolution or pool in between changes its rows. A chain that does not fit the budget as a whole is split into consecutive chains that do.

### Height back-propagation

Starting from the last stage's output tile height, the compiler walks backward through the chain. Each spatial op's kernel, stride, and dilation determine the required input tile height:

```
input_tile_h = output_tile_h * stride + max((kernel - 1) * dilation + 1 - stride, 0)
```

All intermediate tile sizes are computed this way.

### Binary search for tile height

The compiler binary-searches for the maximum output tile height such that the sum of all intermediate tile sizes (with alignment padding) fits within the SRAM budget. Larger tile heights mean fewer iterations and less halo overhead.

### Memory layout during chain execution

Arena during one chain tile iteration:

```text
Low addr                            High addr
+----------+----------+----------+----------+
| Stage 0  | Stage 0  | Stage 1  | Scratch  |
| input    | output / | output   | buffers  |
| tile     | Stage 1  | tile     |          |
|          | input    |          |          |
+----------+----------+----------+----------+
```

Nothing is freed within a tile iteration: the input tile and every operator output tile stay allocated, and that sum is what the binary search fits to the budget. Only the final output tile is spilled to slow memory, and the arena resets before the next tile.

When a chain member has a halo, the compiler marks the chain line-buffered. Each intermediate buffer is then allocated once, at its largest tile height, and keeps its overlap rows from one tile to the next, so an interior tile computes only its new rows instead of recomputing the halo. A stage that holds a Concat recomputes them.
