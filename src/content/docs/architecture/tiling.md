---
title: "Tiling"
description: "Tiling is how TiGrIS fits large models into small SRAM: processing feature maps in horizontal strips cuts peak activation memory in proportion to strip height."
sidebar:
  order: 320
---

Tiling is TiGrIS's core technique for fitting large models into small SRAM. A Conv2D at position (x, y) reads only a local neighborhood of the input. By processing feature maps in horizontal strips, peak activation memory drops in proportion to the strip height.

## Why tiling

A 96x96 feature map with 64 channels occupies 576 KB in int8, already larger than the total SRAM on most Cortex-M and ESP32 targets. Without tiling, the entire feature map must be live in SRAM simultaneously. With tiling, only a strip of height `tile_h + halo` needs to be resident.

The tradeoff: halo rows at tile boundaries are recomputed, adding a small amount of redundant work. In practice the overhead is low, typically under 5% for models with moderate receptive fields.

## Execution strategies

TiGrIS applies three strategies in preference order. The compiler selects the simplest strategy that fits the budget.

### 1. Normal execution

The stage fits in the SRAM budget without tiling. Load inputs from slow memory, execute all ops, spill outputs to slow memory, reset the arena.

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

Two or more consecutive tileable stages form a chain when each stage has exactly one output feeding the next stage's sole input. Chain tiling fuses the tile loops so all stages in the chain execute per tile iteration, and intermediate activations never leave SRAM.

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

The compiler classifies every op to determine whether a stage can be tiled.

| Category | Operators | Tileable | Notes |
|---|---|---|---|
| Spatial | Conv, DepthwiseConv, MaxPool, AvgPool, GlobalAveragePool | Yes | Contribute to receptive field and halo |
| Pointwise | Add, Sub, Mul, Relu, Relu6, Sigmoid, HardSwish | Yes | No spatial extent, tile passthrough |
| Channel-wise | Concat, BatchNorm | Yes | Operate along channel axis only |
| Resample | Resize, Pad | Yes | Scale factors adjust tile geometry |
| Structural | Flatten, Reshape, Transpose | No | Require full spatial extent |
| Dense | Gemm, MatMul, Softmax | No | Operate on entire flattened vector |

A stage is tileable if:
- It contains **zero** untileable ops.
- It contains **at most one** spatial op (Conv, DepthwiseConv, Pool).

Stages with multiple spatial ops (e.g., a backbone block with Conv + MaxPool) fall back to normal execution with slow-memory overflow if they exceed the budget.

## Receptive field and halo

The receptive field determines how many input rows are needed to produce one output row across the entire stage.

### Calculation

Walk spatial ops in reverse execution order, accumulating the receptive field:

```
RF = 1
cumulative_stride = 1

for each spatial op in reverse order:
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

Given the stage's peak memory `peak` at full height `H`, the target tile height is:

```
tile_h = floor(budget * H / peak) - halo
```

The number of tiles:

```
num_tiles = ceil(H / tile_h)
```

Tiled peak memory:

```
tiled_peak ~= peak * (tile_h + halo) / H
```

Redundant computation overhead:

```
overhead_bytes = halo * input_width * channels * (num_tiles - 1)
```

The compiler verifies `tiled_peak <= budget` and adjusts `tile_h` downward if needed.

## Chain tiling details

### Chain detection

The compiler scans consecutive stages looking for chains:

1. Stage N is tileable.
2. Stage N+1 is tileable.
3. Stage N has exactly one output tensor.
4. That tensor is consumed only by Stage N+1 (no fan-out).
5. The intermediate tensor has spatial dimensions (not a 1D vector).

Chains are extended greedily: if Stage N+2 also qualifies, it joins the chain.

### Height back-propagation

Starting from the last stage's output tile height, the compiler walks backward through the chain. Each spatial op's kernel, stride, and dilation determine the required input tile height:

```
input_tile_h = output_tile_h * stride + (kernel - 1) * dilation
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

After Stage 0 finishes, its input tile is freed. Stage 1 reuses that arena space. Only the final output tile is spilled to slow memory.
