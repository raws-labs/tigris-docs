---
title: "Core Concepts"
description: "The key ideas behind how TiGrIS compiles and executes ML models on memory-constrained embedded devices."
sidebar:
  order: 130
---

The key ideas behind how TiGrIS compiles and executes models on memory-constrained devices.

## Execution plan (.tgrs)

A `.tgrs` file is a binary artifact containing the compiler-chosen operator schedule, stage and tiling strategy, tensor metadata, memory budget, and weights. It is not code, but a data structure that the C runtime interprets. One plan can run on any target that has a compatible kernel backend.

```
ds_cnn.tgrs (26 KB)
├── header        magic, version, memory requirements
├── stages[]      ordered list of execution stages
│   └── ops[]     operator descriptors (type, params, tensor refs)
├── tile plans    compiler-chosen streaming strategy and bounds
├── tensors[]     shapes, dtypes, and sizes
└── weights       model weights (f32 or int8), read via XIP or copied to SRAM
```

The plan fixes the execution order and bounds, but it does not contain absolute
runtime tensor addresses. The caller supplies fast and slow buffers, and the
runtime assigns aligned addresses with bounded arena allocators while following
the compiled schedule. It resets and, when needed, compacts those arenas. The
core runtime does not call a general-purpose heap allocator during inference,
but allocation operations can still return an error when the supplied buffers
are insufficient.

## Memory pools

TiGrIS manages three distinct memory regions:

| Pool | Speed | Typical size | Used for |
|------|-------|-------------|----------|
| **SRAM** | Fast | Kilobytes | Activation tensors, scratch buffers |
| **PSRAM** | Slow | Megabytes | Spill/reload between stages |
| **Flash** | Read-only | Megabytes | Weights (XIP), plan metadata |

The `-m` flag sets the SRAM activation budget used by the compiler's schedule
and tiling checks. The application must additionally account for compressed
weight workspace, backend scratch, alignment, and the required slow-buffer
capacity. For multi-stage models, PSRAM is required because intermediate
tensors spill there between stages. Pass a second `-m` value to set the PSRAM
budget. Models that fit in a single stage can run with SRAM only.

## Stages

The compiler splits the model graph into stages: groups of operators whose
modeled activation working set fits within the SRAM budget. Within a normal
stage, the runtime allocates tensors from the fast arena, reclaims dead tensors,
and can compact before retrying an allocation. It may overflow an allocation to
the slow arena where that execution path permits. Between stages, intermediate
tensors that are still needed later spill to PSRAM and reload when consumed. If
the available fast and slow capacity cannot satisfy an operation,
`tigris_run()` returns `TIGRIS_EXEC_ERR_MEM`.

A small model whose peak activation fits within the SRAM budget compiles into a single stage with no spilling. A larger model whose activations exceed the budget compiles into multiple stages with spill/reload between them.

## Tiling

The compiler uses three tiling strategies to fit models into limited SRAM:

**Temporal partitioning.** The compiler splits the model graph into stages whose combined activations fit in SRAM. Between stages, intermediate tensors spill to PSRAM and reload when needed. This is the coarsest strategy and applies automatically when the full model does not fit.

**Spatial tiling.** When a single stage's peak activation still exceeds the SRAM budget, the compiler tiles along the height dimension, processing horizontal strips instead of the full output tensor. Each strip only needs its receptive-field input rows plus halo overlap.

**Chain tiling.** When consecutive stages are all spatially tileable, the compiler fuses them into a chain. Tiles flow through the entire chain in SRAM, so intermediate tensors are never fully materialized and never hit PSRAM. This is the lowest-overhead strategy but requires every operator in the chain to support spatial decomposition.

For implementation details and formulas, see [Tiling](/architecture/tiling/).

## XIP (Execute in Place)

With `--xip` on an uncompressed plan, weights stay in flash and are read via memory-mapped I/O at inference time instead of being copied wholesale to SRAM. If weight compression is enabled, each stage's compressed weights are stored in flash and decompressed into the fast arena as that stage runs.

This is critical for larger models whose weights alone would exceed the SRAM budget. With XIP, SRAM is reserved exclusively for activations and scratch buffers.

The tradeoff is target-dependent latency: flash reads are slower than SRAM reads, while cache behavior depends on the device and the model's access pattern.

## Kernel backends

The execution plan specifies *what* to compute (operator type, tensor shapes, quantization parameters), but not *how*. The kernel backend provides the actual operator implementations.

| Backend | Target | Notes |
|---------|--------|-------|
| `reference` | Any | Portable C99. Correct but slow. Handles both float32 and int8 models. |
| `cmsis-nn` | Arm Cortex-M | Arm's optimized kernels for Cortex-M family (DSP on M4/M7, Helium on M55, plain C fallback elsewhere). |
| `esp-nn` | ESP32 family | Xtensa SIMD intrinsics, ~20x faster than reference. |

The same `.tgrs` plan works with any backend. Swapping `reference` for `esp-nn` on ESP32-S3 can yield 10-20x speedups with the same plan and identical results.

## Quantization

TiGrIS handles both float32 and int8 quantized models. For int8, the compiler folds ONNX QDQ (QuantizeLinear / DequantizeLinear) nodes at compile time, precomputing the fixed-point multipliers and right-shift values each operator needs.
Both symmetric and asymmetric quantization are supported. Quantize your model with any standard tool and TiGrIS handles the rest.

Int8 is recommended for embedded deployment: it cuts memory by 4x and unlocks SIMD kernel backends (ESP-NN, CMSIS-NN) that only support integer arithmetic.
