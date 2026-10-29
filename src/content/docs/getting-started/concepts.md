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
ds_cnn_i8.tgrs (30 KiB)
+-- header        magic, version, memory requirements
+-- stages[]      ordered list of execution stages
|   +-- ops[]     operator descriptors (type, params, tensor refs)
+-- tile plans    compiler-chosen streaming strategy and bounds
+-- tensors[]     shapes, dtypes, and sizes
+-- weights       model weights (f32 or int8), read in place or decompressed to SRAM
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

The `-m` flag bounds the fast arena used by the compiler's schedule and tiling
checks, including any compressed-weight reservation. The application also
accounts for backend scratch, alignment, and the required slow-buffer capacity.
Multi-stage models need writable slow storage for intermediate tensors, such as
PSRAM or another SRAM bank. Pass a second `-m` value to bound that pool.
A single-stage model can use SRAM for both its working set and its input and
output storage.

## Stages

The compiler splits the model graph into stages: groups of operators whose
modeled activation working set fits within the SRAM budget. Within a normal
stage, the runtime allocates tensors from the fast arena, reclaims dead tensors,
and can compact before retrying an allocation. It may overflow an allocation to
the slow arena where that execution path permits. Between stages, intermediate
tensors that are still needed later spill to PSRAM and reload when consumed. If
the available fast and slow capacity cannot satisfy an operation,
`tigris_run()` returns `TIGRIS_EXEC_ERR_MEM`. A plan of schema 11 or later states
the fast memory its largest stage or chain holds, and its run refuses a fast arena
below `tigris_fast_arena_required()` with that error before any stage runs.

A small model whose peak activation fits within the SRAM budget compiles into a single stage with no spilling. A larger model whose activations exceed the budget compiles into multiple stages with spill/reload between them.

## Tiling

The compiler uses three tiling strategies to fit models into limited SRAM:

**Temporal partitioning.** The compiler splits the model graph into stages whose combined activations fit in SRAM. Between stages, intermediate tensors spill to PSRAM and reload when needed. This is the coarsest strategy and applies automatically when the full model does not fit.

**Spatial tiling.** When a single stage's peak activation still exceeds the SRAM budget, the compiler tiles along the height dimension, processing horizontal strips instead of the full output tensor. Each strip only needs its receptive-field input rows plus halo overlap. When no strip fits, or when tiles of rows and columns load fewer input bytes than the strips, the compiler tiles height and width together.

**Chain tiling.** When consecutive stages are all spatially tileable, the compiler fuses them into a chain. Tiles flow through the entire chain in SRAM, so intermediate tensors are never fully materialized and never hit PSRAM. This is the lowest-overhead strategy but requires every operator in the chain to support spatial decomposition.

For implementation details and formulas, see [Tiling](/architecture/tiling/).

## XIP (Execute in Place)

`--xip` marks weights for execute-in-place deployment. The runtime reads uncompressed weights from the loaded plan buffer, so placing that buffer in memory-mapped flash keeps the weights out of SRAM. The flag does not map flash or copy weights; the application owns plan placement. Compressed weights are decompressed into the fast arena as each stage runs.

This is critical for larger models whose weights alone would exceed the SRAM budget. With XIP, SRAM is reserved exclusively for activations and scratch buffers.

The tradeoff is target-dependent latency: flash reads are slower than SRAM reads, while cache behavior depends on the device and the model's access pattern.

## Kernel backends

The execution plan specifies *what* to compute (operator type, tensor shapes, quantization parameters), but not *how*. The kernel backend provides the actual operator implementations.

| Backend | Target | Notes |
|---------|--------|-------|
| `reference` | Any | Portable C. Correct but slow. Handles both float32 and int8 models. |
| `cmsis-nn` | Arm Cortex-M | Arm's optimized kernels for Cortex-M family (DSP on M4/M7, Helium on M55, plain C fallback elsewhere). |
| `esp-nn` | ESP32 family | Espressif's optimized int8 kernels for the ESP32 family. |

The same `.tgrs` plan works with any backend, and the int8 results of the
accelerated kernels match the reference kernels. On an ESP32-S3, the int8 DS-CNN
keyword-spotting plan runs in 20.31 ms with ESP-NN and 565.3 ms on the reference
kernels, from the same plan
([tigris-bench](https://github.com/raws-labs/tigris-bench), ESP32-S3 latency results).

## Quantization

TiGrIS handles both float32 and int8 quantized models. An int8 TFLite model is read with the quantization the converter wrote. For an ONNX model, the compiler folds QDQ (QuantizeLinear / DequantizeLinear) nodes at compile time, precomputing the fixed-point multipliers and right-shift values each operator needs. Activations stated as uint8, which is what an ONNX quantizer emits by default, are restated as int8 at compile time; that is exact, since the zero point moves by the same 128. An int8 plan can include float32 operators after Dequantize, up to Quantize or a float model output. The interface it presents is the one the model file declares: the plan records that dtype per input and output, and `tigris_input_write()` and `tigris_output_read()` convert across the boundary. Hand over and read back exactly what the model file states, whatever encoding the plan executes on.
Both symmetric and asymmetric quantization are supported. Quantize your model with any standard tool and TiGrIS handles the rest.

Int8 is recommended for embedded deployment: it cuts memory by 4x and unlocks SIMD kernel backends (ESP-NN, CMSIS-NN) that only support integer arithmetic.
