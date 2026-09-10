---
title: "Memory Model"
description: "TiGrIS's three-region memory model — a compile-time execution schedule over caller-provided arenas allocated and compacted by the runtime."
sidebar:
  order: 330
---

TiGrIS uses a three-region memory model designed for embedded devices with heterogeneous memory: small fast SRAM, large slow PSRAM, and read-only flash. The compiler fixes the operator schedule, stage and tile strategy, and activation-memory bounds. The runtime then assigns addresses from caller-provided fast and slow arenas with bounded bump allocation, reset, and compaction. The core executor does not call a general-purpose heap allocator during inference; it can still report out-of-memory when either arena is too small.

PSRAM is required for any model that compiles to more than one stage. Intermediate tensors spill from SRAM to PSRAM between stages because flash is read-only and cannot serve as spill storage. Models that fit in a single stage can run with SRAM only.

## Memory regions

### SRAM (fast arena)

The primary working memory for inference. All activation tensors during op execution live here.

- Bump allocator, pointer advances forward with alignment padding.
- Reset per stage or per tile iteration.
- Size set by the `-m` flag (e.g., `-m 256K`).
- An optional `fast_reserved` prefix at the start of the arena holds decompressed weight blocks and survives arena resets.

### PSRAM (slow buffer)

External RAM for inter-stage tensor storage. Required for any multi-stage plan. Present on targets like ESP32-S3 (2-16 MB PSRAM depending on variant).

- Bump allocator, persistent across stages.
- Compacted after each stage: dead tensors are reclaimed and live tensors are shifted down.
- Stage inputs are loaded from here into the fast arena. Stage outputs are spilled from the fast arena to here.
- Model inputs are pre-allocated here by the caller before inference.
- Model outputs remain here after inference completes.

### Flash (read-only, XIP)

Model weights are stored in flash and can be accessed via execute-in-place (XIP) memory mapping. With `--xip` on an uncompressed plan, the runtime reads weights directly from their flash addresses instead of copying the full weight blob to RAM. With weight compression enabled, stage weight blocks are still stored in flash but are decompressed into the fast arena as each stage runs.

```bash
tigris compile model.onnx -m 256K -f 4M --xip -o model.tgrs
```

The `-f` flag validates that the compiled plan (weights + metadata) fits within the flash budget.

### Executor working storage

Executor bookkeeping is not a tensor-memory region and is not included in the
compiled SRAM budget. New integrations provide one caller-owned
`tigris_executor_workspace_t` to `tigris_run_with_workspace()`. Its bounded
size is derived from the runtime's compile-time plan limits; query
`sizeof(tigris_executor_workspace_t)` or
`tigris_executor_workspace_size()` for the current build. Keep it in static,
global, or heap-backed storage unless its full size is intentionally included
in the task-stack budget. The compatibility `tigris_run()` entry point instead
uses one process-global, non-re-entrant workspace.

## Arena allocation

The fast arena is a contiguous block of SRAM. Allocation works as follows:

```text
Arena layout:

Low addr                                 High addr
┌───────────┬──────────┬──────────┬──────────────┐
│ reserved  │ Tensor A │ Tensor B │ free space   │
│ (weights) │ (aligned)│ (aligned)│              │
└───────────┴──────────┴──────────┴──────────────┘
             ^                     ^
             arena_base            bump_ptr
```

- The bump pointer starts after `fast_reserved` and advances with each allocation.
- Each allocation is padded to `TIGRIS_TENSOR_ALIGN` bytes.
- On arena reset (between stages or tiles), the bump pointer rewinds to `arena_base`, but `fast_reserved` is preserved.

### OOM handling

For a normal stage, if an activation allocation exceeds the remaining fast
arena space:

1. **Compact**: Scan for tensors whose last consumer has run. Reclaim their space by shifting subsequent live tensors down. Retry the allocation.
2. **Overflow**: If compaction is insufficient, try the slow pool (PSRAM). A successful fallback incurs latency from slow memory access.

This is a bounded fallback, not a guarantee of success. Slow-arena exhaustion,
tiled or chain workspace exhaustion, compressed-weight workspace exhaustion,
and backend scratch failures are reported to the caller. The core runtime does
not fall back to an unbounded heap allocation.

## Tensor lifetime management

### Intra-stage (within a stage)

The executor derives each tensor's last consumer from the compiled operator
order. After that consumer executes, the tensor pointer is invalidated and its
arena range becomes eligible for reclamation during compaction.

### Inter-stage (between stages)

- **Stage outputs**: After a stage completes, tensors consumed by later stages are spilled (copied) to the slow buffer.
- **Stage inputs**: Before a stage executes, tensors produced by earlier stages are loaded from the slow buffer into the fast arena.
- The fast arena is reset between stages, so all intra-stage temporaries are reclaimed automatically.

### Special tensors

- **Model inputs**: Allocated in the slow buffer by the caller before invoking the executor. The first stage loads them.
- **Model outputs**: Left in the slow buffer after the last stage. The caller reads them directly.
- **Constants/weights**: Reside in flash. Not allocated in either arena. Accessed by pointer.

## Alignment

Tensor addresses are aligned to `TIGRIS_TENSOR_ALIGN` bytes. The value is auto-detected at compile time based on the target architecture.

| Architecture | TIGRIS_TENSOR_ALIGN | Notes |
|---|---|---|
| Xtensa (ESP32-S3) | 8 bytes | Matches `ee.vld.l.64.ip` load alignment |
| AArch64 (Cortex-A, RPi) | 16 bytes | NEON 128-bit vector alignment |
| x86_64 | 32 bytes | AVX2 256-bit vector alignment |
| Arm Cortex-M with DSP (`__ARM_FEATURE_DSP`) | 16 bytes | CMSIS-NN DSP kernels and wide loads |
| Arm Cortex-M without DSP | 4 bytes | Portable fallback alignment |

Override with `-DTIGRIS_TENSOR_ALIGN=N` at build time. Backend-specific scratch buffers (e.g., ESP-NN SIMD) may require stricter alignment and are handled separately by the backend adapter.

The `.tgrs` buffer has a related base-address contract. Optimized CMSIS-NN
kernels can read uncompressed XIP weights directly from offsets relative to the
plan base, so the plan buffer itself must be aligned to
`TIGRIS_TENSOR_ALIGN`—16 bytes on DSP-enabled Cortex-M. Embedded arrays,
linker sections, custom flash mappings, and loaded buffers must all preserve
that base alignment.

## Measured fast-arena peak

`tigris_mem_t.fast_peak` is the high-water mark of `fast_used` since the last
`tigris_mem_init()`. It includes ordinary fast-arena allocations and direct
compressed-weight bumps made by the executor, so inspect it after
an executor call to observe the measured core-arena demand. Account for
compressed-weight overhead when comparing it with the compiler's activation
bound. Backend workspace reserved outside the usable arena must be counted
separately.

## Memory budget sizing

The `-m` flag sets the SRAM activation budget used during compilation. Choosing the right budget involves a tradeoff:

| Budget | Stages | Spill traffic | Tiling overhead | Inference speed |
|---|---|---|---|---|
| Large | Fewer | Less | Less | Faster |
| Small | More | More | More | Slower, but frees SRAM |

A larger budget means fewer temporal partitions, less data movement between SRAM and PSRAM, and fewer tiled iterations. A smaller budget frees SRAM for other uses like DMA buffers, RTOS stacks, or backend scratch buffers.

### Backend scratch buffers

Accelerated kernel backends need setup before inference. Call exactly one
backend preparation function after `tigris_mem_init()` and before the first
executor call, and treat a nonzero result as fatal:

- `tigris_esp_nn_prepare(&plan, &mem)` scans the plan and obtains aligned
  ESP-NN workspace through the platform allocator during initialization.
- `tigris_cmsis_nn_prepare(&plan, &mem)` reserves one 16-byte-aligned CMSIS-NN
  scratch region from the top of the fast buffer and reduces `mem.fast_size`.

The reference float32 and int8 backends do not require a preparation call.
After successful preparation, neither accelerated dispatch path allocates
model-dependent workspace during inference.

```text
Fast-buffer layout after CMSIS-NN preparation:

Low addr                                   High addr
┌───────────┬──────────────────┬──────────────────┐
│ reserved  │ activation space │ backend scratch  │
│ (weights) │ (bump allocator) │ (SIMD scratch)   │
└───────────┴──────────────────┴──────────────────┘
```

`tigris_cmsis_nn_scratch_required()` returns the exact aligned CMSIS-NN
reservation, including vendor scratch and scalar-to-per-channel quantization
expansion. `tigris_cmsis_nn_fast_arena_required()` combines it with the core
activation and compressed-weight requirement, including alignment between the
core arena and adapter reservation, so a tight static fast buffer can be sized
before `tigris_mem_init()`. The buffer base must already satisfy
`TIGRIS_TENSOR_ALIGN`. Both helpers return `UINT32_MAX` for invalid or
unrepresentable input.

The sweet spot depends on the model and backend. Example from a large int8 model on ESP32-S3:

| Configuration | Arena | Scratch | Inference latency |
|---|---|---|---|
| Minimal scratch | 232 KB | 23 KB | ~40 s |
| Balanced | 64 KB | 180 KB | ~11 s |

The balanced configuration is roughly 4x faster because ESP-NN's SIMD kernels use the scratch space effectively, and the additional tiling overhead from a smaller arena is more than offset by faster kernel execution.

### Sizing guideline

1. Run `tigris analyze model.onnx -m <budget>` with different budgets to see the stage count and tiling plan.
2. Reserve enough SRAM for the backend's scratch requirements.
3. Start with the smallest budget that keeps the stage count reasonable (under ~20 stages for most models), then increase if latency is too high.
