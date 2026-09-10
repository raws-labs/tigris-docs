---
title: API Reference
description: Core execution, memory, loader, and accelerated-backend integration
  APIs for the TiGrIS runtime.
sidebar:
  order: 420
slug: 0.6.0/runtime/api-reference
---

Complete C API reference for the TiGrIS runtime. The core API is defined in
`tigris.h`, `tigris_loader.h`, `tigris_mem.h`, and `tigris_executor.h`.
Reference and accelerated dispatch functions are declared in the corresponding
`tigris_kernels*.h` headers.

## Loader

### tigris\_plan\_load

```c
tigris_error_t tigris_plan_load(
    const uint8_t *buf,
    uint32_t       buf_len,
    tigris_plan_t *out_plan);
```

Parse a `.tgrs` binary plan from a buffer. Zero-copy, zero-alloc: all pointers in `out_plan` point directly into `buf`. The caller must keep `buf` alive for the lifetime of `out_plan`.

For uncompressed XIP plans used with optimized CMSIS-NN kernels, `buf` must be
aligned to `TIGRIS_TENSOR_ALIGN`. That is 16 bytes when
`__ARM_FEATURE_DSP` is defined on Cortex-M. Weight offsets are aligned relative
to the plan base, so an unaligned base would make those direct weight pointers
unaligned too. Embedded arrays, linker sections, custom flash mappings, and
loaded buffers must provide the required base alignment.

### Tensor alignment

Unless `TIGRIS_TENSOR_ALIGN` is set by the build, the runtime selects this
minimum tensor alignment from the target architecture:

| Target condition | Default alignment (bytes) |
|------------------|--------------------------:|
| `__ARM_FEATURE_SVE` | 64 |
| `__aarch64__` | 16 |
| `__x86_64__` | 32 |
| `__riscv_vector` | 16 |
| `__XTENSA__` | 8 |
| `__ARM_NEON` | 16 |
| `__ARM_FEATURE_DSP` | 16 |
| `otherwise` | 4 |

An override supplied as `-DTIGRIS_TENSOR_ALIGN=N` must be a power of two.

### tigris\_error\_str

```c
const char *tigris_error_str(tigris_error_t err);
```

Return a human-readable string for a loader error code.

## Memory Manager

### tigris\_mem\_init

```c
tigris_mem_error_t tigris_mem_init(
    tigris_mem_t *mem,
    void        **tensor_ptrs,
    uint16_t      num_tensors,
    void         *fast_buf,
    uint32_t      fast_size,
    void         *slow_buf,
    uint32_t      slow_size);
```

Initialize the memory manager. Zeroes all tensor pointers and aligns the
initial fast and slow bump offsets to `TIGRIS_TENSOR_ALIGN`. The buffers and
pointer table remain caller-owned. The function returns
`TIGRIS_MEM_ERR_OOM` if an unaligned buffer is too small to reach its first
aligned address.

### tigris\_mem\_alloc\_fast

```c
tigris_mem_error_t tigris_mem_alloc_fast(
    tigris_mem_t *mem,
    uint16_t      tensor_idx,
    uint32_t      size_bytes);
```

Bump-allocate in the fast arena. Sets `tensor_ptrs[tensor_idx]` to the allocated address.

### tigris\_mem\_alloc\_slow

```c
tigris_mem_error_t tigris_mem_alloc_slow(
    tigris_mem_t *mem,
    uint16_t      tensor_idx,
    uint32_t      size_bytes);
```

Bump-allocate in the slow buffer. Sets `tensor_ptrs[tensor_idx]` to the allocated address.

### tigris\_mem\_load

```c
tigris_mem_error_t tigris_mem_load(
    tigris_mem_t *mem,
    uint16_t      tensor_idx,
    uint32_t      size_bytes);
```

Allocate in the fast arena and copy data from the tensor's current (slow) location. Updates `tensor_ptrs[tensor_idx]` to point to the fast copy.

### tigris\_mem\_spill

```c
tigris_mem_error_t tigris_mem_spill(
    tigris_mem_t *mem,
    uint16_t      tensor_idx,
    uint32_t      size_bytes);
```

Allocate in the slow buffer and copy data from the tensor's current (fast) location. Updates `tensor_ptrs[tensor_idx]` to point to the slow copy.

### tigris\_mem\_load\_tile

```c
tigris_mem_error_t tigris_mem_load_tile(
    tigris_mem_t        *mem,
    const tigris_plan_t *plan,
    uint16_t             tensor_idx,
    int32_t              h_start,
    int32_t              h_end);
```

Load a height-band of an NHWC tensor from slow to fast. Allocates `N * tile_h * W * C * elem_size` bytes in the fast arena and copies rows `[h_start, h_end)`.

### tigris\_mem\_spill\_tile

```c
tigris_mem_error_t tigris_mem_spill_tile(
    tigris_mem_t        *mem,
    const tigris_plan_t *plan,
    uint16_t             tensor_idx,
    void                *slow_base,
    int32_t              h_start,
    int32_t              h_end);
```

Spill a height-band from fast to slow. Writes rows `[h_start, h_end)` of the full tensor in `slow_base`, then restores `tensor_ptrs[tensor_idx]` to `slow_base`.

### tigris\_mem\_reset\_fast

```c
void tigris_mem_reset_fast(tigris_mem_t *mem);
```

Reset the fast arena bump pointer to `fast_reserved`. Does not touch `tensor_ptrs`. Called by the executor between stages.

### tigris\_mem\_tensor\_ptr

```c
static inline void *tigris_mem_tensor_ptr(
    const tigris_mem_t *mem,
    uint16_t            idx);
```

Return the current data pointer for a tensor. Returns NULL if `mem` is NULL or `idx` is out of range.

### tigris\_mem\_error\_str

```c
const char *tigris_mem_error_str(tigris_mem_error_t err);
```

Return a human-readable string for a memory error code.

## Executor

### tigris\_run

```c
tigris_exec_error_t tigris_run(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats);
```

Compatibility entry point for running inference on a loaded plan. It uses one
process-global executor workspace and is therefore not re-entrant or safe for
concurrent inference. Define `TIGRIS_ENABLE_DEFAULT_EXECUTOR_WORKSPACE=0` to
omit that storage; `tigris_run()` then returns
`TIGRIS_EXEC_ERR_WORKSPACE`. New integrations should use the plan-sized
`tigris_run_with_workspace_buffer()` API.

The caller must have loaded the plan, initialized `tigris_mem_t` with fast and
slow buffers, and allocated model input tensors in slow with data filled in.
After return, model output tensors are in the slow buffer. `stats` may be NULL.

The executor allocates within the supplied arenas, compacts live fast tensors
when required, and can overflow some normal-stage allocations to the slow
arena. If the bounded fallback cannot satisfy an allocation, this function
returns `TIGRIS_EXEC_ERR_MEM`; it does not acquire more memory from a
general-purpose heap.

### tigris\_run\_with\_workspace

```c
tigris_exec_error_t tigris_run_with_workspace(
    const tigris_plan_t         *plan,
    tigris_mem_t                *mem,
    tigris_kernel_fn             kernel,
    void                        *user_ctx,
    tigris_exec_stats_t         *stats,
    tigris_executor_workspace_t *workspace);
```

Re-entrant executor entry point using caller-owned working storage. The
workspace can be static, global, heap-backed, or task-local, but one instance
must remain exclusively owned for the complete call. A NULL workspace returns
`TIGRIS_EXEC_ERR_WORKSPACE`. Other inputs, outputs, and error behavior match
`tigris_run()`.

### tigris\_executor\_workspace\_required

```c
size_t tigris_executor_workspace_required(const tigris_plan_t *plan);
```

Return the caller-owned workspace capacity needed by a loaded plan. The result
includes alignment headroom, so a byte buffer of exactly this size may begin at
any address. A result of zero means the plan metadata is incomplete.

### tigris\_run\_with\_workspace\_buffer

```c
tigris_exec_error_t tigris_run_with_workspace_buffer(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace,
    size_t               workspace_size);
```

Re-entrant executor entry point using an arbitrary caller-owned byte buffer.
This is the plan-sized API used by generated code: the runtime neither
allocates nor retains the buffer. Pass the value returned by
`tigris_executor_workspace_required()` as its minimum size. A NULL or
undersized buffer returns `TIGRIS_EXEC_ERR_WORKSPACE`.

### tigris\_executor\_workspace\_size

```c
size_t tigris_executor_workspace_size(void);
```

Return `sizeof(tigris_executor_workspace_t)` for the current build. The size is
derived from the configured executor limits and is useful to language bindings
or allocators that cannot use the C type directly.

### Kernel callback signature

```c
typedef int (*tigris_kernel_fn)(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);
```

Called once per op during stage execution. Input/output tensor pointers are already set in `mem`. The kernel reads inputs and writes outputs via `mem->tensor_ptrs`. Returns 0 on success, negative on error.

### tigris\_exec\_error\_str

```c
const char *tigris_exec_error_str(tigris_exec_error_t err);
```

Return a human-readable string for an executor error code.

### Reference backends

The float32 and int8 reference dispatchers implement `tigris_kernel_fn` and do
not require a preparation or deinitialization call:

```c
int tigris_dispatch_kernel(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);

int tigris_dispatch_kernel_s8(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);
```

## Accelerated int8 backends

The accelerated backends require a preparation call after
`tigris_mem_init()` and before the first executor call. Call the function once
for the selected backend and do not continue after a nonzero result. The
reference float32 and int8 dispatch functions require no preparation.

### ESP-NN

Declared by `tigris_kernels_esp_nn.h` when `TIGRIS_HAS_ESP_NN` is defined:

```c
int tigris_esp_nn_prepare(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem);

void tigris_esp_nn_deinit(void);

int tigris_dispatch_kernel_esp_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);
```

`tigris_esp_nn_prepare()` scans the plan and obtains 16-byte-aligned backend
workspace through the platform allocator during initialization. Optional
workspace failure makes affected operations use the reference backend; the
function returns -1 for invalid or overflowing metadata or an unavailable
mandatory depthwise-output buffer.
`tigris_esp_nn_deinit()` releases that process-global adapter workspace after
the last inference. Preparation and dispatch are not concurrently re-entrant.

### CMSIS-NN

Declared by `tigris_kernels_cmsis_nn.h` when `TIGRIS_HAS_CMSIS_NN` is defined:

```c
uint32_t tigris_cmsis_nn_scratch_required(
    const tigris_plan_t *plan);

uint32_t tigris_cmsis_nn_fast_arena_required(
    const tigris_plan_t *plan);

int tigris_cmsis_nn_prepare(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem);

int tigris_cmsis_nn_deinit(tigris_mem_t *mem);

int tigris_dispatch_kernel_cmsis_nn(
    const tigris_plan_t *plan,
    const tigris_op_t   *op,
    uint16_t             op_index,
    tigris_mem_t        *mem,
    void                *user_ctx);
```

`tigris_cmsis_nn_scratch_required()` returns the exact aligned adapter
reservation for vendor scratch and scalar-to-per-channel quantization
expansion. `tigris_cmsis_nn_fast_arena_required()` combines it with the core
activation and compressed-weight requirement, plus the alignment needed to
separate the core arena from the adapter reservation. The fast-buffer base
must already satisfy `TIGRIS_TENSOR_ALIGN`. Both helpers return
`UINT32_MAX` for invalid or unrepresentable input.

`tigris_cmsis_nn_prepare()` reserves that workspace at the top of the fast
buffer and reduces `mem->fast_size` so activation allocations cannot overlap
it. It returns 0 on success and -1 for an invalid argument or insufficient
capacity. `tigris_cmsis_nn_deinit()` releases the reservation and restores the
original fast-arena size. The adapter has one process-global prepared context,
so preparation and dispatch are not concurrently re-entrant.

## Key Types

### tigris\_plan\_t

Parsed plan handle. All fields are zero-copy pointers into the loaded buffer.

```c
typedef struct {
    const tigris_file_header_t  *header;
    const tigris_tensor_t       *tensors;         /* [num_tensors] */
    const tigris_op_t           *ops;             /* [num_ops] */
    const tigris_stage_t        *stages;          /* [num_stages] */
    const tigris_tile_plan_t    *tile_plans;      /* [num_tile_plans] */
    const uint16_t              *index_pool;
    const int32_t               *shape_pool;
    const char                  *strings;
    const tigris_weight_entry_t *weight_entries;   /* [num_weights] or NULL */
    const uint8_t               *weight_blob;      /* contiguous weight data */

    /* Compressed weight blocks (optional) */
    const tigris_weight_block_t *weight_blocks;
    const uint8_t               *weight_blocks_data;
    uint16_t                     num_weight_blocks;
    uint16_t                     weight_compression;

    /* Optional variable-length per-operator attributes. */
    const tigris_op_attribute_t *op_attributes;
    const uint8_t               *op_attribute_data;
    uint16_t                     num_op_attributes;

    /* Quantization (optional) */
    const tigris_quant_param_t  *quant_params;
    const int32_t               *quant_data;
    uint16_t                     num_quant_params;

    /* Convenience: model I/O */
    const uint16_t              *model_inputs;     /* [num_model_inputs] */
    const uint16_t              *model_outputs;    /* [num_model_outputs] */
} tigris_plan_t;
```

### tigris\_mem\_t

Memory manager state. Manages a fast arena (SRAM) and a slow buffer (PSRAM).

```c
typedef struct {
    void       **tensor_ptrs;
    uint16_t     num_tensors;
    uint8_t     *fast_base;
    uint32_t     fast_size;
    uint32_t     fast_used;
    uint32_t     fast_reserved;
    uint32_t     fast_peak;
    uint8_t     *slow_base;
    uint32_t     slow_size;
    uint32_t     slow_used;
    tigris_tile_ctx_t tile;
} tigris_mem_t;
```

`fast_peak` is the high-water mark of `fast_used` since
`tigris_mem_init()`. It includes executor-managed compressed-weight bumps as
well as ordinary fast-arena allocations. Inspect `mem.fast_peak` after
inference; backend workspace outside the usable arena is not included.

### tigris\_executor\_workspace\_t

Opaque, naturally aligned executor working storage. Its compile-time size is
derived from these plan limits:

| Definition | Default | Accepted values |
|---|---:|---:|
| `TIGRIS_MAX_TENSORS` | 512 | 1-65,535 |
| `TIGRIS_MAX_STAGE_INPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_STAGE_OUTPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_CHAIN_STAGES` | 16 | 2-65,535 |
| `TIGRIS_MAX_SPATIAL_OPS_PER_STAGE` | 8 | 1 or greater |

All runtime translation units must use the same definitions. The loader
returns `TIGRIS_ERR_PLAN_LIMITS` for a plan outside the selected bounds.
`TIGRIS_EXECUTOR_WORKSPACE_BYTES` may be overridden only to add
target-specific headroom; a compile-time assertion rejects a value too small
for the configured limits.

The type is the build-wide worst-case workspace. For generated or other
plan-specific integrations, use `tigris_executor_workspace_required()` and
`tigris_run_with_workspace_buffer()` to reserve only what that plan needs.

### tigris\_exec\_stats\_t

Execution statistics populated by either executor entry point.

| Field | Type | Description |
|-------|------|-------------|
| `stages_normal` | `uint16_t` | Stages executed without tiling |
| `stages_tiled` | `uint16_t` | Stages executed with spatial tiling |
| `stages_chain` | `uint16_t` | Stages executed as chain tiles |
| `total_tiles` | `uint16_t` | Total tile iterations across all stages |
| `slow_overflow_count` | `uint32_t` | Intra-stage allocs that overflowed to slow |
| `slow_overflow_bytes` | `uint32_t` | Total bytes overflowed to slow |
| `loads_bytes` | `uint32_t` | Total bytes loaded slow to fast |
| `spills_bytes` | `uint32_t` | Total bytes spilled fast to slow |
| `compactions` | `uint32_t` | Number of fast-pool compactions |
| `slow_peak` | `uint32_t` | High-water mark for slow\_used |

## Error Enums

### tigris\_error\_t (Loader)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_OK` | Success |
| -1 | `TIGRIS_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_ERR_TOO_SMALL` | Buffer smaller than header |
| -3 | `TIGRIS_ERR_BAD_MAGIC` | Invalid magic bytes |
| -4 | `TIGRIS_ERR_BAD_VERSION` | Unsupported schema version |
| -5 | `TIGRIS_ERR_BAD_SIZE` | file\_size field != buf\_len |
| -6 | `TIGRIS_ERR_BAD_SECTION` | Section offset out of bounds |
| -7 | `TIGRIS_ERR_MISSING_SEC` | Required section not found |
| -8 | `TIGRIS_ERR_ENDIAN` | Platform is not little-endian |
| -9 | `TIGRIS_ERR_PLAN_LIMITS` | Plan exceeds compiled executor limits |
| -10 | `TIGRIS_ERR_BAD_TENSOR` | Tensor metadata is not executable |
| -11 | `TIGRIS_ERR_BAD_OPERATOR` | Operator contract is not executable |

### tigris\_mem\_error\_t (Memory)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_MEM_OK` | Success |
| -1 | `TIGRIS_MEM_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_MEM_ERR_OOM` | Out of memory in arena |
| -3 | `TIGRIS_MEM_ERR_BAD_INDEX` | Tensor index >= num\_tensors |
| -4 | `TIGRIS_MEM_ERR_NOT_SET` | Tensor pointer is NULL (load/spill source) |

### tigris\_exec\_error\_t (Executor)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_EXEC_OK` | Success |
| -1 | `TIGRIS_EXEC_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_EXEC_ERR_MEM` | Memory allocation failed |
| -3 | `TIGRIS_EXEC_ERR_KERNEL` | Kernel callback returned error |
| -4 | `TIGRIS_EXEC_ERR_NO_STAGES` | Plan has no stages |
| -5 | `TIGRIS_EXEC_ERR_TILE` | Tiled execution error |
| -6 | `TIGRIS_EXEC_ERR_WORKSPACE` | Executor workspace unavailable or NULL |

## Op Types

```c
typedef enum {
    TIGRIS_OP_CONV           = 1,
    TIGRIS_OP_DEPTHWISE      = 2,
    TIGRIS_OP_RELU           = 3,
    TIGRIS_OP_RELU6          = 4,
    TIGRIS_OP_MAX_POOL       = 5,
    TIGRIS_OP_AVG_POOL       = 6,
    TIGRIS_OP_ADD            = 7,
    TIGRIS_OP_MUL            = 8,
    TIGRIS_OP_FULLY_CONN     = 9,
    TIGRIS_OP_SOFTMAX        = 10,
    TIGRIS_OP_CLIP           = 11,
    TIGRIS_OP_SIGMOID        = 12,
    TIGRIS_OP_CONCAT         = 13,
    TIGRIS_OP_PAD            = 14,
    TIGRIS_OP_GLOBAL_AVG     = 15,
    TIGRIS_OP_FLATTEN        = 16,
    TIGRIS_OP_RESHAPE        = 17,
    TIGRIS_OP_SUB            = 18,
    TIGRIS_OP_DIV            = 19,
    TIGRIS_OP_TANH           = 20,
    TIGRIS_OP_LEAKY_RELU     = 21,
    TIGRIS_OP_BATCH_NORM     = 22,
    TIGRIS_OP_INST_NORM      = 23,
    TIGRIS_OP_CONV_TRANSPOSE = 24,
    TIGRIS_OP_MATMUL         = 25,
    TIGRIS_OP_REDUCE_MEAN    = 26,
    TIGRIS_OP_SQUEEZE        = 27,
    TIGRIS_OP_UNSQUEEZE      = 28,
    TIGRIS_OP_TRANSPOSE      = 29,
    TIGRIS_OP_RESIZE         = 30,
    TIGRIS_OP_GLOBAL_MAX     = 31,
    TIGRIS_OP_CONV1D         = 32,
    TIGRIS_OP_UNKNOWN        = 255,
} tigris_op_type_t;
```

## Helpers

These are declared in `tigris.h`. The pointer-access helpers are `static
inline`; the arena-sizing helpers are implemented by the runtime library.

| Function | Returns | Description |
|----------|---------|-------------|
| `tigris_tensor_name(plan, t)` | `const char *` | Tensor name from string table |
| `tigris_op_attribute_data(plan, op, type, size)` | `const uint8_t *` | Variable-length operator attribute data, or NULL |
| `tigris_tensor_shape(plan, t)` | `const int32_t *` | Shape array for a tensor |
| `tigris_op_name(plan, op)` | `const char *` | Op name from string table |
| `tigris_op_inputs(plan, op)` | `const uint16_t *` | Input tensor index array |
| `tigris_op_outputs(plan, op)` | `const uint16_t *` | Output tensor index array |
| `tigris_stage_inputs(plan, stage)` | `const uint16_t *` | Stage input tensor index array |
| `tigris_stage_outputs(plan, stage)` | `const uint16_t *` | Stage output tensor index array |
| `tigris_weight_data(plan, weight)` | `const void *` | Weight entry data pointer |
| `tigris_weight_name(plan, weight)` | `const char *` | Weight entry name from the string table |
| `tigris_op_weight(plan, op)` | `const void *` | Weight data pointer (or NULL) |
| `tigris_op_bias(plan, op)` | `const void *` | Bias data pointer (or NULL) |
| `tigris_tensor_quant(plan, t)` | `const tigris_quant_param_t *` | Quantization params (or NULL) |
| `tigris_stage_ops(plan, stage)` | `const uint16_t *` | Op index array for a stage |
| `tigris_stage_tile_plan(plan, stage)` | `const tigris_tile_plan_t *` | Tile plan (or NULL) |
| `tigris_model_name(plan)` | `const char *` | Model name string |
| `tigris_weight_decompression_overhead(plan)` | `uint32_t` | Extra fast-arena bytes needed for LZ4 decompression |
| `tigris_fast_arena_required(plan)` | `uint32_t` | Activation budget plus simultaneous compressed-weight storage; assumes an aligned arena base |

The two arena-sizing helpers are library functions with these signatures:

```c
uint32_t tigris_weight_decompression_overhead(const tigris_plan_t *plan);
uint32_t tigris_fast_arena_required(const tigris_plan_t *plan);
```
