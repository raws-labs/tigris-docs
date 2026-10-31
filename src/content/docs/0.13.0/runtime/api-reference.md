---
title: "API Reference"
description: "Core execution, memory, loader, and accelerated-backend integration APIs for the TiGrIS runtime."
sidebar:
  order: 420
slug: 0.13.0/runtime/api-reference
---

Complete C API reference for the TiGrIS runtime. The core API is defined in
`tigris.h`, `tigris_loader.h`, `tigris_mem.h`, `tigris_iface.h`, and
`tigris_executor.h`.
Reference and accelerated dispatch functions are declared in the corresponding
`tigris_kernels*.h` headers.

## Loader

### tigris_plan_load

```c
tigris_error_t tigris_plan_load(
    const uint8_t *buf,
    uint32_t       buf_len,
    tigris_plan_t *out_plan);
```

Parse a `.tgrs` binary plan from a buffer. Zero-copy, zero-alloc: all pointers in `out_plan` point directly into `buf`. The caller must keep `buf` alive for the lifetime of `out_plan`.

The loader accepts plan schemas from `TIGRIS_SCHEMA_VERSION_MIN` (4) to
`TIGRIS_SCHEMA_VERSION` (11) and refuses any other with
`TIGRIS_ERR_BAD_VERSION`. It validates the structure, tensor metadata and
built-in operator semantics of every schema it accepts.

For uncompressed XIP plans used with optimized CMSIS-NN kernels, `buf` must be
aligned to 16 bytes, whatever `TIGRIS_TENSOR_ALIGN` is. Weight offsets are
aligned to 16 bytes relative to the plan base, so an unaligned base would make
those direct weight pointers unaligned too. Embedded arrays, linker sections,
custom flash mappings, and loaded buffers must provide the required base
alignment.

A tile record of schema 11 or later states the stage's tiling, and the loader
derives the same facts from the plan with
[`tigris_stage_tile_facts()`](#tigris_stage_tile_facts). A plan whose record
disagrees is refused with `TIGRIS_ERR_TILE_FACTS`.

The loader also refuses a chained stage that reads a tensor outside its band
with `TIGRIS_ERR_CHAIN_BAND`, and a tiled stage that reads or writes a tensor
outside its band with `TIGRIS_ERR_TILE_BAND`. Plans with such stages from
earlier releases load only after they are compiled again.

### tigris_plan_load_ex

```c
tigris_error_t tigris_plan_load_ex(
    const uint8_t        *buf,
    uint32_t              buf_len,
    tigris_plan_t        *out_plan,
    tigris_plan_limits_t *out_required);

void tigris_build_limits(tigris_plan_limits_t *out);
```

`tigris_plan_load_ex()` loads like `tigris_plan_load()` and also fills
`out_required` with the build limits the plan needs: `tensors`,
`stage_inputs`, `stage_outputs`, `chain_stages` and `spatial_ops_per_stage`, in
the terms of the `TIGRIS_MAX_*` definitions. On `TIGRIS_ERR_PLAN_LIMITS` it
holds everything read up to the first limit this build cannot meet. Pass NULL
to ignore it. `tigris_build_limits()` reports the limits the build allows.

### tigris_stage_tile_facts

```c
int tigris_stage_tile_facts(
    const tigris_plan_t *plan,
    const tigris_stage_t *stage,
    tigris_tile_facts_t *out);
```

Derive what a stage's tile record states about its tiling, by the rules the
executor bands the stage with: `band_axis`, the serialized axis the bands run
along; `original_extent`, the extent they divide; `tile_extent`, the band;
`num_tiles`, the bands of `tile_extent` (times the column tiles of a 2D tile);
`halo`, the rows a band reads beyond its own; and `band`, the kind the stage is
banded by (see [`tigris_stage_band_kind`](#tigris_stage_band_kind)) in the low
four bits and a leading band's mode in the high four. A height stripe divides
the rows of its spatial operator's output, or of its first output without one,
into the stated output band, and `num_tiles` is the number of bands the
executor runs. A layout conversion sizes its band in the run, so its
`tile_extent` is the record's, bounded by the extent. An untileable record
yields all zero. Returns 1 when derived and 0 when the stage's tiling has no
such description.

### tigris_stage_band_kind

```c
tigris_band_kind_t tigris_stage_band_kind(
    const tigris_plan_t  *plan,
    const tigris_stage_t *stage,
    tigris_stage_band_t  *band);
```

How the executor bands a standalone stage: `TIGRIS_BAND_KIND_RESHAPE`,
`_LEADING`, `_ROWS`, `_TRANSPOSE`, `_BY_INPUT`, `_HW` or `_STRIPE`, or
`TIGRIS_BAND_KIND_NONE`. From schema 11 a tileable record states the kind,
which is returned when the stage meets that kind's conditions; the loader
refuses a stage that does not. A stage without a tileable record has no kind.
Earlier plans derive the kind in the order the executor decides. Whether the
stage is banded at all also depends on its record and the arena. `band` may be
NULL; otherwise it receives the extent the kind divides. Chain members and
control-flow stages are not classified here.

### Tensor alignment

Unless `TIGRIS_TENSOR_ALIGN` is set by the build, the runtime selects this
minimum tensor alignment from the target architecture:

| Target condition | Default alignment (bytes) |
|------------------|--------------------------:|
| `__ARM_FEATURE_SVE` | 64 |
| `__aarch64__` | 16 |
| `__x86_64__` | 32 |
| `__riscv_vector` | 16 |
| `__XTENSA__` | 16 |
| `__ARM_NEON` | 16 |
| `__ARM_FEATURE_DSP` | 16 |
| `otherwise` | 4 |

An override supplied as `-DTIGRIS_TENSOR_ALIGN=N` must be a power of two.
Compiled plans size allocations for an alignment of at most 32 bytes, so an
auto-detected value above 32 (SVE) stops the build; define
`-DTIGRIS_TENSOR_ALIGN=32` for such a target.

### Runtime version

`tigris.h` states the runtime release as `TIGRIS_RUNTIME_VERSION_MAJOR`,
`TIGRIS_RUNTIME_VERSION_MINOR` and `TIGRIS_RUNTIME_VERSION_PATCH`, so code built
against the headers can check the version it needs at compile time.

### tigris_error_str

```c
const char *tigris_error_str(tigris_error_t err);
```

Return a human-readable string for a `tigris_error_t` code.

## Memory Manager

### tigris_mem_init

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
aligned address. `slow_buf` may be NULL when `slow_size` is 0, as for a plan
for which `tigris compile` prints no `slow memory` row; a slow allocation then
returns `TIGRIS_MEM_ERR_OOM`.

### tigris_mem_alloc_slow

```c
tigris_mem_error_t tigris_mem_alloc_slow(
    tigris_mem_t *mem,
    uint16_t      tensor_idx,
    uint32_t      size_bytes);
```

Bump-allocate in the slow buffer. Sets `tensor_ptrs[tensor_idx]` to the allocated address.

The executor's own fast-arena allocation, load, spill, tile and reset functions
are internal to the runtime sources and not part of this API.

### tigris_mem_tensor_ptr

```c
static inline void *tigris_mem_tensor_ptr(
    const tigris_mem_t *mem,
    uint16_t            idx);
```

Return the current data pointer for a tensor. Returns NULL if `mem` is NULL or `idx` is out of range.

### tigris_mem_error_str

```c
const char *tigris_mem_error_str(tigris_mem_error_t err);
```

Return a human-readable string for a memory error code.

## Model interface

A tensor's stored dtype can differ from the dtype the model file declares at
its boundary: an ONNX graph can state float32 and quantize inside itself, and
the compiler folds that quantization into the boundary tensor. An int8 plan
can also contain float32 sections after Dequantize. The
plan records the declared dtype per model input and output, and these calls
move data across the boundary, so an application hands over and reads back
exactly what the model declares.

### tigris_iface_bytes

```c
uint32_t tigris_iface_bytes(
    const tigris_plan_t *plan,
    uint16_t             tensor_idx);
```

Bytes the caller supplies for a model input, or receives for a model output, in
the dtype the model declares. Returns 0 when the tensor is not a model boundary
tensor, or when its declared dtype has no conversion.

### tigris_input_alloc

```c
tigris_error_t tigris_input_alloc(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    uint16_t             tensor_idx);
```

Place a model input where the run reads it. A plan reads its inputs in place
when it has schema 5 or later and a single stage, without a tile plan, chain,
control flow or state, that reads every model input and writes every model
output. Such a plan's input goes in the fast arena, and the first of its inputs
leaves `tigris_weight_decompression_overhead()` bytes free directly below it,
where the run decompresses the stage's weights. Place every input of such a
plan this way, one after another, so they sit together at the top of the fast
arena. Any other plan's input goes in the slow buffer. An input that already
has a buffer keeps it. Returns `TIGRIS_ERR_NULL` when `plan` or `mem` is NULL
or `mem` has no pointer table, `TIGRIS_ERR_BAD_TENSOR` when `tensor_idx` is not
a model input, `TIGRIS_ERR_TENSOR_SLOTS` when `mem` has no tensor slot for it,
and `TIGRIS_ERR_MEM` when the arena cannot hold the input. A failed
call leaves `mem` unchanged.

### tigris_input_write

```c
tigris_error_t tigris_input_write(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    uint16_t             tensor_idx,
    const void          *src,
    uint32_t             src_bytes);
```

Convert `src` into the model input tensor. An input with no buffer yet is
placed with [`tigris_input_alloc()`](#tigris_input_alloc) first, and its errors
are returned as they are. A buffer the caller allocated beforehand is used as
it is. `src_bytes` must equal `tigris_iface_bytes()`. Returns
`TIGRIS_ERR_BAD_INTERFACE` when the declared dtype has no conversion to what
the plan stores, so an unsupported interface fails rather than being
approximated.

### tigris_output_read

```c
tigris_error_t tigris_output_read(
    const tigris_plan_t *plan,
    const tigris_mem_t  *mem,
    uint16_t             tensor_idx,
    void                *dst,
    uint32_t             dst_bytes);
```

Convert a model output tensor into `dst` after a run. `dst_bytes` must equal
`tigris_iface_bytes()`.

## Executor

### tigris_run

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
`TIGRIS_EXEC_ERR_WORKSPACE`. New integrations should use
[`tigris_run_with_workspace()`](#tigris_run_with_workspace), or
`tigris_run_with_workspace_buffer()` for a plan-sized byte buffer.

The caller must have loaded the plan, initialized `tigris_mem_t` with a fast
buffer and, unless the plan needs none, a slow buffer, and placed
every model input with `tigris_input_alloc()` or `tigris_input_write()`, or
allocated it in slow, and filled it. `tigris_input_write()` converts from the
dtype the model declares and places the input with `tigris_input_alloc()`. After
return, model output tensors are in the slow buffer, and `tigris_output_read()`
converts them back to that dtype. A plan that reads its inputs in place (see
[`tigris_input_alloc`](#tigris_input_alloc)) consumes the inputs placed in the
fast arena, decompresses its weights into the room left below them, and leaves
its outputs in the fast arena, valid until the next allocation in that arena:
read them before writing the next inputs. Inputs placed another way are read in
place only with that room free. `stats` may be NULL.

A plan of schema 11 or later states its scheduled peak. Its run returns
`TIGRIS_EXEC_ERR_MEM` before any stage runs when the fast arena above the run's
floor is smaller than `tigris_fast_arena_required()`, and leaves `mem` as it
was. The floor is the arena's fill level on entry, or, for inputs read in
place, the start of the room below them. A plan of an earlier schema states no
peak and is not checked up front.

The executor allocates within the supplied arenas, compacts live fast tensors
when required, and can overflow some normal-stage allocations to the slow
arena. If the bounded fallback cannot satisfy an allocation, this function
returns `TIGRIS_EXEC_ERR_MEM`; it does not acquire more memory from a
general-purpose heap.

### tigris_run_with_workspace

```c
tigris_exec_error_t tigris_run_with_workspace(
    const tigris_plan_t         *plan,
    tigris_mem_t                *mem,
    tigris_kernel_fn             kernel,
    void                        *user_ctx,
    tigris_exec_stats_t         *stats,
    tigris_executor_workspace_t *workspace);
```

The primary, re-entrant executor entry point, using caller-owned working
storage. Concurrent inferences each pass their own workspace and memory
manager, with a kernel dispatcher that is itself safe to call concurrently. The
workspace can be static, global, heap-backed, or task-local, but one instance
must remain exclusively owned for the complete call. A NULL workspace returns
`TIGRIS_EXEC_ERR_WORKSPACE`. Other inputs, outputs, and error behavior match
`tigris_run()`.

### tigris_executor_workspace_required

```c
size_t tigris_executor_workspace_required(const tigris_plan_t *plan);
```

Return the caller-owned workspace capacity needed by a loaded plan. The result
includes alignment headroom, so a byte buffer of exactly this size may begin at
any address. A result of zero means the plan metadata is incomplete.

### tigris_run_with_workspace_buffer

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

Convenience form of `tigris_run_with_workspace()` for an arbitrary
caller-owned byte buffer instead of the fixed workspace type. This is the
plan-sized API used by generated code: the runtime neither
allocates nor retains the buffer. Pass the value returned by
`tigris_executor_workspace_required()` as its minimum size. A NULL or
undersized buffer returns `TIGRIS_EXEC_ERR_WORKSPACE`.

### tigris_run_with_state

```c
tigris_exec_error_t tigris_run_with_state(
    const tigris_plan_t *plan,
    tigris_mem_t        *mem,
    tigris_kernel_fn     kernel,
    void                *user_ctx,
    tigris_exec_stats_t *stats,
    void                *workspace,
    size_t               workspace_size,
    void                *state,
    size_t               state_size);
```

Runs a plan that keeps variables from one invocation to the next, such as
resource variables or the state of an SVDF or LSTM. The state buffer holds every
variable between runs: each run reads the values the last one left and writes
back the ones it assigns. Prepare the buffer once with `tigris_state_init()`,
which is also how it is reset. A plan without state runs here too, with `state`
NULL and `state_size` 0. Returns `TIGRIS_EXEC_ERR_STATE` when the plan keeps
state and the buffer is NULL or smaller than `tigris_state_required()`;
otherwise it behaves as `tigris_run_with_workspace_buffer()`. The other entry
points refuse a stateful plan with `TIGRIS_EXEC_ERR_STATE`.

### tigris_state_required

```c
size_t tigris_state_required(const tigris_plan_t *plan);
```

Bytes of state a plan keeps across invocations; 0 for a plan without.

### tigris_state_init

```c
tigris_exec_error_t tigris_state_init(const tigris_plan_t *plan, void *state, size_t state_size);
```

Writes every variable's initial value into a state buffer, which prepares it for
the first run and resets it after any number of runs. Returns
`TIGRIS_EXEC_ERR_STATE` when the buffer is NULL or too small.

### tigris_plan_data_dtype

```c
uint8_t tigris_plan_data_dtype(const tigris_plan_t *plan);
```

The plan's default dispatch dtype: int8 (3) when any activation is int8, else
float32 (1). The int8 dispatcher routes float32 section operators to the float
reference kernels. The loader checks that float32 sections follow a typed
conversion or control-flow boundary. Other typed float outputs, such as
detection results, follow their operator's signature.

Dtypes are ONNX `TensorProto.DataType` codes. `tigris.h` names the ones a plan
tensor or model boundary can carry: `TIGRIS_DTYPE_FLOAT32` (1),
`TIGRIS_DTYPE_UINT8` (2), `TIGRIS_DTYPE_INT8` (3), `TIGRIS_DTYPE_INT16` (5),
`TIGRIS_DTYPE_INT32` (6), `TIGRIS_DTYPE_INT64` (7) and `TIGRIS_DTYPE_BOOL` (9).

### tigris_executor_workspace_size

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

### tigris_exec_error_str

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
mandatory depthwise-output buffer, and -2 for misaligned Conv or depthwise
weights.
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

## Execution trace

Configure the runtime with `-DTIGRIS_TRACE=ON` to report every allocation, load,
spill, move, kernel call and reset as it happens. Without the option the hooks
compile to nothing.

```c
void tigris_mem_set_trace(tigris_mem_t *mem, tigris_trace_fn fn, void *ctx);
typedef void (*tigris_trace_fn)(const tigris_trace_event_t *event, void *ctx);
```

The callback runs synchronously; the event is valid only during the call. NULL
stops reporting. Each `tigris_trace_event_t` (`include/tigris_trace.h`) carries its
kind, the pool, stage, tensor and operator it concerns, destination and source
offsets, the bytes, the rows and columns of a tile, and the fast and slow memory
in use after the event.

| Kind | Reported when |
|---|---|
| `TIGRIS_TRACE_STAGE_BEGIN`, `TIGRIS_TRACE_STAGE_END` | A stage starts, with its execution path, and ends |
| `TIGRIS_TRACE_TILE_BEGIN` | A tile starts, with the input rows and columns it reads, halo included |
| `TIGRIS_TRACE_ALLOC` | A tensor gets its place in a pool |
| `TIGRIS_TRACE_LOAD`, `TIGRIS_TRACE_SPILL` | Bytes move from slow to fast memory, or back |
| `TIGRIS_TRACE_MOVE` | A pool compacts, or a chain keeps rows the next tile shares |
| `TIGRIS_TRACE_OP` | A kernel is called |
| `TIGRIS_TRACE_RESET` | Fast memory is reset to its reserved prefix |
| `TIGRIS_TRACE_WEIGHTS` | A weight block is written to fast memory |
| `TIGRIS_TRACE_COPY` | State or control-flow operands are copied |

`tigris analyze --trace` prints these events from the host library.

## Key Types

### tigris_plan_t

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

    /* Variables kept across invocations (optional, schema v10+) */
    const tigris_state_entry_t  *state_entries;    /* [num_state] or NULL */
    const uint8_t               *state_section;
    uint16_t                     num_state;
    uint32_t                     state_bytes;      /* size of the caller's state buffer */
    const tigris_subgraph_t     *subgraphs;        /* [num_subgraphs] or NULL */
    uint16_t                     num_subgraphs;
    uint16_t                     subgraph_depth;   /* deepest control-flow nesting */

    const tigris_tile_plan_v10_t *tile_plans_v10;  /* schema 10 only, otherwise NULL */

    /* Convenience: model I/O */
    const uint16_t              *model_inputs;     /* [num_model_inputs] */
    const uint16_t              *model_outputs;    /* [num_model_outputs] */

    uint32_t                     scheduled_peak;   /* schema 11+, 0 before */
    const tigris_tile_plan_v11_t *tile_plans_v11;  /* schema 11+, otherwise NULL */
} tigris_plan_t;
```

`scheduled_peak` is the fast-arena bytes the plan's largest stage or chain
holds at once, which the compiler checked against the budget; decompressed
weights are not included. Every plan of schema 11 or later carries it in its
memory section, and the loader refuses such a plan without one.

A schema 11 tile record, `tigris_tile_plan_v11_t`, is 48 bytes: the schema 10
scheduling choices followed by a `tigris_tile_facts_t` that states
`tile_extent`, `num_tiles` and `original_extent` (each `uint32_t`), `halo`
(`uint16_t`), `band_axis` and `band` (each `uint8_t`), as
[`tigris_stage_tile_facts()`](#tigris_stage_tile_facts) defines them.
`TIGRIS_TILE_BAND_KIND(band)` and `TIGRIS_TILE_BAND_MODE(band)` read the two
halves of `band`.

### tigris_mem_t

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
    uint32_t     slow_peak;
    struct tigris_exec_stats *stats;
#ifdef TIGRIS_TRACE
    tigris_trace_fn trace_fn;
    void *trace_ctx;
    uint16_t trace_stage;
    uint8_t trace_path;
#endif
    tigris_tile_ctx_t tile;
} tigris_mem_t;
```

`fast_peak` is the high-water mark of `fast_used` since
`tigris_mem_init()`. It includes executor-managed compressed-weight bumps as
well as ordinary fast-arena allocations. Inspect `mem.fast_peak` after
inference; backend workspace outside the usable arena is not included.

### tigris_executor_workspace_t

Opaque, naturally aligned executor working storage. Its compile-time size is
derived from these plan limits:

| Definition | Default | Accepted values |
|---|---:|---:|
| `TIGRIS_MAX_TENSORS` | 512 | 1-65,535 |
| `TIGRIS_MAX_STAGE_INPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_STAGE_OUTPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_CHAIN_STAGES` | 16 | 2-65,535 |
| `TIGRIS_MAX_SPATIAL_OPS_PER_STAGE` | 8 | 1 or greater |
| `TIGRIS_MAX_SUBGRAPHS` | 16 | 1-65,535 |
| `TIGRIS_MAX_SUBGRAPH_DEPTH` | 4 | 1-65,535 |

`TIGRIS_MAX_SUBGRAPHS` counts the graphs of a plan with control flow, the main
graph included; `TIGRIS_MAX_SUBGRAPH_DEPTH` bounds how many `If` and `While`
operators run one inside another.

All runtime translation units must use the same definitions. The loader
returns `TIGRIS_ERR_PLAN_LIMITS` for a plan outside the selected bounds.
`TIGRIS_EXECUTOR_WORKSPACE_BYTES` may be overridden only to add
target-specific headroom; a compile-time assertion rejects a value too small
for the configured limits.

The type is the build-wide worst-case workspace. For generated or other
plan-specific integrations, use `tigris_executor_workspace_required()` and
`tigris_run_with_workspace_buffer()` to reserve only what that plan needs.

### tigris_exec_stats_t

Execution statistics populated by every executor entry point, on every execution
path: normal, tiled, chained and control flow.

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
| `compactions` | `uint32_t` | Number of fast- and slow-pool compactions |
| `slow_peak` | `uint32_t` | High-water mark for slow_used |
| `weight_bytes` | `uint32_t` | Uncompressed weight bytes written to fast memory |
| `copy_bytes` | `uint32_t` | State and control-flow bytes copied |

## Error Enums

### tigris_error_t (Loader and interface)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_OK` | Success |
| -1 | `TIGRIS_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_ERR_TOO_SMALL` | Buffer smaller than header |
| -3 | `TIGRIS_ERR_BAD_MAGIC` | Invalid magic bytes |
| -4 | `TIGRIS_ERR_BAD_VERSION` | Unsupported schema version |
| -5 | `TIGRIS_ERR_BAD_SIZE` | file_size field != buf_len |
| -6 | `TIGRIS_ERR_BAD_SECTION` | Section offset out of bounds |
| -7 | `TIGRIS_ERR_MISSING_SEC` | Required section not found |
| -8 | `TIGRIS_ERR_ENDIAN` | Platform is not little-endian |
| -9 | `TIGRIS_ERR_PLAN_LIMITS` | Plan exceeds compiled executor limits |
| -10 | `TIGRIS_ERR_BAD_TENSOR` | Tensor metadata is not executable |
| -11 | `TIGRIS_ERR_BAD_OPERATOR` | Operator contract is not executable |
| -12 | `TIGRIS_ERR_BAD_INTERFACE` | Declared model interface is not convertible |
| -13 | `TIGRIS_ERR_KERNEL_DISABLED` | Build leaves out a kernel the plan uses |
| -14 | `TIGRIS_ERR_MEM` | Arena cannot hold the tensor |
| -15 | `TIGRIS_ERR_TENSOR_SLOTS` | Memory manager has fewer tensor slots than the plan |
| -16 | `TIGRIS_ERR_TILE_FACTS` | A tile record's stated tiling disagrees with the plan |
| -17 | `TIGRIS_ERR_CHAIN_BAND` | A chained stage reads a tensor outside its band; recompile the model |
| -18 | `TIGRIS_ERR_TILE_BAND` | A tiled stage reads or writes a tensor outside its band; recompile the model |

### tigris_mem_error_t (Memory)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_MEM_OK` | Success |
| -1 | `TIGRIS_MEM_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_MEM_ERR_OOM` | Out of memory in arena |
| -3 | `TIGRIS_MEM_ERR_BAD_INDEX` | Tensor index >= num_tensors |
| -4 | `TIGRIS_MEM_ERR_NOT_SET` | Tensor pointer is NULL |

### tigris_exec_error_t (Executor)

| Value | Name | Description |
|-------|------|-------------|
| 0 | `TIGRIS_EXEC_OK` | Success |
| -1 | `TIGRIS_EXEC_ERR_NULL` | Null pointer argument |
| -2 | `TIGRIS_EXEC_ERR_MEM` | Arena too small for the plan, found up front or in the run |
| -3 | `TIGRIS_EXEC_ERR_KERNEL` | Kernel callback returned error |
| -4 | `TIGRIS_EXEC_ERR_NO_STAGES` | Plan has no stages |
| -5 | `TIGRIS_EXEC_ERR_TILE` | Tiled execution error |
| -6 | `TIGRIS_EXEC_ERR_WORKSPACE` | Executor workspace unavailable, NULL or too small |
| -7 | `TIGRIS_EXEC_ERR_STATE` | Plan keeps state and no state buffer, or too small a one, was given |

## Op Types

The first opcode assignments are shown below. The full enum is declared in
`tigris.h`; the [operator table](/0.13.0/runtime/operator-support/) lists executable
routes.

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
| `tigris_op_attribute_data(plan, op_index, type, out_len)` | `const uint8_t *` | Variable-length operator attribute data, or NULL |
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
| `tigris_quant_multipliers(plan, qp)` | `const int32_t *` | The `num_channels` Q0.31 requantization multipliers of a quant param |
| `tigris_quant_shifts(plan, qp)` | `const int32_t *` | The `num_channels` requantization shifts of a quant param |
| `tigris_quant_page_start(qp)` | `uint32_t` | Index in `quant_data` where the quant param's page starts |
| `tigris_tensor_declared_dtype(t)` | `uint8_t` | Dtype the model declares for a boundary tensor, its own dtype when none is recorded |
| `tigris_stage_ops(plan, stage)` | `const uint16_t *` | Op index array for a stage |
| `tigris_stage_tile_plan(plan, stage)` | `const tigris_tile_plan_t *` | Schema 9 tile record (or NULL) |
| `tigris_stage_tiles(plan, stage)` | `tigris_tile_plan_v10_t` | The stage's tiling choices from any record layout; zero when untiled |
| `tigris_model_name(plan)` | `const char *` | Model name string |
| `tigris_weight_decompression_overhead(plan)` | `uint32_t` | Extra fast-arena bytes needed for LZ4 decompression |
| `tigris_fast_arena_required(plan)` | `uint32_t` | Scheduled peak (schema 11+) or activation budget (earlier plans) plus simultaneous compressed-weight storage; assumes an aligned arena base |

The two arena-sizing helpers are library functions with these signatures:

```c
uint32_t tigris_weight_decompression_overhead(const tigris_plan_t *plan);
uint32_t tigris_fast_arena_required(const tigris_plan_t *plan);
```
