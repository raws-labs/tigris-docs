---
title: Integration
description: A step-by-step guide to integrating the TiGrIS C99 runtime into
  your embedded application.
sidebar:
  order: 410
slug: 0.6.0/runtime/integration
---

Step-by-step guide to integrating the TiGrIS C99 runtime into your embedded application.

## 1. Add to your build

**CMake (as a subdirectory):**

```cmake
add_subdirectory(tigris-runtime)
target_link_libraries(my_app PRIVATE tigris_runtime)
```

**Manual:** Copy `tigris-runtime/src/` and `tigris-runtime/include/` into your project. Add the source files to your build system and set the include path.

## 2. Include headers

```c
#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"       /* int8 reference backend */
#include "tigris_kernels.h"          /* f32 reference backend */
```

Pick the kernel header matching your model's dtype. For accelerated int8 backends, use `tigris_kernels_esp_nn.h` (ESP32 family) or `tigris_kernels_cmsis_nn.h` (Cortex-M family) instead of `tigris_kernels_s8.h`.

## 3. Load the plan

```c
tigris_plan_t plan;
tigris_error_t err = tigris_plan_load(plan_buf, plan_buf_len, &plan);
if (err != TIGRIS_OK) {
    /* report tigris_error_str(err) and stop */
}
```

`plan_buf` is the `.tgrs` file content, either memory-mapped from flash or loaded into a buffer. The loader is zero-copy and zero-alloc: all pointers in `plan` refer directly into `plan_buf`. Keep `plan_buf` alive for the lifetime of `plan`.

Align the plan base to `TIGRIS_TENSOR_ALIGN` when an optimized backend reads
uncompressed XIP weights directly from it. This is 16 bytes on DSP-enabled
Cortex-M (`__ARM_FEATURE_DSP`). Give embedded arrays and linker sections an
explicit alignment, and preserve it for custom flash mappings or loaded
buffers.

## 4. Initialize memory

Allocate a fast buffer (SRAM), a slow buffer (PSRAM), and a tensor pointer array. Then initialize the memory manager:

```c
uint32_t required_fast = tigris_fast_arena_required(&plan);
#if defined(TIGRIS_HAS_CMSIS_NN)
required_fast = tigris_cmsis_nn_fast_arena_required(&plan);
#endif
if (required_fast == UINT32_MAX) {
    /* invalid or unrepresentable fast-buffer requirement; stop */
}
if (fast_capacity < required_fast) {
    /* caller-provided fast buffer cannot cover the requirement; stop */
}

tigris_mem_t mem;
static void *tensor_ptrs[TIGRIS_MAX_TENSORS];
tigris_mem_error_t merr = tigris_mem_init(
    &mem, tensor_ptrs, plan.header->num_tensors,
    fast_buf, fast_capacity, slow_buf, slow_size);
if (merr != TIGRIS_MEM_OK) {
    /* report tigris_mem_error_str(merr) and stop */
}
```

The compiled budget covers modeled activations. `tigris_fast_arena_required()`
adds compressed-weight storage. Both helpers assume that the fast-buffer base
satisfies `TIGRIS_TENSOR_ALIGN`; align the allocation rather than relying on
extra capacity to absorb leading padding. For CMSIS-NN,
`tigris_cmsis_nn_fast_arena_required()` also adds the exact adapter workspace.
Use the helper that matches the selected backend rather than reproducing its
arithmetic in application code.

## 5. Prepare an accelerated backend

```c
#if defined(TIGRIS_HAS_ESP_NN)
if (tigris_esp_nn_prepare(&plan, &mem) != 0) {
    /* backend workspace allocation failed; stop */
}
#elif defined(TIGRIS_HAS_CMSIS_NN)
if (tigris_cmsis_nn_prepare(&plan, &mem) != 0) {
    /* arena too small for CMSIS-NN scratch; stop */
}
#endif
```

Call the preparation function for the selected accelerated backend exactly once
after `tigris_mem_init()` and before inference, and always check its result. The
reference int8 and float32 dispatch functions need no preparation.

CMSIS-NN preparation carves a 16-byte-aligned region from the top of the fast
buffer for vendor scratch and scalar-to-per-channel quantization expansion.
Query that reservation with `tigris_cmsis_nn_scratch_required()`, or query the
complete buffer with `tigris_cmsis_nn_fast_arena_required()`. ESP-NN obtains
aligned workspace through the platform allocator during initialization.
Neither backend allocates model-dependent workspace during inference after
successful preparation. Call `tigris_cmsis_nn_deinit()` or
`tigris_esp_nn_deinit()` after the last inference when the workspace is no
longer needed.

## 6. Set model inputs

Allocate input tensors in the slow buffer and fill with your data:

```c
for (uint16_t i = 0; i < plan.header->num_model_inputs; i++) {
    uint16_t tidx = plan.model_inputs[i];
    uint32_t sz = plan.tensors[tidx].size_bytes;
    tigris_mem_error_t merr = tigris_mem_alloc_slow(&mem, tidx, sz);
    if (merr != TIGRIS_MEM_OK) {
        /* report tigris_mem_error_str(merr) and stop */
    }

    int8_t *input = (int8_t *)mem.tensor_ptrs[tidx];
    /* fill input with your preprocessed data */
}
```

## 7. Run inference

```c
static tigris_executor_workspace_t executor_workspace;

tigris_exec_stats_t stats;
tigris_exec_error_t eerr = tigris_run_with_workspace(
    &plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats,
    &executor_workspace);
if (eerr != TIGRIS_EXEC_OK) {
    /* report tigris_exec_error_str(eerr) and stop */
}
```

Pass the dispatch function for your chosen backend. The `user_ctx` parameter (NULL above) is forwarded to every kernel call.

Provide one workspace for each inference that may run concurrently. Static
storage is appropriate for a single inference task; separate tasks or model
instances need separate workspaces. The compatibility entry point
`tigris_run()` owns one process-global workspace and is therefore not
re-entrant or safe for concurrent inference.

After the call, `mem.fast_peak` is the measured high-water mark of the core
fast arena since initialization. Backend workspace reserved outside that arena
must be counted separately.

### Executor limits and task-stack sizing

The executor workspace is bounded by compile-time plan limits:

| Definition | Default | Accepted values |
|---|---:|---:|
| `TIGRIS_MAX_TENSORS` | 512 | 1-65,535 |
| `TIGRIS_MAX_STAGE_INPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_STAGE_OUTPUTS` | 16 | 1-65,535 |
| `TIGRIS_MAX_CHAIN_STAGES` | 16 | 2-65,535 |
| `TIGRIS_MAX_SPATIAL_OPS_PER_STAGE` | 8 | 1 or greater |

Override these definitions consistently for every runtime translation unit.
Lower limits reduce the workspace; higher limits accept larger plans and
increase it. The loader returns `TIGRIS_ERR_PLAN_LIMITS` before execution when
a plan exceeds the configured limits. `sizeof(tigris_executor_workspace_t)`
and `tigris_executor_workspace_size()` are the authoritative workspace size
for a particular build.

Keep the workspace in static, global, or heap-backed storage unless its full
size is deliberately included in the task-stack budget. Runtime sources reject
variable-length arrays. Build with `-DTIGRIS_STACK_USAGE=ON` to generate and
check GCC stack-usage reports; CI rejects an unbounded dynamic frame or any
individual runtime function frame above 1,024 bytes. Those reports exclude
caller and vendor-library frames, C-library internals, RTOS context, interrupt
nesting, and instrumentation. Size the final task stack from the complete call
chain and confirm it with the target's stack high-water measurement under the
largest supported model and worst interrupt load.

## 8. Read outputs

```c
for (uint16_t i = 0; i < plan.header->num_model_outputs; i++) {
    uint16_t tidx = plan.model_outputs[i];
    int8_t *output = (int8_t *)mem.tensor_ptrs[tidx];
    uint32_t size = plan.tensors[tidx].size_bytes;
    if (output == NULL) { /* inference did not produce this output; stop */ }
    /* process output */
}
```

Model outputs are located in the slow buffer after inference completes.

## Complete example

Minimal POSIX integration that loads a plan from file, runs inference, and prints first output values. This example uses the int8 reference backend; for f32, replace `tigris_dispatch_kernel_s8` with `tigris_dispatch_kernel`:

```c
#define _POSIX_C_SOURCE 200112L

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tigris.h"
#include "tigris_loader.h"
#include "tigris_mem.h"
#include "tigris_executor.h"
#include "tigris_kernels_s8.h"

static tigris_executor_workspace_t executor_workspace;

int main(int argc, char **argv) {
    FILE *f = NULL;
    void *plan_storage = NULL;
    void *fast_buf = NULL;
    void *slow_buf = NULL;
    void **tensor_ptrs = NULL;
    int result = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s model.tgrs\n", argv[0]);
        return 2;
    }

    f = fopen(argv[1], "rb");
    if (f == NULL || fseek(f, 0, SEEK_END) != 0) {
        fprintf(stderr, "could not open or seek plan\n");
        goto cleanup;
    }
    long end = ftell(f);
    if (end <= 0 || (unsigned long)end > UINT32_MAX ||
        fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "invalid plan size\n");
        goto cleanup;
    }
    uint32_t file_size = (uint32_t)end;

    size_t alignment = TIGRIS_TENSOR_ALIGN;
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if (posix_memalign(&plan_storage, alignment, file_size) != 0) {
        fprintf(stderr, "could not allocate aligned plan buffer\n");
        goto cleanup;
    }
    if (fread(plan_storage, 1, file_size, f) != file_size) {
        fprintf(stderr, "could not read complete plan\n");
        goto cleanup;
    }
    if (fclose(f) != 0) {
        f = NULL;
        fprintf(stderr, "could not close plan file\n");
        goto cleanup;
    }
    f = NULL;

    /* Parse plan (zero-copy into the aligned storage) */
    tigris_plan_t plan;
    tigris_error_t err = tigris_plan_load(
        (const uint8_t *)plan_storage, file_size, &plan);
    if (err != TIGRIS_OK) {
        fprintf(stderr, "load failed: %s\n", tigris_error_str(err));
        goto cleanup;
    }
    if (plan.header->num_model_inputs == 0 ||
        plan.header->num_model_outputs == 0) {
        fprintf(stderr, "plan has no model input or output\n");
        goto cleanup;
    }

    uint32_t fast_size = tigris_fast_arena_required(&plan);
    if (fast_size == UINT32_MAX) {
        fprintf(stderr, "invalid fast-buffer requirement\n");
        goto cleanup;
    }
    uint32_t slow_size = 512u * 1024u;
    if (fast_size == 0 ||
        posix_memalign(&fast_buf, alignment, fast_size) != 0 ||
        posix_memalign(&slow_buf, alignment, slow_size) != 0) {
        fprintf(stderr, "could not allocate aligned arenas\n");
        goto cleanup;
    }
    tensor_ptrs = calloc(plan.header->num_tensors, sizeof(void *));
    if (tensor_ptrs == NULL) {
        fprintf(stderr, "could not allocate tensor pointer table\n");
        goto cleanup;
    }

    tigris_mem_t mem;
    tigris_mem_error_t merr = tigris_mem_init(
        &mem, tensor_ptrs, plan.header->num_tensors,
        fast_buf, fast_size, slow_buf, slow_size);
    if (merr != TIGRIS_MEM_OK) {
        fprintf(stderr, "memory init failed: %s\n",
                tigris_mem_error_str(merr));
        goto cleanup;
    }

    /* Allocate and fill every model input. */
    for (uint16_t i = 0; i < plan.header->num_model_inputs; i++) {
        uint16_t in_idx = plan.model_inputs[i];
        merr = tigris_mem_alloc_slow(
            &mem, in_idx, plan.tensors[in_idx].size_bytes);
        if (merr != TIGRIS_MEM_OK) {
            fprintf(stderr, "input allocation failed: %s\n",
                    tigris_mem_error_str(merr));
            goto cleanup;
        }
        memset(mem.tensor_ptrs[in_idx], 1, plan.tensors[in_idx].size_bytes);
    }

    tigris_exec_stats_t stats;
    tigris_exec_error_t eerr = tigris_run_with_workspace(
        &plan, &mem, tigris_dispatch_kernel_s8, NULL, &stats,
        &executor_workspace);
    if (eerr != TIGRIS_EXEC_OK) {
        fprintf(stderr, "inference failed: %s\n", tigris_exec_error_str(eerr));
        goto cleanup;
    }

    uint16_t out_idx = plan.model_outputs[0];
    int8_t *output = (int8_t *)mem.tensor_ptrs[out_idx];
    if (output == NULL || plan.tensors[out_idx].size_bytes < 5) {
        fprintf(stderr, "output is missing or shorter than five bytes\n");
        goto cleanup;
    }
    if (printf("Output[0..4]: %d %d %d %d %d\n"
               "Fast arena peak: %lu bytes\n",
               output[0], output[1], output[2], output[3], output[4],
               (unsigned long)mem.fast_peak) < 0) {
        fprintf(stderr, "could not write output\n");
        goto cleanup;
    }

    result = 0;

cleanup:
    if (f != NULL && fclose(f) != 0) result = 1;
    free(tensor_ptrs);
    free(slow_buf);
    free(fast_buf);
    free(plan_storage);
    return result;
}
```

## ESP-IDF deployment

On ESP32 targets, store the `.tgrs` plan on a dedicated flash partition and use
`esp_partition_mmap()` so the loader can reference it without a RAM copy.
Allocate the fast arena with `heap_caps_malloc(MALLOC_CAP_INTERNAL)` and, when
available, the slow buffer with `heap_caps_malloc(MALLOC_CAP_SPIRAM)`. Keep the
executor workspace outside the task stack unless it is explicitly budgeted
there. Do not use a fixed stack recommendation across models, backends, and
ESP-IDF releases: start from compiler stack reports for the complete firmware,
include RTOS and interrupt margin, and validate the chosen task size with the
ESP-IDF high-water-mark APIs under worst-case inference load.

## Error handling

Check every return code. All API functions return typed error enums:

**Loader errors** (`tigris_error_t`): See [API Reference](/0.6.0/runtime/api-reference/) for the full error enum.

**Memory errors** (`tigris_mem_error_t`): See [API Reference](/0.6.0/runtime/api-reference/) for the full error enum.

**Executor errors** (`tigris_exec_error_t`): See [API Reference](/0.6.0/runtime/api-reference/) for the full error enum.

Use `tigris_error_str()`, `tigris_mem_error_str()`, and `tigris_exec_error_str()` to convert error codes to human-readable strings.
