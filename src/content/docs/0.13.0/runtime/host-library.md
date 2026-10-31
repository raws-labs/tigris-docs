---
title: "Host Library"
description: "The tigris_host shared library runs compiled .tgrs plans on a desktop or server through a small allocating C API; it backs tigris run and the Python Session."
sidebar:
  order: 430
slug: 0.13.0/runtime/host-library
---

`tigris_host` is a shared library built from the runtime sources for desktop and
server machines. Unlike the embedded runtime, which works in arenas the caller
owns, it allocates everything itself behind an opaque session. It is what
[`tigris run`](/0.13.0/toolchain/run/) and `tigris.runtime.Session` call, and it is
bundled in the platform wheels of `tigris-ml`. When the library is available,
`tigris compile` also loads every plan it writes with its loader and writes no
plan the loader refuses.

It runs the portable float32 and int8 reference kernels. A successful run shows
that a plan loads and executes on that path; output correctness takes a
numerical comparison against a reference. It does not exercise ESP-NN or
CMSIS-NN or model device latency.

## Build

```bash
git clone https://github.com/raws-labs/tigris-runtime.git
cd tigris-runtime
cmake -S . -B build-host -DTIGRIS_BUILD_HOST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --config Release --target test_host host_archive
ctest --test-dir build-host -C Release -R host_library --output-on-failure
```

`--config Release` and `-C Release` select the configuration on multi-config
generators such as Visual Studio, which ignore `CMAKE_BUILD_TYPE`; there the
library is written to `build-host/host/Release/`.

This builds `libtigris_host.so` (`.dylib` on macOS, `tigris_host.dll` on
Windows) and, through `host_archive`, a release archive with the library, its
header `tigris_host.h`, the license and a manifest. The manifest records the
runtime version, host ABI, platform, source revision and library checksum. Each
runtime release attaches these archives for the platforms `tigris-ml` ships
native wheels for. On other platforms, set `TIGRIS_HOST_LIBRARY` to a library
you built to use it from `tigris run`; see
[Using your own host library](/0.13.0/toolchain/run/#using-your-own-host-library).

## API

All functions are declared in `host/tigris_host.h`. Calls that can fail return
`NULL` on success or a static error string.

### Version

```c
uint32_t    tigris_host_abi(void);
const char *tigris_host_version(void);
```

`tigris_host_abi()` returns the host ABI number, which changes only when this C
interface changes. `tigris_host_version()` returns the runtime release the
library was built from.

### Sessions

```c
const char *tigris_host_create(const void *data, uint32_t size,
                               uint32_t slow_capacity, tigris_host_t **out);
void        tigris_host_destroy(tigris_host_t *host);
```

`tigris_host_create` copies the plan and allocates the arenas and executor
workspace. A `slow_capacity` of 0 selects a conservative host allocation; it is
not a slow-memory requirement from the compiler. An int8 plan can also execute
float32 sections between Dequantize and Quantize or a float model output.

### Check

```c
const char *tigris_host_check(const void *data, uint32_t size, uint16_t *required);
```

Loads a plan as a target's loader does, without creating a session. It returns
`NULL` when the plan loads and the loader's error message otherwise. The library
is built with every `TIGRIS_MAX_*` limit at 65535, so a refusal is never about
the build. When `required` is not NULL, its five entries receive what the plan
asks of a target build: tensors, stage inputs, stage outputs, chain stages and
spatial operators per chained stage, as `TIGRIS_MAX_*` values; all zero when the
plan is refused.

### Interface

```c
uint32_t    tigris_host_tensor_count(const tigris_host_t *host, int output);
const char *tigris_host_tensor_name (const tigris_host_t *host, int output, uint32_t index);
uint32_t    tigris_host_tensor_bytes(const tigris_host_t *host, int output, uint32_t index);
uint32_t    tigris_host_tensor_dtype(const tigris_host_t *host, int output, uint32_t index);
uint32_t    tigris_host_tensor_rank (const tigris_host_t *host, int output, uint32_t index);
int32_t     tigris_host_tensor_dim  (const tigris_host_t *host, int output, uint32_t index,
                                     uint32_t axis);
```

`output` is 0 for inputs and 1 for outputs. Tensors are described in the plan's
stored axis order and declared interface dtype, the same view
[`tigris inspect`](/0.13.0/toolchain/inspect/) prints. Dtypes use the ONNX codes: 1 is
float32, 2 is uint8, 3 is int8. The interface dtype can differ from the dtype a
plan executes in: an int8 plan can expose float32 or uint8 inputs and outputs.
An invalid query returns 0 or `NULL`.

### Run

```c
const char *tigris_host_run(tigris_host_t *host,
                            const void *const *inputs, const uint32_t *input_sizes,
                            uint32_t input_count,
                            void *const *outputs, const uint32_t *output_sizes,
                            uint32_t output_count);
```

Buffers are passed in interface order with their byte sizes, which must match
`tigris_host_tensor_bytes`. Every run resets the arenas and allocates nothing.
Separate sessions can run concurrently; calls on one session must be
serialized by the caller.

### Metrics

```c
uint64_t tigris_host_metric(const tigris_host_t *host, uint32_t metric);
```

| Metric | Value |
|--------|-------|
| `TIGRIS_HOST_FAST_CAPACITY` | Fast arena size in bytes |
| `TIGRIS_HOST_SLOW_CAPACITY` | Slow arena size in bytes |
| `TIGRIS_HOST_FAST_PEAK` | Fast arena peak of the last run |
| `TIGRIS_HOST_SLOW_PEAK` | Slow arena peak of the last run |
| `TIGRIS_HOST_WORKSPACE_BYTES` | Executor workspace size |
| `TIGRIS_HOST_LOAD_BYTES` | Bytes loaded from slow to fast memory in the last run |
| `TIGRIS_HOST_SPILL_BYTES` | Bytes spilled from fast to slow memory in the last run |
| `TIGRIS_HOST_WEIGHT_BYTES` | Weight bytes written to fast memory in the last run |
| `TIGRIS_HOST_COPY_BYTES` | State and control-flow bytes copied in the last run |
| `TIGRIS_HOST_COMPACTIONS` | Fast- and slow-pool compactions in the last run |
| `TIGRIS_HOST_TILES` | Tiles executed in the last run |
| `TIGRIS_HOST_TENSOR_ALIGN` | Tensor alignment of this build, in bytes |

The peaks cover the execution arenas, not the process memory.

### State

```c
const char *tigris_host_reset_state(tigris_host_t *host);
uint32_t    tigris_host_state_bytes(const tigris_host_t *host);
const char *tigris_host_save_state(const tigris_host_t *host, void *state, uint32_t size);
const char *tigris_host_restore_state(tigris_host_t *host, const void *state, uint32_t size);
```

A session keeps a stateful plan's variables from one run to the next.
`tigris_host_reset_state` returns every variable to its initial value. A
stateless plan has nothing to reset. `tigris_host_state_bytes` is the size of
those variables, 0 for a stateless plan. `tigris_host_save_state` copies them
into a caller-owned buffer of exactly that size, and
`tigris_host_restore_state` copies them back, so the next run starts from the
saved values.

### Trace

```c
void     tigris_host_trace_buffer(tigris_host_t *host, void *events, uint32_t capacity);
uint32_t tigris_host_trace_count (const tigris_host_t *host);
```

`tigris_host_trace_buffer` records the next runs' execution events into a
caller-owned array of `capacity` `tigris_trace_event_t` records
(`include/tigris_trace.h`, see [Execution trace](/0.13.0/runtime/api-reference/#execution-trace));
NULL or a zero capacity stops recording. A run never writes past `capacity`, and
`tigris_host_trace_count` reports how many events the last run emitted, so a count
above the capacity means the array was too small. The host library is built with
tracing on; `tigris analyze --trace` reads these events.
