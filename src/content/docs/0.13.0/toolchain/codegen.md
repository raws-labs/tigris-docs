---
title: "tigris codegen"
description: "Use tigris codegen to generate backend-specific C source from a compiled .tgrs plan, wiring up runtime memory and kernel dispatch."
sidebar:
  order: 230
slug: 0.13.0/toolchain/codegen
---

Generate backend-specific C source code from a compiled `.tgrs` plan. The generated harness sets up runtime memory, selects the requested kernel dispatch, and loads the plan from the target's normal deployment location.

See [Operator and Backend Support](/0.13.0/runtime/operator-support/)
for the generated native, fallback, and unsupported routes behind each
selection.

## Usage

```bash
tigris codegen PLAN [OPTIONS]
```

## Options

| Flag | Type | Description |
|------|------|-------------|
| `PLAN` | path | Compiled `.tgrs` plan |
| `-b`, `--backend` | `reference`, `cmsis-nn`, `esp-nn` | Kernel backend (default `reference`). The accelerated backends are int8 only; a float32 plan on them gets the reference kernels. |
| `-o`, `--output` | path | C file to write. The plan's kernel selection goes next to it as `<stem>_kernels.h`. |
| `--format` | `app`, `core` | `app` is a standalone harness (default), `core` is glue for an existing application |
| `--header` | path | Header for `--format core`; defaults to a sibling `.h` file |
| `--name` | C identifier | Public-symbol prefix for `--format core` (default `tigris_codegen`) |

## Examples

A standalone harness for a Cortex-M target with CMSIS-NN kernels:

```bash
tigris codegen vww.tgrs --backend cmsis-nn -o vww.c
```

```text
wrote vww.c   app, cmsis-nn backend, 5.43 KiB, kernels vww_kernels.h
```

The plan stays a separate file; the generated C loads it at run time.
`vww_kernels.h` enables only the kernels this plan uses; see
[Link only the kernels a plan uses](/0.13.0/runtime/integration/#link-only-the-kernels-a-plan-uses).

Portable C99 on the reference kernels, and the ESP32 family with ESP-NN:

```bash
tigris codegen model.tgrs -o model.c
tigris codegen model.tgrs --backend esp-nn -o model_esp.c
```

Generate an embeddable deployment core for an existing application:

```bash
tigris codegen model.tgrs --backend cmsis-nn --format core \
  -o generated/tigris_codegen_core.c \
  --header generated/tigris_codegen_core.h \
  --name model_codegen
```

## Output formats

### `app` (default)

`--format app` emits a self-contained example harness with an entry point and
the backend's normal plan-location convention. It is the quickest path for a
new deployment or for verifying a generated plan. The harness reserves exactly
the fast and slow arenas the plan needs.

### `core`

`--format core` emits a C source file and public header for embedding in an
application that already owns its entry point, flash placement, arena buffers,
input source, or telemetry. It is backend-specific but platform-neutral: no
hardware target is selected by codegen.

The generated core provides functions to load the supplied plan bytes, initialize
runtime memory and the selected backend, reset activations between inferences,
obtain its dispatcher, and run inference. The embedding application supplies plan bytes, fast and slow arenas,
the tensor-pointer table, and optionally an input-initialization callback. This
fits bare-metal firmware, RTOS applications, and custom instrumentation without
copying generated runtime setup into application code.

The generated header exports model-specific constants for tensor-table capacity,
plan budget, compressed-weight reserve, and the fast and slow arenas the plan
needs (`<NAME>_CORE_FAST_ARENA_BYTES`, `<NAME>_CORE_SLOW_ARENA_BYTES`). When the slow
figure is 0, pass a NULL slow arena of size 0. Read the outputs before the next
reset: reset drops every tensor, and when the plan reads its inputs in place its
outputs stay in the fast arena, where the next inputs go. Use `--name` to choose a C-identifier
prefix for its functions, types, and macros; this lets multiple generated cores
coexist in the same firmware. `--header` and `--name` apply only to
`--format core`.

Code of either format places model inputs with
[`tigris_input_alloc()`](/0.13.0/runtime/api-reference/#tigris_input_alloc) and stops
the build with `#error` on runtime headers that do not define
`TIGRIS_RUNTIME_VERSION_MAJOR`, which predate that call. The core's init and
reset return a `tigris_error_t`: `TIGRIS_ERR_TENSOR_SLOTS` when the tensor
capacity is below the plan's tensor count, and `TIGRIS_ERR_MEM` when an arena
cannot hold the inputs or the backend's reservation. For a plan of schema 11
or later `<NAME>_CORE_FAST_ARENA_BYTES` is the plan's scheduled peak plus the
compressed-weight reserve, the arena a run of that plan requires on a runtime
whose `TIGRIS_TENSOR_ALIGN` is at most 32. The header stops the build with
`#error` when `TIGRIS_TENSOR_ALIGN` is above 32.

## When to Use Codegen vs. Loading .tgrs

There are two ways to deploy a compiled plan to your device:

### Loading .tgrs at runtime

The plan is stored in a flash partition and loaded at runtime with `tigris_plan_load()`.

**Advantages:**
- Swap models without recompiling firmware by reflashing the plan partition
- Smaller firmware binary (plan is separate)
- Multiple models can share the same firmware

**When to use:** Devices with a flash partition table, applications that need model updates in the field, development/iteration workflows.

```bash
# Compile the plan
tigris compile model.onnx -m 256K -o model.tgrs

# Write the plan to the partition named plan (ESP-IDF)
parttool.py write_partition --partition-name plan --input model.tgrs
```

### Codegen `app` harness

The generated C file contains the runtime setup and backend dispatch glue. The plan remains a `.tgrs` binary: ESP-IDF builds load it from a flash partition, Cortex-M examples reference linker-provided flash symbols, and POSIX examples read it from a file path.

**Advantages:**
- Keeps target-specific runtime setup in generated code
- Lets the same `.tgrs` plan be used with different backend harnesses
- Supports targets with different plan placement conventions

**When to use:** Firmware projects that want generated runtime glue while keeping the plan as a deployable `.tgrs` artifact.

```bash
# Compile the plan, then generate C code
tigris compile model.onnx -m 256K -o model.tgrs
tigris codegen model.tgrs --backend cmsis-nn -o model.c

# Add model.c to your firmware build and deploy model.tgrs for your target
```

Use `--format core` instead when the application must retain control of its
entry point, plan placement, memory ownership, or measurement/reporting loop.

### Comparison

| | Manual .tgrs loading | Codegen harness |
|---|---|---|
| Plan location | Application-defined | Backend-specific: file, partition, or linker symbols |
| Model swap | Reflash or replace plan artifact | Reflash or replace plan artifact |
| Build complexity | Hand-written setup code | Generated setup code (`app` or embeddable `core`) |
| Firmware size | Plan separate | Plan separate |
| Flash partition needed | Target-dependent | Target-dependent |
