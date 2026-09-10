---
title: "tigris codegen"
description: "Use tigris codegen to generate backend-specific C source from a compiled .tgrs plan, wiring up runtime memory and kernel dispatch."
sidebar:
  order: 230
---

Generate backend-specific C source code from a compiled `.tgrs` plan. The generated harness sets up runtime memory, selects the requested kernel dispatch, and loads the plan from the target's normal deployment location.

See [Operator and Backend Support](/runtime/operator-support/)
for the generated native, fallback, and unsupported routes behind each
selection.

## Usage

```bash
tigris codegen PLAN [OPTIONS]
```

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `PLAN` | path | yes | Compiled `.tgrs` plan file |
| `-b`, `--backend` | choice | no | Kernel backend: `reference`, `cmsis-nn`, `esp-nn` (default: `reference`) |
| `-o`, `--output` | path | no | Output C file path |
| `--format` | choice | no | Output shape: `app` (standalone harness, default) or `core` (embeddable deployment glue) |
| `--header` | path | no | Header output path for `--format core`; defaults to a sibling `.h` file |
| `--name` | C identifier | no | Public-symbol prefix for `--format core` (default: `tigris_codegen`) |

## Examples

Generate reference C code:

```bash
tigris codegen model.tgrs -o model.c
```

Generate for ESP-NN backend:

```bash
tigris codegen model.tgrs --backend esp-nn -o model_esp.c
```

Generate for CMSIS-NN backend:

```bash
tigris codegen model.tgrs --backend cmsis-nn -o model_cmsis.c
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
new deployment or for verifying a generated plan.

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
plan budget, and compressed-weight reserve. Use `--name` to choose a C-identifier
prefix for its functions, types, and macros; this lets multiple generated cores
coexist in the same firmware. `--header` and `--name` apply only to
`--format core`.

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

# Flash to partition
scripts/flash_plan.sh model.tgrs
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
