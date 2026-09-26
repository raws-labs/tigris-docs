---
title: "tigris inspect"
description: "Use tigris inspect to read the interface, operators, schedule and memory records of an ONNX model or a compiled .tgrs plan without compiling or running it."
sidebar:
  order: 205
slug: 0.11.0/toolchain/inspect
---

Read what an ONNX model declares, or what a compiled `.tgrs` plan records,
without compiling or executing anything.

## Usage

```bash
tigris inspect MODEL [OPTIONS]
```

The format is detected from the file contents, so the extension does not matter.

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | ONNX model or `.tgrs` plan |
| `-v`, `--verbose` | flag | no | Add operators, tensors and execution-plan details |
| `--json` | flag | no | Print complete, versioned metadata as JSON |

## ONNX models

For an ONNX file, `inspect` reports the declared graph: IR version, opsets,
inputs and outputs with their declared shapes, the operator mix and the
initializers.

```bash
tigris inspect model.onnx
```

```
╭───────────────────────────── TiGrIS Inspect - g ─────────────────────────────╮
│ Format        ONNX graph                                                     │
│ File size     991 B                                                          │
│ IR version    13                                                             │
│ Opsets        ai.onnx: 13                                                    │
│ Input         x  float32 [1, 3, 32, 32]                                      │
│ Output        y  float32 [1, 8, 32, 32]                                      │
│ Operators     1: Conv                                                        │
│ Initializers  1 dense, 0 sparse, 0 external dense                            │
╰──────────────────────────────────────────────────────────────────────────────╯
Declared graph metadata. Shapes are not inferred; external tensor data is not
loaded.
```

Shapes are shown as the model declares them. `inspect` does not infer shapes and
does not load external tensor data. To check whether a model fits a memory
budget, use [`tigris analyze`](/0.11.0/toolchain/analyze/).

## Compiled plans

For a `.tgrs` plan, `inspect` reports the plan schema, the interface in stored
axis order and declared dtype, the operators, the schedule and the memory the
compiler recorded. This is the interface [`tigris run`](/0.11.0/toolchain/run/)
expects its inputs in.

```bash
tigris inspect downloaded-model/model.tgrs
```

```
╭─────────────────────────── TiGrIS Inspect - model ───────────────────────────╮
│ Format     TiGrIS execution plan                                             │
│ File size  16.19 KiB                                                         │
│ Schema     7                                                                 │
│ Input      input  float32  (stored: float32 [1, 168], model order)           │
│ Output     output  float32  (stored: float32 [1, 24], model order)           │
│ Operators  1: Gemm                                                           │
│ Schedule   1 stage, 0 tiled, 0 chains                                        │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────────── Memory ───────────────────────────────────╮
│ Fast budget               1.00 KiB                                           │
│ Recorded graph peak       768 B                                              │
│ Weights (including bias)  15.84 KiB                                          │
│ Weight storage            Read in place (XIP), uncompressed                  │
╰──────────────────────────────────────────────────────────────────────────────╯
Memory values are compiler records, not measured runtime peak or total RAM.
```

The memory panel shows what the compiler recorded in the plan. It is not a
measured runtime peak and not the total RAM of an application. A plan's schema
number says which plan format it uses; it does not by itself name a runtime
version.

## JSON output

`--json` prints everything `inspect` reads, with exact byte counts and without
weight values. The top-level `inspection_version` field versions the document,
so scripts can check it before reading further.

```bash
tigris inspect model.tgrs --json
```

## What inspect does not do

- It does not compile, execute or validate kernel numerics.
- It does not infer shapes or load external ONNX tensor data.
- It needs no native library, so it works on every platform `tigris-ml`
  installs on. See [Installation](/0.11.0/getting-started/installation/).
