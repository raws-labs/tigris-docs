---
title: "tigris zoo"
description: "Use tigris zoo to find and download precompiled .tgrs models with their runtime requirements, evaluation and license."
sidebar:
  order: 250
---

Find and download precompiled `.tgrs` models from the
[TiGrIS model zoo](https://huggingface.co/raws-labs/tigris-zoo). Downloads need
no account.

## Usage

```bash
tigris zoo list [OPTIONS]
tigris zoo fetch [MODEL] [OPTIONS]
tigris run downloaded-model/model.tgrs --input downloaded-model/example-input.bin --output prediction.bin
tigris codegen downloaded-model/model.tgrs --format core -o model.c
```

## Filters

`list` and `fetch` take the same filters. All are optional; unspecified
resources stay unconstrained.

| Flag | Description |
|------|-------------|
| `--model` | Model name (`fetch` takes it as the first argument) |
| `--category` | Task category, e.g. `time-series-forecasting` |
| `--runtime` | A runtime release the build must be compatible with, e.g. `0.11.4` |
| `--backend` | Kernel backend, e.g. `reference` |
| `--quantization`, `--quant` | Weight and activation precision, e.g. `float32` |
| `-m`, `--mem` | Maximum fast arena; repeat for the slow arena |
| `-f`, `--flash` | Maximum plan size |

`-m` and `-f` bound the plan's own arenas and size, not total application RAM or
flash. They take a byte count or a number with a `K` or `M` suffix, such as `64K`,
and a size of 0 is a usage error. The newest published build that matches wins.

## `tigris zoo list`

| Flag | Description |
|------|-------------|
| `--json` | Print the matching catalog entries as JSON |
| `-v`, `--verbose` | Show artifact IDs, runtime compatibility and publication details |

Without `--verbose` it prints a table of model, category, precision and memory:

```bash
tigris zoo list
```

```text
model                category               precision    fast RAM    slow RAM         plan
vww-mobilenetv1      Person detection       int8        64.00 KiB   27.03 KiB   283.56 KiB
vww-mobilenetv1      Person detection       int8        10.00 KiB   45.00 KiB   283.97 KiB
resnet8              Image classification   int8        48.00 KiB    3.03 KiB    87.56 KiB
har-cnn1d            Activity recognition   int8         4.00 KiB       800 B    38.00 KiB
kws-ds-cnn           Keyword spotting       int8        16.00 KiB       544 B    39.64 KiB
electricity-hourly   Forecasting            float32      1.00 KiB       768 B    16.19 KiB
```

With `-v`, each build gets its artifact ID, runtime range, tested runtime releases,
plan schema, backends and publication date:

```text
resnet8-s8-s9-48k-3b7f5d5e956844e0   resnet8, Image classification, int8
  runtime     >= 0.11.1, no known upper bound
  tested      0.11.3, 0.11.4, 0.12.0, 0.12.1
  schema      9, backends reference
  memory      fast 48.00 KiB, slow 3.03 KiB, plan 87.56 KiB
  published   2026-09-28T06:44:18.327025Z
```

## `tigris zoo fetch`

| Flag | Description |
|------|-------------|
| `--artifact ID` | Fetch an exact build, including a withdrawn one (with a warning) |
| `-o`, `--output` | New destination directory; defaults to the artifact ID |

`fetch` verifies every file hash and never replaces an existing directory.

## Global options

| Flag | Description |
|------|-------------|
| `--offline` | Use the Hugging Face cache without network requests |
| `--catalog FILE` | Read a local zoo snapshot instead of the model repository |
| `--revision` | Catalog revision to read; defaults to `main` |

## What a download contains

| File | Content |
|------|---------|
| `model.tgrs` | The plan |
| `readme.md` | Input and output conventions, normalization, evaluation and limitations |
| `evaluation.json` | Task measurements and runtime parity results |
| `example-input.bin`, `example-output.bin` | One input tensor and the reference output the model was validated against |
| `license.txt` | The model's license and data attribution |
| `manifest.json` | The build record as published: compiler, sources, file hashes |
| `download.json` | The current runtime constraints, tested runtime releases and catalog revision |

Use `download.json` for dependency decisions. Runtime ranges include both ends;
a missing maximum means no known upper bound. Tested releases are listed
separately, and matching a range does not claim every release in it was tested.

## Run a download

On a native wheel, [`tigris run`](/toolchain/run/) executes a download directly
on your machine with the bundled reference runtime:

```bash
tigris zoo fetch electricity-hourly -o downloaded-model
tigris inspect downloaded-model/model.tgrs
tigris run downloaded-model/model.tgrs \
    --input downloaded-model/example-input.bin --output prediction.bin
```

Before executing, `run` checks the plan's checksum against `manifest.json` and
the loaded runtime against the model's runtime range, taken from `download.json`
(which must belong to the same artifact) or else from the manifest.
The model's `readme.md` names the reference behind `example-output.bin`, and
`evaluation.json` records the parity measured against it, exact or within a
tolerance, for the runtime releases the model was tested with. Compare
`prediction.bin` with it accordingly.

For a device, the runtime is supplied separately: the C runtime from
[tigris-runtime](https://github.com/raws-labs/tigris-runtime) or the ESP-IDF
component, at a release inside the model's runtime range.
