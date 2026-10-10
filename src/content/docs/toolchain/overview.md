---
title: "CLI Overview"
description: "The tigris command line: six commands, each answering one question about a model, with flags that add depth instead of more commands."
sidebar:
  order: 200
  label: "Overview"
---

`tigris` has six commands. Each one answers one question about a model or a plan,
and flags take the same command further instead of adding commands.

| Question | Command |
|---|---|
| What is in this file, and will it convert? | [`tigris inspect`](/toolchain/inspect/) |
| Does it fit my memory? If not, what stops it? | [`tigris analyze`](/toolchain/analyze/) |
| What does the plan do at run time, load by load? | [`tigris analyze --trace`](/toolchain/analyze/#execution-trace) |
| Write the plan. | [`tigris compile`](/toolchain/compile/) |
| What does the plan compute on my machine? | [`tigris run`](/toolchain/run/) |
| Give me the C code for my target. | [`tigris codegen`](/toolchain/codegen/) |
| Is there a precompiled model I can start from? | [`tigris zoo`](/toolchain/zoo/) |

## A session

From a TFLite or ONNX file to C source, with a check on the host in between:

```bash
tigris inspect model.tflite                        # interface, operators, does it convert
tigris analyze model.tflite -m 32K -m 1M           # verdict, memory, budgets that also fit
tigris analyze model.tflite -m 32K -m 1M --trace   # every load and spill, measured
tigris compile model.tflite -m 32K -m 1M -o model.tgrs
tigris run model.tgrs --input input.bin --output prediction.bin
tigris codegen model.tgrs --backend cmsis-nn -o model.c
```

`analyze` and `compile` take the same model and the same budgets, and `compile`
prints the same report before it writes. Everything up to `codegen` runs on your
machine; no board is needed until you deploy. The
[Quickstart](/getting-started/quickstart/) walks through this sequence with a real
model and its output.

## One command, more depth

The default output of every command is a short report. Flags add to it:

| Flag | Commands | Adds |
|---|---|---|
| `-v` | `analyze`, `inspect`, `zoo list` | The detail table: stages, steps and tensors, or every event of a trace |
| `--json` | `analyze`, `inspect`, `run`, `zoo list` | The same facts as a JSON document for scripts, with exact byte counts |
| `--trace` | `analyze` | Runs the plan once on the host runtime and reports what it did instead of what was planned |
| `--input` | `analyze --trace`, `run` | Real input data instead of zeros |

For example, `analyze` goes from a verdict to a full account of one inference:

```bash
tigris analyze model.tflite -m 32K            # fits or not, and why
tigris analyze model.tflite -m 32K -v         # plus one row per stage
tigris analyze model.tflite -m 32K --trace    # per-stage traffic and peaks, measured
tigris analyze model.tflite -m 32K --trace -v # every allocation, load, spill and kernel call
tigris analyze model.tflite -m 32K --json     # all of it for a script
```

## Memory budgets

`analyze` and `compile` take the memory of the target, not tuning knobs:

| Flag | Meaning |
|---|---|
| `-m SIZE` | Fast memory for activations, usually on-chip SRAM. The compiler partitions and tiles the model until every stage fits it. |
| `-m SIZE` again | Slow memory, such as PSRAM or another writable RAM region. Tensors that cross a stage boundary wait there. Without it, `analyze` reports what slow memory the plan needs and does not check it. |
| `-m FAST+SLOW` | Both in one flag: `-m 256K+8M` is `-m 256K -m 8M`. |
| `-f SIZE` | Flash for the plan. `analyze` reports whether the plan fits; `compile` writes nothing when it does not. |

Sizes are bytes, with optional `K` or `M` suffixes (binary, case-insensitive):
`262144`, `256K`, `256KB`, `4M` and `2.5M` all work.

`--input-shape NAME:1x3x224x224` fixes the shape of an input the model leaves
open, such as a free batch dimension. Without it, an open dimension is bound to 1
and a warning says so.

In `tigris zoo`, `-m` and `-f` mean something else: they filter the catalog for
plans whose fast arena, slow arena and size stay within those numbers.

## Files

The format is read from the file contents, so the extension does not matter.

| Command | ONNX | TFLite | `.tgrs` plan |
|---|---|---|---|
| `inspect` | yes | yes | yes |
| `analyze`, `compile` | yes | yes | |
| `run`, `codegen` | | | yes |

## Output

- The report goes to standard output. Warnings start with `warning:` and errors
  with `Error:`, both on standard error.
- `compile` exits with status 1 and writes no file when the plan does not fit a
  budget it was given. `analyze` exits 0 whenever it can read the model, whatever
  the verdict; read the verdict line or `--json` to act on it. A file that cannot
  be read exits with 1, a wrong flag or a missing file with 2.
- Colors appear only on a terminal. Set `NO_COLOR` to turn them off there too.
- JSON documents carry a version: `report` and `version` in `analyze --json`,
  `inspection_version` in `inspect --json`. Check it before reading further.

`tigris --version` prints the compiler version and the host runtime it loads.

## Replaced commands

Two older commands still run, print a note on standard error and are removed in
the next release:

| Instead of | Use |
|---|---|
| `tigris plan` | `tigris compile`, then `tigris inspect PLAN --json` |
| `tigris simulate` | `tigris analyze --trace` |

`--trace` measures on the runtime what `simulate` used to estimate.
