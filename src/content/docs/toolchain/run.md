---
title: "tigris run"
description: "Use tigris run to execute a compiled .tgrs plan on the runtime bundled in tigris-ml, a TFLite model on TFLite Micro or an ExecuTorch Cortex-M program on ExecuTorch, on your own machine."
sidebar:
  order: 235
---

Execute a model on your own machine, with no board attached. The file picks
the runtime:

| File | Runs on | Needs |
|---|---|---|
| `.tgrs` plan | The TiGrIS runtime bundled in `tigris-ml`, on its reference kernels | A platform wheel, or [your own host library](#using-your-own-host-library) |
| TFLite model | TFLite Micro's reference kernels | The `tflite-micro` package |
| ExecuTorch program (`.pte`) lowered for Cortex-M | ExecuTorch with its cortex_m operators on CMSIS-NN's portable C kernels | `--executorch`, an ExecuTorch 1.6 checkout |

A plan runs the same runtime code a device runs on its reference kernels. A
TFLite model and an ExecuTorch program run on the runtime
[`tigris export`](/toolchain/export/) writes for, so you can run a model and
its export on the same input and
[compare the outputs byte for byte](/toolchain/export/#check-the-outputs).

## Usage

```bash
tigris run MODEL --input FILE --output FILE [--executorch PATH] [--json]
```

## Options

| Flag | Type | Description |
|------|------|-------------|
| `MODEL` | path | Compiled `.tgrs` plan, TFLite model or ExecuTorch program |
| `--input` | `FILE` or `NAME=FILE`, repeatable, required | Input tensor as `.bin` or `.npy`; name each file with `NAME=FILE` when the model has several inputs |
| `--output` | path, required | `.bin` or `.npy` for one output tensor, `.npz` for named outputs; the file must not exist |
| `--executorch PATH` | path | ExecuTorch programs: an ExecuTorch 1.6 checkout with its submodules; the host runner is built from it once and cached |
| `--json` | flag | Execution metadata as JSON |

The format is read from the file contents, so the extension does not matter.
`--executorch` with a plan or a TFLite model is a usage error. An ONNX model or
a PyTorch `.pt2` program runs as a plan: compile it with
[`tigris compile`](/toolchain/compile/) first, as `run` says:

```text
Error: model.onnx is an ONNX model; compile it with tigris compile and run the plan
```

## Inputs and outputs

Tensors use the axis order and dtype that [`tigris inspect`](/toolchain/inspect/)
reports for the file: for a plan, the stored axis order and the declared
interface dtype. A `.bin` file holds the contiguous little-endian elements; a
`.npy` file carries that shape and dtype. An ExecuTorch program also takes a
`.npy` array in its PyTorch sizes, see [Input layout](#input-layout).
`run` does no preprocessing: images, audio or CSV data have to be converted to
the model's input tensor first.

## Run a zoo model

A [zoo](/toolchain/zoo/) download ships an example input and the matching
reference output:

```bash
tigris zoo fetch electricity-hourly -o downloaded-model
tigris run downloaded-model/model.tgrs \
    --input downloaded-model/example-input.bin --output prediction.bin
```

```text
model.tgrs   runtime 0.13.0, bundled, host reference backend
  fast arena peak   768 B   measured; plan and workspace excluded
  slow arena peak     0 B

wrote prediction.bin   1x24 float32
```

`example-output.bin` is the ONNX Runtime output for the example input, so compare
numerically, not byte for byte. Float results differ in the last bits:

```python
import numpy as np

prediction = np.fromfile("prediction.bin", dtype="<f4")
reference = np.fromfile("downloaded-model/example-output.bin", dtype="<f4")
print(np.abs(prediction - reference).max())   # 1.2e-07 for electricity-hourly
```

The tolerance each zoo model is validated against is recorded in its
`evaluation.json`.

When the plan sits next to a zoo `manifest.json` that lists the plan's file
name, `run` also checks the plan's checksum and that the loaded runtime, bundled
or selected with `TIGRIS_HOST_LIBRARY`, is within the model's runtime range. The
range comes from `download.json` when present, whose artifact ID must match the
manifest, and from the manifest otherwise. `run` refuses to execute if a check
fails.

## TFLite models

`run` loads a TFLite model into TFLite Micro's Python interpreter, from the
`tflite-micro` package, which has wheels for Linux x86-64:

```bash
pip install tflite-micro
tigris run vww_96_int8.tflite --input image.bin --output prediction.bin
```

```text
vww_96_int8.tflite   TFLite Micro 0.dev20261023044614, reference kernels

wrote prediction.bin   1x2 int8
```

The first line names the `tflite-micro` release. `run` reports no memory
figures: the arena a model takes on a device is what
[`tigris export`](/toolchain/export/#tflite-models) reports for each
[target](/toolchain/export/#targets). `run` sizes TFLite Micro's arena with the
same arena model, for the 64-bit build in the package, plus what TFLite Micro
takes while it plans the arena or while a kernel prepares. A model outside
that arena model, for example one with several subgraphs, an LSTM or a custom
operator, gets an upper bound instead: every tensor and scratch buffer of
every subgraph placed at once, plus every persistent allocation. A kernel the
package lacks stops TFLite Micro while it loads the model, and its own message
follows `TFLite Micro cannot run MODEL:`:

```text
Error: TFLite Micro cannot run float_gelu_approximate.tflite: Didn't find op for builtin opcode 'GELU'
```

## ExecuTorch programs

`run` takes a program lowered by ExecuTorch's Cortex-M backend and runs its
method `forward` on a host build of ExecuTorch 1.6:

```bash
tigris run mbv2.pte --executorch ~/executorch --input image.bin --output prediction.bin
```

```text
mbv2.pte   ExecuTorch 1.6.0 at 4776f71b6f0d, CMSIS-NN v8.0.0 portable kernels

wrote prediction.bin   1x1000 int8
```

The first line names the ExecuTorch release and commit the runner was built
from and the CMSIS-NN release it links. `run` reports no memory figures: the
RAM the program takes on a 32-bit Cortex-M target is what
[`tigris inspect`](/toolchain/inspect/#executorch-programs) shows.

### The host runner

On the first run against a checkout, `run` builds a host runner, the ExecuTorch
runtime with its cortex_m operators on CMSIS-NN, and says so:

```text
building the ExecuTorch host runner at 4776f71b6f0d, once; it is kept in ~/.cache/tigris/pte-runner
```

The runner stays in the user's cache directory, on Linux
`$XDG_CACHE_HOME/tigris/pte-runner` or `~/.cache/tigris/pte-runner`, and `run` builds it again only
for another commit or other uncommitted changes in the checkout, or for a
`tigris-ml` release that changes the runner. The build
needs CMake, a C++ compiler, network access, since ExecuTorch's Cortex-M
backend fetches CMSIS-NN from GitHub, and torch and PyYAML in the Python that
runs `tigris`, for ExecuTorch's code generator. A checkout of the release tag,
in a directory named `executorch`, works:

```bash
git clone --branch v1.6.0 https://github.com/pytorch/executorch.git ~/executorch
git -C ~/executorch submodule update --init --recursive
```

### Kernels

The host runs CMSIS-NN's portable C kernels. `run` takes programs lowered for
scalar kernels (Cortex-M0, M0+, M3, M23) or DSP kernels (Cortex-M4, M7, M33,
M35P). CMSIS-NN's DSP kernels write the same bytes as its portable C kernels,
so the host reproduces the outputs of a DSP target: a ResNet-8 program gave the
same 200 output bytes for 20 inputs on the host and on a NUCLEO-H753ZI. The DSP
code itself runs only on the target. A program lowered for MVE kernels runs on
its Cortex-M55 or M85 target:

```text
Error: bands_m55.pte was lowered for MVE kernels (Cortex-M55, M85); the host runs programs lowered for scalar or DSP kernels
```

A program that calls an operator the runner does not register stops with
ExecuTorch's own message after `the host runner stopped on MODEL:`.

### Input layout

The method takes one tensor. `inspect` shows its shape in memory order, with
its dim order when that differs from PyTorch's order:

```text
  input    0   1x96x96x3 int8, dim order 0,2,3,1
```

The Cortex-M backend keeps this 1x3x96x96 PyTorch input channels last, so its
bytes in memory run as 1x96x96x3. A `.bin` file holds the bytes in that order.
A `.npy` array can take either view:

| `.npy` shape | Read as |
|---|---|
| The PyTorch sizes, 1x3x96x96, as `torch` prints the tensor | Laid out in the dim order |
| The sizes in memory order, 1x96x96x3, or one dimension | Taken as it is |

Dimensions of size 1 are ignored. An array whose shape reads both ways with
different bytes is refused; give that input as a `.bin` file or a
one-dimensional `.npy`. Messages name a tensor as `inspect` shows it, its shape
in memory order and its dim order:

```text
Error: short.bin holds 100 bytes; the method input takes 27648 (1x96x96x3 int8, dim order 0,2,3,1)
```

A `.bin` output holds the output bytes in memory order, a `.npy` output the
array in its PyTorch sizes, and an `.npz` output one array per method output,
named `0`, `1` and so on.

## JSON output

```bash
tigris run model.tgrs --input input.bin --output prediction.bin --json
```

```json
{
  "runtime_version": "0.13.0",
  "runtime_source": "bundled",
  "backend": "reference",
  "output": "prediction.bin",
  "memory": {
    "fast_capacity_bytes": 1024,
    "slow_capacity_bytes": 1664,
    "fast_peak_bytes": 768,
    "slow_peak_bytes": 0,
    "executor_workspace_bytes": 63
  },
  "tensors": [
    {
      "name": "output",
      "dtype": "float32",
      "shape": [
        1,
        24
      ]
    }
  ]
}
```

For a plan, `runtime_source` is `bundled`, or `TIGRIS_HOST_LIBRARY=` followed by the resolved
absolute path of the library that was loaded, for example
`TIGRIS_HOST_LIBRARY=/path/to/libtigris_host.so`. The arena peaks are what the runtime used in this run. They
exclude the plan itself, the executor workspace and the Python process.

For a TFLite model and an ExecuTorch program the document has no `memory`
and adds `runtime`, `tflite-micro` or `executorch`. A TFLite model's document
has no `runtime_source`. An ExecuTorch program's names the runner's path in
`runtime_source`, adds `runtime_commit`, `runtime_changes` (the hash of
uncommitted changes in the checkout, or `null`) and `cmsis_nn_version`, has
`backend` `cmsis-nn`, and each tensor carries its `dim_order`.

## Python

The same runtime is available as `tigris.runtime.Session`:

```python
import numpy as np
from tigris.runtime import Session

with Session("model.tgrs") as session:
    outputs = session.run({"input": np.load("input.npy", allow_pickle=False)})
    print(session.runtime_version, session.memory)
```

`run` takes a dict of arrays keyed by input name and returns a dict keyed by
output name. `session.inputs` and `session.outputs` describe the interface.
Calls on one session are serialized; separate sessions run independently.
`Session(model, slow_capacity=N)` sets the slow arena size in bytes; the default
is a conservative host allocation, not a compiled requirement.

## What run proves, and what it does not

`run` executes a plan on the portable float32 and int8 reference kernels, a
TFLite model on TFLite Micro's reference kernels and an ExecuTorch program on
CMSIS-NN's portable C kernels. A successful run shows that the file loads and
executes on that path. A model and its export that write the same bytes on an
input compute the same result for that input on that path. For a plan,
correctness takes a numerical comparison against a reference, as above. Parity results recorded for a zoo model apply to the runtime releases
it was tested with, not to every compatible release. On-device latency, memory placement and the ESP-NN and CMSIS-NN optimized
kernels are measured on the target, with the target backends and
[the tutorials](/tutorials/esp-idf-deployment/).

## Availability

The host library ships in the platform wheels for Linux x86-64 and ARM64 (glibc
2.28 or newer), macOS 11 or newer on x86-64 and ARM64, and Windows x86-64. Elsewhere pip
installs the pure Python wheel, where every other command works and `run` on a
plan stops with "Bundled host runtime is unavailable". `run` on a TFLite model
or an ExecuTorch program does not use the host library, so it also works in the
pure Python wheel.

## Using your own host library

On any platform the runtime compiles on, build the host library yourself and
point `TIGRIS_HOST_LIBRARY` at it. This also works with the pure Python wheel.

On Linux and macOS:

```bash
git clone --branch vX.Y.Z https://github.com/raws-labs/tigris-runtime.git
cd tigris-runtime
cmake -S . -B build-host -DTIGRIS_BUILD_HOST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target tigris_host
export TIGRIS_HOST_LIBRARY=$PWD/build-host/host/libtigris_host.so   # .dylib on macOS
tigris --version
```

On Windows with the Visual Studio generator, the configuration is chosen at
build time and the DLL lands in a configuration subdirectory (PowerShell):

```powershell
git clone --branch vX.Y.Z https://github.com/raws-labs/tigris-runtime.git
cd tigris-runtime
cmake -S . -B build-host -DTIGRIS_BUILD_HOST=ON
cmake --build build-host --config Release --target tigris_host
$env:TIGRIS_HOST_LIBRARY = "$PWD\build-host\host\Release\tigris_host.dll"
tigris --version
```

```
tigris, version X.Y.Z (runtime X.Y.Z; TIGRIS_HOST_LIBRARY=/path/to/libtigris_host.so)
```

Use the runtime release that matches your `tigris-ml` version. The library must
implement the host ABI `tigris-ml` expects; a mismatch is refused. When the
variable is set, `run` on a plan uses that library or fails with an error naming it; it
never falls back to the bundled one. `tigris --version`, the first line of `run` output and
`run --json` all show which library ran. See [Host Library](/runtime/host-library/)
for the C API.
