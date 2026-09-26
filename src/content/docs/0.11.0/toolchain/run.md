---
title: "tigris run"
description: "Use tigris run to execute a compiled .tgrs plan on your own machine with the reference runtime bundled in tigris-ml, from the command line or Python."
sidebar:
  order: 235
slug: 0.11.0/toolchain/run
---

Execute a compiled `.tgrs` plan on your own machine, with no board attached.
Platform wheels of `tigris-ml` bundle the portable C runtime as a host library,
so `tigris run` executes the same runtime code a device runs on its reference
kernels.

## Usage

```bash
tigris run MODEL --input FILE --output FILE [OPTIONS]
```

## Options

| Flag | Type | Required | Description |
|------|------|----------|-------------|
| `MODEL` | path | yes | Compiled `.tgrs` plan |
| `--input` | `FILE` or `NAME=FILE` (multiple) | yes | Input tensor as `.bin` or `.npy`; name each file with `NAME=FILE` when the model has several inputs |
| `--output` | path | yes | `.bin` or `.npy` for one output tensor, `.npz` for named outputs; the file must not exist |
| `--json` | flag | no | Print execution metadata as JSON |

## Inputs and outputs

Tensors use the stored axis order and the declared interface dtype that
[`tigris inspect`](/0.11.0/toolchain/inspect/) reports for the plan. A `.bin` file holds
the contiguous little-endian elements; a `.npy` file must carry the declared
shape and dtype. `run` does no preprocessing: images, audio or CSV data have to
be converted to the model's input tensor first.

## Run a zoo model

A [zoo](/0.11.0/toolchain/zoo/) download ships an example input and the matching
reference output:

```bash
tigris zoo fetch electricity-hourly -o downloaded-model
tigris run downloaded-model/model.tgrs \
    --input downloaded-model/example-input.bin --output prediction.bin
```

```
╭───────────────────────────────── TiGrIS Run ─────────────────────────────────╮
│ Runtime          0.10.2                                                      │
│ Source           bundled                                                     │
│ Backend          Host reference                                              │
│ Output           prediction.bin                                              │
│ Fast arena peak  768 B                                                       │
│ Slow arena peak  768 B                                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
Arena peaks exclude the plan, executor workspace, and Python process memory.
```

`example-output.bin` is the ONNX Runtime output for the example input, so compare
numerically, not byte for byte. Float results differ in the last bits:

```python
import numpy as np

prediction = np.fromfile("prediction.bin", dtype="<f4")
reference = np.fromfile("downloaded-model/example-output.bin", dtype="<f4")
print(np.abs(prediction - reference).max())   # 2.4e-07 for electricity-hourly
```

The tolerance each zoo model is validated against is recorded in its
`evaluation.json`.

When the plan sits next to a zoo `manifest.json` that lists the plan's file
name, `run` also checks the plan's checksum and that the loaded runtime, bundled
or selected with `TIGRIS_HOST_LIBRARY`, is within the model's runtime range. The
range comes from `download.json` when present, whose artifact ID must match the
manifest, and from the manifest otherwise. `run` refuses to execute if a check
fails.

## JSON output

```bash
tigris run model.tgrs --input input.bin --output prediction.bin --json
```

```json
{
  "runtime_version": "0.10.2",
  "runtime_source": "bundled",
  "backend": "reference",
  "output": "prediction.bin",
  "memory": {
    "fast_capacity_bytes": 1024,
    "slow_capacity_bytes": 1664,
    "fast_peak_bytes": 768,
    "slow_peak_bytes": 768,
    "executor_workspace_bytes": 51
  },
  "tensors": [
    {"name": "output", "dtype": "float32", "shape": [1, 24]}
  ]
}
```

`runtime_source` is `bundled`, or `TIGRIS_HOST_LIBRARY=` followed by the resolved
absolute path of the library that was loaded, for example
`TIGRIS_HOST_LIBRARY=/path/to/libtigris_host.so`. The arena peaks are what the runtime used in this run. They
exclude the plan itself, the executor workspace and the Python process.

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

`run` executes the plan on the portable float32 and int8 reference kernels. A
successful run shows that the plan loads and executes on that path. It does not
compare outputs: correctness takes a numerical comparison against a reference,
as above. Parity results recorded for a zoo model apply to the runtime releases
it was tested with, not to every compatible release. `run` does not exercise ESP-NN or CMSIS-NN kernels and says nothing about
on-device latency or memory placement; use the target backends and
[the tutorials](/0.11.0/tutorials/esp-idf-deployment/) for that.

## Availability

The host library ships in the platform wheels for Linux x86-64 and ARM64 (glibc
2.28 or newer), macOS 11 or newer on x86-64 and ARM64, and Windows x86-64. Elsewhere pip
installs the pure Python wheel, where every other command works and `run` stops
with "Bundled host runtime is unavailable".

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
variable is set, `run` uses that library or fails with an error naming it; it
never falls back to the bundled one. `tigris --version`, the run panel and
`run --json` all show which library ran. See [Host Library](/0.11.0/runtime/host-library/)
for the C API.
