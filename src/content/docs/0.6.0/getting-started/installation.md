---
title: Installation
description: Install the TiGrIS Python toolchain and C99 runtime to compile and
  deploy ML models onto embedded devices.
sidebar:
  order: 115
slug: 0.6.0/getting-started/installation
---

## Python toolchain

Requires Python 3.10 or later.

```bash
pip install tigris-ml
```

This pulls in all runtime dependencies automatically.

### Development install

Clone the repo and install with dev extras for testing:

```bash
git clone https://github.com/raws-labs/tigris.git
cd tigris
pip install -e ".[dev]"
```

The `[dev]` extra adds `pytest` and `onnxruntime` (used for reference output validation).

### Verify

```bash
tigris --version
```

## C runtime

The runtime is a self-contained C99 project. No external dependencies, no package manager.

```bash
git clone https://github.com/raws-labs/tigris-runtime.git
cd tigris-runtime
cmake -B build && cmake --build build
```

Run the test suite:

```bash
./build/test_kernels    # unit tests for all kernel backends
./build/test_e2e test/fixtures/mobilenetv2.tgrs test/fixtures/mobilenetv2.reference.bin
```

## Platform notes

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (x86\_64, aarch64) | Supported | Primary development platform |
| macOS (Apple Silicon, Intel) | Supported | |
| Windows | Untested | Native Windows builds are untested. Using WSL is recommended. |

## Next step

Ready to deploy? Follow the [Quickstart](/0.6.0/getting-started/quickstart/).
