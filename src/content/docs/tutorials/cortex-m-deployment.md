---
title: "Deploy on Arm Cortex-M"
description: "Run a TiGrIS-compiled INT8 model on an Arm Cortex-M board with CMSIS-NN, from ONNX to on-device inference."
sidebar:
  order: 100
---

This guide takes an INT8 model onto an Arm Cortex-M microcontroller with the
CMSIS-NN kernels, from an ONNX file to an inference running on the board and
printing its output over serial. It uses the
[`tigris-cortex-m`](https://github.com/raws-labs/tigris-cortex-m) target-support
package, which supplies the board bring-up and the `cmsis-nn` build that the
runtime's kernel adapter links against.

The primary board here is the **NUCLEO-H753ZI** (Cortex-M7). The
[Adapting to other boards](#adapting-to-other-boards) section covers the
NUCLEO-F446RE and the Raspberry Pi Pico 2 (RP2350).

## What you'll do

1. Build and flash the package's ready-made example, to confirm your toolchain
   and board work end to end.
2. Compile your own model, generate a CMSIS-NN deployment core, embed it, and
   build firmware for it.
3. Read the on-device output and check it against the host.

## Prerequisites

- Python 3.10+ with `tigris-ml` installed (`pip install tigris-ml`)
- The Arm GNU toolchain (`arm-none-eabi-gcc`) and CMake 3.20+
- A NUCLEO-H753ZI board and a way to flash it (ST-LINK via `st-flash`, the
  STM32CubeProgrammer CLI, or OpenOCD)
- A serial terminal (for example `screen`, `minicom`, or `tio`) for the
  ST-LINK virtual COM port

The package fetches its remaining dependencies (the TiGrIS runtime, CMSIS-NN,
CMSIS-Core, and the STM32 CMSIS-Device pack) by pinned commit at configure
time, so an internet connection is needed for the first build.

## Step 1: Get the target-support package

```bash
git clone https://github.com/raws-labs/tigris-cortex-m.git
cd tigris-cortex-m
```

It ships a pre-generated example under `examples/ds_cnn/` (a DS-CNN keyword
spotter compiled for a 64 KiB budget), so you can build a known-good firmware
before touching your own model.

## Step 2: Build and flash the example

```bash
cmake -B build -DTIGRIS_BOARD=nucleo_h753zi
cmake --build build
```

This produces `build/tigris_firmware.bin`. Flash it to the start of flash:

```bash
st-flash write build/tigris_firmware.bin 0x08000000
```

Open the ST-LINK virtual COM port at 115200 baud. The firmware runs one
inference on a fixed input and prints the INT8 output vector and the cycle
count. On the H753 at 480 MHz, DS-CNN runs in roughly 11 ms; exact numbers are
compiler and toolchain results and may shift between releases.

If this works, your toolchain, board, and flashing setup are good, and the rest
of the guide only swaps in a different model.

## Step 3: Compile your model

Produce a binary plan sized to the board's SRAM:

```bash
tigris compile model.onnx -m 128K -o model.tgrs
```

The `-m` budget is the fast (SRAM) pool. It must fit the board's usable SRAM,
because the CMSIS-NN fast arena is provisioned at the **budget**, not at the
(smaller) activation peak. The H753 has ample SRAM; on a smaller part, lower the
budget until the plan fits. See the [`compile`](/toolchain/compile/)
reference for the full flag set and [Operator and Backend
Support](/runtime/operator-support/) for what the
CMSIS-NN backend accelerates.

## Step 4: Generate the CMSIS-NN deployment core

```bash
tigris codegen model.tgrs --backend cmsis-nn --format core \
  --output tigris_codegen_core.c --header tigris_codegen_core.h
```

`--format core` emits a self-contained deployment core. It loads the embedded
plan, prepares the CMSIS-NN backend, and runs the schedule, rather than a full
example application. The board firmware calls into it. See
[`codegen`](/toolchain/codegen/) for the other formats
and backends.

## Step 5: Embed the plan

The `.tgrs` plan is data, not code, so convert it to a C array the firmware can
carry in flash:

```bash
python tools/bin2c.py model.tgrs model_blob.c --symbol g_tigris_plan
```

## Step 6: Swap in your model and rebuild

Replace the three files in `examples/ds_cnn/` with the ones you just generated,
`tigris_codegen_core.c`, `tigris_codegen_core.h`, and `model_blob.c`, then
rebuild, sizing the arenas to your plan and board:

```bash
cmake -B build -DTIGRIS_BOARD=nucleo_h753zi \
  -DTIGRIS_APP_FAST_ARENA_BYTES=131072
cmake --build build
```

Set `-DTIGRIS_APP_FAST_ARENA_BYTES` to at least your plan's `-m` budget, and
`-DTIGRIS_APP_SLOW_ARENA_BYTES` if the plan uses a slow pool. If the linker
reports the `.bss` section overflowing SRAM, the arenas are larger than the
board holds. Lower the budget and the arena together.

## Step 7: Flash and read the output

```bash
st-flash write build/tigris_firmware.bin 0x08000000
```

Read the serial port again: the firmware prints your model's INT8 output vector
and its cycle count.

## Step 8: Check the output against the host

The runtime is designed to produce the same INT8 output on the device as on the
host for the same plan and input. To confirm your deployment, run the same model
and input through the host, for example the `reference` backend, or ONNX
Runtime on the original float model, and compare the INT8 vectors. They should
match exactly; a mismatch usually means the input fed on-device differs from the
one you compared against.

## Adapting to other boards

- **NUCLEO-F446RE** (Cortex-M4F, 128 KiB SRAM): build with
  `-DTIGRIS_BOARD=nucleo_f446re`. The smaller SRAM is the binding constraint:
  keep the `-m` budget and the fast arena within 128 KiB.
- **Raspberry Pi Pico 2 / RP2350** (Cortex-M33): a separate pico-sdk build
  producing a `.uf2`. See the package's
  [README](https://github.com/raws-labs/tigris-cortex-m#readme) for the
  `PICO_SDK_PATH` invocation and flashing.

To bring up a board the package doesn't cover, implement `include/tigris_hal.h`
for it (clock, a UART for `printf`, a cycle counter) and add a board file; the
package README's "Porting a board" section walks through it.

## Troubleshooting

- **`.bss` overflows SRAM at link time**: the arenas exceed the board's SRAM.
  Lower the plan's `-m` budget and `-DTIGRIS_APP_FAST_ARENA_BYTES` together.
- **`Unknown board` at configure time**: pass a supported `-DTIGRIS_BOARD`
  value (`nucleo_f446re`, `nucleo_h753zi`, or the pico2 project).
- **First configure fails to fetch dependencies**: the pinned sources are
  fetched over the network; for an offline build, point each at a local mirror
  with `-DFETCHCONTENT_SOURCE_DIR_<NAME>=<path>` (see the package README).

## Next steps

- [`compile`](/toolchain/compile/) and
  [`codegen`](/toolchain/codegen/): the full CLI
  reference.
- The [`tigris-cortex-m`](https://github.com/raws-labs/tigris-cortex-m) package:
  board files, the example, and porting notes.
