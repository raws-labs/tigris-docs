---
title: Running a Plan on an Emulated Cortex-M55
description: Execute a TiGrIS plan on a bare-metal Cortex-M55 under QEMU, with
  no board, and read back what it used.
sidebar:
  order: 170
slug: 0.9.0/guides/emulated-cortex-m55
---

QEMU emulates an Arm Cortex-M55 well enough to run a TiGrIS plan as bare-metal
firmware and report what it used. That is enough to answer the question a
deployment usually starts with, which is whether a model fits a part at all, and
it answers it before any hardware exists.

What it cannot answer is how fast the model runs. QEMU's Cortex-M55 is
functional, not cycle-accurate, so treat anything it says about time as
meaningless. Use it for correctness and memory placement, and a board for
latency.

## What you need

* `qemu-system-arm` 8.2 or later, which carries the `mps3-an547` machine, an
  Arm Corstone-300 with a Cortex-M55.
* `arm-none-eabi-gcc` 13.2 or later.
* A compiled `.tgrs` plan.

Check the machine is there before anything else:

```bash
qemu-system-arm -machine help | grep an547
```

```
mps3-an547           ARM MPS3 with AN547 FPGA image for Cortex-M55
```

## The memory map is the point

The interesting decision is where each tier lives, because that is what decides
whether a model fits. The AN547 gives you on-chip SRAM and an external DDR
window, and a plan's two arenas can go in either:

| Region | Address | Stands for |
|---|---|---|
| DTCM | `0x20000000` | tightly coupled memory, 512 KiB |
| SRAM | `0x21000000` | on-chip SRAM, 4 MiB |
| DDR | `0x60000000` | external memory |

Putting the plan itself at `0x60000000` and reading it in place models
execute-in-place from external flash, so the weights never occupy SRAM. Put the
fast arena in on-chip SRAM always. The slow tier is the question: on-chip if the
model's cross-stage tensors are small enough, external if they are not.

QEMU loads the plan to that address without any firmware involvement:

```bash
qemu-system-arm -M mps3-an547 -cpu cortex-m55 -nographic \
  -semihosting-config enable=on,target=native \
  -kernel firmware.elf \
  -device loader,file=model.tgrs,addr=0x60000000
```

Semihosting is what carries `printf` back to your terminal, so the firmware
needs no UART.

## A ready-made harness

[tigris-bench](https://github.com/raws-labs/tigris-bench) carries a working
firmware under `cortex-m/m55-qemu/`: a `main.c` that loads the plan from
`0x60000000`, allocates both arenas from linker-placed buffers, runs one
inference and prints what it used. Two scripts wrap it, one to build against
the runtime sources and one to run:

```bash
cd cortex-m/m55-qemu
./build.sh path/to/model.tgrs "My model" -DFAST_KB=192 -DSLOW_KB=256
./run.sh   path/to/model.tgrs
```

`-DFAST_KB` and `-DSLOW_KB` size the two arenas in on-chip SRAM, and
`-DSLOW_DDR` moves the slow tier to external memory instead, for a model whose
cross-stage tensors will not fit on-chip.

Build for `cortex-m55+nomve`. The portable int8 kernels do not use Helium, and
leaving MVE out of the build keeps what you measure honest about which kernels
ran.

## Reading the result

A DS-CNN keyword spotter, small enough to be a single stage:

```
BENCH_START DS-CNN @ Cortex-M55 (QEMU mps3-an547)
Ops: 12  Stages: 1  Tensors: 13  Budget: 20480 bytes
OUTPUT n=12 checksum=10244 min=-128 max=127 nonmin=9
ARENA fast_peak=16496 slow_used=16 total_sram=16512 bytes (fast_cap=20480 slow_cap=65536)
```

`fast_peak` is the high-water mark the runtime actually reached, not the
capacity you gave it. It is the number to carry into a hardware decision: this
model needs 16.5 KiB of SRAM, so it fits a part with 32 KiB and leaves room for
everything else the firmware does.

The same harness on MobileNetV2-0.35 at 224x224 shows why tiling is worth
having:

```
BENCH_START MobileNetV2-0.35 @ Cortex-M55 (QEMU mps3-an547)
Ops: 65  Stages: 17  Tensors: 66  Budget: 131072 bytes
OUTPUT n=10 checksum=4869 min=-128 max=127 nonmin=6
ARENA fast_peak=130176 slow_used=16 total_sram=130192 bytes (fast_cap=131072 slow_cap=262144)
```

That model's naive peak is 735.00 KiB, with a single 1x48x112x112 activation
accounting for 588.00 KiB of it. Seventeen stages bring the working set down to
130,192 bytes, on-chip, with the slow tier barely touched. A runtime that
allocates one contiguous arena needs the full 735 KiB and does not run here at
all.

## Checking the numbers mean something

The emulator will happily run a plan that computes nonsense, so compare the
output against a host reference on the same input rather than trusting that it
ran. The firmware prints a checksum and the range of the output vector, which is
enough to catch the two failures worth catching: a degenerate output, where
every element is identical, and a mismatch against the host.

Do not treat a checksum printed in someone else's README as the target. It is a
property of the exact plan bytes that produced it, and a plan rebuilt from a
model with a freshly initialized head will give a different one while being
equally correct. Generate the reference from the plan you are running.
