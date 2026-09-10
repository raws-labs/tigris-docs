---
title: "YOLOv5n on ESP32-S3: Object Detection on a $10 MCU"
date: 2026-03-04T00:00:00+00:00
excerpt: "800KB peak activations. 512KB of SRAM. Tiling makes it work — 11 seconds to inference."
---

YOLOv5n is a real object detector. 80 COCO classes, multi-scale anchor-free heads, bounding boxes with confidence scores. Not a toy classifier — the kind of model you'd actually use to detect people, vehicles, or animals in a camera feed. At 320x320 int8, it has 238 ops, 239 tensors, and a peak activation memory of 800 KB.

The ESP32-S3 has 512 KB of SRAM, of which ~384 KB is usable. Any runtime that allocates all tensors upfront can't fit this model — 800 KB simply doesn't go into 384 KB. TiGrIS tiles the computation down to a 64 KB arena, runs all 238 ops in 11 seconds, and produces correct detections.

## The model

YOLOv5n (ultralytics anchor-free format) quantized to int8 at 320x320 input. The architecture is non-trivial: a backbone with C3 blocks (each containing multiple Conv + SiLU + residual connections) and an SPPF module with three cascaded MaxPool ops, an FPN neck with Concat and Resize for multi-scale fusion, and three detection heads outputting at 40x40, 20x20, and 10x10 grids.

| Property | Value |
|----------|-------|
| Ops | 238 |
| Tensors | 239 |
| Plan binary | 2,726,804 bytes (2.6 MiB) |
| Peak activation memory | 819,200 bytes (800 KB) |
| Weights | from flash (XIP) |
| Model outputs | 6 (3 scales x 2 tensors) |

This model required five new runtime kernels beyond what DS-CNN and MobileNetV1 needed (Sigmoid, Mul, Sub, Resize nearest, Concat with requantization) plus a compiler pass to decompose SiLU into Sigmoid + Mul. It's the first model to exercise the full TiGrIS op set.

## Why this model doesn't fit conventionally

Standard ML runtimes (TFLite Micro, etc.) allocate all live tensors into a single contiguous arena before inference starts. The planner computes the minimum arena size from the model's tensor lifetime graph, and if the result exceeds available SRAM, allocation fails.

800 KB peak > 384 KB usable SRAM. With a flat memory planner, there's no way to start inference — the model needs more memory than exists on the chip. This is the fundamental constraint that tiling addresses.

## How TiGrIS tiles it

The TiGrIS compiler partitions the compute graph into stages, each planned to fit within the SRAM budget. Three strategies, applied in order of preference: **normal** execution (no tiling, op fits flat), **chain tiling** (fuse consecutive ops into a streaming pipeline), and **spatial tiling** (process feature maps in horizontal strips with PSRAM spill).

The budget is a compile-time parameter, and smaller budgets produce more stages:

| Budget | Stages | Tile plans | Strategy breakdown |
|--------|-------:|----------:|-------------------|
| 232 KB | 132 | 69 | Chain + spatial, all stages fit without overflow |
| 64 KB | 238 | 156 | Aggressive tiling, norm=66, tile=90, chain=82 |

The 64 KB budget is not an accident — it's a deliberate trade-off. Read on.

## ESP-NN and the scratch buffer trade-off

ESP-NN's SIMD convolution kernels use an im2col strategy: they copy the convolution input into a contiguous scratch buffer, then run vectorized multiply-accumulate on the columns. The scratch buffer size depends on the layer geometry — `f(IC, KH, KW, OH, OW)` — and ranges from under 1 KB for pointwise convolutions to 425 KB for the largest layers in YOLOv5n.

The scratch buffer must live in fast memory (SRAM) for the SIMD speedup to materialize. If it's in PSRAM, the memory access latency dominates and ESP-NN is no faster than the reference C kernels.

This creates a direct conflict: **the model arena and the ESP-NN scratch buffer compete for the same SRAM.** A larger arena leaves less room for scratch. A smaller arena leaves more.

TiGrIS uses a two-tier scratch allocation: carve as much SRAM as possible after the arena, then fall back to PSRAM for ops whose scratch exceeds what's available. The firmware measures free heap after allocating the arena and uses that as the SRAM scratch pool.

Here's what happens at different budget levels:

| Config | Arena | SRAM scratch | Conv in SRAM | Conv in PSRAM | Latency |
|--------|------:|------------:|------------:|-------------:|--------:|
| s8_ref @232K | 232 KB | — | — | — | 40,127 ms |
| ESP-NN @232K | 232 KB | 23 KB | 69 | 49 | 40,127 ms |
| ESP-NN @64K | 64 KB | 180 KB | 259 | 5 | **11,116 ms** |

At 232 KB arena, only 23 KB of SRAM remains for scratch. 42% of convolution dispatches (49 of 118) fall back to PSRAM scratch — and those ops run at PSRAM speed, negating the SIMD benefit. The ESP-NN configuration matches reference C latency exactly.

At 64 KB arena, 180 KB of SRAM is available for scratch. 98% of convolution dispatches (259 of 264) use fast SRAM scratch. The result: **3.6x speedup** over reference, from 40.1 seconds to 11.1 seconds. The 5 remaining PSRAM ops each need 325–425 KB of scratch — they'll never fit in SRAM on this chip.

The counterintuitive result: **giving the model *less* memory makes inference *faster***, because the freed SRAM goes to ESP-NN scratch buffers where it has more impact than arena headroom.

## Detection results

The proof is in the bounding boxes. We sent two test images to the device over serial and compared against ONNX Runtime on the host.

**Living room scene** (multi-object):

| ORT reference | ESP32-S3 device |
|:---:|:---:|
| ![ORT living room](./ort_result.jpg) | ![ESP living room](./esp_result.jpg) |

TV, person, chairs, potted plant — near-identical detections between ORT and the device. The int8 quantization shifts some confidence scores by a few percent, but the bounding boxes land in the same places.

**Bear** (single object):

| ORT reference | ESP32-S3 device |
|:---:|:---:|
| ![ORT bear](./ort_bear.jpg) | ![ESP bear](./esp_bear.jpg) |

Same bounding box, same class, same confidence range.

## The pipeline

The full workflow from PyTorch model to device detection:

1. **Export**: `export_yolov5n.py` loads the ultralytics YOLOv5n, exports to ONNX, trims the post-processing (NMS stays on the host), and quantizes to int8 using calibration images
2. **Compile**: `tigris compile yolov5n_i8.onnx -m 64K -f 4M --xip -o yolov5n_i8_64k.tgrs` — tiles the model to fit a 64 KB arena
3. **Flash**: the plan binary is stored in a flash partition, memory-mapped at runtime via ESP-IDF's partition mmap API
4. **Inference**: the host sends a 320x320x3 int8 image over serial, the device runs inference and returns 6 raw output tensors, the host runs NMS and draws bounding boxes

The original host/device demo assets are not part of the current public
benchmark tree, so this post does not advertise a one-command reproduction.
The maintained compiler and runtime integration paths are covered in the
[Quickstart](/getting-started/quickstart/).

## What's next

- **160x160 input** for ~3s inference — practical for periodic detection tasks
- **Camera module** integration for continuous real-time detection
- **CMSIS-NN backend** to bring the same pipeline to Cortex-M targets (STM32H7, nRF5340)
- **Compile-time weight transpose** — depthwise conv weights are currently transposed at runtime from CHW to HWC; moving this to the compiler eliminates a startup cost

---

*YOLOv5n demo built with [TiGrIS](https://github.com/raws-labs/tigris) on ESP32-S3-DevKitC-1 (N16R8) at 240 MHz. ESP-IDF v5.4, ESP-NN SIMD kernels.*
