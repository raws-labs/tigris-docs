---
title: "What Is TiGrIS"
description: "TiGrIS is an ahead-of-time compiler that plans ML inference to fit the memory of an embedded device, on its own runtime or in front of the runtime you ship."
sidebar:
  order: 101
---

TiGrIS, the Tiled Graph Inference Scheduler, is an ahead-of-time compiler for ML
inference on embedded devices. You give it a trained model and the memory your device
has. It plans the inference so that it fits, or tells you what stops it.

## The problem

Weights can stay in flash, but activations need RAM while the model runs. A single
intermediate tensor can outgrow the device's SRAM even when every weight fits, and
reusing an arena cannot shrink a tensor below its own size. The usual answer is a
smaller model. TiGrIS keeps the model and changes how its computation is laid out
in memory.

## How it works

The compiler splits the graph into stages and cuts spatial operators into tiles, by
rows or as a 2D grid, so only a small working set is live at any moment. Consecutive
tiled stages form chains whose intermediates never leave fast memory. Every decision
is made on the host, against the budget you set: fast memory, an optional slower
pool such as PSRAM, and flash. Nothing is analyzed or replanned on the device, and a
model that cannot meet the budget is refused with the reason instead of compiled
best-effort.

<svg class="wt-flow" viewBox="0 0 480 460" role="img" aria-label="A model in ONNX, TFLite, PyTorch or ExecuTorch form goes into the TiGrIS compiler on the host, which plans it against fast, slow and flash budgets. The compiler writes either a .tgrs plan for the TiGrIS runtime or a banded TFLite or ExecuTorch file for the runtime you already ship. Both run on an embedded device with fixed RAM." style="display:block;width:100%;max-width:480px;margin:2rem auto">
  <style>
    .wt-flow .wt-box { fill: var(--sl-color-gray-6); stroke: var(--sl-color-gray-4); stroke-width: 1; }
    .wt-flow .wt-core { fill: var(--sl-color-accent-low); stroke: var(--sl-color-accent); stroke-width: 1.5; }
    .wt-flow .wt-line { fill: none; stroke: var(--sl-color-gray-3); stroke-width: 1.5; }
    .wt-flow .wt-head { fill: var(--sl-color-gray-3); }
    .wt-flow text { font-family: var(--sl-font, system-ui, sans-serif); text-anchor: middle; }
    .wt-flow .wt-t { fill: var(--sl-color-text); font-size: 14px; }
    .wt-flow .wt-b { font-weight: 600; }
    .wt-flow .wt-s { fill: var(--sl-color-gray-2); font-size: 13px; }
  </style>
  <defs>
    <marker id="wt-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
      <path d="M0,0 L10,5 L0,10 z" class="wt-head" />
    </marker>
  </defs>
  <rect class="wt-box" x="40" y="8" width="400" height="48" rx="6" />
  <text class="wt-s" x="240" y="27">your model</text>
  <text class="wt-t" x="240" y="46">ONNX, TFLite, PyTorch .pt2, ExecuTorch .pte</text>
  <path class="wt-line" d="M240,56 V86" marker-end="url(#wt-arrow)" />
  <rect class="wt-core" x="40" y="90" width="400" height="84" rx="6" />
  <text class="wt-t wt-b" x="240" y="116">TiGrIS compiler, on the host</text>
  <text class="wt-s" x="240" y="138">budget: fast memory, slow memory, flash</text>
  <text class="wt-s" x="240" y="158">stages, row and 2D tiles, chains</text>
  <path class="wt-line" d="M240,174 V192 M120,192 H360" />
  <path class="wt-line" d="M120,192 V214" marker-end="url(#wt-arrow)" />
  <path class="wt-line" d="M360,192 V214" marker-end="url(#wt-arrow)" />
  <rect class="wt-box" x="10" y="218" width="220" height="44" rx="6" />
  <text class="wt-t" x="120" y="245">plan .tgrs</text>
  <rect class="wt-box" x="250" y="218" width="220" height="44" rx="6" />
  <text class="wt-t" x="360" y="245">banded .tflite / .pte</text>
  <path class="wt-line" d="M120,262 V292" marker-end="url(#wt-arrow)" />
  <path class="wt-line" d="M360,262 V292" marker-end="url(#wt-arrow)" />
  <rect class="wt-box" x="10" y="296" width="220" height="64" rx="6" />
  <text class="wt-t wt-b" x="120" y="322">TiGrIS runtime</text>
  <text class="wt-s" x="120" y="344">portable C, arenas you own</text>
  <rect class="wt-box" x="250" y="296" width="220" height="64" rx="6" />
  <text class="wt-t wt-b" x="360" y="322">TFLite Micro / ExecuTorch</text>
  <text class="wt-s" x="360" y="344">the runtime you already ship</text>
  <path class="wt-line" d="M120,360 V380 H360 V360 M240,380 V404" />
  <path class="wt-line" d="M240,392 V404" marker-end="url(#wt-arrow)" />
  <rect class="wt-box" x="120" y="408" width="240" height="44" rx="6" />
  <text class="wt-t" x="240" y="435">embedded device, fixed RAM</text>
</svg>

The result runs one of two ways today:

- **On the TiGrIS runtime.** The compiler writes a plan, a `.tgrs` file, and the
  portable C runtime executes it in arenas you own, with no heap allocation during
  inference. The kernel library is chosen when the C glue is generated: reference
  C, CMSIS-NN or ESP-NN.
- **As a step before the runtime you ship.** The compiler rewrites the model itself
  into row bands and writes the same format back: a TFLite file for TFLite Micro, or
  an ExecuTorch program for Cortex-M. That runtime then runs it in less RAM with the
  same outputs.

The compiler reads ONNX, TFLite and PyTorch `.pt2` files.

## When it helps little

- **The model already fits.** The plan then holds about what a good tensor planner
  holds; the gain is kilobytes, not a deployment.
- **Latency comes first.** Tiles recompute the rows they share, and a slow pool adds
  traffic.
- **The model is outside what the compiler accepts:** dynamic shapes, int16
  activations, or operators missing from the [operator support](/runtime/operator-support/)
  table.

TiGrIS does not train or fine-tune, does not quantize a float model for you, and does
not replace kernel libraries: it calls them.

Next: [Quickstart](/getting-started/quickstart/) takes a model from file to C source,
and [Core Concepts](/getting-started/concepts/) covers plans, memory pools and tiling
in more depth.
