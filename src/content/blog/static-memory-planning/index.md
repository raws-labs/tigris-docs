---
title: "Static Memory Planning for Embedded Neural Network Inference"
date: 2026-06-11T00:00:00+00:00
excerpt: "Peak working memory depends on the execution schedule and allocation, so the schedule can be optimized ahead of time and executed with bounded runtime arenas."
---

Picture a device that has been in the field for years: a sensor node, a
metering unit, a small controller. It was designed long before
machine learning was part of its requirements, and it cannot be redesigned
now. The fleet is deployed, the hardware is fixed, and a board revision is out
of the question. What can change is the firmware. So when a model is supposed
to be retrofitted onto it, it has to live in whatever memory the existing
firmware leaves over. Suppose the model is ready: accurate, already quantized
to int8, and the file is small. But deployment still fails. The runtime cannot
allocate its working buffers, and the standard advice is to shrink the network
until it fits. This post is about why that happens, and why shrinking the
model is like sawing the legs off a piano to get it through the door: it fits
now, but it is not really the instrument you meant to deliver.

The reason is that model size and memory demand are two different quantities.
The size of a model is the size of its parameters. Whether inference runs on a
device like this depends on peak working memory: the largest amount of
intermediate data alive at any single point during execution. And that peak
depends not only on the model, but also on the order its operators run in and
on how memory is allocated. Both of these are decisions, not givens, so whether
a model fits is a planning problem, solvable ahead of time on a workstation.

## 1. Model size is the wrong number

A quantized network might occupy a few hundred kilobytes of parameters. On a
resource-constrained device this is still not the number that matters, because
parameters are immutable. They are written once at build time and only read during
inference, so they can stay in flash and be fetched, or executed in place,
without taking up RAM.

What has to sit in RAM is what the network writes while it computes: the
intermediate tensors between operators, plus the scratch memory some kernels
need. These buffers appear and disappear during a single forward pass. What the
budget has to hold is the largest set of them that is alive at the same time,
not their sum, and not the parameter count. A network with few parameters can
have a large activation peak, and a network with many parameters can have a
small one. Most deployment failures are the first case: the weights sit
comfortably in flash, and the activations do not fit in RAM.

## 2. The memory classes

It helps to separate the memory an inference workload consumes into classes,
because they differ in size, in mutability, and in where they can live.

| Class | Mutable? | Natural home | Reducible by planning? |
|-------|----------|--------------|------------------------|
| Weights and constants | No | Flash (read / XIP) | No (read-only; not the binding constraint when flash-resident) |
| Input / output tensors | Yes | RAM | Marginally (must persist across the pass) |
| Intermediate activations | Yes | RAM | **Yes, substantially** |
| Operator scratch | Yes | RAM (must be fast) | Indirectly (kernel-dependent) |
| Runtime / arena bookkeeping | n/a | RAM | Bounded separately from activations |

Most of this post is about intermediate activations: the largest mutable
class, and the one planning can do the most about, because their lifetimes are
short and overlapping. Operator scratch is different. It depends on which
kernel implementation sits behind each operator, not on the graph alone, so it
cannot be planned at this level. The activation budget discussed below must be
combined with backend-specific workspace sizing and checked during runtime
initialization.

## 3. Peak memory is a property of the schedule

Model the network as a directed acyclic graph of operators producing tensors.
Each intermediate tensor has a lifetime: it is born when its producer runs, and
it is dead after its last consumer has run. At any step of an execution order,
the live tensors are the ones born and not yet dead. Their total size is the
working set, and the maximum over the schedule is what the arena must hold.

This peak is not fixed by the graph. It follows from two decisions:

1. **The schedule**, the order the operators run in, which decides what is live
   at the same time. An order that interleaves independent branches keeps more
   tensors live at once than one that finishes a branch before starting the
   next.
2. **The allocation**, the byte offsets assigned in the arena. Once a tensor is
   dead its bytes can be reused, so a good offset assignment packs tensors with
   disjoint lifetimes into the same space.

Graphs with branches are where this matters most. If a tensor feeds both the
next operator and a residual connection further downstream, it stays live
across the whole branch in between and holds its space, no matter how cheap it
was to produce. Minimizing the arena under fixed lifetimes is a packing
problem, similar to register allocation, and NP-hard in general; in practice
simple greedy heuristics come close to the optimum. The important point is the
framing: peak memory is a quantity you can minimize by choosing well, not a
constant you measure and accept.

## 4. Why naive execution wastes RAM

A runtime that allocates tensors as it encounters them, or sizes a flat arena
from the natural execution order with conservative lifetimes, wastes memory in
two ways. It keeps buffers alive when it cannot prove they are dead, so they
stay around past their last use. And it follows an execution order that was
chosen for correctness, not for minimal concurrency, so more tensors are live
at the same time than necessary. The result is a model that should fit but does
not. The usual response is to shrink the network until allocation succeeds,
which pays with model quality for what is really a planning problem.

## 5. Static execution plans

The schedule should be decided where compute is cheap: ahead of time, on the
workstation. A compiler can read the whole graph, derive lifetimes, choose an
operator order, split it into stages, and select a tiling strategy within a
requested activation budget. TiGrIS records those decisions and tensor sizes
in a static plan; it does not bake absolute runtime tensor addresses into the
file.

On the device, the core runtime walks that fixed schedule and assigns aligned
addresses from caller-provided fast and slow arenas. Its bump allocation,
reset, and compaction operations are bounded by those buffers. This avoids a
general-purpose heap in the core inference path, but it does not remove runtime
allocation bookkeeping or make allocation infallible. The compiler rejects a
schedule whose modeled activation requirement exceeds the requested budget;
the runtime can still report an error for insufficient physical buffers,
compressed-weight workspace, backend scratch, or slow-arena capacity.

That distinction is useful in safety-oriented embedded software: a fixed
schedule and caller-owned arenas make memory behavior easier to bound and
audit. They are inputs to a resource-safety argument, not a substitute for
checking initialization and inference return codes on the deployed target.

The compiler's modeled activation requirement is known before firmware is
flashed. For MobileNetV1 (α = 1.0, 128×128 input, int8), condensed:

```bash
$ tigris analyze mobilenet_v1_i8.onnx -m 256K -f 16M

Operators            30
Tensors              114 (31 activations)
Peak memory (naive)  256.00 KiB
Largest tensor       1x64x64x64 (256.00 KiB)

SRAM   budget 256.00 KiB, 1 stage          PASS - fits in budget
Flash  weights 3.09 MiB, plan 3.10 MiB     PASS - plan fits
```

These numbers show the point from section 1: 3.09 MiB of weights sit in flash
and never touch the activation budget, while modeled fit is decided by a
256 KiB activation peak dominated by a single 64×64×64 tensor early in the
network. The reported peak is an activation figure. The physical fast buffer
must also cover alignment, compressed-weight workspace, plan-sized executor
metadata, and accelerated-backend scratch. The runtime exposes exact CMSIS-NN
scratch and total-fast-arena queries so integrations can size that buffer
before allocation. A PASS therefore settles the compiled activation strategy,
while target initialization remains a checked operation.

## 6. When planning is not enough: tiling

Scheduling and allocation rearrange and reuse memory, but they cannot make a
single tensor smaller. If one operator's output, or one layer's working set,
exceeds the budget on its own, no offset assignment will fit it. The remaining
lever is the shape of the computation.

Tiling changes that shape. Large intermediates are produced and consumed in
pieces instead of all at once, which trades extra execution steps for a smaller
peak working set. The weights, the arithmetic, and the numerical result stay
identical; only the granularity and order of the computation change. This is
the difference between a different execution shape of the same model and a
different, smaller model. The first keeps the network you trained; the second
does not.

TiGrIS treats this as multi-strategy scheduling. It splits the graph into
stages and picks, per stage, the strategy that meets the budget: run an
operator directly when it already fits, fuse a sequence of operators into a
streaming pipeline so their intermediates never fully materialize, or process a
feature map in spatial strips, spilling to slower memory, when a single
operator is too large on its own. The budget is a compile-time parameter; a
tighter budget yields more and finer stages. The result is a single knob that
trades inference latency against memory footprint, while the model at both ends
of the curve stays the one you trained.

Tighter budgets therefore produce more and finer stages; the exact latency
cost depends on the model, backend, and memory system. We keep measured claims
separate from this planning example. The [benchmark section in Introducing
TiGrIS](/blog/introducing-tigris/#benchmarks) publishes the matched-model
hardware comparison and identifies the exact `tigris-bench` commit and summary
digest from which every table row is derived.

This is what makes planning interesting for the retrofit case from the
beginning. On deployed hardware the budget is not negotiable.
A compile-time knob that adapts the execution shape to that budget,
instead of the model to the device, is the difference between shipping
a firmware update and redesigning the product.

## 7. Summary

The memory wall in embedded inference is usually treated as a model-size
problem and answered with compression. A large part of it is a planning
problem. Peak working memory is determined by the execution schedule and the
allocation. Lifetime analysis and compile-time scheduling reduce the required
working set, while bounded runtime arena allocation, reuse, and compaction
realize that schedule in caller-provided memory. Tiling extends the reachable
budget by changing the execution shape instead of the model. TiGrIS is built
around this division: an ahead-of-time compiler emits a fixed binary plan, and
a small C99 core executes it without acquiring memory from a general-purpose
heap. The activation strategy is settled statically; buffer provisioning,
backend preparation, and inference remain checked operations on the target.

---

*TiGrIS is developed at [RAWS Labs](https://www.raws.at), an applied
research and engineering lab. If you are working against a memory budget
like the one above, we are interested in hearing about it.*
