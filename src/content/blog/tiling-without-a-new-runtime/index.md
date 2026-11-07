---
title: "Tiling Without a New Runtime"
date: 2026-11-05T00:00:00+00:00
excerpt: "TiGrIS 0.13 rewrites a .tflite so that its early layers run in row bands of stock operators. The TensorFlow Lite Micro build you already ship runs the new file, after at most four op resolver registrations, in a smaller arena with identical output, for a few percent more multiply-accumulates."
---

An int8 MobileNetV2 with width multiplier 0.35 and a 224x224 input has an activation peak of 735 KiB. It occurs at the fifth operator, a depthwise convolution that reads a 112x112x48 feature map (588 KiB) while it writes its 56x56x48 output (147 KiB). After that operator, the model never needs more than 319 KiB.

The peak, not the average, sets the arena TensorFlow Lite Micro (TFLM) needs, so it sets how much RAM the model costs on an embedded device.

TiGrIS is an ahead-of-time compiler that tiles ONNX and TFLite models to fit the memory of embedded devices. This post shows how `tigris export`, new in TiGrIS 0.13, rewrites the `.tflite` file so that the peak never forms. The rewritten file uses only stock TFLite operators and runs on the TFLM build you already ship. For this model, TFLM's total arena, measured on the host, shrinks by 47% for 3.39% more multiply-accumulates (MACs), and the output stays identical.

## Keeping the runtime

TiGrIS started as a compiler for its own runtime: it tiles a model, and a small C runtime executes the plan. That asks you to replace the interpreter, and if your firmware is built and validated around TFLM, that is a real cost. For 0.13 we wanted the tiling without the runtime change. That set four constraints:

- A `.tflite` goes in and a `.tflite` comes out. TiGrIS reads TFLite files since 0.12, and 0.13 adds the writer.
- Every operator in the output is a stock TFLite builtin. The firmware changes only in the model array and in at most four op resolver registrations: PAD, SLICE and CONCATENATION, plus PADV2 when a banded region holds a padded max pool.
- The output is identical to the original's.
- The saving is known before anything runs.

## Where the peak comes from

`tigris analyze` reads the model and reports what TFLM's planner will need. For the MobileNetV2 with a 480 KiB budget (`tigris analyze mbv2.tflite -m 480K -v`), the relevant lines are:

```
memory
  TFLite Micro arena     735.00 KiB   its planner and reduction scratch; persistent allocations and other kernel scratch excluded
  TFLite Micro export   -386.11 KiB   head 735.00 KiB -> 327.69 KiB, persistent +21.20 KiB, +3.39% MACs, reference kernels on arm-none-eabi
  largest tensor         588.00 KiB   1x112x112x48

stages
  stage    ops   untiled peak   in   out   tiling
      0    0-2     392.00 KiB    1     1   chain of 3, tile height 18
      1      3     686.00 KiB    1     1   in chain 0
      2      4     735.00 KiB    1     1   in chain 0
      3   5-64     318.50 KiB    1     1   untiled
```

The first line is the activation memory TFLM's planner lays out, its head: 735.00 KiB. The stage table is TiGrIS's plan for its own runtime at that budget. Only its untiled peak column matters here, and it shows where the head comes from. TiGrIS numbers operators from 0. Operator 3 writes a 1x112x112x48 tensor (602,112 B), and operator 4, a stride-2 depthwise convolution, reads it while it writes its own 150,528 B output. Both are live at once, and no planner can place them in less than their sum. From operator 5 on, the model never needs more than 318.50 KiB. `analyze` splits the model's final MEAN into two operators, so it counts 65 where the file has 64. Operators 0 to 61 carry the same numbers in both.

On top of the head, TFLM allocates persistent records for every tensor and operator. Measured with its own recording allocator on a 64-bit host, the model needs 824,496 B in all.

The second line predicts what the rewrite will do: the head drops from 735.00 to 327.69 KiB, the persistent part grows by 21.20 KiB, and the model does 3.39% more MACs.

## Row bands

`tigris export` computes the region around the peak in horizontal bands. For each band it emits a SLICE of the input rows the band reads, PAD operators for the padding the original operators applied at that band's edges, and copies of the region's operators. One CONCATENATION joins the band outputs into the original tensor, so the rest of the network does not change:

```
mbv2_orig.tflite   64 operators
operators 0-5 computed in 10 row bands, +3.39% MACs; 159 operators after export

TFLite Micro arena, reference kernels on arm-none-eabi
  head            327.69 KiB   735.00 KiB before
  persistent      +21.20 KiB
  total change   -386.11 KiB
```

Here operators 0 to 5 become 10 bands. Each band reads 26 to 31 rows of the 224-row input and produces 5 or 6 rows of the 56x56x8 tensor that operator 5 writes. The 112x112x48 tensor that set the peak never exists in full. Only one band's slice of it is live at a time.

The new file has 159 operators instead of 64: 36 more CONV_2D, 18 more DEPTHWISE_CONV_2D, 30 PAD, 10 SLICE and 1 CONCATENATION, all stock builtins. It grows by 13,952 B (2.2%).

## Choices and their costs

### Recompute the halo

A 3x3 convolution needs one row above and below each output row, and a chain of them needs more. So each band's SLICE includes halo rows that the neighboring band reads too, and the band's convolutions compute those rows again. That recomputation is the 3.39% of extra MACs. It buys the memory: a band shares no intermediate tensor with its neighbors, so its intermediates are freed before the next band starts.

### Move the padding out of the convolutions

A convolution with SAME padding pads the borders of its full input. Inside a band, the top and bottom edges are real rows of the image, not borders. So the band copies run with VALID padding, and an explicit PAD adds exactly the rows and columns the original operator padded at that band's edges. In front of a max pool the fill is -128 instead of the zero point, so a padded tap never wins the max. If the input model already has an explicit PAD that equals the SAME padding of the convolution reading it, export also tries the model with that PAD folded into the convolution.

### Search the region and the band count

Export tries every run of up to 12 operators it can band that contains a strided or pooling operator and whose intermediate tensors are read only inside the run. For each run it tries 2 to 10 bands. Ten is the most inputs TFLM's CONCATENATION accepts. For MobileNetV2 the winner is operators 0 to 5 in 10 bands.

### Score with TFLM's own arithmetic

Each candidate is scored by the arena TFLM would plan for it with its reference kernels: the head, which includes the scratch of reduction kernels such as MEAN, plus how much the persistent allocations grow on the target, 32-bit arm-none-eabi by default. Export writes the best candidate only if it lowers that total. Ties go to fewer operators. Before it returns, export reads the written file back and checks that its arena matches the plan.

The prediction holds. On the host, TFLM's recording allocator reports a head of 752,640 B for the original and 335,552 B for the export. Those are the 735.00 and 327.69 KiB that `analyze` printed, to the byte.

## Results

On the MobileNetV2 and three MLPerf Tiny reference models, measured on the host with TFLM's recording allocator:

| Model | Rewrite | MACs | TFLM arena, host |
|---|---|---|---|
| MobileNetV2 0.35/224 | operators 0-5, 10 bands | +3.39% | 824,496 B -> 434,416 B (-47.3%) |
| Visual Wake Words | operators 2-3, 4 bands | +0.25% | 103,680 B -> 79,296 B (-23.5%) |
| ResNet-8 | operators 0-7, 2 bands | +9.25% | 55,984 B -> 50,544 B (-9.7%) |
| Keyword spotting DS-CNN | none | | 24,272 B |

The trade varies by model. Visual Wake Words loses almost a quarter of its arena for 0.25% more MACs. ResNet-8 pays 9.25% more MACs for about a tenth. For the keyword-spotting model, no candidate lowers the arena, so export says so and writes no file.

In all three rewritten models, TFLM gives identical outputs for the original and the exported file on 16 random int8 inputs. The MobileNetV2's untrained classification head saturates many outputs, so for it we also compared the tensor the band region produces, which is not saturated: identical on all 16 inputs.

On a board, the tigris-bench suite runs the exported MobileNetV2 on a NUCLEO-H753ZI. Its largest contiguous RAM bank is the 512 KiB AXI SRAM, smaller than the original model's 735 KiB arena head. With TFLM, CMSIS-NN kernels and a 491,520 B (480 KiB) arena in that bank, the exported file uses 423,684 B. On the bench's fixed input, its output on the board is identical to the TiGrIS runtime's output for the same model.

## Where it stops

Export handles single-subgraph int8 models. It refuses a model with an operator whose TFLM scratch or persistent allocations it does not model, and says which. A region holds at most 12 operators in at most 10 bands. In these three models the rewrite costs extra MACs, grows the persistent part of the arena, and grows the file by 1 to 5 percent.

What the rewrite costs, you can read before you take it: `export` prints the arena change and the extra MACs, and it writes nothing unless the arena goes down.

## What the numbers measure

Three memory figures appear in this post, and they are not interchangeable.

- **TFLM arena, measured on the board.** The bytes TFLM reports from `arena_used_bytes()` after `AllocateTensors()` on a NUCLEO-H753ZI. They cover the planned activation tensors and TFLM's persistent allocations: tensor and node records, operator data and kernel scratch. Weights stay in flash, and the stack is not counted.
- **TFLM arena, measured on the host.** The same allocator run through the `tflite-micro` Python package on a 64-bit x86 machine, using its recording allocator. The package builds TFLM's reference kernels, and its records hold 64-bit pointers, so its persistent part is larger than on a 32-bit board. The exported MobileNetV2 needs 434,416 B on the host and 423,684 B on the board.
- **Arena head, predicted by TiGrIS.** The largest set of activation tensors TFLM's planner keeps live at once, plus the scratch of its reduction kernels, printed by `tigris analyze` and `tigris export` before anything runs. It leaves out the persistent part. For MobileNetV2, VWW and ResNet-8, before and after export, it equals the recording allocator's planned bytes exactly.

## How the numbers were produced

The board result comes from [tigris-bench](https://github.com/raws-labs/tigris-bench/tree/be65cfb88efb414d3520bb9380dd0b2e9bd248c8/cortex-m/deployability-hil), run with TiGrIS 0.13.0. TFLM is built from source and linked with the same CMSIS-NN revision the TiGrIS runtime uses. The exported file is produced by `tigris export` with default settings from the bench's MobileNetV2 `.tflite`, and the firmware's op resolver registers PAD, SLICE and CONCATENATION. The board was flashed and captured remotely on the [SiliconRig](https://siliconrig.dev) hardware-in-the-loop lab.

The MobileNetV2 is a 0.35-width, 224x224 model with an ImageNet backbone and an untrained 10-class head, quantized to int8. Its memory needs do not depend on what the head has learned, but its outputs saturate often, which is why the host check also compares the unsaturated band output. The board comparison uses the bench's fixed input.

The host figures use `tigris-ml` 0.13.0 and `tflite-micro` 0.dev20261023044614, the version `tigris-ml` 0.13.0 pins for its tests. That package builds TFLM's reference kernels on x86-64. The MLPerf Tiny models are the reference int8 files from [mlcommons/tiny](https://github.com/mlcommons/tiny/tree/4addd0fa08d216e20637637874e084895f289da4/benchmark/training).

## Try it

The host check runs on Linux x86-64, where the `tflite-micro` package is available:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install tigris-ml==0.13.0 tflite-micro==0.dev20261023044614 numpy
curl -sSfLO https://raw.githubusercontent.com/mlcommons/tiny/4addd0fa08d216e20637637874e084895f289da4/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
tigris analyze vww_96_int8.tflite
tigris export vww_96_int8.tflite -o vww_export.tflite
```

Then let TFLM measure both files and compare their outputs:

```python
import numpy as np
from tflite_micro.python.tflite_micro import runtime

a = runtime.Interpreter.from_file("vww_96_int8.tflite", arena_size=1 << 20)
b = runtime.Interpreter.from_file("vww_export.tflite", arena_size=1 << 20)
a.print_allocations()
b.print_allocations()

rng = np.random.default_rng(0)
same = 0
for _ in range(16):
    x = rng.integers(-128, 128, size=a.get_input_details(0)["shape"], dtype=np.int8)
    for m in (a, b):
        m.set_input(x, 0)
        m.invoke()
    same += np.array_equal(a.get_output(0), b.get_output(0))
print(f"{same} of 16 outputs identical")
```

The allocator reports a total of 103,680 bytes for the original and 79,296 for the export, and the last line reads `16 of 16 outputs identical`. To deploy, embed `vww_export.tflite` in place of the original and register PAD, SLICE and CONCATENATION if your resolver does not have them. The [export documentation](https://tigris-ml.dev/toolchain/export/) lists the options.

---

*TiGrIS is developed at [RAWS Labs](https://www.raws.at), an applied research and engineering lab. If you are shipping TFLM and a model does not fit its arena, we are interested in hearing about it.*
