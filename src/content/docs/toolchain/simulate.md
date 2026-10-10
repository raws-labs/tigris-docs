---
title: "tigris simulate"
description: "Deprecated: the execution trace is now tigris analyze --trace."
sidebar:
  order: 240
---

`tigris simulate` is deprecated and is removed in the next release. The execution
trace it printed is now part of `analyze`:

```bash
tigris analyze model.onnx -m 64K -m 8M --trace
```

`simulate` still runs, prints a deprecation note on stderr, and then prints the same
trace. See [`tigris analyze`](/toolchain/analyze/#execution-trace).
