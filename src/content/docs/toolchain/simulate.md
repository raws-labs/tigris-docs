---
title: "tigris simulate"
description: "Deprecated: the execution trace is now tigris analyze --trace."
sidebar:
  order: 240
  hidden: true
---

`tigris simulate` is deprecated and is removed in the next release. The execution
trace it printed is now part of `analyze`, measured on the host runtime:

```bash
tigris analyze model.onnx -m 64K -m 8M --trace
```

`simulate` still runs, prints a deprecation note on standard error, and then prints
the same trace. See [`tigris analyze`](/toolchain/analyze/#execution-trace) and the
other [replaced commands](/toolchain/overview/#replaced-commands).
