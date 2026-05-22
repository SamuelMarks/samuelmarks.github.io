---
orphan: true
---

# WASM Compiler

ONNX9000 features a built-in WASM (WebAssembly) compiler engine that allows for executing ONNX models entirely in the browser with near-native performance.

## Features

- **Zero-Dependency:** Uses built-in browser WebAssembly APIs, without requiring external C/C++ toolchains like Emscripten during runtime.
- **JIT Compilation:** Compiles the ONNX execution graph into WASM bytecode dynamically for optimal performance on the target architecture.
- **Fast Execution:** Specifically optimized for mathematical operations typical in ML workloads.

## Interactive Demo

You can test the WASM compiler directly by launching the standalone interactive demo:

```bash
onnx9000 serve --demo wasm
```
