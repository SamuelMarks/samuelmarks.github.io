---
orphan: true
---

# Understanding the onnx9000 MLIR Lowering Pipeline

> **Ecosystem Context:** `onnx9000` operates as a **Zero-dependency**, **WASM-First**, **WebGPU-Native**, **Polyglot Monorepo**. It enables high-performance inference through a sophisticated `@onnx9000/compiler` pipeline that targets WebAssembly and WebGPU.

The compilation from an ONNX model into WebGPU shader code or WebAssembly bytecode undergoes a multi-stage lowering process managed by the `@onnx9000/compiler`.

## 1. The Compilation Pipeline

The pipeline leverages IREE and MLIR integration to ensure optimal performance on modern hardware.

1. **ONNX to MHLO (High-Level Dialect):**
   - Operations like `Add`, `Conv`, and `MatMul` are mapped to target-agnostic MHLO operations with resolved data types and shapes using the core `@onnx9000/compiler` frontend.
2. **MHLO to Linalg (Structural Dialect):**
   - Implicit looping patterns are expanded into explicit `linalg.generic` iterations. Elementwise fusion occurs here to minimize memory bandwidth.
3. **Bufferization:**
   - Value semantics (`Tensor`) are lowered into memory semantics (`MemRef`). Allocations are explicitly scheduled and optimized for the target device's memory hierarchy.
4. **Linalg to HAL (Hardware Abstraction Layer):**
   - Nested loops are compiled into native executables (WGSL for WebGPU or WASM for CPU). Command buffers are built to schedule dispatches and copy operations efficiently.
5. **HAL to VM (Virtual Machine Control Flow):**
   - The command buffer sequence is flattened into bytecode. Dynamic shapes and symbolic variables are bound. Register allocation maps SSA definitions into a small flat array of VM registers.

## 2. Targeting WebGPU-Native Execution

This multi-layer architecture enables `@onnx9000/compiler` to emit fully self-contained Standalone JS payloads. These payloads include WGSL shader string literals that run directly on the GPU, bypassing the need for heavy external runtimes.

## 3. Zero-Dependency Real-Time Transpilation

Because the entire compiler is available as a **WASM-First** package, you can perform these lowering steps directly in the browser or via a lightweight CLI. This enables a truly **Polyglot** development experience where models from any framework can be lowered to the metal with minimal latency.

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
