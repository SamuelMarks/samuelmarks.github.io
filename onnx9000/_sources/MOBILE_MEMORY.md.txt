---
orphan: true
---

# Mobile & Edge Memory Best Practices

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. It specializes in high-efficiency, zero-copy inference for mobile and edge devices using `@onnx9000/core`, `@onnx9000/coreml`, and `onnx9000-tflite-exporter`.

## Zero-Dependency Strategy

Unlike traditional runtimes that bundle heavy protobuf libraries and C++ runtimes, `onnx9000` uses a modular, zero-dependency approach:

- **Python (`onnx9000-core`):** Direct `mmap` and `struct` based parsing of `.onnx`, `.tflite`, and `.safetensors`.
- **JS/TS (`@onnx9000/core`):** Pure `ArrayBuffer` and `DataView` implementations for raw memory access without external overhead.

## Optimizing for Mobile Memory

1. **Static Memory Arenas:** Use `onnx9000-backend-native` (Python) or `@onnx9000/backend-web` (JS) to pre-allocate memory blocks, eliminating dynamic allocations during inference.
2. **Quantization:** Leverage `onnx9000-optimizer` to convert models to INT8, INT4, or GGUF formats, drastically reducing memory footprint.
3. **Progressive Loading:** Use `@onnx9000/core` to stream model weights into memory only when needed, as detailed in `docs/PROGRESSIVE_LOADING.md`.
4. **Platform Native Backends:** Export models specifically for target hardware using `@onnx9000/coreml` (iOS/macOS) or `onnx9000-tflite-exporter` (Android/EdgeTPU).

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
