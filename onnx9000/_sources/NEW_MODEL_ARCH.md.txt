---
orphan: true
---

# Supporting a New Model Architecture

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. Supporting a new architecture involves mapping framework-specific graphs into the unified `onnx9000-core` (Python) or `@onnx9000/core` (JS) Intermediate Representation (IR).

## 1. Defining the Mapping (`onnx9000-converters`)

Support for new architectures typically starts in the `onnx9000-converters` package. Here, you define how source framework operators and weights are translated into the standard `onnx9000` IR.

- **Direct AST Parsing:** Map framework-specific nodes (e.g., PyTorch's `nn.Conv2d`) directly to `onnx9000-core` IR nodes.
- **Zero-Dependency Philosophy:** Your converter must avoid depending on the original framework's binary runtime for conversion or inference.

## 2. Validating the Core IR

All architectures, whether in Python or JavaScript, share the same IR principles managed by the core packages:

- **Strict Topological Sorting:** All nodes in the graph must be strictly ordered.
- **Static Shape Inference:** Use `onnx9000-core.shape_inference` to validate tensor dimensions across the new architecture's graph.
- **Metadata Management:** Store model-specific parameters (e.g., vocabulary size, sequence length) in the standard IR metadata structures.

## 3. Cross-Platform Backend Support

Once an architecture is mapped to the core IR, it is automatically compatible with the entire ecosystem:

- **Native Execution:** via `onnx9000-backend-native` for high-performance CPU/CUDA inference.
- **Web-Native:** via `@onnx9000/backend-web` for zero-dependency WebGPU and WebNN execution.
- **AOT Compilation:** via `@onnx9000/compiler` to generate `.wvm` or C++ binaries.
- **Format Export:** via `onnx9000-onnx2gguf`, `@onnx9000/coreml`, or `onnx9000-tflite-exporter`.

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
