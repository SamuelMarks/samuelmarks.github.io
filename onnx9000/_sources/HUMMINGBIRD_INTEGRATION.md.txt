# Hummingbird Integration

> **Ecosystem Context:** `onnx9000` integrates Hummingbird technology natively into its optimization pipeline, allowing traditional machine learning models (like Random Forests and Gradient Boosted Trees) to be transpiled into tensor operations.

This guide explains how to use Hummingbird features within the ecosystem.

## Overview

Traditional ML models from scikit-learn, XGBoost, and LightGBM are often executed via specialized CPU libraries. The `onnx9000-optimizer` incorporates Hummingbird to transpile these rule-based tree ensemble models into ONNX matrix multiplications (`MatMul`), additions, and relational operators (`Less`, `Greater`).

This allows traditional models to be executed on GPUs via ONNX Runtime or WebGPU in the browser.

## Features

- **Tree to Tensor Transpilation:** Converts `TreeEnsembleClassifier` and `TreeEnsembleRegressor` nodes into standard mathematical ONNX operators.
- **Hardware Acceleration:** Allows traditional ML models to run efficiently on GPUs and accelerators.
- **Zero-Dependency Web Support:** The transpiler runs natively in Python or via WebAssembly for client-side conversions.

## Usage

### CLI

To apply the Hummingbird transpilation strategy to an existing ONNX model that contains traditional ML operators (from `ai.onnx.ml`):

```bash
onnx9000 hummingbird model.onnx -o model_tensor.onnx
```

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.optimizer.hummingbird import TranspilationEngine, Strategy, TargetHardware

graph = load("model.onnx")

engine = TranspilationEngine()
transpiled_graph = engine.transpile(
    graph,
    strategy=Strategy.PERFECT_TREE,
    hardware=TargetHardware.WEBGPU
)

transpiled_graph.save("model_tensor.onnx")
```
