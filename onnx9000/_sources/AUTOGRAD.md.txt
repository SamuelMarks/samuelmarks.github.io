# Autograd Engine

> **Ecosystem Context:** The `onnx9000-toolkit` includes a fully functional Autograd engine that allows differentiating ONNX graphs without relying on PyTorch or TensorFlow.

This guide explains how to use the Autograd features in ONNX9000.

## Overview

The `onnx9000-toolkit` features a built-in reverse-mode automatic differentiation engine. It traverses the ONNX graph from the output nodes backwards to the inputs, generating the necessary gradient operators (`ai.onnx.training` ops or mathematical equivalents) directly within the graph.

This is essential for on-device training or fine-tuning models on edge devices where PyTorch is not available.

## Features

- **Reverse-Mode Autodiff:** Automatically computes gradients for all differentiable ONNX operators.
- **Graph Augmentation:** Injects the gradient subgraph directly into the existing ONNX model, outputting a new, standalone `.onnx` file that calculates both the forward pass and gradients.
- **Zero-Dependency:** Written entirely in pure Python and TypeScript.

## Usage

### CLI

To augment an existing ONNX model with its gradient subgraph:

```bash
onnx9000 autograd model.onnx --loss CrossEntropyLoss -o model_with_gradients.onnx
```

### Python API

```python
from onnx9000.core.parser.core import load
from onnx9000.toolkit.autograd import AutogradEngine

graph = load("model.onnx")

# Initialize the engine
engine = AutogradEngine()

# Append gradient nodes to the graph with respect to a specific loss node
grad_graph = engine.build_gradient_graph(graph, loss_node_name="loss")

grad_graph.save("model_with_gradients.onnx")
```
