---
orphan: true
---

# PyTorch Code Generation (`pytorch-codegen`)

ONNX9000 allows you to transpile ONNX models back into native, readable PyTorch code. By walking the `onnx9000` Intermediate Representation (IR), this tool constructs a pure PyTorch `nn.Module` definition that matches the topological structure of the original ONNX graph.

This feature is available via the Python SDK, JavaScript/TypeScript SDK, CLI, and Web Demo.

## Command Line Interface (CLI)

Use the `pytorch-codegen` command to parse an `.onnx` model and generate the corresponding `.py` code:

```bash
# Print generated PyTorch code to standard output
onnx9000 pytorch-codegen my_model.onnx

# Write the generated code to a file
onnx9000 pytorch-codegen my_model.onnx -o my_model.py
```

## Python SDK

In Python, use the `ONNXToPyTorchVisitor` from the `onnx9000.core.codegen.pytorch` module:

```python
from onnx9000.core.parser.core import load
from onnx9000.core.codegen.pytorch import ONNXToPyTorchVisitor

# 1. Parse the ONNX file into the Core IR
graph = load("my_model.onnx")

# 2. Instantiate the code generation visitor
visitor = ONNXToPyTorchVisitor(graph)

# 3. Generate the PyTorch code
pytorch_code = visitor.generate()
print(pytorch_code)
```

## JavaScript/TypeScript SDK

In the browser or Node.js, the same transpilation engine is natively available without a Python runtime:

```typescript
import { load, ONNXToPyTorchVisitor } from '@onnx9000/core';
import * as fs from 'fs';

// 1. Load the model from an ArrayBuffer
const arrayBuffer = fs.readFileSync('my_model.onnx').buffer;
const graph = await load(arrayBuffer);

// 2. Generate the PyTorch code
const visitor = new ONNXToPyTorchVisitor(graph);
const pytorchCode = visitor.generate();

console.log(pytorchCode);
```

## Interactive Web Demo

To interactively view generated PyTorch code side-by-side with your ONNX models locally:

```bash
onnx9000 serve
```

And navigate to `/pytorch-codegen` in your web browser.
