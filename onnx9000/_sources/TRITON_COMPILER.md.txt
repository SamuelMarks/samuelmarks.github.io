---
orphan: true
---

# Triton Compiler and Code Generation (`triton-compiler`)

ONNX9000 provides the `@onnx9000/triton-compiler` module to natively transpile ONNX Intermediate Representation (IR) graphs into highly optimized custom **OpenAI Triton** kernels.

This enables dynamic generation of fused GPU kernels without requiring ahead-of-time Python template configurations. By inspecting the ONNX AST natively in WebAssembly or Node.js, we emit raw Python/Triton (`@triton.jit`) strings ready to be fed to `torch.compile` or executed directly.

## Command Line Interface (CLI)

Both the Python and JS CLI allow you to quickly generate Triton code from an `.onnx` model offline:

```bash
# Generate Triton code and print to standard output
onnx9000 triton my_model.onnx
```

## Python SDK

In Python, the CLI exposes a convenient entrypoint for the compilation pipeline.

```python
from onnx9000_cli.main import triton_cmd
import argparse

args = argparse.Namespace(model="model.onnx")
triton_cmd(args)
```

## JavaScript/TypeScript SDK

In a JavaScript environment, you can parse the AST and compile it synchronously:

```typescript
import { Graph, Node } from '@onnx9000/core';
import { generateTriton } from '@onnx9000/triton-compiler';

// 1. Build or parse your ONNX Graph
const g = new Graph('custom_fused_kernel');
g.inputs.push({ name: 'A', shape: [1024], type: null as any });
g.inputs.push({ name: 'B', shape: [1024], type: null as any });
g.outputs.push({ name: 'C', shape: [1024], type: null as any });

const addNode = new Node('Add');
addNode.inputs = ['A', 'B'];
addNode.outputs = ['C'];
g.nodes.push(addNode);

// 2. Generate the kernel string with custom block config
const code = generateTriton(g, { blockM: 128 });
console.log(code);
```

## Interactive Web Demo

To interactively parse graphs and view the generated Python/Triton AST mapping locally:

```bash
onnx9000 serve
```

And navigate to `/triton` in your web browser.
