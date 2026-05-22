---
orphan: true
---

# CoreML Integration (`coreml`)

ONNX9000 provides the `@onnx9000/coreml` module to compile ONNX Intermediate Representation (IR) graphs directly into Apple's MIL (Model Intermediate Language) AST.

This enables you to export ONNX models to the `.mlpackage` format statically, optimizing them natively for Apple Neural Engine (ANE) constraints and memory requirements—completely within a JS/WebAssembly environment without depending on the heavy `coremltools` Python library.

## Command Line Interface (CLI)

Both the Python and JS CLI allow you to orchestrate the CoreML MIL generation offline. Note that the Python CLI delegates to the internal JS tooling.

```bash
# Export ONNX model to CoreML/MIL
onnx9000 coreml export my_model.onnx
```

## JavaScript/TypeScript SDK

In a JavaScript environment, you can invoke the exporter directly:

```typescript
import { Graph, Node } from '@onnx9000/core';
import { convertToCoreML } from '@onnx9000/coreml/src/api.js';

// Build a mock ONNX graph
const g = new Graph('mock_model');
g.inputs.push({ name: 'input1', shape: [1, 3, 224, 224], type: null as any });
g.outputs.push({ name: 'output1', shape: [1, 1000], type: null as any });

const reluNode = new Node('Relu');
reluNode.inputs = ['input1'];
reluNode.outputs = ['output1'];
g.nodes.push(reluNode);

const milAst = convertToCoreML(g);
console.log(milAst);
```

## Interactive Web Demo

To interactively parse an ONNX graph and view the generated Apple MIL AST natively in the browser:

```bash
onnx9000 serve
```

And navigate to `/coreml` in your web browser.
