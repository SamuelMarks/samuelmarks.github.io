# ONNX9000 GraphSurgeon / Optimizer Engine

ONNX9000 integrates a powerful graph rewriting and optimization engine. In Python, this is provided by `onnx9000-optimizer` and acts similarly to NVIDIA's TensorRT GraphSurgeon. In JavaScript, this is provided natively by `@onnx9000/modifier`.

## Key Features

- **Topological Rewriting:** Easily identify subgraphs (e.g. `MatMul` + `Add`) and fuse them into highly optimized kernels (e.g. `Gemm`).
- **Constant Folding & Pruning:** Automatically evaluate sub-graphs whose inputs are all constant, removing redundant layers and minimizing graph size.
- **Polyglot Execution:** Both the Python and JS APIs allow direct manipulation of the `onnx.ModelProto` memory structures without requiring disk I/O.

## Usage in Python (Optimizer)

```python
from onnx9000.optimizer.simplifier.api import simplify
from onnx9000.core.parser.core import load, save

model = load("model.onnx")

# Apply algebraic simplifications, constant folding, and shape inference
simplified_model = simplify(model)

save(simplified_model, "model_optimized.onnx")
```

## Usage in the Browser (Modifier)

```typescript
import { GraphMutator } from '@onnx9000/modifier';

const modelBuffer = await fetch('model.onnx').then((r) => r.arrayBuffer());
const modelProto = parseONNX(modelBuffer);

const mutator = new GraphMutator(modelProto);

// Remove dropout layers for inference
for (const node of modelProto.graph.node) {
  if (node.opType === 'Dropout') {
    mutator.deleteNode(node.name);
  }
}

const optimizedBuffer = serializeONNX(modelProto);
```

## CLI Usage

Optimize an ONNX model directly from the command line:

```bash
# Simplify a model (constant folding, dead code elimination)
onnx9000 simplify my_model.onnx -o my_model_opt.onnx

# Apply specific optimizations
onnx9000 optimize my_model.onnx --passes fuse_bn_into_conv
```
