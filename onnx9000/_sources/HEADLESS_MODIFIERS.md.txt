# Headless Graph Modifiers

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. The `onnx9000-optimizer` provides several headless modifiers for quick, programmatic alterations to ONNX computational graphs.

The `onnx9000-optimizer` library provides utility APIs designed to programmatically modify the `onnx9000-core` Intermediate Representation (IR). These actions can be triggered via the CLI or instantiated directly via the Python SDK.

## Available Modifiers

- **Rename Input**: Change the name of a specific input feature.
- **Change Batch Size**: Programmatically update the outer dimension of inputs/outputs.
- **Mutate**: Apply complex structural changes defined in a `.json` configuration file.

## Python SDK Usage

```python
from onnx9000.core.parser.core import load
from onnx9000.core.serializer import save
from onnx9000.optimizer.surgeon.headless import rename_input, change_batch, mutate

# Load your model graph
graph = load("model.onnx")

# 1. Rename an input tensor
graph = rename_input(graph, old_name="images", new_name="input_tensor")

# 2. Change the batch size to 32
graph = change_batch(graph, new_size=32)

# 3. Apply arbitrary mutations from a JSON script
graph = mutate(graph, "mutations.json")

# Save modifications
save(graph, "modified_model.onnx")
```

## CLI Usage

These utilities are also available directly from the unified `onnx9000` CLI:

```bash
# Rename Input
onnx9000 rename-input model.onnx images input_tensor -o updated_model.onnx

# Change Batch Size
onnx9000 change-batch model.onnx 32 -o updated_model.onnx

# Mutate
onnx9000 mutate model.onnx --script mutations.json -o updated_model.onnx
```

### Mutation JSON Schema

The `mutate` command accepts a JSON array specifying actions. Currently supported actions include `remove_node`.

```json
[
  {
    "action": "remove_node",
    "node_name": "Dropout_14"
  }
]
```
