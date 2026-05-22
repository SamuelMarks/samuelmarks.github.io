# onnx9000-toolkit: Fluent Scripting

`onnx9000-toolkit` provides a **zero-dependency**, **WASM-First** authoring environment for ONNX. Built as a core pillar of our polyglot monorepo, it allows developers to define complex model architectures using pure Python without the overhead of official C++ Protobuf bindings or heavy framework dependencies.

> **Ecosystem Context:** By leveraging our internal IR, `onnx9000-toolkit` supports real-time transpilation and offline conversions across C++, PyTorch, MLIR, CoreML, and Caffe targets directly in browser environments (like Pyodide/WASM).

## Features

- **Zero-Dependency**: Operates entirely on raw Python structures, making it extremely lightweight for edge deployment.
- **Dynamic Op Namespace**: Use `op.Add(A, B)` or `op.Relu(X)` with full IDE autocomplete support.
- **Operator Overloading**: Use standard Python operators (`A + B`, `A * B`, `A > B`).
- **Control Flow**: Supports mapping `if`, `for`, and `while` statements natively into ONNX `If` and `Loop` subgraphs.
- **Type Annotations**: Annotate inputs to strictly bind them to the internal `onnx9000-core` IR.

## Usage

```python
from onnx9000.toolkit.script import script, op
from onnx9000.core.dtypes import DType

@script
def my_model(x):
    # Arithmetic is automatically converted to ONNX ops
    return op.Relu(x + 1.0)

# Compiles the Python AST into a zero-dependency onnx9000-core Graph
model = my_model.to_graph()

# Serialize to standard .onnx format (no external libraries needed)
model.save("my_model.onnx")
```
.. interactive-demo::

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
