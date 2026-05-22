---
orphan: true
---

# Graph Simplification

> **Ecosystem Context:** `onnx9000-optimizer` includes a `simplify` command designed to reduce the size and complexity of ONNX graphs before inference. It uses constant folding, redundant node elimination, and dead code removal.

## Usage via CLI

```bash
# Simplify a model
onnx9000 simplify model.onnx -o model_sim.onnx
```

## Interactive Web Demo

.. interactive-demo::
   :initial-source: script
   :initial-target: onnx
