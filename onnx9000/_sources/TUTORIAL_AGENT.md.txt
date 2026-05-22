---
orphan: true
---

# Tutorial: Agentic Workflows with onnx9000-toolkit

> **Ecosystem Context:** `onnx9000` operates as a **Zero-dependency**, **WASM-First**, **WebGPU-Native**, **Polyglot Monorepo**. It enables high-performance inference and agentic orchestration across browsers and edge devices without the overhead of heavy native runtimes.

This tutorial demonstrates how to use the `onnx9000-toolkit` (Python) to build autonomous agentic workflows that can manipulate, optimize, and execute ONNX graphs using a tool-calling paradigm.

## 1. Initializing the Agent Runner

The `onnx9000-toolkit` provides a high-level `AgentRunner` designed for "Reason+Act" (ReAct) loops. It can bind to both native Python functions and remote ONNX-based tools.

```python
from onnx9000.toolkit.agent import AgentRunner

# Define a custom tool for graph surgery
def optimize_graph(graph_path: str):
    # Logic to call onnx9000-optimizer
    print(f"Optimizing graph at {graph_path}...")
    return "Graph optimized successfully."

agent = AgentRunner()
agent.register_tool(
    name="optimize_graph",
    func=optimize_graph,
    description="Optimizes an ONNX graph by fusing operators and pruning weights."
)
```

## 2. Tool-Calling with ONNX Graphs

Agents in `onnx9000` can interact directly with model IRs. For example, you can ask an agent to "make this model 20% smaller" and it will coordinate between the `onnx9000-optimizer` and the core IR.

```python
# Run an agentic loop to reduce model size
agent.run_loop("Load the model at './model.onnx', prune it to 80% sparsity, and save it as './pruned.onnx'.")
```

## 3. Defining Multi-Agent DAGs

For complex tasks, such as a "Planner" followed by a "Coder", `onnx9000-toolkit` supports structured Directed Acyclic Graphs (DAGs).

```python
dag = {
    "nodes": [
        {"id": "planner", "type": "llm", "prompt": "Create an optimization plan for this ResNet50 model."},
        {"id": "optimizer", "type": "tool", "tool_name": "optimize_graph"},
    ],
    "edges": [
        {"from": "planner", "to": "optimizer"}
    ]
}

agent.execute_dag(dag, initial_input="./resnet50.onnx")
```

## 4. Integration with onnx9000-core

The toolkit is designed to be fully compatible with `onnx9000-core` for final serialization and `onnx9000-backend-native` for local validation. This ensures that your agentic workflows are grounded in actual hardware performance metrics.

You now have a powerful agentic system built on top of the most efficient ONNX stack available.
