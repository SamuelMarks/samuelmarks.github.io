---
orphan: true
---

# `onnx9000-core` GenAI API Reference

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, **Polyglot Monorepo**. Through its modular design, it supports real-time transpilation and offline conversions across C++, PyTorch, MLIR, CoreML, and GGUF targets without requiring heavy official bindings or native dependencies.

## AgentRunner API (JavaScript/TypeScript)

The Agent API in `@onnx9000/core` handles orchestration of zero-dependency Local AI agents via the `AgentRunner` paradigm.

### `globalAgent.registerTool(tool: IAgentTool)`

Registers a functional closure as an executable agent tool within the `@onnx9000/core` runtime.

- **`name`**: Unique string identifier mapped internally to action requests.
- **`description`**: Context fed directly into the system prompt guiding the LLM selection mechanism.
- **`execute`**: An asynchronous `(args: string) => Promise<string>` callback executing sandbox logic.

### `globalAgent.runAgentLoop(prompt: string, signal?: AbortSignal)`

Initiates an autonomous Reason+Act iterative loop using the `@onnx9000/core` inference engine.

- Interacts exclusively with the `globalEvents` PubSub architecture (`agentLog`).
- Stops execution automatically upon encountering an `AbortSignal.abort()`.

### `globalAgent.executeDAG(dag: IAgentDAG, initialInput: string)`

Runs complex topologies involving conditional looping ("critic", "coder") natively via Javascript asynchronous chains, passing state top-down through the `@onnx9000/core` IR.
