# CLI Utility Tools

> **Ecosystem Context:** `onnx9000` is a **Zero-dependency, WASM-First, and WebGPU-Native Polyglot Monorepo**. Alongside model compilation and graph surgery, the CLI provides auxiliary commands to interact with system hardware, orchestrate workspaces, and track project coverage.

## `info webnn`

Prints diagnostic information regarding the host system's neural processing unit (NPU) and WebNN API capability.

```bash
onnx9000 info webnn
```
*(Note: Full detailed NPU metrics are more reliably retrieved from the browser context by running `onnx9000 serve` and viewing the WebNN dashboard.)*

## `chat`

Launches a Text-User Interface (TUI) within your terminal. This interfaces with your downloaded conversational AI models (e.g. LLaMA) providing an interactive CLI chatbot experience without needing a web browser.

```bash
onnx9000 chat
```

## `workspace`

Scaffolds a new standard ONNX9000 testing or integration workspace. It creates necessary boilerplate files allowing you to start injecting models and verifying them rapidly.

```bash
onnx9000 workspace ./my_new_project
```

## `coverage`

A maintenance utility to recalculate and summarize the test and framework-mapping coverage matrix for the ONNX9000 monorepo.

```bash
onnx9000 coverage
```
