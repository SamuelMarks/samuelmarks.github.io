# Memory and Execution Profiler

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, **Polyglot Monorepo**. Through its modular design, it supports real-time transpilation and offline conversions across targets without requiring heavy official bindings or native dependencies.

## Overview

The Profiler is a critical component for diagnosing execution bottlenecks, visualizing memory arenas, and understanding hardware utilization (such as NPU/GPU metrics).

## Features

- **Memory Arena Visualization:** Understand how static memory planners allocate tensors.
- **Execution Tracing:** Fine-grained operation timing.
- **Export Formats:** Export traces in standard formats like Chrome Trace Format.

## Usage

Use the CLI command to profile models:

```bash
onnx9000 profiler <model_path>
```
