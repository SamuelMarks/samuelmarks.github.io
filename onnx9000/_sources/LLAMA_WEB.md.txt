# LLaMA-Web Integration

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, **Polyglot Monorepo**. Through its modular design, it supports real-time transpilation and offline conversions across targets without requiring heavy official bindings or native dependencies.

## Overview

LLaMA-Web provides web-native LLaMA inference, utilizing WebGPU and WebAssembly to run models efficiently directly in the browser.

## Features

- **WebGPU Acceleration:** Native performance for LLaMA in the browser.
- **Zero-Dependency:** No external dependencies required for running LLaMA on the web.
- **KV-Caching:** Native KV-caching support.
- **W4A16 Quantization:** Built-in W4A16 weight quantization.

## Usage

You can use the CLI to interact with LLaMA-Web:

```bash
onnx9000 llama-web <model_path>
```
