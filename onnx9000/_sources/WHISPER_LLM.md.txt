# Whisper-LLM Integration

> **Ecosystem Context:** `onnx9000` operates as a zero-dependency, **Polyglot Monorepo**. Through its modular design, it supports real-time transpilation and offline conversions across targets without requiring heavy official bindings or native dependencies.

## Overview

Whisper-LLM provides web-native inference for Whisper models, leveraging WebGPU and WebAssembly for efficient audio transcription directly in the browser.

## Features

- **WebGPU Acceleration:** Native performance for Whisper in the browser.
- **Zero-Dependency:** No external dependencies required.
- **Optimized Decoding:** Custom logit processors and beam search tailored for speech-to-text.

## Usage

You can use the CLI to interact with Whisper-LLM:

```bash
onnx9000 whisper-llm <model_path>
```
