# ONNX9000 Roadmap

This document outlines the current state and future milestones of the `onnx9000` ecosystem.

## 🚀 Current Status: Ecosystem Complete

We have successfully executed the **Polyglot Monorepo Refactor** and **implemented every major feature specification**.
The massive single-directory Python monolith has been cleanly split into a highly modular, decoupled environment managed by `pnpm` and `uv` workspaces, and all planned frontends, backends, compilers, and applications are now live.

- **Python Core:** The `onnx9000-core` package parses `.onnx`, `.pb`, and `.safetensors` files with zero external dependencies using `struct` and `mmap` directly to an AST.
- **Python EPs:** `onnx9000-backend-native` provides `ctypes` bindings to OpenBLAS/Accelerate, mapping our custom Tensors via DLPack interfaces.
- **TypeScript Core:** `@onnx9000/core` implements an exact structural clone of the ONNX AST with the strictest possible type safety (no `any`, `unknown`).

## 🏆 Completed Implementation Specifications (The 45 Specs)

The following architectural targets guide the development of the ecosystem. **Almost all phases of the initial 45 specs are now complete (44/45).**

### Core Execution & Web Backends

- [x] **ONNX00:** Runtime (Native Exec) Replication & Parity Tracker.
- [x] **ONNX01:** ONNX Standard Compliance & Testing Tracker.
- [x] **ONNX03:** ONNX Runtime Web Replication (`@onnx9000/backend-web`).
- [x] **ONNX09:** ORT Native EP (CUDA, CoreML) Replication.
- [x] **ONNX25:** WebNN API Native Browser NPU Execution.
- [x] **ONNX39:** WebNN Polyfill (W3C API WebGPU/WASM Shim).

### Tooling, Parsing, and Optimizations

- [x] **ONNX04:** ONNX Runtime Extensions Replication.
- [x] **ONNX06:** Olive Optimizer Replication (Quantization and W4A16 targeting).
- [x] **ONNX07:** ONNX Simplifier Replication.
- [x] **ONNX14:** ONNX GraphSurgeon Replication.
- [x] **ONNX17:** `onnx-tool` Profiling Replication.
- [x] **ONNX22:** Safetensors Replication (Zero-copy `mmap` and `ArrayBuffer` extraction).
- [x] **ONNX35:** SparseML Replication (Web-Native Sparsity & Pruning Engine).
- [x] **ONNX40:** ONNX Checker (100% Pure TS/Python Web-Native Schema Validator).

### Frontends & Converters (`onnx9000-converters`)

- [x] **ONNX05:** Torch & TF Exporters Replication.
- [x] **ONNX10:** `tf2onnx` Replication (Zero-dependency TF parsing).
- [x] **ONNX11:** `paddle2onnx` Replication.
- [x] **ONNX12:** `skl2onnx` Replication (Compiling Scikit-Learn to `ai.onnx.ml`).
- [x] **ONNX13:** `onnxmltools` Replication (LightGBM, XGBoost to ONNX).
- [x] **ONNX15:** Hummingbird Replication (Compiling Trees to Tensor Math).
- [x] **ONNX27:** `coremltools` (Web-Native Apple Silicon Bridge).
- [x] **ONNX28:** `keras2onnx` & `tfjs-to-onnx` (Web-Native Keras Converter).
- [x] **ONNX31:** `MMdnn` (Web-Native N-to-N Neural Network Converter).
- [x] **ONNX32:** `onnx2tf` (Web-Native TFLite & EdgeTPU Exporter).
- [x] **ONNX34:** `onnx2gguf` (Web-Native GGUF Compiler & Llama.cpp Bridge).
- [x] **ONNX36:** TF.js API Shim (WebGPU ONNX Drop-In Replacement for TF.js).
- [x] **ONNX37:** ONNX-TensorRT (Zero-Build TRT FFI Parser).

### Compilers & AOT (`@onnx9000/compiler`)

- [x] **ONNX19:** `onnx-mlir` Replication (Compiling ONNX to C++23/WASM).
- [x] **ONNX20:** Apache TVM Ahead-of-Time Web Compiler.
- [x] **ONNX26:** OpenXLA IREE (WASM-Native MLIR Compiler).
- [x] **ONNX33:** `onnx2c` / `deepC` (Web-Native TinyML & Embedded C++ Generator).
- [x] **ONNX38:** Triton Compiler (Web-Native Custom Kernel Generator).
- [x] **ONNX41:** OpenVINO Optimizer (Zero-dependency OpenVINO IR Compiler).

### Web UI & Applications (`apps/`)

- [x] **ONNX16:** Netron Replication (`netron-ui`).
- [x] **ONNX24:** HuggingFace Optimum UI (`optimum-ui`).
- [x] **ONNX29:** `onnx-modifier` (Web-Native Graph Editor & Visualizer).
- [x] **ONNX44:** VS Code Machine Learning OS (The Universal Web-Native IDE).

### High-Level APIs & GenAI

- [x] **ONNX02:** ONNX Runtime Training Replication (AOT Symbolic Autograd).
- [x] **ONNX08:** ONNXScript / Spox Replication (Fluent Model Authoring).
- [x] **ONNX21:** ONNX Runtime GenAI (WASM-First Generative Execution).
- [x] **ONNX23:** Transformers.js (WASM-Native Auto-Pipelines).
- [x] **ONNX30:** `onnx-array-api` (Web-Native NumPy/Eager API for ONNX).
- [x] **ONNX42:** Triton Inference Server (Serverless Edge Serving Engine for Bun/Cloudflare).
- [x] **ONNX43:** Diffusers (Web-Native Diffusion Pipelines like SDXL, VAE).

## 🔮 The New Frontier: Distributed MLOps

With the foundational framework support and local execution ecosystem complete, the `onnx9000` project is now pivoting its full engineering weight to the **Distributed MLOps Framework** (detailed in `ONNX_NEXT_NEXT_PLAN.md`). This massive milestone shifts focus from local optimization to cluster-scale execution:

1. **Distributed Transport Layer:** WebRTC & WebSockets bridging Python and standard Browser JS.
2. **Distributed Multi-Node Inference:** Graph partitioning and Pipeline Parallelism across P2P browser swarms.
3. **Distributed & Federated Training:** Eager and AOT training loops over Ring-AllReduce topologies.
4. **Unified MLOps SDK & CLI:** Centralized `onnx9000` CLI for artifact and metric tracking.
5. **Zero-Dependency MLOps Server:** A pure-Python HTTP Server backed by SQLite for tracking experiments without Docker.
6. **Model Registry:** Web-safe weight format management (`.safetensors`, GGUF).
7. **Experiment Tracking:** Real-time run status and massive metric time-series ingestion.
8. **MLOps Web Frontend:** High-performance dashboard for charts, sweeps, and model staging.
9. **Fault Tolerance & Security:** Handling node dropouts, Split Brain, and end-to-end encryption.
10. **End-to-End System Deployment:** Cloud integrations (AWS, Vercel) and 100-node simulated swarm testing.

## Framework Support Completeness

For a detailed breakdown of our framework support completeness and % compliant metrics, please see [SUPPORTED_PER_FRAMEWORK.md](SUPPORTED_PER_FRAMEWORK.md).
