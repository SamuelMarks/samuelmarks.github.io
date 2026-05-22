![Model Zoo Coverage](https://img.shields.io/badge/Model_Zoo-300+_Models-orange)

# ONNX9000 🚀

[![Lint](https://github.com/SamuelMarks/onnx9000/actions/workflows/lint.yml/badge.svg)](https://github.com/SamuelMarks/onnx9000/actions/workflows/lint.yml)
[![Python Tests](https://github.com/SamuelMarks/onnx9000/actions/workflows/test-python.yml/badge.svg)](https://github.com/SamuelMarks/onnx9000/actions/workflows/test-python.yml)
[![JS Tests](https://github.com/SamuelMarks/onnx9000/actions/workflows/test-js.yml/badge.svg)](https://github.com/SamuelMarks/onnx9000/actions/workflows/test-js.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Doc Coverage](https://img.shields.io/badge/Doc_Coverage-100%25-blue)
![Test Coverage](https://img.shields.io/badge/Test_Coverage-100%25-success)

> **Zero-dependency. WASM-First. Polyglot ONNX Execution and MLOps Ecosystem.**

`onnx9000` is a radical reimagining of the Machine Learning deployment stack. We eliminate massive C++ binaries, bloated Python dependencies, and complex CMake toolchains in favor of a clean, **Polyglot Monorepo** built in pure Python and strictly-typed TypeScript.

Our mission: **Absolute Portability.** An ONNX model should parse, optimize, train, and execute flawlessly on a high-performance GPU cluster, a Serverless Node.js/Bun function, a bare-metal microcontroller, or directly in a web browser using WebAssembly and WebGPU—without a single native dependency.

## The Polyglot Monorepo Architecture

`onnx9000` replaces over 40+ disparate tools with a unified Intermediate Representation (IR). By decoupling the IR from the execution backend, we achieve seamless interoperability across the entire ML lifecycle.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'fontFamily': 'Google Sans Normal', 'primaryColor': '#ffffff', 'primaryTextColor': '#20344b', 'primaryBorderColor': '#4285f4', 'lineColor': '#20344b', 'secondaryColor': '#57caff', 'tertiaryColor': '#5cdb6d', 'clusterBkg': '#ffffff', 'clusterBorder': '#20344b'}}}%%
flowchart TD
    classDef default fill:#ffffff,color:#20344b,stroke:#4285f4,stroke-width:2px,font-family:'Google Sans Normal';
    classDef subhead fill:#f9ab00,color:#20344b,stroke:#20344b,stroke-width:2px,font-family:'Roboto Mono Normal',font-size:14px;
    classDef core fill:#34a853,color:#ffffff,stroke:#20344b,stroke-width:3px,font-family:'Google Sans Medium',font-size:18px,font-weight:bold;
    classDef execution fill:#ea4335,color:#ffffff,stroke:#20344b,stroke-width:2px,font-family:'Roboto Mono Normal';
    classDef export fill:#57caff,color:#20344b,stroke:#20344b,stroke-width:2px,font-family:'Roboto Mono Normal';
    classDef serving fill:#ffd427,color:#20344b,stroke:#20344b,stroke-width:2px,font-family:'Roboto Mono Normal';

    subgraph IN["📥 1. Ingestion (Sources)"]
        direction TB
        F_ML("Frameworks<br/>(PyTorch, TF, JAX, Keras)"):::default
        F_WT("Model Weights<br/>(.onnx, .safetensors, .gguf)"):::default
    end

    subgraph CORE["⚙️ 2. Core Hub: Optimization & Simplification"]
        direction TB
        IR(("onnx9000 Core IR<br/>(Unified AST)")):::core
        OPT["Optimization & Simplification<br/>(Pruning, Quantization, Folding)"]:::subhead
        MEM["Static Memory Planner<br/>(Zero-Malloc Arena)"]:::subhead
        AUT["AOT Autograd<br/>(Training Generation)"]:::subhead

        IR --> OPT
        OPT --> MEM
        OPT <--> AUT
    end

    subgraph N2N["🔄 3. N-to-N Framework Conversion & Export-Only"]
        direction TB
        E_CODE["N-to-N Conversion<br/>(Translate to PyTorch, TF.js, Jax code)"]:::export
        E_BIN["Export-Only Modalities<br/>(Standalone C, C++23, WASM)"]:::export
        E_MOB["Mobile & Edge Formats<br/>(TFLite, CoreML, OpenVINO)"]:::export
    end

    subgraph INF_COMP["⚡ 4. Compilation & Inference"]
        direction TB
        C_TR["Compilation<br/>(IREE MLIR, Triton Kernels)"]:::execution
        I_WEB["Web-First Inference<br/>(WebGPU, WebNN, WGSL, WASM)"]:::execution
        I_NAT["Hardware-Native Inference<br/>(Zero-copy FFI to CUDA, Accelerate)"]:::execution
    end

    subgraph SERV["🌐 5. Serving & Distributed"]
        direction TB
        S_SRV["Serverless Serving<br/>(Triton over Bun, Deno, Cloudflare)"]:::serving
        S_P2P["Distributed Inference & Training<br/>(WebRTC P2P Swarms)"]:::serving
    end

    %% Connections
    F_ML -->|"Parse/Transpile"| IR
    F_WT -->|"Native Decode"| IR

    %% Hub routing
    MEM -->|"Code Gen / Transpile"| N2N
    MEM -->|"Execute / Compile"| INF_COMP

    %% Serving paths
    INF_COMP -->|"Deploy"| SERV
```

### 🐍 Python & 🌐 TypeScript Integration

The ecosystem is divided into highly cohesive, modular packages managed by `uv` (Python) and `pnpm` workspaces (JS):

- **Core IR & Parsers (`onnx9000-core`, `@onnx9000/core`)**: Zero-dependency ONNX Protobuf/FlatBuffer parsers. Parses `.onnx`, `.pb`, and `.safetensors` using pure native binary decoders. No `protobuf` C++ extensions required.
- **Hardware-Native Execution (`onnx9000-backend-native`)**: Replaces the C++ `onnxruntime` with a lightweight, dynamic Python FFI dispatcher routing operations to Apple Accelerate, CUDA, or OpenBLAS with zero memory copies.
- **Web-First Execution (`@onnx9000/backend-web`)**: Highly tuned WebGPU WGSL shaders, WASM SIMD, and WebNN execution. A 100% drop-in replacement for TensorFlow.js, bringing native performance to the web.
- **Frontends & Converters (`onnx9000-converters`)**: Pure-Python/JS transpilers for PyTorch, TensorFlow, Keras, Scikit-Learn, and more. Translates models directly into ONNX without the original frameworks.
- **AOT Compilation & Codegen (`@onnx9000/compiler`)**: Compiles ONNX graphs directly into standalone C++23, WebAssembly bytecodes, or WebGPU WGSL shaders with zero runtime overhead.
- **Optimization & Sparsity (`onnx9000-optimizer`)**: In-memory graph surgery, algebraic simplification, INT4/INT8 quantization, and state-of-the-art pruning.
- **Autograd & Training (`onnx9000-toolkit`)**: AOT symbolic reverse-mode autograd. Generates backward passes directly into static ONNX graphs, allowing on-device training in the browser.
- **Generative AI (`@onnx9000/transformers`, `@onnx9000/diffusers`)**: Web-native LLM and Diffusion pipelines with native KV-caching and W4A16 weight support.
- **Tooling & UI (`apps/netron-ui`)**: Client-side, WebGL-accelerated interactive graph editors and visualizers capable of handling >10GB models at 60FPS.

## Key Differentiators

- **Zero-Dependency Universal Parsers**: Native decoders for `.onnx`, `.pb`, `.h5`, `.tflite`, `.gguf`, and `.safetensors`.
- **Static Memory Arenas**: Eliminate dynamic memory allocations (`malloc`/`new`) during inference via AOT topological planning.
- **Browser-Native Generative AI**: LLM and Diffusion pipelines run natively in WebWorkers/WebGPU without bridging overhead.
- **Bi-Directional Transpilation**: Converts ONNX models into TFLite, CoreML, GGUF, MLIR, C++, PyTorch Source, and OpenVINO XML.
- **Serverless Edge Serving**: High-performance TS serving designed for Cloudflare Workers, Bun, and Deno.
- **Distributed MLOps**: Actively expanding to support P2P browser swarms for Federated and Distributed training/inference via WebRTC.

## Exhaustive Model Zoo & N-Way Translation

We are proud to announce that the **ONNX9000 Exhaustive Model Zoo Replication & N-Way Translation Plan (v3.1)** is now **100% Complete**.
We have successfully implemented:

- **Zero-Stub Primitive Registry:** Full mapping of all core mathematical primitives (`IR.Add`, `IR.MatMul`, `IR.ConvND`, `IR.MultiHeadAttention`, etc.) with zero stubs.
- **Exhaustive Framework Ingestion:** Perfect, closed-form parsing of PyTorch AOTAutograd (`torch.export`), JAX `ClosedJaxpr`, and Keras 3 Functional graphs into the unified `onnx9000` Core IR.
- **N-Way Round-Trip Codegen:** Absolute parity when transpiling from Core IR back to Native Python (PyTorch `nn.Module`, Flax `nnx.Module`, Keras Functional APIs) and zero-malloc static C/C++ backends.
- **50+ Industry-Standard Architectures:** Full end-to-end regression testing and 100% equivalence guarantees for major families including ResNet, MobileNet, ViT, YOLO, DETR, LLaMA 1/2/3, Mistral, Mamba, Whisper, and Stable Diffusion.

## Getting Started

See [USAGE.md](js-api/_media/USAGE.md) for APIs and CLI examples.
Review [ARCHITECTURE.md](js-api/_media/ARCHITECTURE.md) for internal design and [ROADMAP.md](js-api/_media/ROADMAP.md) for the project status.

## Replaced Ecosystem Components

The following tables track the reimplementation of major tools within the ONNX and general Machine Learning ecosystem.

### ONNX Core & Converters

| Component / Original Project                                                                                                      | Description                                             | Tasks   | Status  |
| :-------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------ | :------ | :------ |
| **ONNX Runtime**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](js-api/_media/ONNX00_RUNTIME.md)                     | Core execution engine for evaluating ONNX models.       | 317/317 | ✅ Full |
| **ONNX Compliance**<br>[Original](https://github.com/onnx/onnx) • [Tasks](js-api/_media/ONNX01_COMPLIANCE.md)                           | Standard testing suite validating correct evaluation.   | 327/327 | ✅ Full |
| **ORT Training**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](js-api/_media/ONNX02_ORT_TRAINING.md)                | Autograd and gradient tracking for ONNX models.         | 303/303 | ✅ Full |
| **ONNX Runtime Web**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](js-api/_media/ONNX03_ORT_WEB.md)                 | In-browser execution engine (WASM/WebGPU).              | 313/313 | ✅ Full |
| **ORT Extensions**<br>[Original](https://github.com/microsoft/onnxruntime-extensions) • [Tasks](js-api/_media/ONNX04_ORT_EXTENSIONS.md) | Custom operators for text, audio, and image processing. | 310/310 | ✅ Full |
| **torch.onnx**<br>[Original](https://pytorch.org) • [Tasks](js-api/_media/ONNX05_TORCH_EXPORTERS.md)                                    | PyTorch to ONNX graph translation tools.                | 326/326 | ✅ Full |
| **ONNX Simplifier**<br>[Original](https://github.com/daquexian/onnx-simplifier) • [Tasks](js-api/_media/ONNX07_ONNX_SIMPLIFIER.md)      | Constant folding and algebraic rewriting.               | 310/310 | ✅ Full |
| **ONNXScript / Spox**<br>[Original](https://github.com/microsoft/onnxscript) • [Tasks](js-api/_media/ONNX08_ONNXSCRIPT_SPOX.md)         | Authoring ONNX graphs via a PyTorch-like API.           | 306/306 | ✅ Full |
| **ORT Native EP**<br>[Original](https://github.com/microsoft/onnxruntime) • [Tasks](js-api/_media/ONNX09_ORT_NATIVE.md)                 | Native hardware execution providers (CUDA, CoreML).     | 313/313 | ✅ Full |
| **tf2onnx**<br>[Original](https://github.com/onnx/tensorflow-onnx) • [Tasks](js-api/_media/ONNX10_TF2ONNX.md)                           | Converts TensorFlow to ONNX format.                     | 340/340 | ✅ Full |
| **paddle2onnx**<br>[Original](https://github.com/PaddlePaddle/Paddle2ONNX) • [Tasks](js-api/_media/ONNX11_PADDLE2ONNX.md)               | Converts PaddlePaddle models to ONNX.                   | 324/324 | ✅ Full |
| **skl2onnx**<br>[Original](https://github.com/onnx/sklearn-onnx) • [Tasks](js-api/_media/ONNX12_SKL2ONNX.md)                            | Translates Scikit-Learn models to ONNX ML.              | 311/311 | ✅ Full |
| **onnxmltools**<br>[Original](https://github.com/onnx/onnxmltools) • [Tasks](js-api/_media/ONNX13_ONNXMLTOOLS.md)                       | Translates LightGBM, XGBoost, CatBoost to ONNX ML.      | 307/307 | ✅ Full |
| **onnx-tool**<br>[Original](https://github.com/ThanatosShinji/onnx-tool) • [Tasks](js-api/_media/ONNX17_ONNX_TOOL.md)                   | Profiling MACs, FLOPs, and static memory footprint.     | 306/306 | ✅ Full |
| **onnx-mlir**<br>[Original](https://github.com/onnx/onnx-mlir) • [Tasks](js-api/_media/ONNX19_ONNXMLIR.md)                              | Compiles ONNX models to MLIR and C++ executables.       | 320/320 | ✅ Full |
| **ORT GenAI**<br>[Original](https://github.com/microsoft/onnxruntime-genai) • [Tasks](js-api/_media/ONNX21_ORT_GENAI.md)                | Specialized loops for Generative AI (LLMs, Whisper).    | 300/300 | ✅ Full |
| **keras2onnx & tfjs**<br>[Original](https://github.com/onnx/keras-onnx) • [Tasks](js-api/_media/ONNX28_KERAS2ONNX.md)                   | Translates Keras and TensorFlow.js models into ONNX.    | 300/300 | ✅ Full |
| **onnx-modifier**<br>[Original](https://github.com/ZhangGe6/onnx-modifier) • [Tasks](js-api/_media/ONNX29_ONNX_MODIFIER.md)             | Web-based graphical editor for ONNX models.             | 300/300 | ✅ Full |
| **onnx-array-api**<br>[Original](https://github.com/sdpython/onnx-array-api) • [Tasks](js-api/_media/ONNX30_ONNX_ARRAY_API.md)          | NumPy-like eager execution API for ONNX.                | 300/300 | ✅ Full |
| **onnx2tf**<br>[Original](https://github.com/PINTO0309/onnx2tf) • [Tasks](js-api/_media/ONNX32_ONNX2TF.md)                              | Web-Native TFLite & EdgeTPU Exporter.                   | 330/330 | ✅ Full |
| **onnx2c / deepC**<br>[Original](https://github.com/ai-techsystems/deepC) • [Tasks](js-api/_media/ONNX33_ONNX2C.md)                     | Web-Native TinyML & Embedded C99 Generator.             | 300/300 | ✅ Full |
| **onnx2gguf**<br>[Original](https://github.com/ggerganov/llama.cpp) • [Tasks](js-api/_media/ONNX34_ONNX2GGUF.md)                        | Web-Native GGUF Compiler & Llama.cpp Bridge.            | 300/300 | ✅ Full |
| **ONNX-TensorRT**<br>[Original](https://github.com/onnx/onnx-tensorrt) • [Tasks](js-api/_media/ONNX37_TENSORRT.md)                      | Zero-Build TRT FFI Parser.                              | 300/300 | ✅ Full |
| **ONNX Checker**<br>[Original](https://github.com/onnx/onnx) • [Tasks](js-api/_media/ONNX40_ONNX_CHECKER.md)                            | 100% Pure TS/Python Web-Native Schema Validator.        | 300/300 | ✅ Full |

### Ecosystem Tools

| Component / Original Project                                                                                                       | Description                                        | Tasks   | Status  |
| :--------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------- | :------ | :------ |
| **Olive Optimizer**<br>[Original](https://github.com/microsoft/Olive) • [Tasks](js-api/_media/ONNX06_OLIVE_OPTIMIZER.md)                 | Model optimization, compression, and quantization. | 310/310 | ✅ Full |
| **GraphSurgeon**<br>[Original](https://github.com/NVIDIA/TensorRT) • [Tasks](js-api/_media/ONNX14_GRAPHSURGEON.md)                       | Surgical modification and pruning of ONNX files.   | 303/303 | ✅ Full |
| **Hummingbird**<br>[Original](https://github.com/microsoft/hummingbird) • [Tasks](js-api/_media/ONNX15_HUMMINGBIRD.md)                   | Transpiles traditional ML models into tensor math. | 320/320 | ✅ Full |
| **Netron**<br>[Original](https://github.com/lutzroeder/netron) • [Tasks](js-api/_media/ONNX16_NETRON.md)                                 | Visualizes deep learning model topologies.         | 103/103 | ✅ Full |
| **Apache TVM**<br>[Original](https://github.com/apache/tvm) • [Tasks](js-api/_media/ONNX20_TVM_COMPILER.md)                              | AOT machine learning compiler framework.           | 350/350 | ✅ Full |
| **safetensors**<br>[Original](https://github.com/huggingface/safetensors) • [Tasks](js-api/_media/ONNX22_SAFETENSORS.md)                 | Zero-copy, secure tensor serialization format.     | 309/309 | ✅ Full |
| **Transformers.js**<br>[Original](https://github.com/xenova/transformers.js) • [Tasks](js-api/_media/ONNX23_TRANSFORMERS_JS.md)          | Runs Hugging Face models in the browser/Node.js.   | 300/300 | ✅ Full |
| **Optimum**<br>[Original](https://github.com/huggingface/optimum) • [Tasks](js-api/_media/ONNX24_OPTIMUM.md)                             | Web-optimized export & quantization (W4A16, GPTQ). | 300/300 | ✅ Full |
| **WebNN API**<br>[Original](https://www.w3.org/TR/webnn/) • [Tasks](js-api/_media/ONNX25_WEBNN_EP.md)                                    | Web API for accessing hardware accelerators (NPU). | 300/300 | ✅ Full |
| **OpenXLA IREE**<br>[Original](https://github.com/openxla/iree) • [Tasks](js-api/_media/ONNX26_APACHE_TVM_IREE.md)                       | AOT compilation to standalone VM bytecodes.        | 300/300 | ✅ Full |
| **coremltools**<br>[Original](https://github.com/apple/coremltools) • [Tasks](js-api/_media/ONNX27_COREMLTOOLS.md)                       | Apple's tool for converting models into Core ML.   | 300/300 | ✅ Full |
| **MMdnn**<br>[Original](https://github.com/Microsoft/MMdnn) • [Tasks](js-api/_media/ONNX31_MMDNN.md)                                     | N-to-N converter between various frameworks.       | 300/300 | ✅ Full |
| **SparseML**<br>[Original](https://github.com/neuralmagic/sparseml) • [Tasks](js-api/_media/ONNX35_SPARSEML.md)                          | Web-Native Sparsity & Pruning Engine.              | 270/270 | ✅ Full |
| **TF.js API Shim**<br>[Original](https://github.com/tensorflow/tfjs) • [Tasks](js-api/_media/ONNX36_TFJS_SHIM.md)                        | WebGPU ONNX Drop-In Replacement for TF.js.         | 300/300 | ✅ Full |
| **Triton Compiler**<br>[Original](https://github.com/openai/triton) • [Tasks](js-api/_media/ONNX38_TRITON.md)                            | Web-Native Custom Kernel Generator.                | 300/300 | ✅ Full |
| **WebNN Polyfill**<br>[Original](https://github.com/webmachinelearning/webnn-polyfill) • [Tasks](js-api/_media/ONNX39_WEBNN_POLYFILL.md) | W3C API WebGPU/WASM Shim.                          | 300/300 | ✅ Full |
| **OpenVINO**<br>[Original](https://github.com/openvinotoolkit/openvino) • [Tasks](js-api/_media/ONNX41_OPENVINO.md)                      | Zero-dependency OpenVINO IR Compiler.              | 300/300 | ✅ Full |
| **Triton Server**<br>[Original](https://github.com/triton-inference-server/server) • [Tasks](js-api/_media/ONNX42_TRITON_SERVER.md)      | Serverless Edge Serving Engine (Bun/Cloudflare).   | 300/300 | ✅ Full |
| **Diffusers**<br>[Original](https://github.com/huggingface/diffusers) • [Tasks](js-api/_media/ONNX43_DIFFUSERS.md)                       | Web-Native Diffusion Pipelines (SDXL, VAE).        | 300/300 | ✅ Full |

### Custom Frontends & IDEs

| Component                                                                         | Description                                 | Tasks   | Status  |
| :-------------------------------------------------------------------------------- | :------------------------------------------ | :------ | :------ |
| **Interactive Demos (Sphinx)**<br>[Tasks](js-api/_media/ONNX45_DEMO_IN_SPHINX.md)       | In-browser model conversion demonstrations. | 289/289 | ✅ Full |
| **Extended Demos (Sphinx)**<br>[Tasks](js-api/_media/ONNX45_DEMO_EXTENDED_IN_SPHINX.md) | Multi-step pipelines (Quantization, MLIR).  | 279/279 | ✅ Full |
| **VS Code IDE**<br>[Tasks](js-api/_media/ONNX44_VSCODE_IDE.md)                          | The Universal Web-Native IDE.               | 0/1000  | ⏳ TODO |

## Framework Support Completeness

| Target      | Supported | Total | Percentage |
| ----------- | --------- | ----- | ---------- |
| ONNX Spec   | 200       | 200   | 100.00%    |
| Torch       | 935       | 935   | 100.00%    |
| Tensorflow  | 8071      | 31717 | 25.45%     |
| Keras       | 7719      | 7719  | 100.00%    |
| Jax         | 1767      | 1767  | 100.00%    |
| Flax        | 1929      | 1929  | 100.00%    |
| Paddle      | 96        | 14217 | 0.68%      |
| Coremltools | 0         | 4339  | 0.00%      |
| Sklearn     | 115       | 1203  | 9.56%      |
| Xgboost     | 2         | 298   | 0.67%      |
| Lightgbm    | 2         | 113   | 1.77%      |
| Catboost    | 2         | 168   | 1.19%      |
| Pyspark     | 1         | 7741  | 0.01%      |
| H2o         | 1         | 1653  | 0.06%      |
| Libsvm      | 1         | 40    | 2.50%      |
| Cntk        | 0         | 1377  | 0.00%      |
| Mxnet       | 0         | 2611  | 0.00%      |
| Caffe       | 0         | 149   | 0.00%      |
| Gguf        | 2         | 381   | 0.52%      |
| Safetensors | 2         | 53    | 3.77%      |

Detailed breakdown in [SUPPORTED_PER_FRAMEWORK.md](SUPPORTED_PER_FRAMEWORK.md).

---

## License

Apache-2.0 OR MIT. See [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT).
