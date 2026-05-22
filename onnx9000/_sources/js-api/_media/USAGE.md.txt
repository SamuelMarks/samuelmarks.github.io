# onnx9000: Usage Guide & Presentation Handout

> **Zero-dependency. WASM-First. Polyglot ONNX Execution and MLOps Ecosystem.**

`onnx9000` is a radical reimagining of the Machine Learning deployment stack. By completely eliminating massive C++ binaries, bloated Python dependencies, and complex CMake toolchains, we offer a truly cross-platform, zero-dependency environment for modern ML.

---

## 🚀 The Vision: Why onnx9000?

1. **Zero-Dependency by Default**: Parse, optimize, and execute models without `protobuf` C++ bindings or `onnxruntime`.
2. **Polyglot Monorepo**: Python handles data science tooling & native FFI; strictly-typed TypeScript powers the Edge, WebGPU, and UI.
3. **Write Once, Run Anywhere**: From Apple Accelerate / CUDA, to Cloudflare Workers, to WebGPU and WebAssembly in the browser.
4. **Static Memory Arenas**: Total elimination of dynamic memory allocations (`malloc`/`new`) during inference via AOT topological planning.

---

## 📦 Installation

### Python (via `uv` or `pip`)

```bash
# Minimal core for parsing/editing
uv pip install onnx9000-core

# Hardware execution, optimizers, and compilers
uv pip install onnx9000-backend-native onnx9000-optimizer onnx9000-converters

# Install specific backend accelerators
uv pip install onnx9000-openvino onnx9000-tensorrt

# Install everything
uv pip install "onnx9000[all]"
```

### TypeScript / JavaScript (via `pnpm` or `npm`)

```bash
# Core AST, parsers, and Edge serving
pnpm add @onnx9000/core @onnx9000/serve

# Web backends (WebGPU / WebNN / WASM SIMD)
pnpm add @onnx9000/backend-web

# Higher-level Generative AI pipelines (LLMs, Diffusion)
pnpm add @onnx9000/transformers @onnx9000/diffusers
```

---

## 🐍 Python Ecosystem

### 1. Zero-Dependency Parsers & Graph Surgery

`onnx9000-core` reads `.onnx`, `.pb`, `.h5`, `.tflite`, `.gguf`, and `.safetensors` using raw Python data structures.

```python
from onnx9000.core.parser.core import load
from onnx9000.core.shape_inference import infer_shapes_and_types
from onnx9000.core.graph_surgeon import extract_subgraph

# Parses the structure into an in-memory Graph AST without C++ Protobuf
graph = load("mobilenetv2.onnx")

# Run strict static shape inference
infer_shapes_and_types(graph)

# Extract a subgraph surgically
sub_graph = extract_subgraph(graph, input_names=["conv1_out"], output_names=["layer3_out"])
```

### 2. Hardware-Native Execution & Compilers

`onnx9000-backend-native` routes operations via a dynamic Python FFI dispatcher to native math libraries, or compiles down using our suite of plugins (OpenVINO, TensorRT, Triton, IREE).

```python
import numpy as np
from onnx9000.core.parser.core import load
from onnx9000.backends.session import InferenceSession
from onnx9000.backends.cuda.executor import CUDAExecutionProvider

# Integrates with enterprise backends effortlessly
from onnx9000.openvino import OpenVINOExecutionProvider

graph = load("model.onnx")

# Orchestrate execution
session = InferenceSession(
    graph,
    providers=[OpenVINOExecutionProvider(), CUDAExecutionProvider()]
)

# Run inference
input_data = {"input_1": np.random.randn(1, 3, 224, 224).astype(np.float32)}
outputs = session.run(output_names=["output_1"], input_feed=input_data)
```

### 3. Optimization & Quantization

`onnx9000-optimizer` offers in-memory graph surgery, algebraic simplification, and INT4/INT8 quantization.

```python
from onnx9000.optimizer import optimize, quantize, QuantizationConfig

# Apply Level 3 fusions (GELU, RoPE, LayerNorm)
optimized_graph = optimize(graph, level=3)

# INT8/INT4 Quantization
q_config = QuantizationConfig(weight_type="int8", activation_type="int8")
quantized_graph = quantize(optimized_graph, q_config)
```

---

## 🌐 TypeScript & Web Ecosystem

### 1. WebGPU / WebNN Browser Inference

`@onnx9000/backend-web` turns your AST into optimized WebGPU WGSL shaders or leverages WebNN for NPU access directly in the browser.

```typescript
import { load } from '@onnx9000/core';
import { InferenceSession, WebGPUProvider, WebNNProvider } from '@onnx9000/backend-web';

async function runModel(modelUrl: string) {
  const buffer = await (await fetch(modelUrl)).arrayBuffer();
  const graph = load(buffer); // No ONNX Runtime Web required

  const provider = new WebNNProvider(); // or WebGPUProvider
  await provider.initialize();

  const session = new InferenceSession(graph, [provider]);

  const inputData = new Float32Array(1 * 3 * 224 * 224).fill(0.5);
  const results = await session.run(['output'], {
    input: { data: inputData, shape: [1, 3, 224, 224], dtype: 'float32' },
  });
}
```

### 2. High-level GenAI (Diffusers & Transformers)

Run full Generative AI pipelines in WebWorkers or Node.js using `@onnx9000/diffusers` and `@onnx9000/transformers`.

```typescript
import { StableDiffusionPipeline } from '@onnx9000/diffusers';

// Loads optimized weights, builds WebGPU shaders, and generates in browser
const pipe = await StableDiffusionPipeline.fromPretrained('runwayml/stable-diffusion-v1-5');
const image = await pipe.generate('a futuristic city skyline at sunset');
```

### 3. Serverless Edge Serving

Serve models directly from Cloudflare Workers, Bun, or Deno using our zero-dependency server package.

```typescript
import { serve } from '@onnx9000/serve';
import { load } from '@onnx9000/core';

// Blazing fast inference on Edge functions
const graph = load('model.onnx');
serve(graph, { port: 3000, runtime: 'bun' });
```

### 4. TensorFlow.js Drop-in Shim

Migrate `@tensorflow/tfjs` projects with zero code changes and gain WebGPU performance.

```typescript
import * as tf from '@onnx9000/tfjs-shim';

const model = await tf.loadGraphModel('model.json');
const tensor = tf.tensor([1, 2, 3, 4], [2, 2]);
const output = model.predict(tensor);
output.print();
```

---

## 💻 Unified CLI (`onnx9000`) & Developer Apps

`onnx9000` ships with a powerful CLI and a suite of UI apps (built in TS) for a seamless developer experience.

```bash
# 🔍 Inspect a model's architecture
onnx9000 inspect ./model.onnx

# ⚡ Optimize and Quantize
onnx9000 optimize ./model.onnx --level 3 --output ./optimized.onnx
onnx9000 quantize ./model.onnx --format int8 --output ./quantized.onnx

# 🔄 N-Way Universal Transpilation (No original frameworks required)
onnx9000 convert --src keras --dst onnx ./model.h5
onnx9000 convert --src onnx --dst gguf ./model.onnx --output model.gguf
onnx9000 convert --src tflite --dst onnx ./model.tflite --output model.onnx

# 🛠 AOT Codegen: Compile directly to C++ source
onnx9000 compile --src onnx --dst c++ ./model.onnx --output model.cpp

# 🚀 Serve models directly via REST/gRPC
onnx9000 serve ./model.onnx --port 8080

# 🎨 Launch Interactive Browser Demos & UI Tools
onnx9000 ui netron         # Launch the native Graph Editor
onnx9000 ui llama-web      # Launch the WebGPU LLM Chat UI
onnx9000 ui whisper-web    # Launch WebGPU Whisper Audio Transcription
onnx9000 ui onnx2gguf      # Launch Visual GGUF Converter Tool
```
