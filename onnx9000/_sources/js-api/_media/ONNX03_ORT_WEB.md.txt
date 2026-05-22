# ONNX Runtime Web Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `ONNX Runtime Web` (`onnxruntime-web`) within the `onnx9000` ecosystem.
The original `onnxruntime-web` compiles the massive C++ ONNX Runtime into WebAssembly via Emscripten, resulting in large bundle sizes and heavy initialization times.
Our `onnx9000` Web engine is written natively in strict TypeScript/JavaScript. It parses ONNX graphs dynamically in the browser and dispatches execution directly to **WebGPU** (via native WGSL compute shaders) or lightweight **WASM** (generated AOT or via pure JS SIMD polyfills). This architecture enables true zero-overhead startup, progressive weight streaming directly to GPU memory, and deep integration with native Web APIs like WebWorkers, OffscreenCanvas, and WebNN.

## Exhaustive Parity Checklist

### 1. Core JS/TS API & Session Management (40+ items)

- [xx] Implement `onnxruntime-web` compatible `InferenceSession` class API
- [xx] Implement `InferenceSession.create(modelPath, options)`
- [xx] Implement `InferenceSession.create(buffer, options)`
- [xx] Implement `InferenceSession.run(feeds, fetches, options)`
- [xx] Implement `InferenceSession.startProfiling()`
- [xx] Implement `InferenceSession.endProfiling()`
- [xx] Implement `Tensor` abstraction matching `ort.Tensor` exactly
- [xx] Support `Tensor` instantiation from `Float32Array`
- [xx] Support `Tensor` instantiation from `Uint8Array`
- [xx] Support `Tensor` instantiation from `Int32Array`
- [xx] Support `Tensor` instantiation from `BigInt64Array`
- [xx] Support `Tensor` instantiation from native JS `Array`
- [xx] Support `Tensor` string variants (arrays of JS strings)
- [xx] Support `Tensor` explicit reshaping (`tensor.reshape([N, C, H, W])`)
- [xx] Implement global `env` configuration object (`ort.env`)
- [xx] Support `env.wasm.numThreads` configuration natively
- [xx] Support `env.wasm.simd` configuration natively
- [xx] Support `env.wasm.wasmPaths` overrides natively
- [xx] Support `env.webgl.pack` (legacy compatibility) natively
- [xx] Support `env.logLevel` (`'verbose'`, `'info'`, `'warning'`, `'error'`, `'fatal'`)
- [xx] Expose `SessionOptions.executionProviders` natively (`['webgpu', 'wasm']`)
- [xx] Expose `SessionOptions.graphOptimizationLevel` configuration
- [xx] Expose `SessionOptions.logSeverityLevel`
- [xx] Expose `SessionOptions.logId`
- [xx] Implement topological sorting of the graph dynamically in JS
- [xx] Implement dynamic input validation (verifying shapes/types before execution)
- [xx] Support overriding node shapes via `SessionOptions.freeDimensionOverrides`
- [xx] Catch WebGPU unsupported warnings and fallback to WASM automatically
- [xx] Provide TypeScript `.d.ts` type definitions matching the official ORT specs
- [xx] Wrap all async initialization logic safely to prevent browser UI freezing
- [xx] Implement asynchronous tensor data downloading for external `.bin` weights
- [xx] Expose `getInputs()` returning structured metadata arrays
- [xx] Expose `getOutputs()` returning structured metadata arrays
- [xx] Map internal ONNX graph errors to clear, stack-traced JavaScript `Error` instances
- [xx] Support Webpack/Vite/Rollup tree-shaking natively (ESM modules)
- [xx] Support CommonJS (`require()`) fallback for Node.js compatibility
- [xx] Support CDN-based loading (e.g., `<script src="https://cdn..."></script>`)
- [xx] Guarantee zero dependencies in `package.json` for the core inference engine
- [xx] Prevent usage of `eval()` or `new Function()` in JS parser to comply with strict CSP (Content Security Policy)
- [xx] Avoid `SharedArrayBuffer` usage implicitly unless specifically enabled by user (due to COOP/COEP restrictions)

### 2. WebGPU Execution Provider: Core Engine (35+ items)

- [xx] Detect `navigator.gpu` natively
- [xx] Request `GPUDevice` and `GPUAdapter` securely
- [xx] Support requesting high-performance GPU (`powerPreference: "high-performance"`)
- [xx] Support requesting low-power GPU (`powerPreference: "low-power"`)
- [xx] Implement `GPUBuffer` memory arena for intermediate tensors
- [xx] Track buffer lifetimes and execute `destroy()` to prevent GPU OOM
- [xx] Optimize intermediate buffer reuse (reusing `GPUBuffer` for disjoint tensors)
- [xx] Allocate `GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST` safely
- [xx] Pad `GPUBuffer` sizes to strict WebGPU alignment rules (multiples of 4 bytes / 256 bytes for uniforms)
- [xx] Manage `GPUBindGroupLayout` caching to reduce compilation overhead
- [xx] Manage `GPUPipelineLayout` caching statically per op-type
- [xx] Implement WGSL Shader compilation cache (`createComputePipelineAsync`)
- [xx] Manage `GPUComputePassEncoder` dynamically for topological execution
- [xx] Support submitting massive graphs via chunked `GPUCommandBuffer` (preventing TDR timeouts)
- [xx] Map JS `Float32Array` to `GPUBuffer` natively via `writeBuffer`
- [xx] Map `GPUBuffer` back to JS `Float32Array` via `mapAsync(GPUMapMode.READ)`
- [xx] Expose explicit WebGPU IOBinding (allow users to feed pre-existing `GPUBuffer` to `.run()`)
- [xx] Expose explicit WebGPU IOBinding for outputs (skip `mapAsync` download)
- [xx] Analyze device `limits` (e.g., `maxStorageBufferBindingSize`, `maxComputeWorkgroupSizeX`) dynamically
- [xx] Fallback/chunk executions if intermediate tensors exceed `maxStorageBufferBindingSize`
- [xx] Inject dynamic uniform buffers for scalar constants (e.g., shapes, strides)
- [xx] Support WGSL `f16` extension (`shader-f16`) explicitly if the device supports it
- [xx] Fallback to `f32` emulation if `shader-f16` is unavailable natively
- [xx] Pack `int8` weights natively into `u32` WGSL buffers to save memory
- [xx] Unpack `int8` to `f32` inside the WGSL compute shader
- [xx] Handle 64-bit integer shapes dynamically in WGSL (emulated via 32-bit math to comply with WGSL limits)
- [xx] Use `read-only` storage buffers aggressively for performance
- [xx] Track pipeline compilation latency using `performance.now()`
- [xx] Implement "warmup" execution passes to trigger async pipeline compilations ahead of time
- [xx] Manage WebGPU lost device events (`device.lost`) natively and attempt recovery
- [xx] Integrate with WebXR frame loops without breaking WebGL/WebGPU context locks
- [xx] Output native WebGPU error scopes (`pushErrorScope`) for WGSL debugging
- [xx] Compile shaders explicitly using WGSL (no SPIR-V or internal translations)

### 3. WebGPU Execution Provider: WGSL Shaders (50+ items)

- [xx] Implement WGSL `Add` (broadcasting, 1D/2D/3D/4D)
- [xx] Implement WGSL `Sub`
- [xx] Implement WGSL `Mul`
- [xx] Implement WGSL `Div`
- [xx] Implement WGSL `MatMul` (Naive generic)
- [xx] Implement WGSL `MatMul` (Workgroup tile-cached, highly optimized for `16x16` or `8x8`)
- [xx] Implement WGSL `MatMul` (Batched 3D/ND)
- [xx] Implement WGSL `Gemm` (with `alpha`, `beta`, `transA`, `transB`)
- [xx] Implement WGSL `Conv` (Standard 2D, naive loop)
- [xx] Implement WGSL `Conv` (Im2Col + Gemm WGSL translation)
- [xx] Implement WGSL `Conv` (Depthwise optimization)
- [xx] Implement WGSL `ConvTranspose`
- [xx] Implement WGSL `MaxPool`
- [xx] Implement WGSL `AveragePool`
- [xx] Implement WGSL `GlobalAveragePool`
- [xx] Implement WGSL `BatchNormalization`
- [xx] Implement WGSL `LayerNormalization`
- [xx] Implement WGSL `Relu`
- [xx] Implement WGSL `LeakyRelu`
- [xx] Implement WGSL `Sigmoid`
- [xx] Implement WGSL `Tanh`
- [xx] Implement WGSL `Softmax` (Safe implementation subtracting max)
- [xx] Implement WGSL `LogSoftmax`
- [xx] Implement WGSL `Gelu` (Erf approximation)
- [xx] Implement WGSL `Erf`
- [xx] Implement WGSL `Exp`
- [xx] Implement WGSL `Log`
- [xx] Implement WGSL `Pow`
- [xx] Implement WGSL `Sqrt`
- [xx] Implement WGSL `Abs`
- [xx] Implement WGSL `Clip`
- [xx] Implement WGSL `ReduceSum` (Parallel reduction tree in WGSL)
- [xx] Implement WGSL `ReduceMean`
- [xx] Implement WGSL `ReduceMax`
- [xx] Implement WGSL `ReduceMin`
- [xx] Implement WGSL `Transpose` (Shared memory tiling optimization)
- [xx] Implement WGSL `Reshape` (Metadata update, no-op in buffers)
- [xx] Implement WGSL `Concat`
- [xx] Implement WGSL `Split`
- [xx] Implement WGSL `Slice`
- [xx] Implement WGSL `Gather`
- [xx] Implement WGSL `GatherND`
- [xx] Implement WGSL `ScatterND`
- [xx] Implement WGSL `Pad`
- [xx] Implement WGSL `Tile`
- [xx] Implement WGSL `Expand`
- [xx] Implement WGSL `Where`
- [xx] Implement WGSL `Cast`
- [xx] Implement WGSL `NonMaxSuppression` (Compute-heavy filtering)
- [xx] Implement WGSL `Resize` (Bilinear and Nearest)

### 4. WebGPU Optimizations & Tuning (30+ items)

- [xx] Tune `workgroup_size` dynamically based on tensor dimensionality (e.g. `[64, 1, 1]` vs `[8, 8, 1]`)
- [xx] Vectorize WGSL memory loads (e.g., using `vec4<f32>` instead of `f32` for 4x bandwidth)
- [xx] Vectorize WGSL memory stores (`vec4<f32>`)
- [xx] Prevent Out-of-Bounds memory accesses explicitly in WGSL via array-length clamping
- [xx] Fuse `Conv` + `Relu` dynamically by generating a combined WGSL shader string
- [xx] Fuse `Conv` + `Add` (Bias) dynamically into a single shader
- [xx] Fuse `MatMul` + `Add` dynamically
- [xx] Fuse `Gemm` + `Relu` dynamically
- [xx] Fuse sequential elementwise ops (e.g. `Add` + `Mul` + `Sigmoid` -> single WGSL shader)
- [xx] Implement shader template caching (deduplicating identical WGSL strings dynamically)
- [xx] Cache WGSL bindings natively based on signature
- [xx] Evaluate `dispatchWorkgroups` XYZ limits natively and wrap execution if exceeded
- [xx] Emulate multidimensional indexing (e.g., `[n][c][h][w]`) natively using fast WGSL modulo/division math
- [xx] Optimize 1D flat access for contiguous memory paths natively in WGSL
- [xx] Generate WGSL constants directly into the shader text (e.g. `#define BATCH_SIZE 32` -> `const BATCH: u32 = 32;`) to allow WebGPU compiler unrolling
- [xx] Compile explicit `INT8` quantized matrix multiplications (DP4A emulation in WGSL)
- [xx] Unroll small inner loops dynamically during JS WGSL template rendering
- [xx] Handle WebGPU coordinate system bounds gracefully (top-left vs bottom-left differences if any)
- [xx] Verify WebGPU timestamp queries for microsecond-accurate shader profiling (`timestamp-query` extension)
- [xx] Disable profiling extensions automatically in production environments
- [xx] Expose WebGPU debug labels (`label: "Conv_23"`) for clean Chrome/Firefox GPU debugging
- [xx] Minimize `writeBuffer` calls by packing multiple scalars into a single Uniform buffer upload
- [xx] Sync GPU execution with `requestAnimationFrame` optimally
- [xx] Verify GPU context loss recovery works seamlessly for long-running tabs
- [xx] Use `createReadyComputePipeline` (if API exists) vs `createComputePipelineAsync` optimally
- [xx] Dynamically rewrite unsupported `mod` math in WGSL explicitly
- [xx] Dynamically rewrite 64-bit comparisons natively to avoid WGSL type errors
- [xx] Handle `Inf` and `NaN` propagation securely within WebGPU
- [xx] Enforce strict IEEE 754 compliance in WGSL mathematically
- [xx] Avoid subgroup operations if not universally supported across standard browser WebGPU targets

### 5. WASM Execution Provider (WebAssembly SIMD & Threads) (40+ items)

- [xx] Detect WebAssembly support natively
- [xx] Detect WebAssembly SIMD (`Fixed-Width SIMD`) natively
- [xx] Detect WebAssembly Threads (`SharedArrayBuffer`) natively
- [xx] Compile core ONNX ops explicitly to `.wasm` (via AOT transpiler or lightweight C++ kernels)
- [xx] Instantiate `.wasm` module securely via `WebAssembly.instantiateStreaming`
- [xx] Implement WASM `Add`, `Sub`, `Mul`, `Div` kernels
- [xx] Implement WASM `MatMul` (cache-blocked, utilizing SIMD intrinsics `v128`)
- [xx] Implement WASM `Conv` (spatial and depthwise)
- [xx] Implement WASM `Softmax`, `Relu`, `Gelu`, `Sigmoid`
- [xx] Implement WASM `ReduceSum`, `ReduceMean`, `ReduceMax`, `ReduceMin`
- [xx] Allocate WASM memory arena (`WebAssembly.Memory`) dynamically
- [xx] Expose JS `Float32Array` views mapped directly to WASM memory space
- [xx] Synchronize WASM memory expansion (`memory.grow`) dynamically without crashing
- [xx] Implement multithreaded execution using a WebWorker thread pool (Pthreads emulation)
- [xx] Dispatch specific subgraph branches to distinct WebWorkers for parallel execution
- [xx] Implement `Atomics.wait` and `Atomics.notify` natively for WASM thread synchronization
- [xx] Fallback to sequential execution if `SharedArrayBuffer` is blocked by CORS/COOP
- [xx] Pre-fetch and cache the `.wasm` payload via ServiceWorker natively
- [xx] Limit `.wasm` payload size to <2MB via extreme DCE (Dead Code Elimination)
- [xx] Load pure JavaScript implementations (polyfills) if WASM is entirely disabled
- [xx] Implement JS fallback `MatMul` using naive loops
- [xx] Implement JS fallback `Conv` using naive loops
- [xx] Implement JS fallback `Softmax`
- [xx] Implement JS fallback `Gather`, `Scatter`, `Slice`, `Concat`
- [xx] Profile WASM execution latency natively using `performance.now()`
- [xx] Profile JS fallback execution latency
- [xx] Align WASM memory arrays strictly to 16-byte boundaries for SIMD vectorization
- [xx] Extract string metadata correctly from WASM memory boundaries
- [xx] Validate 32-bit WASM memory limitations (maximum 4GB)
- [xx] Catch WASM `RuntimeError: memory access out of bounds` and map to readable JS errors
- [xx] Free WASM allocated blocks explicitly upon `InferenceSession` destruction
- [xx] Compile WASM with `-Oz` (size optimization) targeting web constraints
- [xx] Compress `.wasm` payload using Brotli/Gzip during network transit
- [xx] Bind WASM execution synchronously for tiny models (blocking main thread briefly)
- [xx] Bind WASM execution asynchronously via WebWorkers for massive models
- [xx] Emulate `Math.erf` securely in JS for Gelu approximations
- [xx] Extract subnormal float values cleanly from WASM arrays
- [xx] Handle Javascript BigInt mappings seamlessly to WASM 64-bit integer arguments
- [xx] Provide explicit WebAssembly instantiation hooks for strict CSP environments
- [xx] Optimize Javascript to WASM function call overhead (batching calls where possible)

### 6. Progressive Loading, Streaming & IO Binding (35+ items)

- [xx] Implement `fetch` with `Range` headers natively
- [xx] Download Model JSON structural definition asynchronously
- [xx] Download Model `.bin` weight payloads progressively
- [xx] Map downloaded chunks directly into `GPUBuffer` instantly
- [xx] Map downloaded chunks directly into WASM `WebAssembly.Memory` instantly
- [xx] Delete JS `ArrayBuffer` instantly after GPU/WASM upload to trigger V8 Garbage Collection
- [xx] Prevent JS Heap Out-Of-Memory (OOM) on massive >2GB models
- [xx] Support loading models natively from IndexedDB (`idb`)
- [xx] Cache downloaded network weights directly into IndexedDB for instant offline re-loads
- [xx] Expose progress callbacks: `onProgress({ loaded, total })` for UI loading bars
- [xx] Support fetching weights directly from HuggingFace Hub (`hf.co`)
- [xx] Implement auto-retry logic for interrupted `fetch` requests
- [xx] Read local files directly via the `File` API (drag-and-drop into browser)
- [xx] Parse `.safetensors` external data natively in JavaScript
- [xx] Support WebCodecs `VideoFrame` as an explicit IOBinding source (zero-copy GPU video processing)
- [xx] Support WebGL `WebGLTexture` as an explicit IOBinding source (interop)
- [xx] Support HTML5 `<canvas>` / `ImageData` as an explicit IOBinding source
- [xx] Map output `GPUBuffer` directly to HTML5 Canvas for instant rendering (no CPU roundtrip)
- [xx] Support `OffscreenCanvas` in WebWorkers natively
- [xx] Stream execution: Execute layer N while layer N+1 is still downloading
- [xx] Track exact memory bounds dynamically to pause fetching if VRAM limits approach
- [xx] Flush WebGPU command queues incrementally during progressive loads
- [xx] Parse ONNX External Data mappings accurately (resolving relative URIs)
- [xx] Implement HTTP Keep-Alive for high-throughput chunk fetching
- [xx] Handle 416 Range Not Satisfiable errors gracefully by falling back to full downloads
- [xx] Decrypt HTTP chunked payloads dynamically if requested
- [xx] Unpack compressed weights (e.g. gzip/brotli) dynamically via `DecompressionStream` API
- [xx] Validate `sha256` checksums of downloaded chunks dynamically using Web Crypto API
- [xx] Pause execution natively if required tensors are not yet fully buffered
- [xx] Expose JS async iterators (`async function*`) for stream generation tasks (e.g., LLM text gen)
- [xx] Allow cancelling a running progressive load cleanly (aborting `fetch` signals)
- [xx] Prevent memory leaks by aborting internal promises cleanly on session close
- [xx] Support Base64 embedded tensors (data URIs)
- [xx] Detect endianness mismatch and byte-swap dynamically in JS if required
- [xx] Zero-pad corrupted chunks safely if `strict=false` is set

### 7. WebNN Execution Provider (Experimental/Future Parity) (25+ items)

- [xx] Detect `navigator.ml` (WebNN API) natively
- [xx] Request WebNN Context (`navigator.ml.createContext()`)
- [xx] Request WebNN Context with GPU preference
- [xx] Request WebNN Context with NPU (Neural Processing Unit) preference
- [xx] Request WebNN Context with CPU preference
- [xx] Map ONNX `Add`, `Sub`, `Mul`, `Div` to WebNN `mlGraphBuilder.add`, etc.
- [xx] Map ONNX `MatMul` to WebNN `mlGraphBuilder.matmul`
- [xx] Map ONNX `Conv` to WebNN `mlGraphBuilder.conv2d`
- [xx] Map ONNX `MaxPool`, `AveragePool` to WebNN equivalents
- [xx] Map ONNX `Relu`, `Sigmoid`, `Tanh`, `Softmax` to WebNN equivalents
- [xx] Map ONNX `Reshape`, `Transpose`, `Concat`, `Slice` to WebNN equivalents
- [xx] Map ONNX `ReduceMean`, `ReduceSum` to WebNN equivalents
- [xx] Map ONNX `BatchNormalization` to WebNN equivalents
- [xx] Build WebNN Graph asynchronously (`mlGraphBuilder.build()`)
- [xx] Compute WebNN Graph asynchronously (`context.compute()`)
- [xx] Handle dynamic shapes via explicit graph recompilation or WebNN dynamic features
- [xx] Fallback to WebGPU/WASM for ONNX operations not supported by the current WebNN spec
- [xx] Map WebNN output buffers to standard `ort.Tensor` abstractions
- [xx] Share memory buffers between WebGPU and WebNN explicitly if APIs permit
- [xx] Validate WebNN operand constraints natively in JS before submission
- [xx] Profile WebNN execution latency natively
- [xx] Handle `NotSupportedError` cleanly and fallback to standard execution paths
- [xx] Detect Chrome specific WebNN flags/status natively
- [xx] Map ONNX `Gemm` explicitly to WebNN `gemm`
- [xx] Track WebNN Specification updates (W3C) for newly added operator bindings

### 8. Environment, Workers & Node.js Compatibility (30+ items)

- [xx] Execute cleanly in Main Browser Thread
- [xx] Execute cleanly in dedicated WebWorker
- [xx] Execute cleanly in SharedWorker
- [xx] Execute cleanly in ServiceWorker (for background inference)
- [xx] Execute cleanly in Chrome Extension / Firefox Add-on environments (Manifest V3 compatible)
- [xx] Execute cleanly in Node.js `worker_threads`
- [xx] Detect Node.js environment automatically (`process.versions.node`)
- [xx] Fallback WebGPU to `@webgpu/types` or native Node.js WebGPU bindings if available
- [xx] Fallback `fetch` to Node.js `https.get` / `node-fetch`
- [xx] Read local files seamlessly in Node.js via `fs.readFile` / `fs.promises`
- [xx] Map Node.js `Buffer` natively to `Uint8Array` / `ort.Tensor`
- [xx] Support execution in `Deno` environments
- [xx] Support execution in `Bun` environments
- [xx] Execute cleanly in Electron apps (Main Process)
- [xx] Execute cleanly in Electron apps (Renderer Process)
- [xx] Execute cleanly in Native WebViews WebViews
- [xx] Export as standard ECMAScript Module (`import { InferenceSession } from 'onnx9000-web'`)
- [xx] Export as CommonJS module (`require('onnx9000-web')`)
- [xx] Export as UMD bundle for direct `<script>` tags
- [xx] Provide TypeScript Source Maps (`.map`) for clean browser debugging
- [xx] Prevent polluting the global `window` object implicitly
- [xx] Manage logging globally via `console.log`, `console.warn`, `console.error` securely
- [xx] Prevent overriding native JS prototype methods (Array, Object)
- [xx] Catch unhandled promise rejections internally
- [xx] Allow overriding `setTimeout` / `setInterval` for deterministic testing
- [xx] Provide an API to explicitly terminate/dispose all active WebWorkers
- [xx] Profile main-thread blocking time natively
- [xx] Ensure the entire JS payload is < 500KB (minified and gzipped)
- [xx] Strip debug logging code aggressively during production builds
- [xx] Ensure CSP (Content Security Policy) compliance: no `unsafe-eval`, no `unsafe-inline` needed

### 9. Testing, Validation & Edge Cases (30+ items)

- [xx] Unit Test: Load and execute a standard CNN (MobileNet) via WebGPU natively
- [xx] Unit Test: Load and execute a standard NLP model (BERT) via WebGPU natively
- [xx] Unit Test: Validate WebGPU outputs against standard Python ONNX Runtime (atol=1e-3)
- [xx] Unit Test: Validate WASM outputs against standard Python ONNX Runtime (atol=1e-5)
- [xx] Unit Test: Dynamic Sequence Generation (LLM auto-regressive loop) entirely in JS
- [xx] Unit Test: Progressive loading of a 1GB model without crashing the JS Heap
- [xx] Test numerical stability of WebGPU `Softmax` on massive values
- [xx] Test numerical stability of WebGPU `Exp` and `Log`
- [xx] Verify WebGPU handling of empty tensors (0 dimensions) natively
- [xx] Verify WebGPU handling of scalar tensors (0D) natively
- [xx] Handle explicit NaNs and Infs securely in WebGPU without crashing the context
- [xx] Test WebWorker serialization of `Tensor` objects (`postMessage`)
- [xx] Test WebWorker serialization of `Tensor` using Transferable Objects (zero-copy)
- [xx] Verify Node.js CI/CD execution pipeline (using headless WebGL/WebGPU simulators)
- [xx] Test Safari (WebKit) compatibility natively
- [xx] Test Firefox (Gecko) compatibility natively
- [xx] Test Chrome (Blink) compatibility natively
- [xx] Test iOS Safari WebGL/WebGPU limits
- [xx] Test Android Chrome WebGL/WebGPU limits
- [xx] Handle WebGL context loss dynamically during execution
- [xx] Handle WebGPU device loss dynamically during execution
- [xx] Ensure strict adherence to JS `eslint` configurations natively
- [xx] Run `prettier` auto-formatting on all TS files
- [xx] Expose an interactive test-harness UI (HTML page) to run tests manually in any browser
- [xx] Validate `Uint8` quantized evaluation natively in WebGPU
- [xx] Validate `Float16` typed arrays natively in JS
- [xx] Catch explicitly invalid ONNX Opsets and throw readable JS errors
- [xx] Catch missing operations dynamically and list the failing Node name explicitly
- [xx] Profile Garbage Collection pauses specifically during heavy inference loops
- [xx] Validate deterministic execution behavior across multiple consecutive `.run()` calls
