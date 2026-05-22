# ONNX26: Apache TVM IREE (WASM-Native MLIR Compiler)

## Original Project Description

IREE (Intermediate Representation Execution Environment) is an end-to-end MLIR-based compiler and runtime built by Google and the open-source community. It was designed to replace heavy inference frameworks with a tiny, bare-metal capable runtime. It takes ML models (like ONNX or TensorFlow), lowers them through multiple dialects of MLIR (Linalg, Flow, HAL, VM), and compiles them into standalone CPU/GPU executables or FlatBuffer modules. It represents the pinnacle of "Ahead-of-Time" (AOT) compilation for Machine Learning, aggressively optimizing memory planning, kernel scheduling, and execution overhead down to kilobytes instead of megabytes.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of relying on LLVM and Google's massive C++ MLIR toolchain to compile models offline, `onnx9000.iree` introduces a lightweight, web-native MLIR equivalent directly within the monolith.

- **Web-MLIR Dialects:** It implements its own subset of MLIR-like dialects (e.g., `web.linalg`, `web.hal`) written entirely in TypeScript and Python, bypassing LLVM.
- **Browser-Based Lowering:** The entire lowering pipeline—from ONNX to Linalg, to Loops, to WebAssembly Text (WAT)/WGSL—can run directly in the browser.
- **Zero-Dependency Bytecode VM:** Emits a tiny, specialized bytecode format (`.wvm` - Web Virtual Machine) that a minuscule (<50kb) WASM interpreter can execute, bypassing the need for even the standard `onnx9000` execution engine on extreme edge devices.
- **AOT WebGPU Pre-compilation:** IREE-style aggressive AOT planning pre-calculates every WebGPU buffer offset and shader dispatch order during the build phase, emitting a single, flat JavaScript execution queue with zero runtime overhead.

---

## Exhaustive Implementation Checklist

### Phase 1: High-Level Dialect (`web.mhlo` / `web.tensor`)

- [x] 1. Define base `Operation` class (op code, operands, results, attributes).
- [x] 2. Define `Region` and `Block` structures for MLIR-style nested control flow.
- [x] 3. Implement `web.tensor.extract` operation.
- [x] 4. Implement `web.tensor.insert` operation.
- [x] 5. Implement `web.tensor.splat` operation.
- [x] 6. Implement `web.tensor.pad` operation.
- [x] 7. Implement `web.mhlo.add` (broadcastable addition).
- [x] 8. Implement `web.mhlo.subtract`.
- [x] 9. Implement `web.mhlo.multiply`.
- [x] 10. Implement `web.mhlo.divide`.
- [x] 11. Implement `web.mhlo.maximum`.
- [x] 12. Implement `web.mhlo.minimum`.
- [x] 13. Implement `web.mhlo.exponential`.
- [x] 14. Implement `web.mhlo.log`.
- [x] 15. Implement `web.mhlo.cosine`.
- [x] 16. Implement `web.mhlo.sine`.
- [x] 17. Implement `web.mhlo.dot` (matrix multiplication).
- [x] 18. Implement `web.mhlo.convolution` (general N-D convolution).
- [x] 19. Implement `web.mhlo.reduce` (general reduction with reducer block).
- [x] 20. Implement `web.mhlo.reduce_window` (pooling).
- [x] 21. Implement `web.mhlo.select` (ternary/where).
- [x] 22. Implement `web.mhlo.broadcast_in_dim`.
- [x] 23. Implement `web.mhlo.reshape`.
- [x] 24. Implement `web.mhlo.transpose`.
- [x] 25. Implement `web.mhlo.concatenate`.
- [x] 26. Implement `web.mhlo.slice`.
- [x] 27. Implement `web.mhlo.dynamic_slice`.
- [x] 28. Implement `web.mhlo.gather`.
- [x] 29. Implement `web.mhlo.scatter`.
- [x] 30. Create the ONNX-to-MHLO lowering pass (mapping ONNX graphs to this dialect).

### Phase 2: Structural Dialect (`web.linalg`)

- [x] 31. Define `AffineMap` class (for iteration space mapping).
- [x] 32. Define `web.linalg.generic` operation.
- [x] 33. Support `iterator_types` attribute (parallel, reduction).
- [x] 34. Support `indexing_maps` attribute (mapping loops to tensor dimensions).
- [x] 35. Implement `web.linalg.matmul` named op.
- [x] 36. Implement `web.linalg.batch_matmul` named op.
- [x] 37. Implement `web.linalg.conv_2d_nhwc_hwcf` named op.
- [x] 38. Implement `web.linalg.pooling_nhwc_max` named op.
- [x] 39. Implement `web.linalg.fill` named op.
- [x] 40. Implement `web.linalg.yield` (terminator for linalg blocks).
- [x] 41. Create the MHLO-to-Linalg lowering pass.
- [x] 42. Translate `web.mhlo.add` to `web.linalg.generic` (parallel iterator).
- [x] 43. Translate `web.mhlo.reduce` to `web.linalg.generic` (reduction iterator).
- [x] 44. Implement pass: Linalg fusion on tensors (fusing elementwise ops into matmul/conv producers).
- [x] 45. Implement pass: Tiling (breaking large `linalg.generic` ops into smaller tile loops).
- [x] 46. Support custom tile sizes for WebGPU (e.g., 16x16, 64x64).
- [x] 47. Implement pass: Bufferization (lowering from value-semantics/tensors to memory-semantics/buffers).
- [x] 48. Implement `web.memref.alloc` operation.
- [x] 49. Implement `web.memref.dealloc` operation.
- [x] 50. Implement `web.memref.load` and `web.memref.store`.

### Phase 3: Hardware Abstraction Layer Dialect (`web.hal`)

- [x] 51. Define `web.hal.device` abstraction.
- [x] 52. Define `web.hal.buffer` abstraction.
- [x] 53. Define `web.hal.buffer_view` (buffer + shape + element type).
- [x] 54. Define `web.hal.command_buffer`.
- [x] 55. Define `web.hal.executable` (representing a compiled shader/WASM module).
- [x] 56. Implement `web.hal.command_buffer.dispatch` operation.
- [x] 57. Implement `web.hal.command_buffer.copy_buffer` operation.
- [x] 58. Implement `web.hal.command_buffer.fill_buffer` operation.
- [x] 59. Implement `web.hal.buffer.subspan` (aliasing memory).
- [x] 60. Create the Linalg-to-HAL lowering pass.
- [x] 61. Lower tiled `linalg.generic` into distinct `hal.executable` blocks.
- [x] 62. Generate 3D dispatch grids (Workgroups) for WebGPU targets.
- [x] 63. Extract kernel functions from the main control flow graph.
- [x] 64. Implement pass: Static memory planning (converting `alloc`/`dealloc` into static arena offsets).
- [x] 65. Emit `hal.buffer.subspan` based on the static arena layout.
- [x] 66. Implement pass: Command Buffer batching (grouping dispatches to minimize host overhead).
- [x] 67. Generate host-side synchronization points only when crossing hardware boundaries.
- [x] 68. Handle dynamic shapes using HAL symbolic variables (binding shapes at execution time).
- [x] 69. Support multiple target backends within the same HAL graph (e.g., WASM fallback).
- [x] 70. Implement HAL textual printer for debugging dispatch logic.

### Phase 4: Control Flow & VM Dialect (`web.vm`)

- [x] 71. Define `web.vm.module`.
- [x] 72. Define `web.vm.func`.
- [x] 73. Define `web.vm.call`.
- [x] 74. Implement `web.vm.branch` (unconditional jump).
- [x] 75. Implement `web.vm.cond_branch` (conditional jump).
- [x] 76. Implement `web.vm.cmp` (integer/float comparison).
- [x] 77. Implement basic integer arithmetic (`vm.add.i32`, `vm.mul.i32`).
- [x] 78. Implement `web.vm.return`.
- [x] 79. Create the HAL-to-VM lowering pass.
- [x] 80. Convert HAL command buffer recording into a sequence of VM API calls.
- [x] 81. Translate MLIR `Block` structures into flat lists of basic blocks with explicit jumps.
- [x] 82. Implement pass: VM block layout optimization.
- [x] 83. Implement pass: VM register allocation (mapping SSA values to VM registers).
- [x] 84. Lower dynamic shape calculations entirely into VM integer math.
- [x] 85. Expose `vm.import` declarations for bridging to host JS functions (e.g., `console.log`).
- [x] 86. Implement a FlatBuffer-like schema for serializing the VM module.
- [x] 87. Build the `wvm` (Web Virtual Machine) Bytecode Emitter.
- [x] 88. Map VM instructions to custom binary opcodes.
- [x] 89. Encode literal constants (weights) directly into the `wvm` binary payload.
- [x] 90. Build a CLI disassembler to convert `.wvm` binary back to text.

### Phase 5: Executable Translation (WASM CPU)

- [x] 91. Create the `hal.executable` to WASM translator.
- [x] 92. Define the `web.scf` (Structured Control Flow) dialect for nested loops.
- [x] 93. Lower `linalg.generic` inside the executable to `scf.for` loops.
- [x] 94. Lower `scf.for` loops to flat VM jumps or directly to WASM `loop`/`br`.
- [x] 95. Implement loop unrolling pass based on target heuristics.
- [x] 96. Implement vectorization pass (identifying contiguous memory accesses).
- [x] 97. Emit `v128` SIMD intrinsics for vectorized inner loops.
- [x] 98. Emit base WASM scalar operations for non-vectorizable loops.
- [x] 99. Generate an independent WASM module (the "kernel library") containing all executables.
- [x] 100. Provide a stable ABI for the VM to call these WASM functions.
- [x] 101. Support shared linear memory between the VM and the WASM execution module.
- [x] 102. Compile the mathematical kernels to WAT (WebAssembly Text) string representation.
- [x] 103. Parse WAT into final WASM binary within the JS/TS compiler.
- [x] 104. Implement WASM threading via SharedArrayBuffer (generating thread-pool dispatchers).
- [x] 105. Optimize standard convolutions into optimized WASM Im2Col + MatMul sequences natively.

### Phase 6: Executable Translation (WGSL WebGPU)

- [x] 106. Create the `hal.executable` to WGSL translator.
- [x] 107. Map `hal.buffer` inputs to `var<storage, read>`.
- [x] 108. Map `hal.buffer` outputs to `var<storage, read_write>`.
- [x] 109. Map `hal.executable` dispatch shapes to `builtin(global_invocation_id)`.
- [x] 110. Translate inner loop bodies (from `linalg.generic`) to WGSL AST nodes.
- [x] 111. Resolve indexing maps to calculate flat 1D buffer offsets in WGSL.
- [x] 112. Implement memory coalescing optimizations explicitly in the WGSL generator.
- [x] 113. Emit Workgroup (Shared) Memory declarations for tiled MatMul kernels.
- [x] 114. Generate standard WebGPU pipelines directly from the compiled shader strings.
- [x] 115. Implement a standalone JS runner that executes the generated WGSL shaders precisely following the VM's command buffer graph.
- [x] 116. Support mapping `hal` synchronization points to `device.queue.submit()`.
- [x] 117. Implement kernel fusion at the WGSL level (e.g., generating a single shader for MatMul+Relu).
- [x] 118. Handle FP16 WGSL extensions automatically if the HAL executable specifies fp16 math.
- [x] 119. Generate custom shader variations for different workgroup sizes during compilation.
- [x] 120. Strip all WGSL whitespace and minify variables for smaller payload delivery.

### Phase 7: The Minimal IREE-Style Runtime (VM Interpreter)

- [x] 121. Build a pure JavaScript `wvm` interpreter (< 100kb).
- [x] 122. Build a pure WASM `wvm` interpreter (< 50kb compiled).
- [x] 123. Define the runtime `Module` state (holding global variables and memory).
- [x] 124. Define the runtime `Context` (managing execution state and call stack).
- [x] 125. Implement the bytecode dispatch loop (`switch(opcode)`).
- [x] 126. Implement dynamic module loading (`vm.import` resolution).
- [x] 127. Bind the `web.hal` VM instructions to actual WebGPU API calls (`createBuffer`, `createComputePipeline`).
- [x] 128. Bind the `web.hal` VM instructions to standard WASM calls.
- [x] 129. Implement an asynchronous execution mode for the VM (yielding to the browser event loop).
- [x] 130. Implement a synchronous execution mode (for Web Workers).
- [x] 131. Support passing raw ArrayBuffers from JS directly into the VM state.
- [x] 132. Support retrieving output ArrayBuffers from the VM state.
- [x] 133. Provide strict validation of the `.wvm` binary format during instantiation.
- [x] 134. Handle WebGPU context loss inside the VM gracefully, throwing a catchable VM error.
- [x] 135. Integrate a tiny Memory Allocator inside the runtime for resolving dynamic shapes if static planning failed.

### Phase 8: Static Standalone Web Generation (The Ultimate Export)

- [x] 136. Create an exporter that completely bypasses the `.wvm` interpreter.
- [x] 137. Perform full loop-unrolling of the VM control flow graph.
- [x] 138. Emit a highly customized, standalone `index.js` file.
- [x] 139. Embed all compiled WGSL shaders directly as string literals in the JS.
- [x] 140. Embed the static buffer arena allocations natively in the JS logic.
- [x] 141. Emit explicit, hardcoded `device.queue.submit` sequences without any loops or branches (if the model is static).
- [x] 142. Produce an "Executable" size of roughly 5-10KB (plus weights), completely removing `onnx9000` from the loop.
- [x] 143. Support bundling weights securely via Fetch/CacheAPI directly within the generated `index.js`.
- [x] 144. Ensure the standalone script is strictly ES6 Module compliant.
- [x] 145. Create an HTML template combining the standalone script with an `<input type="file">` for instant local testing.

### Phase 9: Model specific MLIR optimization passes

- [x] 146. Implement a pass to detect and optimize `Attention` patterns specifically at the `linalg` level.
- [x] 147. Map specific `linalg` patterns directly to emerging WebNN API calls (bypassing WGSL/WASM generation entirely).
- [x] 148. Implement specific padding removal passes (lowering padded Convolutions to valid Convolutions with manual boundary checks).
- [x] 149. Identify and fuse sequences of elementwise operations across multiple basic blocks.
- [x] 150. Optimize dynamic slice bounds by hoisting shape calculations outside of execution loops.
- [x] 151. Implement a peephole optimizer for the VM dialect (e.g., `vm.add x, 0 -> x`).
- [x] 152. Perform global value numbering (GVN) for common subexpression elimination in the Linalg dialect.
- [x] 153. Implement dead code elimination specifically for unused MLIR attributes and regions.
- [x] 154. Support `linalg` vectorization specific to Apple Neural Engine constraints (if targeting WebNN fallback).
- [x] 155. Provide dynamic dimension size propagation down to the lowest HAL layer.

### Phase 10: Compiler CLI & Tooling (`onnx9000-iree`)

- [x] 156. Implement `onnx9000 iree compile <model.onnx>` command.
- [x] 157. Support `--target-backend=wgsl` flag.
- [x] 158. Support `--target-backend=wasm` flag.
- [x] 159. Support `--target-backend=webnn` flag.
- [x] 160. Support `--target-backend=standalone-js` flag.
- [x] 161. Support `--dump-mlir` flag (saving all intermediate dialect steps to `.mlir` text files).
- [x] 162. Support `--optimize-level=O3` parameter mapping to specific IREE passes.
- [x] 163. Provide a graphical trace visualizer for the generated HAL command buffers.
- [x] 164. Generate an interactive HTML report mapping WGSL shaders back to original ONNX nodes.
- [x] 165. Provide an API to run the MLIR compiler entirely in the browser (via a heavy Web Worker).
- [x] 166. Establish a testing suite that compares native ORT output vs compiled `wvm` output.
- [x] 167. Enable debug logging of VM register states step-by-step.
- [x] 168. Package the compiler and runtime as separate NPM modules (`@onnx9000/iree-compiler`, `@onnx9000/iree-runtime`).
- [x] 169. Write tutorial: "Building a Zero-Dependency 10KB Image Classifier".
- [x] 170. Write tutorial: "Understanding the `onnx9000` MLIR Lowering Pipeline".

### Phase 11: End-to-End Validation (Vision)

- [x] 171. Validate compilation and standalone execution of **MNIST (CNN)**.
- [x] 172. Validate compilation and standalone execution of **MobileNetV2**.
- [x] 173. Validate compilation and standalone execution of **ResNet50**.
- [x] 174. Validate compilation and standalone execution of **SqueezeNet**.
- [x] 175. Validate compilation and standalone execution of **YOLOv8** (Object Detection).
- [x] 176. Ensure post-processing bounding box logic can be baked directly into the `.wvm` bytecode.
- [x] 177. Validate compilation of **ViT** (Vision Transformer).
- [x] 178. Validate memory planning on high-resolution image inputs (e.g., 1024x1024).
- [x] 179. Benchmark standalone JS initialization speed vs standard `onnxruntime-web`.
- [x] 180. Verify precise pixel matching across all vision model outputs.

### Phase 12: End-to-End Validation (NLP & LLMs)

- [x] 181. Validate compilation of **BERT** into standalone WGSL/WVM.
- [x] 182. Validate compilation of **DistilBERT**.
- [x] 183. Validate compilation of **GPT-2**.
- [x] 184. Validate compilation of a miniature **LLaMA** block (e.g., TinyLlama).
- [x] 185. Handle autoregressive control flow (while-loops) explicitly via `web.vm.branch`.
- [x] 186. Pre-calculate KV cache memory layouts during the HAL bufferization pass.
- [x] 187. Bake BPE Tokenization dictionaries directly into the VM module as static read-only buffers.
- [x] 188. Ensure dynamic sequence lengths don't trigger recompilation in the WGSL runners.
- [x] 189. Validate performance of compiled text generation vs ONNX Runtime GenAI.
- [x] 190. Handle extremely large tensor initialization payloads securely via separate weight chunks.

### Phase 13: End-to-End Validation (Audio)

- [x] 191. Validate compilation of **Whisper** (Encoder).
- [x] 192. Validate compilation of **Whisper** (Decoder).
- [x] 193. Implement cross-attention caching across the Encoder/Decoder boundary inside the VM.
- [x] 194. Validate compilation of **Wav2Vec2**.
- [x] 195. Verify Mel-Spectrogram feature extraction can be compiled directly into the VM graph.
- [x] 196. Support compiling streaming audio models utilizing ring buffers inside HAL.
- [x] 197. Validate numerical stability of FFT operations compiled to WGSL.
- [x] 198. Establish benchmark comparisons for real-time factor (RTF) in audio decoding.
- [x] 199. Integrate output of compiled audio graphs directly to Web Audio API Worklets.
- [x] 200. Debug and trace stateful RNN/LSTM cells executing over long audio sequences.

### Phase 14: Dynamic Quantization Lowering

- [x] 201. Support ONNX `DynamicQuantizeLinear` directly in the `web.mhlo` dialect.
- [x] 202. Lower dynamic quantization steps into explicit Linalg min/max/scale/cast operations.
- [x] 203. Optimize `linalg.generic` loops to perform quantization and matmul in a single pass (fusing scales).
- [x] 204. Validate 8-bit dynamic quantization executing entirely inside WebGPU WGSL.
- [x] 205. Support W4A16 (4-bit weight packing) lowering explicitly in the MLIR pipeline.
- [x] 206. Implement the shift/mask unpacking logic directly in the target WGSL executables.
- [x] 207. Handle sub-byte buffer indexing securely within the VM HAL dispatcher.
- [x] 208. Benchmark W4A16 WGSL executables against standard FP16 equivalents.
- [x] 209. Provide detailed size tracking showing binary size before and after lowering quantization.
- [x] 210. Map explicit mixed-precision topologies (some layers INT8, some FP16) seamlessly.

### Phase 15: Target-Specific Autotuning (MetaSchedule Integration)

- [x] 211. Integrate the `onnx9000.tvm` auto-tuner (from ONNX18) into the IREE lowering pipeline.
- [x] 212. Allow the tuner to mutate the `linalg.generic` tiling sizes iteratively.
- [x] 213. Profile generated WGSL shaders rapidly using `device.createQuerySet`.
- [x] 214. Record optimal tile sizes and memory access patterns into an `iree_config.json`.
- [x] 215. Feed the configuration file back into the `Linalg-to-HAL` pass to lock in the optimal shapes.
- [x] 216. Autotune WebGPU `workgroup_size` mapping specifically for Apple M-Series GPUs.
- [x] 217. Autotune WebGPU `workgroup_size` mapping specifically for Nvidia discrete GPUs.
- [x] 218. Display a live tuning dashboard during compilation if `--autotune` is provided.
- [x] 219. Provide heuristic fallbacks if autotuning is skipped.
- [x] 220. Support tuning WASM SIMD unroll factors for optimal V8/SpiderMonkey compilation.

### Phase 16: Interoperability & Import/Export

- [x] 221. Implement standard `.mlir` text file parser.
- [x] 222. Allow importing raw MLIR files generated by Google IREE and executing them on the `wvm`.
- [x] 223. Implement standard `.mlir` text file emitter.
- [x] 224. Support importing TensorFlow SavedModels via bridging through XLA to MHLO.
- [x] 225. Support importing PyTorch models via `torch-mlir`.
- [x] 226. Ensure the `web.mhlo` dialect accurately reflects standard `stablehlo` specification to maximize compatibility.
- [x] 227. Provide a conversion script between `stablehlo` and `web.mhlo` handling any unsupported discrepancies.
- [x] 228. Export the standalone JS bundles into an NPM-publishable format automatically.
- [x] 229. Expose source maps connecting `.wvm` bytecode instructions back to specific ONNX node IDs.
- [x] 230. Integrate cleanly with `onnx9000.transformers` auto-classes to act as a hidden backend provider.

### Phase 17: Security, Sandbox, and Stability

- [x] 231. Ensure the generated `.wvm` interpreter strictly confines memory access to its initialized ArrayBuffer.
- [x] 232. Prevent out-of-bounds reads/writes in the VM via explicit bound checks during development mode.
- [x] 233. In production mode, utilize WASM memory bounds implicitly to ensure zero-overhead security.
- [x] 234. Validate generated WGSL shaders against the WebGPU specification to prevent driver crashes.
- [x] 235. Sanitize model inputs passing through the `vm.import` boundaries.
- [x] 236. Prevent infinite loops by injecting watchdog counters in `web.vm.branch` instructions.
- [x] 237. Ensure execution determinism across multiple runs (assuming identical inputs and seeds).
- [x] 238. Validate VM robustness against corrupted or maliciously crafted `.wvm` bytecode files.
- [x] 239. Handle WebGL context loss as a graceful fallback if WebGPU crashes due to OS issues.
- [x] 240. Implement comprehensive telemetry reporting specific pass times during compilation.

### Phase 18: Ecosystem Demos & Examples

- [x] 241. Provide "Tiny LLM in 20KB JS" example repository.
- [x] 242. Provide "Webcam Object Detection without External Dependencies" example.
- [x] 243. Integrate the standalone JS output directly into an HTML Canvas element for instant visual feedback.
- [x] 244. Provide a Deno/Bun example executing `.wvm` files purely via command line.
- [x] 245. Create an interactive "Compiler Explorer" (like godbolt.org) for ONNX -> MLIR -> WGSL.
- [x] 246. Provide examples of embedding a `.wvm` binary directly inside a Chrome Extension service worker.
- [x] 247. Demonstrate cross-platform parity: running the exact same `.wvm` file on Node.js and Browser.
- [x] 248. Write integration tests mapping `wvm` APIs to standard REST API server logic.
- [x] 249. Publish a gallery of pre-compiled `.wvm` binaries for popular foundational models.
- [x] 250. Create detailed documentation explaining the transition from standard execution to AOT MLIR execution.

### Phase 19: Advanced Graph Diagnostics & Tracing

- [x] 251. Implement a Chrome Tracing (`.json`) generator for the compiler passes.
- [x] 252. Output detailed memory lifecycle graphs (showing peak memory vs active buffers over time).
- [x] 253. Provide a visualization of the HAL command buffers to identify synchronization bottlenecks.
- [x] 254. Track total WGSL shader string size and optimize minification strategies.
- [x] 255. Support injecting profiling counters into the generated WGSL to measure exact GPU ticks per kernel.
- [x] 256. Correlate GPU profiling data back to the original ONNX nodes.
- [x] 257. Provide a "diff" tool to compare the MLIR representation before and after a specific optimization pass.
- [x] 258. Expose all intermediate WGSL shaders to a debug directory during compilation.
- [x] 259. Implement a fallback execution mode that runs the exact graph structure on CPU for numerical debugging.
- [x] 260. Capture WebGPU validation errors and map them precisely to the faulty MLIR lowerings.

### Phase 20: Full Parity & Future Hardening

- [x] 261. Achieve compilation success on the standard ONNX `model_zoo` (top 50 models).
- [x] 262. Verify standard compliance with MLIR upstream dialects where applicable.
- [x] 263. Implement robust error recovery during the parsing of complex `.mlir` text files.
- [x] 264. Support explicit multi-device execution within a single `.wvm` module (e.g., Device 0 = WebGPU, Device 1 = CPU).
- [x] 265. Ensure correct topological sorting handles disconnected graph components appropriately.
- [x] 266. Provide deterministic pseudo-random number generation primitives within the VM dialect.
- [x] 267. Map specialized String handling ops from ONNX (if applicable) into the VM.
- [x] 268. Handle extreme edge-case tensor ranks (e.g., 6D or 7D tensors) gracefully.
- [x] 269. Compile the `wvm` interpreter specifically for Cloudflare Workers (minimizing startup latency).
- [x] 270. Create WebRTC broadcast utilities for distributing `.wvm` tasks across a peer-to-peer browser network.
- [x] 271. Support dynamic dimension patching without recompiling the `.wvm` module.
- [x] 272. Implement graph-level deduplication (merging identical subgraphs across different model partitions).
- [x] 273. Support loading external weights via memory mapping (mmap) equivalents in Node/Deno.
- [x] 274. Verify exact parity of INT8 outputs against ORT QLinearOps.
- [x] 275. Expand CLI flag support (`--emit-mlir`, `--emit-wgsl`, `--emit-wvm`, `--run`).
- [x] 276. Ensure the compiler respects `NODE_ENV=production` for minification vs debugging behaviors.
- [x] 277. Implement a strict "no eval" policy in the interpreter to satisfy rigorous Content Security Policies.
- [x] 278. Establish standard benchmarking CI jobs that block PRs if `.wvm` execution time regresses.
- [x] 279. Prepare the architecture for future WebNN backend support via the HAL dialect.
- [x] 280. Handle `uint64` data types explicitly where WebGPU specifications restrict them.
- [x] 281. Build a custom VSCode extension to provide syntax highlighting for the `web.mlir` dialect.
- [x] 282. Ensure the VM correctly manages Promise resolutions when integrating asynchronous WebGPU readbacks.
- [x] 283. Support `hal.buffer.view` conversions that change data types safely (e.g., bitcasting).
- [x] 284. Handle extremely deeply nested MLIR regions without throwing JS Maximum Call Stack Exceeded errors.
- [x] 285. Provide explicit documentation on the memory safety guarantees of the `.wvm` architecture.
- [x] 286. Optimize the JSON serialization of MLIR ASTs for passing between Web Workers during compilation.
- [x] 287. Map ONNX `Loop` natively using `web.scf.while` control flow.
- [x] 288. Ensure `onnx9000-iree` can compile itself natively if run through a JS-to-WASM transpiler (meta-compilation).
- [x] 289. Support exporting the standalone scripts in UMD format for legacy browser integration.
- [x] 290. Provide a detailed roadmap for tracking upstream Google IREE feature implementations.
- [x] 291. Compile MobileBERT entirely into the standalone JS format.
- [x] 292. Compile TinyBERT entirely into the standalone JS format.
- [x] 293. Verify the generated Standalone JS executes offline with no network requests.
- [x] 294. Map explicit `isInf` and `isNaN` logic natively into WGSL primitives.
- [x] 295. Execute deep memory lifecycle analysis to prove there are zero memory leaks during a `wvm` generation loop.
- [x] 296. Maintain exact numeric parity with reference PyTorch models.
- [x] 297. Support `--disable-webgpu-fp16` for legacy devices.
- [x] 298. Validate precise execution under severe memory constraints (e.g., simulated 512MB RAM limits).
- [x] 299. Write comprehensive API documentation for the `wvm` interpreter.
- [x] 300. Release v1.0 feature complete certification for `onnx9000.iree`.
