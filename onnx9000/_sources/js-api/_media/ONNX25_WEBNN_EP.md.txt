# ONNX25: WebNN API (Native Browser NPU Execution)

## Original Project Description

The ONNX Runtime WebNN Execution Provider (EP) allows web applications to run ONNX models with hardware acceleration utilizing the emerging W3C Web Neural Network API (WebNN). WebNN provides standard low-level browser APIs to access dedicated machine learning accelerators like Neural Processing Units (NPUs), Digital Signal Processors (DSPs), and specialized GPU ML cores (like Apple's Neural Engine or Intel's VPU/NPU). In the standard ORT architecture, the WebNN EP acts as a bridge: compiling C++ ONNX nodes into JavaScript `MLGraphBuilder` calls via WebAssembly interop.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000` eliminates the C++ WebAssembly middleware entirely for graph construction. Since `onnx9000` handles the ONNX graph directly in TypeScript/JavaScript, the mapping to WebNN is direct, native, and synchronous.

- **Zero JS-WASM Boundary Crossing for Compilation:** The `onnx9000` graph compiler traverses the IR in memory and calls `MLGraphBuilder` natively, compiling the NPU graph orders of magnitude faster than a C++ runtime shuttling strings and pointers across the WASM bridge.
- **Granular Sub-Graph Partitioning:** If the host's NPU/WebNN implementation doesn't support a specific ONNX operator (e.g., a custom transformer node), `onnx9000` dynamically partitions the graph. The supported sub-graphs run natively on the NPU via WebNN, while unsupported nodes seamlessly fall back to `onnx9000`'s highly optimized WebGPU or WASM SIMD backends sharing the same memory context.
- **WebNN Polyfill Integration:** Automatically integrates with the WebNN Polyfill for rapid testing on browsers that have not yet fully shipped the W3C spec.
- **First-Class FP16 & INT8:** WebNN is primarily designed for low-power edge NPUs; `onnx9000` strictly maps its Web-Native W4A16 and INT8 quantizations directly to WebNN primitives to maximize NPU throughput.

---

## Exhaustive Implementation Checklist

### Phase 1: Context Initialization & Feature Detection

- [x] [x] 1. Implement `navigator.ml` presence detection.
- [x] [x] 2. Implement graceful fallback if WebNN API is missing.
- [x] [x] 3. Request `MLContext` via `navigator.ml.createContext()`.
- [x] [x] 4. Support `deviceType: 'npu'` preference.
- [x] [x] 5. Support `deviceType: 'gpu'` preference.
- [x] [x] 6. Support `deviceType: 'cpu'` preference.
- [x] [x] 7. Support `powerPreference: 'default'`.
- [x] [x] 8. Support `powerPreference: 'high-performance'`.
- [x] [x] 9. Support `powerPreference: 'low-power'`.
- [x] [x] 10. Implement caching of the `MLContext` singleton.
- [x] [x] 11. Detect supported data types (`float32`, `float16`, `int32`, `int8`, `uint8`).
- [x] [x] 12. Implement capability queries to check if specific ops are supported by the host context.
- [x] [x] 13. Provide a diagnostic CLI command: `onnx9000 info webnn` to list host NPU capabilities.
- [x] [x] 14. Handle context loss/restore events dynamically.
- [x] [x] 15. Support initializing `MLGraphBuilder` strictly bound to the active context.

### Phase 2: Graph Builder (MLGraphBuilder) Core Orchestration

- [x] [x] 16. Initialize the `MLGraphBuilder` instance.
- [x] [x] 17. Define an internal map of ONNX Node IDs to `MLOperand` objects.
- [x] [x] 18. Implement translation of ONNX Graph Inputs to `builder.input(name, type)`.
- [x] [x] 19. Implement translation of ONNX Initializers to `builder.constant(data)`.
- [x] [x] 20. Resolve ONNX dimensions (Array of Numbers) to WebNN dimensions.
- [x] [x] 21. Map `onnx9000` Float32 tensors to WebNN `float32` constants.
- [x] [x] 22. Map `onnx9000` Float16 tensors to WebNN `float16` constants.
- [x] [x] 23. Map `onnx9000` Int32 tensors to WebNN `int32` constants.
- [x] [x] 24. Map `onnx9000` Int8 tensors to WebNN `int8` constants.
- [x] [x] 25. Map `onnx9000` UInt8 tensors to WebNN `uint8` constants.
- [x] [x] 26. Handle dynamic axes in inputs (specifying `-1` or large bounds if required by specific WebNN drafts).
- [x] [x] 27. Track intermediate `MLOperand` instances during the topological traversal.
- [x] [x] 28. Support releasing intermediate `MLOperand` references to aid garbage collection.
- [x] [x] 29. Map ONNX Graph Outputs to final `MLOperand` evaluations.
- [x] [x] 30. Handle cases where an initializer is passed directly as a graph output.

### Phase 3: Unary & Binary Arithmetic Operations

- [x] [x] 31. Map ONNX `Add` to WebNN `builder.add()`.
- [x] [x] 32. Map ONNX `Sub` to WebNN `builder.sub()`.
- [x] [x] 33. Map ONNX `Mul` to WebNN `builder.mul()`.
- [x] [x] 34. Map ONNX `Div` to WebNN `builder.div()`.
- [x] [x] 35. Map ONNX `Max` to WebNN `builder.max()`.
- [x] [x] 36. Map ONNX `Min` to WebNN `builder.min()`.
- [x] [x] 37. Map ONNX `Pow` to WebNN `builder.pow()`.
- [x] [x] 38. Map ONNX `Abs` to WebNN `builder.abs()`.
- [x] [x] 39. Map ONNX `Ceil` to WebNN `builder.ceil()`.
- [x] [x] 40. Map ONNX `Floor` to WebNN `builder.floor()`.
- [x] [x] 41. Map ONNX `Exp` to WebNN `builder.exp()`.
- [x] [x] 42. Map ONNX `Log` to WebNN `builder.log()`.
- [x] [x] 43. Map ONNX `Cos` to WebNN `builder.cos()`.
- [x] [x] 44. Map ONNX `Sin` to WebNN `builder.sin()`.
- [x] [x] 45. Map ONNX `Tan` to WebNN `builder.tan()`.
- [x] [x] 46. Map ONNX `Acos` to WebNN `builder.acos()`.
- [x] [x] 47. Map ONNX `Asin` to WebNN `builder.asin()`.
- [x] [x] 48. Map ONNX `Atan` to WebNN `builder.atan()`.
- [x] [x] 49. Map ONNX `Sqrt` to WebNN `builder.sqrt()`.
- [x] [x] 50. Map ONNX `Erf` to WebNN `builder.erf()`.
- [x] [x] 51. Map ONNX `Sign` to WebNN `builder.sign()`.
- [x] [x] 52. Map ONNX `Neg` to WebNN `builder.neg()`.
- [x] [x] 53. Handle Numpy-style implicit broadcasting in WebNN binary ops automatically.
- [x] [x] 54. Explicitly reshape scalar initializers for WebNN if the spec requires strict rank matching.

### Phase 4: Activation Functions

- [x] [x] 55. Map ONNX `Relu` to WebNN `builder.relu()`.
- [x] [x] 56. Map ONNX `Sigmoid` to WebNN `builder.sigmoid()`.
- [x] [x] 57. Map ONNX `Tanh` to WebNN `builder.tanh()`.
- [x] [x] 58. Map ONNX `Softmax` to WebNN `builder.softmax()`.
- [x] [x] 59. Handle `Softmax` axis parameter mapping.
- [x] [x] 60. Map ONNX `LeakyRelu` to WebNN `builder.leakyRelu()`.
- [x] [x] 61. Parse `alpha` parameter for `LeakyRelu`.
- [x] [x] 62. Map ONNX `Elu` to WebNN `builder.elu()`.
- [x] [x] 63. Parse `alpha` parameter for `Elu`.
- [x] [x] 64. Map ONNX `HardSigmoid` to WebNN `builder.hardSigmoid()`.
- [x] [x] 65. Parse `alpha` and `beta` parameters for `HardSigmoid`.
- [x] [x] 66. Map ONNX `Softplus` to WebNN `builder.softplus()`.
- [x] [x] 67. Map ONNX `Softsign` to WebNN `builder.softsign()`.
- [x] [x] 68. Map ONNX `Gelu` to WebNN `builder.gelu()`.
- [x] [x] 69. Map ONNX `PRelu` to WebNN `builder.prelu()`.
- [x] [x] 70. Support `Clip` via WebNN `builder.clamp()`.
- [x] [x] 71. Handle missing min/max boundaries in `Clip` converting to infinity bounds.

### Phase 5: Matrix Multiplication & Linear Algebra

- [x] [x] 72. Map ONNX `MatMul` to WebNN `builder.matmul()`.
- [x] [x] 73. Map ONNX `Gemm` to WebNN `builder.gemm()`.
- [x] [x] 74. Parse and apply `alpha` scalar for `Gemm`.
- [x] [x] 75. Parse and apply `beta` scalar for `Gemm`.
- [x] [x] 76. Handle `transA` flag correctly in `Gemm` via WebNN options.
- [x] [x] 77. Handle `transB` flag correctly in `Gemm` via WebNN options.
- [x] [x] 78. Support explicit bias addition in `Gemm` via `c` operand.
- [x] [x] 79. If WebNN `matmul` doesn't support n-dimensional batching natively, emulate via `reshape` -> `matmul` -> `reshape` if mathematically equivalent.
- [x] [x] 80. Fallback: Emulate `Gemm` with `MatMul` + `Add` if `builder.gemm` is missing on specific hardware implementations.
- [x] [x] 81. Implement 1D matrix multiplication bounds checking according to WebNN spec.

### Phase 6: Tensor Manipulation & Routing

- [x] [x] 82. Map ONNX `Reshape` to WebNN `builder.reshape()`.
- [x] [x] 83. Extract dynamic shape tensor inputs to static shapes if WebNN requires static `reshape` arguments at build time.
- [x] [x] 84. Map ONNX `Transpose` to WebNN `builder.transpose()`.
- [x] [x] 85. Pass explicit `permutation` array to `builder.transpose()`.
- [x] [x] 86. Map ONNX `Slice` to WebNN `builder.slice()`.
- [x] [x] 87. Resolve dynamic ONNX `Slice` starts/ends/axes/steps to static WebNN options.
- [x] [x] 88. Emulate negative `starts` and `ends` indices since WebNN slice may require positive absolute bounds.
- [x] [x] 89. Map ONNX `Concat` to WebNN `builder.concat()`.
- [x] [x] 90. Handle `axis` mapping for `Concat`.
- [x] [x] 91. Map ONNX `Split` to WebNN `builder.split()`.
- [x] [x] 92. Handle equal splitting (scalar `split` argument).
- [x] [x] 93. Handle unequal splitting (array `split` argument).
- [x] [x] 94. Map ONNX `Squeeze` to WebNN `builder.reshape()` (calculating squeezed shape dynamically).
- [x] [x] 95. Map ONNX `Unsqueeze` to WebNN `builder.reshape()` (calculating unsqueezed shape dynamically).
- [x] [x] 96. Map ONNX `Expand` to WebNN `builder.expand()`.
- [x] [x] 97. Map ONNX `Gather` to WebNN `builder.gather()`.
- [x] [x] 98. Handle `axis` parameter for `Gather`.
- [x] [x] 99. Handle dynamic/variable indices in `Gather` if WebNN supports them.
- [x] [x] 100. Map ONNX `Tile` by composing `expand` or `concat` ops if direct `tile` is unavailable.
- [x] [x] 101. Map ONNX `Pad` to WebNN `builder.pad()`.
- [x] [x] 102. Handle `constant` padding mode.
- [x] [x] 103. Handle `reflect` padding mode.
- [x] [x] 104. Handle `edge` padding mode.
- [x] [x] 105. Transform ONNX pad tensor format `[x1_begin, x2_begin... x1_end, x2_end...]` to WebNN format `[ [x1_begin, x1_end], [x2_begin, x2_end]... ]`.
- [x] [x] 106. Handle `Cast` using WebNN `builder.cast()`.
- [x] [x] 107. Map ONNX `Shape` to a static CPU/WASM computation since WebNN expects static graphs.

### Phase 7: Convolution & Pooling (Vision Architectures)

- [x] [x] 108. Map ONNX `Conv` (2D) to WebNN `builder.conv2d()`.
- [x] [x] 109. Extract `strides` attribute.
- [x] [x] 110. Extract `dilations` attribute.
- [x] [x] 111. Extract `group` attribute (support Depthwise Conv2D via WebNN groups).
- [x] [x] 112. Map explicit `pads` attribute to WebNN options.
- [x] [x] 113. Implement `auto_pad="SAME_UPPER"` calculation mapping to explicit pad values.
- [x] [x] 114. Implement `auto_pad="SAME_LOWER"` calculation mapping.
- [x] [x] 115. Implement `auto_pad="VALID"` mapping.
- [x] [x] 116. Support passing bias as `bias` option in `conv2d()`.
- [x] [x] 117. Convert ONNX weights (`[M, C/group, kH, kW]`) to WebNN expected layout if default varies (`oihw`).
- [x] [x] 118. Handle `inputLayout` explicitly (`nchw` vs `nhwc`).
- [x] [x] 119. Handle `filterLayout` explicitly (`oihw`, `hwio`, etc.).
- [x] [x] 120. Map ONNX `ConvTranspose` to WebNN `builder.convTranspose2d()`.
- [x] [x] 121. Extract `output_padding` attribute for `ConvTranspose`.
- [x] [x] 122. Map ONNX `MaxPool` to WebNN `builder.maxPool2d()`.
- [x] [x] 123. Map ONNX `AveragePool` to WebNN `builder.averagePool2d()`.
- [x] [x] 124. Handle `kernel_shape` for pooling operations.
- [x] [x] 125. Handle pooling `pads`.
- [x] [x] 126. Handle pooling `strides`.
- [x] [x] 127. Emulate 1D Convolution (`Conv1D`) via WebNN `conv2d` by unsqueezing height=1.
- [x] [x] 128. Emulate 1D Pooling via WebNN `pool2d` by unsqueezing height=1.
- [x] [x] 129. Implement `GlobalAveragePool` via WebNN `builder.averagePool2d()` matching entire spatial dim.
- [x] [x] 130. Implement `GlobalMaxPool` via WebNN `builder.maxPool2d()` matching entire spatial dim.

### Phase 8: Reduction Operations

- [x] [x] 131. Map ONNX `ReduceMean` to WebNN `builder.reduceMean()`.
- [x] [x] 132. Handle `axes` parsing.
- [x] [x] 133. Handle `keepdims` mapping.
- [x] [x] 134. Map ONNX `ReduceSum` to WebNN `builder.reduceSum()`.
- [x] [x] 135. Map ONNX `ReduceMax` to WebNN `builder.reduceMax()`.
- [x] [x] 136. Map ONNX `ReduceMin` to WebNN `builder.reduceMin()`.
- [x] [x] 137. Map ONNX `ReduceProd` to WebNN `builder.reduceProduct()`.
- [x] [x] 138. Map ONNX `ReduceL1` to WebNN `builder.reduceL1()`.
- [x] [x] 139. Map ONNX `ReduceL2` to WebNN `builder.reduceL2()`.
- [x] [x] 140. Map ONNX `ReduceLogSumExp` to WebNN `builder.reduceLogSumExp()`.
- [x] [x] 141. Emulate `ArgMax` via WebNN `builder.argMax()` (if available) or via WebGPU fallback.
- [x] [x] 142. Emulate `ArgMin` via WebNN `builder.argMin()`.

### Phase 9: Normalization Operations

- [x] [x] 143. Map ONNX `BatchNormalization` to WebNN `builder.batchNormalization()`.
- [x] [x] 144. Pass `scale` operand to WebNN.
- [x] [x] 145. Pass `B` (bias) operand to WebNN.
- [x] [x] 146. Pass `mean` operand to WebNN.
- [x] [x] 147. Pass `var` operand to WebNN.
- [x] [x] 148. Parse `epsilon` attribute.
- [x] [x] 149. Map ONNX `InstanceNormalization` to WebNN `builder.instanceNormalization()`.
- [x] [x] 150. Handle `scale` and `B` parameters for InstanceNorm.
- [x] [x] 151. Map ONNX `LayerNormalization` to WebNN `builder.layerNormalization()`.
- [x] [x] 152. Resolve `axis` parameter dynamically for LayerNorm.
- [x] [x] 153. Handle `scale` and `B` parameters for LayerNorm.
- [x] [x] 154. Support `LpNormalization` via WebNN `builder.l2Normalization()`.

### Phase 10: Logical & Relational Operations

- [x] [x] 155. Map ONNX `Equal` to WebNN `builder.equal()`.
- [x] [x] 156. Map ONNX `Greater` to WebNN `builder.greater()`.
- [x] [x] 157. Map ONNX `GreaterOrEqual` to WebNN `builder.greaterOrEqual()`.
- [x] [x] 158. Map ONNX `Less` to WebNN `builder.lesser()`.
- [x] [x] 159. Map ONNX `LessOrEqual` to WebNN `builder.lesserOrEqual()`.
- [x] [x] 160. Map ONNX `Not` to WebNN `builder.logicalNot()`.
- [x] [x] 161. Map ONNX `And` to WebNN `builder.logicalAnd()`.
- [x] [x] 162. Map ONNX `Or` to WebNN `builder.logicalOr()`.
- [x] [x] 163. Map ONNX `Xor` to WebNN `builder.logicalXor()`.
- [x] [x] 164. Map ONNX `Where` to WebNN `builder.where()`.
- [x] [x] 165. Ensure output boolean masks cast strictly back to ONNX Float/Int types if downstream ops require it.

### Phase 11: Graph Compilation & Execution Engine

- [x] [x] 166. Implement the `build()` sequence: finalizing the WebNN `MLGraph`.
- [x] [x] 167. Call `await builder.build(outputs)` to trigger the host NPU compilation.
- [x] [x] 168. Track compile times and log NPU startup latency.
- [x] [x] 169. Allocate `ArrayBuffer` objects for WebNN graph inputs natively in JS.
- [x] [x] 170. Allocate `ArrayBuffer` objects for WebNN graph outputs.
- [x] [x] 171. Map `onnx9000.Tensor` data to WebNN input buffers via `TypedArray` views.
- [x] [x] 172. Implement `context.compute(graph, inputs, outputs)` execution cycle.
- [x] [x] 173. Support the newer `context.dispatch(graph, ...)` API utilizing WebGPU `GPUBuffer` interoperability.
- [x] [x] 174. Enable Zero-Copy execution by mapping `onnx9000` WebGPU tensors directly into WebNN via `MLTensor`.
- [x] [x] 175. Handle execution synchronization (awaiting the NPU compute Promise).
- [x] [x] 176. Re-map WebNN `ArrayBuffer` outputs back to `onnx9000.Tensor` objects safely.
- [x] [x] 177. Maintain an LRU Cache of compiled `MLGraph` objects for dynamic shapes.
- [x] [x] 178. Handle graph disposal via `graph.destroy()` or GC FinalizationRegistry.
- [x] [x] 179. Gracefully catch and log NPU timeout or out-of-memory errors.
- [x] [x] 180. Implement asynchronous non-blocking inference in Web Workers.

### Phase 12: Sub-Graph Partitioning & Fallback

- [x] [x] 181. Implement a WebNN capability checker (simulating a build to check for supported nodes).
- [x] [x] 182. Implement an AST traversal to identify contiguous blocks of WebNN-supported ops.
- [x] [x] 183. Partition the `onnx9000` graph into "WebNN Regions" and "WASM/WebGPU Regions".
- [x] [x] 184. Generate distinct sub-graphs (`onnx9000.Graph`) for each region.
- [x] [x] 185. Compile WebNN Regions to separate `MLGraph` instances.
- [x] [x] 186. Compile WASM/WebGPU Regions using the standard `onnx9000` runtime.
- [x] [x] 187. Execute regions sequentially, copying outputs from WebNN to WASM and vice-versa.
- [x] [x] 188. Optimize boundary crossings (using WebGPU buffers to avoid CPU roundtrips if supported by both).
- [x] [x] 189. Provide CLI flag `--disable-webnn-fallback` to force strict NPU execution (throwing errors if unsupported).
- [x] [x] 190. Handle dynamic shape propagation correctly across partitioned sub-graphs.

### Phase 13: Transformer & LLM specific Operators (WebNN Draft Extensions)

- [x] [x] 191. Map explicit `Gelu` fusions to `builder.gelu()`.
- [x] [x] 192. Translate ONNX `Attention` or `FlashAttention` into standard WebNN MatMul+Softmax subgraphs if a native WebNN Attention op is unavailable.
- [x] [x] 193. Check for emerging W3C WebNN Draft ops (e.g., `triangular`, `scaledDotProductAttention`).
- [x] [x] 194. Fallback: Decompose `LayerNorm` into `ReduceMean`, `Sub`, `Pow`, `Add`, `Div` if `builder.layerNormalization` fails or lacks spec compliance.
- [x] [x] 195. Emulate `RoPE` using WebNN standard trigonometric (`Cos`/`Sin`) and arithmetic primitives.
- [x] [x] 196. Handle multi-dimensional dynamic KV cache updates. If WebNN forbids dynamic `concat`, execute cache updates in WebGPU/WASM and only run the dense feed-forward blocks in WebNN.
- [x] [x] 197. Support caching pre-compiled NPU transformer blocks.
- [x] [x] 198. Map NLP vocabulary `Gather` operations efficiently (or offload to CPU if NPUs struggle with embedding lookups).
- [x] [x] 199. Compile MoE (Mixture of Experts) routers natively in WebNN if conditional execution (`builder.if`) becomes supported.
- [x] [x] 200. Execute gating logic on CPU and only send the selected expert matrices to WebNN to save bandwidth.

### Phase 14: Quantization (W8A8 & W4A16 Native WebNN integration)

- [x] [x] 201. Support ONNX `QuantizeLinear` via WebNN `builder.quantizeLinear()`.
- [x] [x] 202. Support ONNX `DequantizeLinear` via WebNN `builder.dequantizeLinear()`.
- [x] [x] 203. Map ONNX `DynamicQuantizeLinear` to WebNN if supported, otherwise emulate via `reduceMax/Min` and `quantize`.
- [x] [x] 204. Detect and utilize `int8` data types natively in `builder.conv2d` and `builder.matmul`.
- [x] [x] 205. Support zero-point shifting explicitly in WebNN matrix multiplications.
- [x] [x] 206. Implement INT4 unpacking via WebNN bitwise ops (`builder.bitwiseAnd`, `builder.shiftRightLogical`) if available.
- [x] [x] 207. Emulate INT4 unpacking via `float32` math if bitwise ops are missing on the NPU host.
- [x] [x] 208. Integrate with `onnx9000.optimum` to export models specifically targeting WebNN Int8 topologies.
- [x] [x] 209. Push QDQ (Quantize-Dequantize) pairs down the graph into the WebNN compiler, allowing the NPU to fuse them into native Int8 MAC instructions.
- [x] [x] 210. Validate quantization accuracy against CPU baseline to ensure NPU driver hasn't applied aggressive lossy compression.

### Phase 15: Edge Cases & Quirks Management

- [x] [x] 211. Emulate ONNX `GatherElements` (often missing in NPUs) using WebGPU.
- [x] [x] 212. Emulate ONNX `ScatterND` using WebGPU fallback.
- [x] [x] 213. Emulate ONNX `NonZero` (dynamic output shape) by executing exclusively on CPU/WASM.
- [x] [x] 214. Emulate ONNX `TopK` using WASM fallback.
- [x] [x] 215. Handle differences in `padding` spec between ONNX and WebNN (explicit symmetric vs asymmetric arrays).
- [x] [x] 216. Ensure 64-bit integer inputs (`int64`) are automatically down-casted to `int32`, as WebNN officially drops `int64` support for portability.
- [x] [x] 217. Ensure 64-bit floats (`float64`) are down-casted to `float32`.
- [x] [x] 218. Handle empty tensor evaluations (e.g., shape `[0, 10]`) without crashing the NPU driver.
- [x] [x] 219. Manage NaNs and Infs propagation explicitly according to WebNN standard guidelines.
- [x] [x] 220. Prevent WebNN memory limit exceeded crashes by chunking massive convolutions iteratively.

### Phase 16: Device-Specific Tuning (Intel VPU, Apple Neural Engine, Snapdragon)

- [x] [x] 221. Implement a hardware-sniffing utility checking user-agent/GPU strings.
- [x] [x] 222. Workaround: If Apple Neural Engine, prefer NHWC layout explicit casting before `conv2d` to prevent catastrophic driver reshapes.
- [x] [x] 223. Workaround: If Intel VPU, pad channel dimensions to multiples of 4 or 16.
- [x] [x] 224. Workaround: Avoid `builder.erf()` on Snapdragon NPUs if known to be unstable, emulating with Tanh polynomials.
- [x] [x] 225. Workaround: Detect driver timeouts on Windows ARM and reduce graph partition sizes automatically.
- [x] [x] 226. Provide a `--webnn-compat-mode` flag enabling all known driver workarounds.
- [x] [x] 227. Profile operator compilation times to dynamically skip WebNN for extremely fast, simple nodes (e.g., a single `Add`), which are faster in WASM.
- [x] [x] 228. Support pre-warming the NPU with dummy data to avoid UI stutters on first inference.
- [x] [x] 229. Expose NPU execution metrics natively via the `onnx9000` Profiler API.
- [x] [x] 230. Submit anonymous telemetry on specific WebNN operator failures to identify broken driver updates.

### Phase 17: Memory Management & Buffer Re-use

- [x] [x] 231. Implement an Arena allocator specifically for WebNN `ArrayBuffer` inputs.
- [x] [x] 232. Prevent garbage collection thrashing by re-using `context.compute` output buffers.
- [x] [x] 233. Map `onnx9000` internal tensor pools directly to WebNN view allocations.
- [x] [x] 234. Handle sub-array views cleanly (when an ONNX tensor is merely a sliced view of another memory block).
- [x] [x] 235. Support zero-initialization of padding buffers to prevent security leaks of old memory.
- [x] [x] 236. Manage WebGPU `MLTensor` lifecycles properly, calling `tensor.destroy()` precisely when the graph is destroyed.
- [x] [x] 237. Ensure asynchronous execution prevents memory mutations from the main thread during NPU execution.
- [x] [x] 238. Fallback to copying buffers securely if SharedArrayBuffer is restricted by CORS/COOP headers.
- [x] [x] 239. Monitor JS heap size vs active WebNN allocations, triggering manual GC hints if nearing OOM.
- [x] [x] 240. Track precise byte alignment requirements (e.g., 4-byte boundaries) for `float16` buffers passed to WebNN.

### Phase 18: Testing & Conformance

- [x] [x] 241. Construct automated test suite passing the standard ONNX Node test dataset directly to the WebNN EP.
- [x] [x] 242. Validate `Add` node outputs against WASM CPU.
- [x] [x] 243. Validate `Conv2d` node outputs against WASM CPU.
- [x] [x] 244. Validate `MatMul` node outputs against WASM CPU.
- [x] [x] 245. Run tests using the `webnn-polyfill` in headless Chrome/Puppeteer.
- [x] [x] 246. Run tests natively on macOS Chrome with `--enable-features=WebMachineLearningNeuralNetwork`.
- [x] [x] 247. Run tests natively on Windows Edge with NPU support enabled.
- [x] [x] 248. Calculate acceptable numerical drift tolerances (e.g., 1e-4) to account for NPU-specific precision drops.
- [x] [x] 249. Create tests for every single pad mode (`constant`, `edge`, `reflect`).
- [x] [x] 250. Create tests for specific broadcast combinations (e.g., `[1, 3, 224, 224] + [3, 1, 1]`).
- [x] [x] 251. Test multi-output nodes (e.g., `Split`, `TopK` fallback) correctness.
- [x] [x] 252. Ensure memory is pristine after 1000 successive iterations (leak testing).
- [x] [x] 253. Build a fuzzing harness generating random ONNX graphs and ensuring the WebNN EP doesn't crash the browser.
- [x] [x] 254. Test dynamic batch sizes (changing input shape from `[1, ...]` to `[4, ...]`) without re-compiling the graph.
- [x] [x] 255. Verify asynchronous execution does not block CSS animations on the main thread.

### Phase 19: Framework & Tooling Integration

- [x] [x] 256. Allow `Transformers.js` pipelines to explicitly target WebNN (`device: 'webnn'`).
- [x] [x] 257. Hook WebNN capability checking into the `AutoConfig` loader.
- [x] [x] 258. Ensure `onnx9000.genai` can offload LLM MatMul blocks natively to the NPU.
- [x] [x] 259. Integrate with `onnx9000.optimum` CLI to allow testing WebNN equivalence directly from the command line (`onnx9000 test webnn model.onnx`).
- [x] [x] 260. Publish a diagnostic web page showing "WebNN Readiness" for a user's current browser.
- [x] [x] 261. Integrate with Native WebViews (when WebNN ships to mobile WebViews).
- [x] [x] 262. Support WebNN EP configuration flags (e.g., setting execution priority).
- [x] [x] 263. Emit standard `onnxruntime` EP log formats for compatibility with legacy debugging tools.
- [x] [x] 264. Support importing generic ONNX JSON (via ORT) and building the WebNN graph.
- [x] [x] 265. Document the complete list of supported ops and their spec version in a generated Markdown file.

### Phase 20: Advanced API Features & Future Specs

- [x] [x] 266. Prepare for W3C WebNN API v2 (dynamic shapes natively).
- [x] [x] 267. Map ONNX `Loop` natively if WebNN introduces control flow APIs.
- [x] [x] 268. Map ONNX `If` natively to WebNN.
- [x] [x] 269. Support specialized WebNN `lstm` and `gru` builder functions for RNN models.
- [x] [x] 270. Support WebNN `builder.resample2d` explicitly for ONNX `Resize` operations.
- [x] [x] 271. Support nearest-neighbor interpolation in WebNN `resample2d`.
- [x] [x] 272. Support linear interpolation in WebNN `resample2d`.
- [x] [x] 273. Support `builder.gatherNd` if added to the WebNN spec.
- [x] [x] 274. Handle WebNN `logicalAnd/Or/Not` applied to multi-dimensional masks.
- [x] [x] 275. Map ONNX `CumSum` to NPU native execution (often tricky, might require scan algorithms).
- [x] [x] 276. Provide hooks for WebNN `builder.gruCell` mapping.
- [x] [x] 277. Provide hooks for WebNN `builder.lstmCell` mapping.
- [x] [x] 278. Support explicit data layout overriding during WebNN graph build (ignoring ONNX constraints).
- [x] [x] 279. Build an automated transpiler: `onnx9000-to-wgsl` for ops rejected by the WebNN context, ensuring no fallback to slow JS math.
- [x] [x] 280. Handle `uint32` data types in WebNN (often required for Gather indices).
- [x] [x] 281. Integrate `onnx9000.image` pre-processing natively into the WebNN graph (fusing Normalize/Resize ops into the NPU).
- [x] [x] 282. Expose `builder.triangular` for specialized causal masking if present.
- [x] [x] 283. Support executing multiple isolated WebNN contexts concurrently for multi-model web apps.
- [x] [x] 284. Implement fallback logic for WebNN unsupported `dilations` values in specific layers.
- [x] [x] 285. Support `builder.dequantizeLinear` executing specifically on NPU vector engines.
- [x] [x] 286. Map ONNX `SpaceToDepth` and `DepthToSpace` to WebNN if supported natively.
- [x] [x] 287. Compile and run YOLO-v8 fully accelerated on the WebNN EP.
- [x] [x] 288. Compile and run MobileViT fully accelerated on the WebNN EP.
- [x] [x] 289. Compile and run Whisper (Encoder) fully accelerated on the WebNN EP.
- [x] [x] 290. Maintain an architecture compatibility matrix tracking exact NPU support levels (Qualcomm vs Apple vs Intel).
- [x] [x] 291. Validate exact compliance with WebNN Draft Spec W3C Working Drafts.
- [x] [x] 292. Support `builder.concat` with more than 5 inputs (handling NPU argument limits).
- [x] [x] 293. Track and bypass known WebNN Polyfill bugs dynamically.
- [x] [x] 294. Optimize constant memory uploads to prevent Chrome UI freezes during `builder.build()`.
- [x] [x] 295. Execute deep layout analysis (NCHW to NHWC) eliminating redundant transpose chains specific to NPU backends.
- [x] [x] 296. Map ONNX `HardSwish` natively using WebNN arithmetic `x * hardSigmoid`.
- [x] [x] 297. Support WebNN native `builder.softplus`.
- [x] [x] 298. Validate precise execution parity between `device: 'webgpu'` and `device: 'webnn'` on the exact same hardware.
- [x] [x] 299. Write comprehensive tutorial: "Deploying ONNX Models to NPUs in the Browser".
- [x] [x] 300. Release v1.0 complete feature parity certification matching the official C++ ONNX Runtime WebNN EP.
