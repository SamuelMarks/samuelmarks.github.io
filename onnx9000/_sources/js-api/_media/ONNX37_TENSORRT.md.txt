# ONNX37: ONNX-TensorRT (Zero-Build TRT FFI Parser)

## Original Project Description

The `onnx-tensorrt` project is NVIDIA's open-source C++ parser that translates ONNX models into TensorRT `INetworkDefinition` graphs. TensorRT uses this definition to perform rigorous kernel auto-tuning and generates a highly optimized execution engine (`.plan` / `.trt` file) tailored for the specific local NVIDIA GPU. Historically, using this parser requires a heavy C++ toolchain, strict alignment of CUDA/cuDNN/TensorRT header versions, and compiling massive native binaries just to ingest an ONNX file.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.tensorrt` eliminates the C++ compilation requirement completely by offering a **100% pure Python and Node.js FFI (Foreign Function Interface) parser**.

- **Zero-Build Compilation:** Instead of building a C++ ONNX parser, `onnx9000` reads the ONNX AST using its own zero-dependency engine and directly invokes the native TensorRT C-API (`libnvinfer.so` / `nvinfer.dll`) via `ctypes` or `node-ffi-napi`.
- **Dynamic Polyglot Deployment:** A Node.js backend server can ingest an ONNX file, translate it natively to a TRT engine using the local GPU driver, and instantly serve inferences without ever invoking Python or C++.
- **Deep Memory Control:** By handling the parsing in Python/JS, `onnx9000` can inject custom weight layouts, apply AST-level fusions (e.g., packing W4A16), or dynamically partition the graph _before_ handing it over to TensorRT, bypassing many of the strict limitations in NVIDIA's native ONNX parser.

---

## Exhaustive Implementation Checklist

### Phase 1: Core FFI Architecture & LibNVInfer Loading

- [x][x] 1. Detect `libnvinfer.so` (Linux) dynamically via `ctypes.util.find_library`.
- [x][x] 2. Detect `nvinfer.dll` (Windows) dynamically.
- [x][x] 3. Extract TensorRT API version natively (e.g., 8.6, 10.0) from the shared library.
- [x][x] 4. Establish FFI fallback policies if different TRT versions expose different function signatures.
- [x][x] 5. Implement `ILogger` C-callback natively in Python to intercept TRT diagnostic messages.
- [x][x] 6. Route TRT `ILogger` `kINFO`, `kWARNING`, `kERROR` events directly to standard Python/JS loggers.
- [x][x] 7. Provide dynamic memory management across the FFI boundary to prevent C-side segfaults.
- [x][x] 8. Implement a global registry of active TRT pointers to ensure explicit destruction (`__del__` / `FinalizationRegistry`).
- [x][x] 9. Implement `createInferBuilder_INTERNAL` FFI binding.
- [x][x] 10. Support explicitly destroying builders via `destroy()` bindings.
- [x][x] 11. Implement `node-ffi-napi` bindings for equivalent Node.js server execution.
- [x][x] 12. Map C `enum` values specifically to Python `IntEnum` structures for precise TRT configurations.
- [x][x] 13. Catch C-level hardware errors and surface them as readable Python `RuntimeError` exceptions.
- [x][x] 14. Support detecting `libnvinfer_plugin.so` automatically for custom operator extensions.
- [x][x] 15. Expose `cudaGetDeviceProperties` via FFI to query GPU compute capability automatically.

### Phase 2: TensorRT Builder & Network Definition

- [x][x] 16. Initialize `IBuilder` explicitly.
- [x][x] 17. Initialize `INetworkDefinition` (Explicit Batch flag required for ONNX).
- [x][x] 18. Initialize `IBuilderConfig`.
- [x][x] 19. Implement `addInput(name, type, dims)` mapping ONNX Inputs to TRT.
- [x][x] 20. Implement `markOutput(tensor)` mapping ONNX Outputs to TRT.
- [x][x] 21. Translate ONNX `FLOAT32` to `DataType::kFLOAT`.
- [x][x] 22. Translate ONNX `FLOAT16` to `DataType::kHALF`.
- [x][x] 23. Translate ONNX `INT32` to `DataType::kINT32`.
- [x][x] 24. Translate ONNX `INT8` to `DataType::kINT8`.
- [x][x] 25. Translate ONNX `BOOL` to `DataType::kBOOL`.
- [x][x] 26. Emulate `INT64` support by explicitly injecting `Cast` nodes (TRT historically lacks strict int64 support).
- [x][x] 27. Map ONNX dimensional arrays `[1, 3, 224, 224]` to TRT `Dims` structs in C memory.
- [x][x] 28. Map ONNX dynamic axes (`-1`) to TRT dynamic dimensions correctly.
- [x][x] 29. Track the mapping between ONNX `NodeArg` strings and TRT `ITensor` pointers dynamically in a dictionary.
- [x][x] 30. Configure explicit memory pools (`config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, size)`).

### Phase 3: Constant & Weight Translation

- [x][x] 31. Implement `addConstant(dims, weights)` mapping ONNX Initializers to TRT.
- [x][x] 32. Pass Python NumPy pointers directly into `Weights` structs securely (`data_ptr`).
- [x][x] 33. Ensure weights outlive the Engine building phase by pinning Python arrays in memory.
- [x][x] 34. Extract scalar ONNX constants correctly to 0-D TRT constants.
- [x][x] 35. Embed large external weights (`.bin`) seamlessly by mapping their raw bytes via `mmap` into the TRT `Weights` struct.
- [x][x] 36. Handle Endianness requirements natively before passing buffers to `addConstant`.
- [x][x] 37. Bypass zero-sized constants natively.
- [x][x] 38. Collapse constant chains explicitly in `onnx9000.modifier` before passing them to TRT to save build time.
- [x][x] 39. Emit specific warnings if a Constant exceeds TRT dimension size maximums.
- [x][x] 40. Prevent memory leaks by explicitly unpinning weight arrays once the `buildSerializedNetwork` call completes.

### Phase 4: Core Math & Matrix Operations (`IElementWiseLayer`)

- [x][x] 41. Map ONNX `Add` to `addElementWise(a, b, ElementWiseOperation::kSUM)`.
- [x][x] 42. Map ONNX `Sub` to `ElementWiseOperation::kSUB`.
- [x][x] 43. Map ONNX `Mul` to `ElementWiseOperation::kPROD`.
- [x][x] 44. Map ONNX `Div` to `ElementWiseOperation::kDIV`.
- [x][x] 45. Map ONNX `Max` to `ElementWiseOperation::kMAX`.
- [x][x] 46. Map ONNX `Min` to `ElementWiseOperation::kMIN`.
- [x][x] 47. Map ONNX `Pow` to `ElementWiseOperation::kPOW`.
- [x][x] 48. Map ONNX `Equal` to `ElementWiseOperation::kEQUAL`.
- [x][x] 49. Map ONNX `Less` to `ElementWiseOperation::kLESS`.
- [x][x] 50. Map ONNX `Greater` to `ElementWiseOperation::kGREATER`.
- [x][x] 51. Handle ONNX implicit broadcasting manually by injecting `IShuffleLayer` or TRT broadacast flags if required.
- [x][x] 52. Map ONNX `MatMul` to `addMatrixMultiply(a, opA, b, opB)`.
- [x][x] 53. Handle `MatrixOperation::kTRANSPOSE` natively based on ONNX transpose structures.
- [x][x] 54. Map ONNX `Gemm` to `addFullyConnected()` if applicable, or decomposed `MatrixMultiply` + `ElementWise`.
- [x][x] 55. Validate multi-dimensional batched MatMul dimensions explicitly.

### Phase 5: Convolution & Pooling Layers

- [x][x] 56. Map ONNX `Conv` (2D) to `addConvolutionNd(input, numOutputs, kernelSize, weights, bias)`.
- [x][x] 57. Map ONNX `Conv` (3D) to `addConvolutionNd`.
- [x][x] 58. Map ONNX `Conv` (1D) to `addConvolutionNd` (Emulating via 2D if required by older TRT versions).
- [x][x] 59. Set TRT `Stride` explicitly from ONNX attributes.
- [x][x] 60. Set TRT `Padding` (pre/post) explicitly.
- [x][x] 61. Set TRT `Dilation` explicitly.
- [x][x] 62. Set TRT `NumGroups` explicitly for Depthwise Convolution mapping.
- [x][x] 63. Map ONNX `ConvTranspose` to `addDeconvolutionNd()`.
- [x][x] 64. Map ONNX `MaxPool` to `addPoolingNd(input, PoolingType::kMAX, windowSize)`.
- [x][x] 65. Map ONNX `AveragePool` to `PoolingType::kAVERAGE`.
- [x][x] 66. Support `AveragePool` `count_include_pad` attribute mapping.
- [x][x] 67. Set pooling `Stride`, `Padding`, and `BlendFactor` dynamically.
- [x][x] 68. Map ONNX `GlobalAveragePool` to `addPoolingNd` spanning the entire spatial dimension.
- [x][x] 69. Map ONNX `GlobalMaxPool` to `addPoolingNd`.
- [x][x] 70. Handle asymmetric padding safely via TRT `setPaddingMode` or explicit `IPaddingLayer`.

### Phase 6: Activation, Normalization, & Unary Ops

- [x][x] 71. Map ONNX `Relu` to `addActivation(input, ActivationType::kRELU)`.
- [x][x] 72. Map ONNX `Sigmoid` to `ActivationType::kSIGMOID`.
- [x][x] 73. Map ONNX `Tanh` to `ActivationType::kTANH`.
- [x][x] 74. Map ONNX `LeakyRelu` to `ActivationType::kLEAKY_RELU` (setting `alpha`).
- [x][x] 75. Map ONNX `Elu` to `ActivationType::kELU`.
- [x][x] 76. Map ONNX `Selu` to `ActivationType::kSELU`.
- [x][x] 77. Map ONNX `Softplus` to `ActivationType::kSOFTPLUS`.
- [x][x] 78. Map ONNX `Clip` to `ActivationType::kCLIP` (setting `alpha` and `beta`).
- [x][x] 79. Map ONNX `HardSigmoid` to `ActivationType::kHARD_SIGMOID`.
- [x][x] 80. Map ONNX `Softmax` to `addSoftMax(input)`.
- [x][x] 81. Map Softmax `axis` cleanly.
- [x][x] 82. Map ONNX `BatchNormalization` to `addScale(input, ScaleMode::kCHANNEL, shift, scale, power)`.
- [x][x] 83. Pre-calculate Batch Norm scale/shift values offline in Python before passing to TRT `ScaleLayer`.
- [x][x] 84. Map ONNX `InstanceNormalization` to `addScaleNd` or native TRT Plugin.
- [x][x] 85. Map ONNX `LayerNormalization` to `addNormalization()` (TRT 10+).
- [x][x] 86. Map Unary `Exp` to `addUnaryOperation(input, UnaryOperation::kEXP)`.
- [x][x] 87. Map Unary `Log` to `UnaryOperation::kLOG`.
- [x][x] 88. Map Unary `Sqrt` to `UnaryOperation::kSQRT`.
- [x][x] 89. Map Unary `Abs` to `UnaryOperation::kABS`.
- [x][x] 90. Map Unary `Neg` to `UnaryOperation::kNEG`.

### Phase 7: Dimension Manipulation & Routing

- [x][x] 91. Map ONNX `Reshape` to `addShuffle(input)`.
- [x][x] 92. Handle `Reshape` dynamic dimensions via `setReshapeDimensions`.
- [x][x] 93. Map ONNX `Transpose` to `IShuffleLayer` via `setFirstTranspose(perm)`.
- [x][x] 94. Map ONNX `Flatten` to `addShuffle` with flattened dims.
- [x][x] 95. Map ONNX `Squeeze` to `addShuffle`.
- [x][x] 96. Map ONNX `Unsqueeze` to `addShuffle`.
- [x][x] 97. Map ONNX `Concat` to `addConcatenation(tensors, numTensors)`.
- [x][x] 98. Handle `Concat` axis parameter natively.
- [x][x] 99. Map ONNX `Split` to `addSlice()` dynamically allocating individual output tensors.
- [x][x] 100. Map ONNX `Slice` to `addSlice(input, start, size, stride)`.
- [x][x] 101. Process negative indices in `Slice` dynamically.
- [x][x] 102. Map ONNX `Gather` to `addGather(data, indices, axis)`.
- [x][x] 103. Map ONNX `GatherND` to `addGatherNd`.
- [x][x] 104. Map ONNX `ScatterND` to `addScatter`.
- [x][x] 105. Map ONNX `ScatterElements` to `addScatter`.
- [x][x] 106. Map ONNX `Shape` to `addShape(input)`.
- [x][x] 107. Map ONNX `Expand` to `IShuffleLayer` broadcast mechanisms.
- [x][x] 108. Map ONNX `Tile` to TRT equivalents (or nested loops if unsupported).
- [x][x] 109. Map ONNX `Pad` to `addPaddingNd` natively.
- [x][x] 110. Evaluate explicit TRT dynamic shape bounds prior to mapping slice/gather.

### Phase 8: Reduction & Logical Operators

- [x][x] 111. Map ONNX `ReduceMean` to `addReduce(input, ReduceOperation::kAVG, keepAxes, keepDims)`.
- [x][x] 112. Map ONNX `ReduceSum` to `ReduceOperation::kSUM`.
- [x][x] 113. Map ONNX `ReduceMax` to `ReduceOperation::kMAX`.
- [x][x] 114. Map ONNX `ReduceMin` to `ReduceOperation::kMIN`.
- [x][x] 115. Map ONNX `ReduceProd` to `ReduceOperation::kPROD`.
- [x][x] 116. Map ONNX `ArgMax` to `addTopK(input, TopKOperation::kMAX, 1, axes)` + Gather.
- [x][x] 117. Map ONNX `ArgMin` to `TopKOperation::kMIN`.
- [x][x] 118. Map ONNX `TopK` directly to `addTopK()`.
- [x][x] 119. Map ONNX `Not` to `UnaryOperation::kNOT`.
- [x][x] 120. Map ONNX `And` to `ElementWiseOperation::kAND`.
- [x][x] 121. Map ONNX `Or` to `ElementWiseOperation::kOR`.
- [x][x] 122. Map ONNX `Xor` to `ElementWiseOperation::kXOR`.
- [x][x] 123. Map ONNX `Where` to `addSelect(condition, thenInput, elseInput)`.
- [x][x] 124. Map ONNX `NonZero` to TRT `addNonZero` (Dynamic Output Shape).
- [x][x] 125. Process boolean casting seamlessly for logic gates.

### Phase 9: Advanced Control Flow & Subgraphs

- [x][x] 126. Map ONNX `If` to TRT `IIfConditional`.
- [x][x] 127. Parse True Branch graph into `IIfConditional->setTrue()`.
- [x][x] 128. Parse False Branch graph into `IIfConditional->setFalse()`.
- [x][x] 129. Bind Subgraph inputs logically using `IIfConditionalInputLayer`.
- [x][x] 130. Extract outputs logically using `IIfConditionalOutputLayer`.
- [x][x] 131. Map ONNX `Loop` to TRT `ILoop`.
- [x][x] 132. Implement loop state variables via `addRecurrenceLayer`.
- [x][x] 133. Implement sequence lengths via `addTripLimit`.
- [x][x] 134. Handle iterators dynamically inside the TRT loop block.
- [x][x] 135. Manage loop body outputs securely.

### Phase 10: Dynamic Shapes & Optimization Profiles

- [x][x] 136. Detect dynamic axes (`-1`) across all Graph Inputs dynamically.
- [x][x] 137. Create `IOptimizationProfile` explicitly via `builder->createOptimizationProfile()`.
- [x][x] 138. Expose Python API `set_dynamic_shape(input_name, min_shape, opt_shape, max_shape)`.
- [x][x] 139. Set Min dimensions via `profile->setDimensions(inputName, OptProfileSelector::kMIN, dims)`.
- [x][x] 140. Set Opt (Optimal) dimensions via `OptProfileSelector::kOPT`.
- [x][x] 141. Set Max dimensions via `OptProfileSelector::kMAX`.
- [x][x] 142. Add profile to config via `config->addOptimizationProfile(profile)`.
- [x][x] 143. Support adding multiple distinct optimization profiles for varying batch sizes.
- [x][x] 144. Validate Min <= Opt <= Max rules natively in Python before calling TRT to prevent cryptic C++ crashes.
- [x][x] 145. Evaluate dynamic constraints globally to auto-generate opt shapes if user omits them.

### Phase 11: Quantization & Precision Control

- [x][x] 146. Enable FP16 execution globally (`config->setFlag(BuilderFlag::kFP16)`).
- [x][x] 147. Enable INT8 execution globally (`config->setFlag(BuilderFlag::kINT8)`).
- [x][x] 148. Set explicit layer precisions (`layer->setPrecision(DataType::kINT8)`).
- [x][x] 149. Map ONNX `QuantizeLinear` and `DequantizeLinear` (QDQ) structures dynamically to TRT implicit INT8 processing.
- [x][x] 150. Emulate Post-Training Quantization (PTQ) Calibration interfaces.
- [x][x] 151. Implement `IInt8EntropyCalibrator2` entirely using Python C-Callbacks.
- [x][x] 152. Implement `IInt8MinMaxCalibrator` via Python callbacks.
- [x][x] 153. Supply calibration dataset batches securely from Python Generators into the TRT C-API pointer buffers.
- [x][x] 154. Read and Write TRT calibration cache files natively.
- [x][x] 155. Enforce strict FP32 types on specific sensitive layers (e.g., Softmax) natively via API hooks.
- [x][x] 156. Handle Web-Native `W4A16` configurations, extracting packed INT4 weights and exposing them as explicit TRT structures if supported, or unpacking to FP16 dynamically.
- [x][x] 157. Toggle `BuilderFlag::kOBEY_PRECISION_CONSTRAINTS` if strict mode is requested.
- [x][x] 158. Enable `BuilderFlag::kTF32` explicitly for Ampere+ hardware.
- [x][x] 159. Enable `BuilderFlag::kFP8` explicitly for Hopper+ hardware.
- [x][x] 160. Parse and extract Dynamic Range attributes directly from ONNX schemas if provided.

### Phase 12: Custom Plugins & Extensibility (`IPluginV2`)

- [x][x] 161. Detect `ai.onnx.contrib` or unknown operators.
- [x][x] 162. Implement `IPluginCreator` bindings.
- [x][x] 163. Map ONNX `GridSample` to TRT standard `GridSamplePlugin`.
- [x][x] 164. Map ONNX `NonMaxSuppression` to TRT `BatchedNMSDynamic_TRT` plugin natively.
- [x][x] 165. Map ONNX `RoiAlign` to TRT `ROIAlign_TRT` plugin.
- [x][x] 166. Handle serialization of plugin configuration variables (`PluginFieldCollection`) via ctypes structs.
- [x][x] 167. Bind explicitly written Python/CUDA custom extensions natively into the TRT build process.
- [x][x] 168. Ensure custom plugin `getSerializationSize()` matches exactly with the provided FFI structs.
- [x][x] 169. Provide fallback implementations: if a node lacks a Plugin, replace it mathematically using standard ONNX ops (e.g., `Gelu` expansion).
- [x][x] 170. Load `libnvinfer_plugin.so` automatically via `initLibNvInferPlugins`.

### Phase 13: Engine Serialization & Caching

- [x][x] 171. Trigger `buildSerializedNetwork(network, config)`.
- [x][x] 172. Output progress logs continuously during the massive compilation loop.
- [x][x] 173. Catch Out-Of-Memory limits securely during the builder phase.
- [x][x] 174. Extract the compiled Engine byte payload (`IHostMemory`).
- [x][x] 175. Stream the engine bytes natively to a `.trt` / `.engine` / `.plan` file.
- [x][x] 176. Implement zero-copy buffer views returning the Engine byte payload directly to Python RAM.
- [x][x] 177. Instantiate `IRuntime` explicitly.
- [x][x] 178. Deserialize the Engine (`runtime->deserializeCudaEngine(data, size)`).
- [x][x] 179. Set the Device Index natively (`cudaSetDevice`) before deserialization.
- [x][x] 180. Track hardware environment constraints (TRT engines are hardware-specific; fail safely if attempting to load an engine built on A100 into a T4).

### Phase 14: Engine Execution Context & Runtime

- [x][x] 181. Instantiate `IExecutionContext` from the Engine.
- [x][x] 182. Implement dynamic shape allocation (`context->setBindingDimensions(index, dims)`).
- [x][x] 183. Resolve exact output dimensions dynamically using `context->getBindingDimensions(index)`.
- [x][x] 184. Extract total required Workspace Size dynamically.
- [x][x] 185. Provide native CUDA memory allocation wrappers (`cudaMalloc`, `cudaFree`) in Python via ctypes.
- [x][x] 186. Support creating CUDA Streams explicitly (`cudaStreamCreate`).
- [x][x] 187. Implement asynchronous enqueue (`context->enqueueV3(stream)`).
- [x][x] 188. Support legacy `enqueueV2` for backward compatibility.
- [x][x] 189. Synchronize explicitly (`cudaStreamSynchronize`).
- [x][x] 190. Wrap the execution context securely inside Python Context Managers (`with trt_session:`) to guarantee cleanup.

### Phase 15: Zero-Copy DLPack Integration (PyTorch / CuPy Bridge)

- [x][x] 191. Implement `__dlpack__` ingestor explicitly for TRT bindings.
- [x][x] 192. Accept PyTorch `torch.Tensor` residing in CUDA memory natively as execution inputs.
- [x][x] 193. Accept CuPy `cp.ndarray` residing in CUDA memory natively.
- [x][x] 194. Extract device pointers seamlessly (`tensor.data_ptr()`).
- [x][x] 195. Create empty PyTorch output tensors on the identical CUDA device dynamically based on `getBindingDimensions`.
- [x][x] 196. Execute TRT engine directly over the PyTorch pre-allocated pointers (True Zero-Copy).
- [x][x] 197. Hook the TRT execution natively into standard PyTorch CUDA Streams.
- [x][x] 198. Ensure asynchronous non-blocking launches return immediately to the Python event loop.
- [x][x] 199. Handle continuous batching setups by swapping input pointer bindings asynchronously.
- [x][x] 200. Evaluate latency bounds natively (comparing C++ TRT exec vs Python FFI TRT exec; target <5% overhead).

### Phase 16: Integration with `onnx9000` Core Ecosystem

- [x][x] 201. Define `TensorrtExecutionProvider` natively within `onnx9000`.
- [x][x] 202. Implement automated Sub-Graph partitioning (Send supported nodes to TRT, keep unsupported nodes in WebGPU/CPU).
- [x][x] 203. Execute `cudaMemcpy` automatically when crossing Execution Provider boundaries.
- [x][x] 204. Enable user flag: `--trt-fallback` to determine strict vs relaxed execution.
- [x][x] 205. Store compiled `TRT` subgraphs in `~/.cache/onnx9000/trt_engines/`.
- [x][x] 206. Read cached engines automatically on session reload based on Model Hash and Node topology.
- [x][x] 207. Generate comprehensive Optimization Fusions natively in `onnx9000` (Level 3) _before_ TRT to speed up TRT compilation times.
- [x][x] 208. Strip Dropout and Identity nodes out of the graph natively before submitting to TRT.
- [x][x] 209. Inject dynamic shapes exclusively around known variable dimensions (Batch, SeqLen).
- [x][x] 210. Expose standard `InferenceSession` APIs globally over the TRT backend.

### Phase 17: Python API & CLI Tooling (`onnx9000 trt`)

- [x][x] 211. Provide CLI: `onnx9000 trt build model.onnx -o model.engine`.
- [x][x] 212. Support flag: `--fp16`.
- [x][x] 213. Support flag: `--int8`.
- [x][x] 214. Support flag: `--dynamic-batch min:opt:max` (e.g., `1:8:32`).
- [x][x] 215. Support flag: `--workspace-size <MB>`.
- [x][x] 216. Provide CLI: `onnx9000 trt run model.engine --inputs data.json`.
- [x][x] 217. Expose an equivalent utility to `trtexec` providing raw performance metrics (Latency, Throughput, GPU mem).
- [x][x] 218. Generate detailed timeline traces natively in Python matching the execution timeline.
- [x][x] 219. Expose an API specifically for loading and running pre-built `.engine` files without needing the original ONNX.
- [x][x] 220. Support logging output directly to a file (`--log-file trt_build.log`).

### Phase 18: Node.js / Serverless API Integration

- [x][x] 221. Replicate the FFI binding layer identically using `node-ffi-napi`.
- [x][x] 222. Expose JS asynchronous API: `const engine = await trt.build(onnxBuffer, { fp16: true })`.
- [x][x] 223. Run the TRT Builder explicitly off the main JS thread using libuv worker pools to prevent blocking incoming HTTP requests.
- [x][x] 224. Wrap `cudaStreamSynchronize` as an asynchronous Javascript Promise.
- [x][x] 225. Support Node.js `Buffer` objects seamlessly translating into CUDA memory via pinned host allocations.
- [x][x] 226. Ensure JS garbage collection successfully triggers `engine->destroy()` safely.
- [x][x] 227. Export an Express.js / Fastify middleware wrapper deploying TRT engines dynamically for REST APIs.
- [x][x] 228. Handle strict Endianness validation natively in Javascript when translating tensors.
- [x][x] 229. Expose `TRTLogger` callbacks directly into `console.log`.
- [x][x] 230. Distribute the Node.js package independently as `@onnx9000/tensorrt`.

### Phase 19: Edge Cases & Specific Architecture Optimizations

- [x][x] 231. Handle extremely large LLMs (e.g., Llama-3 70B) by partitioning the builder explicitly across multiple GPUs natively.
- [x][x] 232. Support Weight-Only Quantization building explicitly (`W4A16` TRT equivalents).
- [x][x] 233. Handle 1D Tensors reliably (TRT historically forced `[N, C, 1, 1]`, map this cleanly).
- [x][x] 234. Map PyTorch specific export markers flawlessly into TRT dynamic bounds.
- [x][x] 235. Automatically correct ONNX `Gather` negative axis parameters to positive, as TRT gathers fail on negative indices in some versions.
- [x][x] 236. Fallback from `Einsum` to explicit explicit matmuls natively inside `onnx9000` before TRT submission, as TRT `Einsum` support is notoriously fragile.
- [x][x] 237. Handle ONNX `Cast` from `float32` to `bool` cleanly.
- [x][x] 238. Detect and patch `Softmax` on massive sequence dimensions natively to prevent `NaN` during TRT FP16 execution.
- [x][x] 239. Fix generic Pad structures dynamically since TRT occasionally rejects asymmetric constants.
- [x][x] 240. Manage multiple Optimization Profiles dynamically inside the Inference Execution.

### Phase 20: Performance Parity & Validation Checks

- [x][x] 241. Unit Test: Build pure `Add` ONNX, execute via TRT, validate output.
- [x][x] 242. Unit Test: Build `MatMul` ONNX, validate output accuracy.
- [x][x] 243. Unit Test: Build `Conv2D` ONNX, validate output accuracy.
- [x][x] 244. Integration Test: Convert MNIST CNN, execute via TRT natively, evaluate latency.
- [x][x] 245. Integration Test: Convert MobileNetV2, evaluate exact FPS.
- [x][x] 246. Integration Test: Convert YOLOv8, evaluate exact bounding box tolerances (atol=1e-3) under FP16.
- [x][x] 247. Validate execution natively across TensorRT 8.4, 8.5, and 8.6+ versions dynamically.
- [x][x] 248. Assert Memory Leak absence during 100+ sequential inference requests.
- [x][x] 249. Compare execution throughput natively against official C++ `trtexec` (Must achieve >98% relative throughput).
- [x][x] 250. Handle out-of-bounds pointer allocations safely without hard-crashing the host process.
- [x][x] 251. Validate multi-threading (calling `.enqueueV3()` from multiple Python threads concurrently).
- [x][x] 252. Validate multi-model multi-session execution natively.
- [x][x] 253. Compile explicit CNN benchmarks directly natively.
- [x][x] 254. Support reading `.safetensors` external data natively during the builder process.
- [x][x] 255. Evaluate LayerNorm numerical drift extensively.
- [x][x] 256. Provide clear visual reports during optimization loops.
- [x][x] 257. Emit a structural graph report identical to TRT's Engine Inspector format.
- [x][x] 258. Identify Subnormal floats natively.
- [x][x] 259. Validate that standard ONNX compliance tests pass natively under TRT execution.
- [x][x] 260. Manage exact execution determinism correctly.
- [x][x] 261. Expose interactive CLI commands for profiling subgraphs natively.
- [x][x] 262. Check specific dimension limits natively in Python before C++ invokes exceptions.
- [x][x] 263. Emulate unsupported CustomOps gracefully.
- [x][x] 264. Compile Transformer MultiHeadAttention safely.
- [x][x] 265. Emulate `GatherElements` securely.
- [x][x] 266. Manage exact padding requirements for `Conv` native execution.
- [x][x] 267. Handle dynamic looping structures inside LLMs correctly.
- [x][x] 268. Extract 1D vectors seamlessly via SIMD hooks if used concurrently.
- [x][x] 269. Render interactive trace reports cleanly natively.
- [x][x] 270. Add support for creating parallel engine instances on the same GPU.
- [x][x] 271. Implement specific memory layouts for HWC image buffering into TRT.
- [x][x] 272. Evaluate exact bounds checking natively.
- [x][x] 273. Validate execution parity natively.
- [x][x] 274. Create custom issue templates mapping TRT failures.
- [x][x] 275. Render graph connections dynamically in console UI.
- [x][x] 276. Ensure all generated pointers are explicitly typed using `ctypes` annotations.
- [x][x] 277. Write comprehensive API documentation mapping TRT configurations.
- [x][x] 278. Establish automated workflows to test compilation locally.
- [x][x] 279. Support generating `.safetensors` explicitly during the debug phases.
- [x][x] 280. Validate complete `--help` documentation parity against `trtexec`.
- [x][x] 281. Develop specific bounds tracking for variable dimensions.
- [x][x] 282. Track peak VRAM usage natively across hardware explicitly.
- [x][x] 283. Support executing `Einsum` unrolled directly in Python before TRT dispatch.
- [x][x] 284. Extract strings as `const char*` strictly.
- [x][x] 285. Support `--builder-optimization-level` natively (TRT 10.0+).
- [x][x] 286. Handle dynamic sequence generation (LLM autoregressive loop) utilizing a continuous TRT session natively.
- [x][x] 287. Provide memory footprint checks warning the user before initiating massive compilation.
- [x][x] 288. Manage explicitly unknown spatial sizes safely natively.
- [x][x] 289. Map explicit `Less` / `Greater` ops inside TRT flawlessly.
- [x][x] 290. Extract specific INT8 quantized execution topologies successfully.
- [x][x] 291. Validate exact mathematical equivalence of `Exp` / `Log` natively.
- [x][x] 292. Enable Python `__call__` explicit binding.
- [x][x] 293. Map Python `__del__` safely across the GC boundary.
- [x][x] 294. Create standard Github Actions for TRT integration checks securely.
- [x][x] 295. Configure explicit fallback logic for missing CUDA dependencies cleanly natively.
- [x][x] 296. Catch memory allocation errors (OOM) explicitly during the context initialization natively.
- [x][x] 297. Support overriding standard configurations explicitly natively.
- [x][x] 298. Validate precise execution under explicit memory bounds checking on massive GPU architectures natively.
- [x][x] 299. Write comprehensive API documentation matching TRT execution targets natively.
- [x][x] 300. Release v1.0 feature complete certification for `onnx9000.tensorrt` allowing zero-build TRT parsing natively.
