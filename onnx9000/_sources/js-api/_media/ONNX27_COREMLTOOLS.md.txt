# ONNX27: coremltools (Web-Native Apple Silicon Bridge)

## Original Project Description

Apple's `coremltools` is a Python package that converts machine learning models from major frameworks (PyTorch, TensorFlow, ONNX) into the Core ML format (`.mlmodel` and the newer `.mlpackage`). This conversion is essential for deploying models natively on Apple hardware (macOS, iOS, iPadOS, watchOS), allowing the operating system to optimally route computations across the CPU, GPU, and the highly efficient Apple Neural Engine (ANE). The tool relies heavily on Python, protobuf generation, and underlying native binaries to optimize the graph intermediate representation (Model Intermediate Language, or MIL).

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of requiring a local macOS machine with a massive Python environment to generate iOS/macOS payloads, `onnx9000.coreml` is entirely implemented in TypeScript and WebAssembly.

- **Browser-Based Generation:** Allows developers to drag-and-drop an ONNX file into a web browser, perform MIL optimizations, and instantly download a signed `.mlpackage` ready for Xcode, entirely client-side.
- **No Native Dependencies:** Bypasses Apple's proprietary C++ parsing libraries by implementing the CoreML protobuf schema and MIL AST directly in TypeScript.
- **Bi-directional (CoreML -> ONNX):** While Apple's tool primarily focuses on _exporting_ to CoreML, `onnx9000` natively supports _importing_ existing `.mlmodel` files and lifting them back up to standard ONNX IR to run on non-Apple web environments.
- **WebNN Integration:** Automatically generates WebNN execution hints tailored for Safari's CoreML-backed WebNN implementation, ensuring the generated graphs hit the ANE fast-paths dynamically in the browser.

---

## Exhaustive Implementation Checklist

### Phase 1: CoreML Schema & Serialization (Web-Native)

- [x] 1. Implement CoreML protobuf parser in pure TypeScript/WASM.
- [x] 2. Implement CoreML protobuf emitter in pure TypeScript/WASM.
- [x] 3. Support `Model` spec version 1 to 7 compatibility flags.
- [x] 4. Define `NeuralNetwork` schema structure.
- [x] 5. Define `NeuralNetworkBuilder` schema structure.
- [x] 6. Define `MILSpec.Program` schema structure (CoreML v4+).
- [x] 7. Define `MILSpec.Function` schema structure.
- [x] 8. Define `MILSpec.Block` schema structure.
- [x] 9. Implement JSON-to-Protobuf serialization for CoreML definitions.
- [x] 10. Implement `.mlmodel` flat file generator.
- [x] 11. Implement `.mlpackage` directory structure generator.
- [x] 12. Utilize JSZip (or WASM equivalent) to package the `.mlpackage` in the browser.
- [x] 13. Generate `Manifest.json` for `.mlpackage`.
- [x] 14. Generate `FeatureDescriptions.json` for `.mlpackage`.
- [x] 15. Implement `weights/weight.bin` external data writer.
- [x] 16. Handle chunked writing for `weight.bin` exceeding browser memory limits.
- [x] 17. Define `FeatureType` representations (Int64, Double, String, Image, MultiArray).
- [x] 18. Implement `ImageFeatureType` mapping (RGB, BGR, Grayscale).
- [x] 19. Implement `DictionaryFeatureType` mapping.
- [x] 20. Implement `SequenceFeatureType` mapping.
- [x] 21. Parse and preserve model metadata (author, license, description).
- [x] 22. Serialize user-defined metadata dictionaries.
- [x] 23. Implement a linter validating the generated CoreML protobuf against Apple's strict schema.
- [x] 24. Expose an API to read metadata from an existing `.mlmodel` without parsing weights.
- [x] 25. Support updating model metadata natively in the browser and re-exporting.

### Phase 2: Model Intermediate Language (MIL) AST

- [x] 26. Define base `mil.Operation` AST node.
- [x] 27. Define base `mil.Value` and `mil.Var` nodes.
- [x] 28. Implement MIL type system (`mil.type.tensor`, `mil.type.scalar`, `mil.type.tuple`).
- [x] 29. Implement `mil.type.fp16`, `mil.type.fp32`, `mil.type.int32`, `mil.type.bool`.
- [x] 30. Implement `mil.Builder` class for programmatic MIL graph construction.
- [x] 31. Implement `mil.Program` and `mil.Function` containers.
- [x] 32. Implement graph topological sort utility for MIL operations.
- [x] 33. Implement MIL constant folding optimization pass.
- [x] 34. Implement MIL dead code elimination (DCE) pass.
- [x] 35. Implement MIL common subexpression elimination (CSE).
- [x] 36. Implement namespace isolation for MIL variables to prevent collision.
- [x] 37. Create the ONNX-to-MIL high-level translation loop.
- [x] 38. Map ONNX input tensors to MIL function inputs.
- [x] 39. Map ONNX initializers to MIL `const` operations.
- [x] 40. Map ONNX output tensors to MIL function outputs.
- [x] 41. Implement shape inference within the MIL AST to resolve dynamic boundaries.
- [x] 42. Translate ONNX dynamic axes to MIL symbolic variables (e.g., `isize1`).
- [x] 43. Handle type casting implicitly if ONNX types (e.g., int64) aren't supported natively by MIL.
- [x] 44. Implement a textual printer for MIL AST debugging (similar to Apple's PyMIL).
- [x] 45. Implement AST node replacement utilities for graph rewriting.
- [x] 46. Validate MIL AST prior to lowering to Protobuf.
- [x] 47. Track original ONNX node names in MIL metadata for debugging traceability.
- [x] 48. Implement loop/conditional block unwrapping into standard MIL execution flow.
- [x] 49. Establish a specific intermediate dialect `onnx9000.apple_ane` for Neural Engine specifics.
- [x] 50. Implement a topological verifier checking against acyclic directed graph rules.

### Phase 3: Unary & Binary Arithmetic Translation (ONNX -> MIL)

- [x] 51. Map ONNX `Add` to MIL `add`.
- [x] 52. Map ONNX `Sub` to MIL `sub`.
- [x] 53. Map ONNX `Mul` to MIL `mul`.
- [x] 54. Map ONNX `Div` to MIL `real_div` / `floor_div`.
- [x] 55. Map ONNX `Pow` to MIL `pow`.
- [x] 56. Map ONNX `Abs` to MIL `abs`.
- [x] 57. Map ONNX `Ceil` to MIL `ceil`.
- [x] 58. Map ONNX `Floor` to MIL `floor`.
- [x] 59. Map ONNX `Round` to MIL `round`.
- [x] 60. Map ONNX `Exp` to MIL `exp`.
- [x] 61. Map ONNX `Log` to MIL `log`.
- [x] 62. Map ONNX `Sqrt` to MIL `sqrt`.
- [x] 63. Map ONNX `Sin` to MIL `sin`.
- [x] 64. Map ONNX `Cos` to MIL `cos`.
- [x] 65. Map ONNX `Tan` to MIL `tan`.
- [x] 66. Map ONNX `Asin` to MIL `asin`.
- [x] 67. Map ONNX `Acos` to MIL `acos`.
- [x] 68. Map ONNX `Atan` to MIL `atan`.
- [x] 69. Map ONNX `Sign` to MIL `sign`.
- [x] 70. Map ONNX `Mod` to MIL `mod`.
- [x] 71. Map ONNX `Max` to MIL `maximum`.
- [x] 72. Map ONNX `Min` to MIL `minimum`.
- [x] 73. Map ONNX `Erf` to MIL `erf`.
- [x] 74. Map ONNX `IsNaN` to MIL `isnan`.
- [x] 75. Handle ONNX implicit broadcasting logic mapping to MIL explicit broadcast ops if needed.

### Phase 4: Tensor Manipulation Translation (ONNX -> MIL)

- [x] 76. Map ONNX `Reshape` to MIL `reshape`.
- [x] 77. Map ONNX `Transpose` to MIL `transpose`.
- [x] 78. Map ONNX `Concat` to MIL `concat`.
- [x] 79. Map ONNX `Slice` to MIL `slice_by_index` or `slice_by_size`.
- [x] 80. Handle dynamic ONNX slice parameters using MIL symbolic bounds.
- [x] 81. Map ONNX `Split` to MIL `split`.
- [x] 82. Map ONNX `Squeeze` to MIL `squeeze`.
- [x] 83. Map ONNX `Unsqueeze` to MIL `expand_dims`.
- [x] 84. Map ONNX `Gather` to MIL `gather`.
- [x] 85. Map ONNX `GatherElements` to MIL `gather_along_axis`.
- [x] 86. Map ONNX `GatherND` to MIL `gather_nd`.
- [x] 87. Map ONNX `Scatter` to MIL `scatter`.
- [x] 88. Map ONNX `ScatterElements` to MIL `scatter_along_axis`.
- [x] 89. Map ONNX `ScatterND` to MIL `scatter_nd`.
- [x] 90. Map ONNX `Tile` to MIL `tile`.
- [x] 91. Map ONNX `Pad` to MIL `pad`.
- [x] 92. Handle `constant` padding mode in MIL.
- [x] 93. Handle `reflect` padding mode in MIL.
- [x] 94. Handle `edge` padding mode in MIL.
- [x] 95. Map ONNX `Expand` to MIL `broadcast_to`.
- [x] 96. Map ONNX `Shape` to MIL `shape`.
- [x] 97. Map ONNX `Size` to MIL `size`.
- [x] 98. Map ONNX `Cast` to MIL `cast`.
- [x] 99. Identify unsupported ONNX type casting (e.g., float64) and insert warning traces.
- [x] 100. Resolve negative axes indexing natively within the MIL translation.

### Phase 5: Neural Network Layers Translation (ONNX -> MIL)

- [x] 101. Map ONNX `Conv` (1D/2D/3D) to MIL `conv`.
- [x] 102. Handle convolution `dilations` translation.
- [x] 103. Handle convolution `strides` translation.
- [x] 104. Handle depthwise convolution via `groups` parameter.
- [x] 105. Handle `auto_pad` string matching in MIL.
- [x] 106. Map ONNX `ConvTranspose` to MIL `conv_transpose`.
- [x] 107. Map ONNX `MaxPool` to MIL `max_pool`.
- [x] 108. Map ONNX `AveragePool` to MIL `avg_pool`.
- [x] 109. Map ONNX `GlobalMaxPool` to MIL `global_max_pool` (or pool with full spatial kernel).
- [x] 110. Map ONNX `GlobalAveragePool` to MIL `global_avg_pool` (or pool with full spatial kernel).
- [x] 111. Map ONNX `BatchNormalization` to MIL `batch_norm`.
- [x] 112. Map ONNX `InstanceNormalization` to MIL `instance_norm`.
- [x] 113. Map ONNX `LayerNormalization` to MIL `layer_norm`.
- [x] 114. Parse and apply epsilon values correctly across all norm layers.
- [x] 115. Map ONNX `Dropout` to a MIL Identity (since CoreML export is inference-only).
- [x] 116. Map ONNX `MatMul` to MIL `matmul`.
- [x] 117. Map ONNX `Gemm` to MIL `linear` (fusing alpha/beta/bias directly).
- [x] 118. Implement padding conversions (ONNX specific spatial pads to MIL format).
- [x] 119. Handle asymmetric padding safely within MIL convolution parameters.
- [x] 120. Emulate ONNX `LocalResponseNormalization` (LRN) if native MIL op varies.
- [x] 121. Emulate ONNX `MaxUnpool` utilizing indices from previous MaxPool operations.
- [x] 122. Emulate `DepthToSpace` via MIL `pixel_shuffle`.
- [x] 123. Emulate `SpaceToDepth` via MIL reshape and transpose sequences.
- [x] 124. Map ONNX `Resize` to MIL `resize_bilinear` or `resize_nearest_neighbor`.
- [x] 125. Parse coordinate transformation modes (`align_corners`, `half_pixel`) for `Resize`.

### Phase 6: Activations & Reduction Ops Translation (ONNX -> MIL)

- [x] 126. Map ONNX `Relu` to MIL `relu`.
- [x] 127. Map ONNX `LeakyRelu` to MIL `leaky_relu`.
- [x] 128. Map ONNX `Sigmoid` to MIL `sigmoid`.
- [x] 129. Map ONNX `Tanh` to MIL `tanh`.
- [x] 130. Map ONNX `Softmax` to MIL `softmax`.
- [x] 131. Map ONNX `LogSoftmax` to MIL `log_softmax`.
- [x] 132. Map ONNX `Elu` to MIL `elu`.
- [x] 133. Map ONNX `HardSigmoid` to MIL `hard_sigmoid`.
- [x] 134. Map ONNX `Softplus` to MIL `softplus`.
- [x] 135. Map ONNX `Softsign` to MIL `softsign`.
- [x] 136. Map ONNX `PRelu` to MIL `prelu`.
- [x] 137. Map ONNX `Gelu` to MIL `gelu`.
- [x] 138. Map ONNX `Clip` to MIL `clip`.
- [x] 139. Map ONNX `ReduceMean` to MIL `reduce_mean`.
- [x] 140. Map ONNX `ReduceSum` to MIL `reduce_sum`.
- [x] 141. Map ONNX `ReduceMax` to MIL `reduce_max`.
- [x] 142. Map ONNX `ReduceMin` to MIL `reduce_min`.
- [x] 143. Map ONNX `ReduceProd` to MIL `reduce_prod`.
- [x] 144. Map ONNX `ReduceLogSumExp` to MIL `reduce_log_sum_exp`.
- [x] 145. Map ONNX `ArgMax` to MIL `argmax`.
- [x] 146. Map ONNX `ArgMin` to MIL `argmin`.
- [x] 147. Map ONNX `NonMaxSuppression` (NMS) to MIL NMS implementations.
- [x] 148. Map ONNX `TopK` to MIL `topk`.
- [x] 149. Map ONNX `NonZero` to MIL `non_zero`.
- [x] 150. Handle default `keepdims` behaviors between ONNX and MIL properly.

### Phase 7: Control Flow, Logicals, and RNNs (ONNX -> MIL)

- [x] 151. Map ONNX `Equal` to MIL `equal`.
- [x] 152. Map ONNX `Greater` to MIL `greater`.
- [x] 153. Map ONNX `GreaterOrEqual` to MIL `greater_equal`.
- [x] 154. Map ONNX `Less` to MIL `less`.
- [x] 155. Map ONNX `LessOrEqual` to MIL `less_equal`.
- [x] 156. Map ONNX `Not` to MIL `logical_not`.
- [x] 157. Map ONNX `And` to MIL `logical_and`.
- [x] 158. Map ONNX `Or` to MIL `logical_or`.
- [x] 159. Map ONNX `Xor` to MIL `logical_xor`.
- [x] 160. Map ONNX `Where` to MIL `select`.
- [x] 161. Map ONNX `If` to MIL `cond`.
- [x] 162. Map ONNX `Loop` to MIL `while_loop`.
- [x] 163. Map ONNX `LSTM` to MIL `lstm`.
- [x] 164. Parse LSTM direction (forward, reverse, bidirectional).
- [x] 165. Implement state tracking for LSTM hidden variables.
- [x] 166. Map ONNX `GRU` to MIL `gru`.
- [x] 167. Parse GRU sequence layouts cleanly.
- [x] 168. Map ONNX `RNN` to MIL `rnn`.
- [x] 169. Support extraction of multiple outputs from RNN layers.
- [x] 170. Handle static unrolling of loops if MIL dynamic control flow is unsupported in earlier spec versions.
- [x] 171. Provide warning traces for control flow conversion potentially impacting ANE performance.
- [x] 172. Implement `is_inf` and `is_nan` boolean mapping.
- [x] 173. Handle ONNX `Scan` operation by unrolling it dynamically into the MIL AST.
- [x] 174. Manage scope variables properly across MIL block boundaries.
- [x] 175. Verify acyclic flow after parsing nested subgraphs from ONNX `If`/`Loop`.

### Phase 8: Apple Neural Engine (ANE) Specific Optimizations

- [x] 176. Identify and rewrite MatMul sequences into 1x1 Convolutions for ANE acceleration.
- [x] 177. Pad hidden dimensions to multiples of 64 or 32 specifically to satisfy ANE lane requirements.
- [x] 178. Split massive convolutions (e.g., > 16384 channels) into smaller concatenated blocks to avoid ANE fallback to GPU.
- [x] 179. Fuse sequence of `Split` -> `Concat` operations out of the graph if they cancel each other out.
- [x] 180. Fuse `Slice` operations with adjacent `Pad` operations.
- [x] 181. Optimize out Gather operations that index into static constants (pre-computing the gather).
- [x] 182. Replace Swish/SiLU activations with ANE-friendly approximations if requested.
- [x] 183. Identify LayerNorms and rewrite them into `reduce_mean`, `sub`, `pow`, `add` if explicit layer_norm causes GPU fallback on older iOS devices.
- [x] 184. Implement an explicit ANE compatibility checker pass before finalizing the MIL AST.
- [x] 185. Rewrite `Einsum` into explicit Transpose + MatMul + Reshape chains natively.
- [x] 186. Pre-transpose weight constants offline to match the expected format for ANE.
- [x] 187. Ensure 5D and 6D tensors are flattened into 4D or lower, as ANE historically struggles with higher rank shapes.
- [x] 188. Force CAST inputs to FP16 since ANE operates almost exclusively in FP16 precision.
- [x] 189. Map standard Transformers Attention into the specific `scaled_dot_product_attention` MIL op (requires CoreML v7/iOS 17).
- [x] 190. Eliminate redundant `Cast` (FP32 -> FP16 -> FP32) boundaries.

### Phase 9: Compression & Quantization (`coremltools.optimize`)

- [x] 191. Implement FP16 casting pass for all weights and biases.
- [x] 192. Implement Palettization compression (k-means clustering of weights).
- [x] 193. Encode LUT (Look-Up Table) weights natively into the CoreML `.weight.bin` format.
- [x] 194. Implement INT8 Weight Quantization (W8A16) natively generating CoreML `constexpr_affine_dequantize`.
- [x] 195. Implement INT4 Weight Quantization (W4A16) specific to CoreML iOS 17 features.
- [x] 196. Implement sparse weight compression (storing non-zero values and bitmasks).
- [x] 197. Support block-wise quantization grouping (e.g., group_size = 32 or 128).
- [x] 198. Allow defining a mixed-precision configuration dictionary per layer.
- [x] 199. Map existing ONNX `QuantizeLinear`/`DequantizeLinear` pairs directly to CoreML quantized weight representations.
- [x] 200. Execute dynamic quantization statistics gathering natively in JS if an un-quantized model needs compression.
- [x] 201. Support Joint-Data-Algorithm (JDA) for pruning.
- [x] 202. Generate a compression report tracking memory reduction per layer.
- [x] 203. Export multi-bitrate weights allowing the Apple OS to select the precision dynamically.
- [x] 204. Handle specific iOS 17 stateful KV Cache quantization mappings.
- [x] 205. Implement decompression validation ensuring the unpacked LUT exactly matches expected logic.

### Phase 10: Input/Output Formatting & iOS Integration

- [x] 206. Support explicitly defining inputs as `ImageType` rather than generic `MultiArray`.
- [x] 207. Define image scaling properties (`blueBias`, `greenBias`, `redBias`, `imageScale`).
- [x] 208. Parse ONNX Vision transforms to automatically inject image bias metadata.
- [x] 209. Map generic integer array outputs to specific iOS `DictionaryType` for classification tasks.
- [x] 210. Generate the standard Core ML Class Labels file based on ONNX attributes or separate text input.
- [x] 211. Inject custom Vision Framework descriptions directly into the generated MLModel metadata.
- [x] 212. Provide configurable outputs mapping (renaming ONNX generic names to Swift-friendly camelCase variables).
- [x] 213. Support defining specific input sequences as `SequenceType` for CoreML RNN wrappers.
- [x] 214. Embed custom vocabulary files inside the `.mlpackage` for internal tokenization usage.
- [x] 215. Configure the generated package to utilize `computeUnits = .all` explicitly by default.

### Phase 11: Bi-Directional Conversion (CoreML -> ONNX)

- [x] 216. Implement `.mlmodel` and `.mlpackage` loader/unzipper in JS.
- [x] 217. Parse `MILSpec.Program` back into the TypeScript AST representation.
- [x] 218. Parse Apple NeuralNetwork V1-V3 layers (legacy protobuf) into the AST representation.
- [x] 219. Inverse Map: MIL `conv` to ONNX `Conv`.
- [x] 220. Inverse Map: MIL `matmul` / `linear` to ONNX `MatMul` / `Gemm`.
- [x] 221. Inverse Map: MIL `scaled_dot_product_attention` to explicit ONNX Subgraph (MatMul, Div, Softmax, MatMul).
- [x] 222. Extract `weight.bin` packed data back into ONNX Float32 / Float16 Initializers.
- [x] 223. Dequantize CoreML INT4/INT8 palettized weights statically during the extraction to ONNX.
- [x] 224. Rebuild the ONNX standard inputs/outputs definitions from the CoreML `FeatureDescription`.
- [x] 225. Handle Swift/Apple specific renaming back to standard ONNX tensor naming conventions.
- [x] 226. Produce a standard valid `model.onnx` payload.
- [x] 227. Create a visual diff checker comparing the original ONNX vs the Round-Trip ONNX.

### Phase 12: GenAI & Stateful Models (iOS 18 / CoreML v8 Spec Prep)

- [x] 228. Implement the newer Core ML Stateful operations mapping (`mil.state`).
- [x] 229. Translate ONNX Runtime GenAI KV Cache patterns directly into CoreML state variables.
- [x] 230. Map explicit KV-cache ring buffer updates into MIL `read_state` and `write_state`.
- [x] 231. Translate LLaMA / Mistral ONNX topologies into stateful MLPackages natively.
- [x] 232. Support exporting models with `Stateful=True` flags.
- [x] 233. Generate appropriate Swift/Objective-C boilerplate text for utilizing the generated stateful model.
- [x] 234. Map Whisper architectures efficiently to CoreML using specialized ANE-friendly audio layers.
- [x] 235. Map Stable Diffusion UNets to CoreML natively, bypassing standard `python-coremltools` pipelines.

### Phase 13: Browser CLI & Execution Environment

- [x] 236. Add CLI command: `onnx9000 coreml export <model.onnx>`.
- [x] 237. Add CLI command: `onnx9000 coreml import <model.mlpackage>`.
- [x] 238. Provide Node.js API: `import { convertToCoreML } from 'onnx9000/coreml'`.
- [x] 239. Enable streaming conversion for files larger than 2GB (bypassing V8 array limits).
- [x] 240. Implement Web Worker distribution for MIL AST optimization passes.
- [x] 241. Provide a UI component: "Drop ONNX -> Get CoreML".
- [x] 242. Display progress bars parsing protobufs natively in the browser.
- [x] 243. Provide WebNN hint generation for macOS Safari utilizing the `coreml` WebNN backend.
- [x] 244. Create debugging logs showing exactly which layers were offloaded to ANE vs GPU vs CPU (via simulation heuristics).
- [x] 245. Establish memory bounds checking inside the WASM exporter to prevent browser tab crashes.

### Phase 14: Quality Assurance & Parity Testing

- [x] 246. Validate complete conversion of ResNet50 (ONNX -> CoreML).
- [x] 247. Validate complete conversion of MobileNetV2 (ONNX -> CoreML).
- [x] 248. Validate complete conversion of YOLOv8 (ONNX -> CoreML).
- [x] 249. Validate complete conversion of BERT (ONNX -> CoreML).
- [x] 250. Validate complete conversion of GPT-2 (ONNX -> CoreML).
- [x] 251. Validate complete conversion of Whisper-Tiny (ONNX -> CoreML).
- [x] 252. Extract test outputs from native `coremltools` Python and compare exact tensor outputs.
- [x] 253. Measure and ensure that generated `.mlpackage` sizes are within 1% of the Python equivalent.
- [x] 254. Run automated tests ensuring all generated files successfully load in Xcode without validation errors.
- [x] 255. Verify output differences of Palettized exports are mathematically acceptable (Cosine Similarity > 0.99).
- [x] 256. Automate iOS simulator execution checking for the generated packages (using external CI/CD wrappers).
- [x] 257. Verify that image classification labels are properly surfaced in macOS Quick Look.

### Phase 15: Edge Case & Exception Handling

- [x] 258. Handle parsing of completely unsupported ONNX ops (e.g., custom user ops) by generating clear error traces.
- [x] 259. Fallback cleanly if an ONNX model utilizes double precision (`float64`), forcibly downcasting to `float32`.
- [x] 260. Manage ONNX `If`/`Loop` constructs that contain operations incompatible with MIL control flow.
- [x] 261. Detect and warn users if their dynamic axes definitions exceed Apple's supported dimension limits.
- [x] 262. Warn users explicitly if a specific graph topology is known to trigger ANE thermal throttling.
- [x] 263. Catch and handle Protobuf decode failures gracefully in the browser.
- [x] 264. Ensure generated filenames for weights inside `.mlpackage` contain no illegal characters.
- [x] 265. Strip `\0` null terminators and non-UTF8 characters from ONNX metadata before serializing to CoreML JSON.

### Phase 16: Ecosystem & Native Integration

- [x] 266. Enable importing HuggingFace models seamlessly: `onnx9000 coreml export hf://gpt2`.
- [x] 267. Map Hub models to CoreML instantly using cached ONNX graphs internally.
- [x] 268. Produce an equivalent to the `coremltools` Python API natively in TypeScript.
- [x] 269. Support exporting multiple models into a single Pipeline `.mlmodelc` natively.
- [x] 270. Create custom Xcode playground templates alongside the exported model.
- [x] 271. Hook into `onnx9000.optimum` to share quantization logic directly with `onnx9000.coreml`.
- [x] 272. Build integration examples showing the `.mlpackage` running in Swift CoreML.
- [x] 273. Provide a Native WebViews component demonstrating inference on the generated file.

### Phase 17: Deep CoreML v6/v7 Optimization Passes

- [x] 274. Implement MIL `constexpr` folding dynamically to resolve constants during the AST building.
- [x] 275. Translate ONNX `Einsum` into explicit tensor ops, bypassing ANE issues with native einsum mappings.
- [x] 276. Identify `LayerNormalization` acting on the last dimension and utilize specific CoreML accelerated primitives.
- [x] 277. Rewrite 1D convolutions into 2D convolutions (with height=1) as ANE highly prefers 2D topologies.
- [x] 278. Rewrite all grouped convolutions into explicit slices/concats if targeting older iOS versions via backwards compatibility flags.
- [x] 279. Support explicit definition of the "Compute Precision" (Float16 vs Float32) for the entire package.
- [x] 280. Extract dynamic sequence padding operations to CPU boundaries to prevent ANE pipeline stalling.

### Phase 18: Security & Sandbox Execution

- [x] 281. Sandbox the JSZip/Archive generation to ensure no cross-site scripting attacks via malicious model metadata.
- [x] 282. Prevent local file-system access (except via standard Browser API prompts) during export.
- [x] 283. Verify that parsing `.mlmodel` files doesn't trigger JS prototype pollution.
- [x] 284. Handle extremely large dimensional definitions (e.g., trying to allocate 100GB tensors) safely without crashing the V8 engine.
- [x] 285. Utilize Subresource Integrity (SRI) on all remote script loading inside the exported HTML demos.

### Phase 19: Documentation & Profiling

- [x] 286. Provide comprehensive API documentation mapping `python-coremltools` functions to their TS equivalents.
- [x] 287. Publish a migration guide: "Moving from CoreMLTools to `onnx9000`".
- [x] 288. Generate a summary table during export showing exactly which layers are structurally modified.
- [x] 289. Develop a mock "Profiler" simulating ANE vs GPU time based on layer topologies in the browser.
- [x] 290. Provide visual diffing of the ONNX graph vs the generated MIL graph inside the web UI.

### Phase 20: Final Polish and Release Readiness

- [x] 291. Implement dynamic graph batching (converting a single-batch ONNX to a multi-batch CoreML package).
- [x] 292. Add support for specialized Apple Vision Pro (visionOS) deployment targets inside the metadata.
- [x] 293. Build a WASM fallback for reading/writing Apple's compiled `.mlmodelc` binary directories directly.
- [x] 294. Ensure full deterministic compilation (same ONNX + same settings = exactly identical byte output for `.mlpackage`).
- [x] 295. Add strict linter enforcing that no proprietary/undocumented Apple MIL opcodes are generated unless explicitly requested.
- [x] 296. Resolve all Typescript strict-mode typing errors inside the CoreML generation logic.
- [x] 297. Configure automated CI checks running against Xcode command-line tools `coremlcompiler`.
- [x] 298. Establish telemetry for recording which ONNX operators fail to translate most frequently.
- [x] 299. Write comprehensive tutorial: "Bringing ONNX LLMs to iOS using `onnx9000`".
- [x] 300. Release v1.0 complete feature parity certification matching official Apple `coremltools`.
