# ONNX28: keras2onnx & tfjs-to-onnx (Web-Native Keras Converter)

## Original Project Description

`keras2onnx` (and its underlying dependencies like `tf2onnx`) is a Python-based conversion tool that translates Keras and TensorFlow models into the standard ONNX format. It parses Keras `.h5`, `.keras`, or SavedModel files, extracting the computational graph and weight tensors, and meticulously maps Keras layer semantics (which default to NHWC layout) into ONNX operator semantics (which default to NCHW layout). It requires a heavy, full-scale Python installation with TensorFlow and ONNX pip packages installed.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.keras` eliminates the Python dependency completely, offering a **pure TypeScript/WebAssembly converter**.

- **Browser-Based Conversion:** Developers can drag and drop a Keras `.h5` file or a TensorFlow.js `model.json` directly into the browser, and `onnx9000` will output a `.onnx` file instantly, strictly client-side.
- **Dual-Format Ingestion:** Natively parses both Keras HDF5 (`.h5`) via a pure-JS HDF5 reader, and TensorFlow.js formats (`LayersModel` and `GraphModel`), providing an automatic bridge from the JS ecosystem to ONNX.
- **Zero-Copy Weight Transposition:** Translating Keras weight layouts (e.g., `[H, W, In, Out]`) to ONNX layouts (`[Out, In, H, W]`) is performed by highly optimized WASM kernels to prevent browser tab crashes on massive models.
- **Direct Execution Pipeline:** Converted models can be instantly routed into the `onnx9000` execution backend (WebGPU/WASM), allowing a user to `await onnx9000.keras.load('model.json')` and run it as if it were a native ONNX model.

---

## Exhaustive Implementation Checklist

### Phase 1: Ingestion & Format Parsing (TF.js & Keras H5)

- [x] 1. Implement `tfjs.LayersModel` (`model.json`) JSON schema parser.
- [x] 2. Implement `tfjs.GraphModel` JSON schema parser.
- [x] 3. Implement external binary weight shard downloader/parser for TF.js.
- [x] 4. Combine chunked `.bin` shards into a contiguous ArrayBuffer.
- [x] 5. Map TF.js weight manifests to specific layer variables.
- [x] 6. Implement pure-JS HDF5 (`.h5`) file reader.
- [x] 7. Parse Keras `model_config` JSON strings embedded within HDF5 files.
- [x] 8. Extract layer weights sequentially from HDF5 datasets.
- [x] 9. Implement parser for the newer Keras 3 `.keras` zip-based format.
- [x] 10. Support reading from local `File`/`Blob` objects in the browser.
- [x] 11. Support fetching from remote URLs (with CORS handling).
- [x] 12. Extract input specifications (shapes, names, types) from Keras config.
- [x] 13. Extract output specifications from Keras config.
- [x] 14. Identify multi-input / multi-output model topologies.
- [x] 15. Build an internal abstract graph of Keras layers before ONNX translation.

### Phase 2: Core Layout Translation Engine (NHWC to NCHW)

- [x] 16. Build the NHWC (Channels Last) to NCHW (Channels First) shape translator.
- [x] 17. Build the `onnx9000` Transpose WASM kernel for 4D Image weights (Conv2D: `[H, W, I, O]` -> `[O, I, H, W]`).
- [x] 18. Build the Transpose WASM kernel for 3D Sequence weights.
- [x] 19. Build the Transpose WASM kernel for 5D Video weights (Conv3D).
- [x] 20. Transpose Keras `Dense` weights (`[In, Out]` -> `[Out, In]`) where ONNX requires explicit GEMM mapping.
- [x] 21. Track layout states dynamically (inserting ONNX `Transpose` ops dynamically if a Keras layer explicitly assumes NHWC data).
- [x] 22. Implement a layout optimizer pass to remove redundant `Transpose` (e.g., `NCHW->NHWC->NCHW` collapses).
- [x] 23. Handle Keras `data_format="channels_first"` layers gracefully (bypassing transpose injection).
- [x] 24. Resolve Spatial padding discrepancies between Keras and ONNX.
- [x] 25. Convert explicit TF `padding="SAME"` behavior to explicit ONNX padding values.
- [x] 26. Convert explicit TF `padding="VALID"` behavior to ONNX padding values.

### Phase 3: Core Keras Layers Mapping (ONNX Emitters)

- [x] 27. Map `InputLayer` to ONNX Graph Inputs.
- [x] 28. Map `Dense` to ONNX `MatMul` + `Add` (Bias) or `Gemm`.
- [x] 29. Extract `Dense` activation and append matching ONNX activation node.
- [x] 30. Map `Activation` layer directly.
- [x] 31. Map `ReLU` activation.
- [x] 32. Map `Softmax` activation (handling axis conversions).
- [x] 33. Map `LeakyReLU` activation.
- [x] 34. Map `PReLU` activation (handling shared axes constraints).
- [x] 35. Map `ELU` activation.
- [x] 36. Map `ThresholdedReLU` activation.
- [x] 37. Map `Softplus` activation.
- [x] 38. Map `Softsign` activation.
- [x] 39. Map `HardSigmoid` activation.
- [x] 40. Map `Swish` / `SiLU` activation.
- [x] 41. Map `GELU` activation (handling approx vs exact flags).
- [x] 42. Map `Dropout` to ONNX `Identity` (or drop entirely for inference).
- [x] 43. Map `SpatialDropout1D`, `SpatialDropout2D`, `SpatialDropout3D` to `Identity`.
- [x] 44. Map `GaussianDropout` to `Identity`.
- [x] 45. Map `GaussianNoise` to `Identity`.
- [x] 46. Map `ActivityRegularization` to `Identity`.
- [x] 47. Map `AlphaDropout` to `Identity`.

### Phase 4: Convolutional Layers Mapping

- [x] 48. Map `Conv1D` to ONNX `Conv`.
- [x] 49. Map `Conv2D` to ONNX `Conv`.
- [x] 50. Map `Conv3D` to ONNX `Conv`.
- [x] 51. Parse and apply `strides` tuple to ONNX.
- [x] 52. Parse and apply `dilation_rate` tuple to ONNX.
- [x] 53. Parse and apply `groups` attribute.
- [x] 54. Map `SeparableConv1D` to Depthwise `Conv` + Pointwise `Conv`.
- [x] 55. Map `SeparableConv2D` to Depthwise `Conv` + Pointwise `Conv`.
- [x] 56. Map `DepthwiseConv2D` to ONNX `Conv` with `groups = in_channels`.
- [x] 57. Map `Conv1DTranspose` to ONNX `ConvTranspose`.
- [x] 58. Map `Conv2DTranspose` to ONNX `ConvTranspose`.
- [x] 59. Map `Conv3DTranspose` to ONNX `ConvTranspose`.
- [x] 60. Calculate ONNX `output_padding` dynamically to match Keras shape inference for Transpose Convs.

### Phase 5: Pooling Layers Mapping

- [x] 61. Map `MaxPooling1D` to ONNX `MaxPool`.
- [x] 62. Map `MaxPooling2D` to ONNX `MaxPool`.
- [x] 63. Map `MaxPooling3D` to ONNX `MaxPool`.
- [x] 64. Map `AveragePooling1D` to ONNX `AveragePool`.
- [x] 65. Map `AveragePooling2D` to ONNX `AveragePool`.
- [x] 66. Map `AveragePooling3D` to ONNX `AveragePool`.
- [x] 67. Map `GlobalMaxPooling1D` to ONNX `GlobalMaxPool`.
- [x] 68. Map `GlobalMaxPooling2D` to ONNX `GlobalMaxPool`.
- [x] 69. Map `GlobalMaxPooling3D` to ONNX `GlobalMaxPool`.
- [x] 70. Map `GlobalAveragePooling1D` to ONNX `GlobalAveragePool`.
- [x] 71. Map `GlobalAveragePooling2D` to ONNX `GlobalAveragePool`.
- [x] 72. Map `GlobalAveragePooling3D` to ONNX `GlobalAveragePool`.
- [x] 73. Handle Keras `keepdims=False` (default in GlobalPools) by inserting ONNX `Squeeze`.

### Phase 6: Recurrent Layers (RNN/LSTM/GRU) Mapping

- [x] 74. Map `SimpleRNN` to ONNX `RNN`.
- [x] 75. Transpose and pack Keras RNN weights (`kernel`, `recurrent_kernel`, `bias`) into ONNX RNN combined weights `W` and `R`.
- [x] 76. Handle Keras `return_sequences=True` (outputting full sequence).
- [x] 77. Handle Keras `return_sequences=False` (outputting last state, slicing ONNX output).
- [x] 78. Handle Keras `return_state=True` (outputting hidden states).
- [x] 79. Map `LSTM` to ONNX `LSTM`.
- [x] 80. Convert Keras LSTM weight gate order (i, f, c, o) to ONNX LSTM gate order (i, o, f, c).
- [x] 81. Map `GRU` to ONNX `GRU`.
- [x] 82. Convert Keras GRU weight gate order (z, r, h) to ONNX GRU gate order (z, r, h).
- [x] 83. Handle GRU `reset_after` flag (mapping to linear_before_reset in ONNX).
- [x] 84. Map `Bidirectional` wrapper for RNN/LSTM/GRU.
- [x] 85. Combine forward and backward Keras weights into ONNX multi-directional weights.
- [x] 86. Implement `merge_mode='concat'` for Bidirectional outputs.
- [x] 87. Implement `merge_mode='sum'` for Bidirectional outputs.
- [x] 88. Implement `merge_mode='mul'` for Bidirectional outputs.
- [x] 89. Implement `merge_mode='ave'` for Bidirectional outputs.
- [x] 90. Handle initial state inputs securely for stateful sequence models.

### Phase 7: Merge Layers Mapping

- [x] 91. Map `Add` to ONNX `Add` (with multi-input accumulation).
- [x] 92. Map `Subtract` to ONNX `Sub`.
- [x] 93. Map `Multiply` to ONNX `Mul` (with multi-input accumulation).
- [x] 94. Map `Average` to ONNX `Mean`.
- [x] 95. Map `Maximum` to ONNX `Max`.
- [x] 96. Map `Minimum` to ONNX `Min`.
- [x] 97. Map `Concatenate` to ONNX `Concat`.
- [x] 98. Resolve negative `axis` properly for `Concatenate` within the NHWC -> NCHW translation context.
- [x] 99. Map `Dot` to ONNX `MatMul` (handling explicit axes parameters via Transpose injections).
- [x] 100. Handle implicit broadcasting differences between Keras and ONNX during merge operations.

### Phase 8: Advanced & Attention Layers

- [x] 101. Map `Attention` to explicit ONNX Subgraph (MatMul + Softmax + MatMul).
- [x] 102. Handle causal masks dynamically inside `Attention` mapping.
- [x] 103. Map `AdditiveAttention` (Bahdanau) to ONNX explicit Ops.
- [x] 104. Map Keras 3 `MultiHeadAttention` to ONNX explicitly (or map to specific ONNX `Attention` op if supported by opset).
- [x] 105. Split multi-head weights out of the Keras dense representations.
- [x] 106. Handle `use_causal_mask` for MHA.
- [x] 107. Map Keras `Embedding` layer to ONNX `Gather`.
- [x] 108. Support `mask_zero=True` in `Embedding` by emitting an explicit boolean mask output.
- [x] 109. Map `ConvLSTM1D` to explicit sequence of Conv + LSTM logic.
- [x] 110. Map `ConvLSTM2D` to explicit sequence.
- [x] 111. Map `ConvLSTM3D` to explicit sequence.
- [x] 112. Map Keras `TimeDistributed` wrapper by reshaping `[batch, time, ...]` -> `[batch * time, ...]` -> Apply Layer -> Reshape back.

### Phase 9: Normalization & Reshaping Layers

- [x] 113. Map `BatchNormalization` to ONNX `BatchNormalization`.
- [x] 114. Extract moving mean, moving variance, beta, and gamma.
- [x] 115. Map `LayerNormalization` to ONNX `LayerNormalization` or `ReduceMean`->`Sub`->`Pow`->`Add`->`Div` if ONNX opset is too low.
- [x] 116. Handle `axis` mapping for `LayerNormalization`.
- [x] 117. Map `UnitNormalization` to ONNX `LpNormalization`.
- [x] 118. Map `GroupNormalization` to ONNX standard operations.
- [x] 119. Map `Reshape` to ONNX `Reshape`.
- [x] 120. Translate Keras implicit `-1` batch dimension inside `Reshape`.
- [x] 121. Map `Flatten` to ONNX `Flatten`.
- [x] 122. Handle `data_format` correctly inside `Flatten`.
- [x] 123. Map `RepeatVector` to ONNX `Expand` or `Tile`.
- [x] 124. Map `Permute` to ONNX `Transpose`.
- [x] 125. Map `ZeroPadding1D` to ONNX `Pad`.
- [x] 126. Map `ZeroPadding2D` to ONNX `Pad`.
- [x] 127. Map `ZeroPadding3D` to ONNX `Pad`.
- [x] 128. Map `Cropping1D` to ONNX `Slice`.
- [x] 129. Map `Cropping2D` to ONNX `Slice`.
- [x] 130. Map `Cropping3D` to ONNX `Slice`.
- [x] 131. Map `UpSampling1D` to ONNX `Resize` (Nearest/Bilinear).
- [x] 132. Map `UpSampling2D` to ONNX `Resize`.
- [x] 133. Map `UpSampling3D` to ONNX `Resize`.

### Phase 10: TF.js GraphModel Specific Ops (tf.\* equivalents)

- [x] 134. Map TF.js `tf.add` to ONNX `Add`.
- [x] 135. Map TF.js `tf.sub` to ONNX `Sub`.
- [x] 136. Map TF.js `tf.mul` to ONNX `Mul`.
- [x] 137. Map TF.js `tf.div` to ONNX `Div`.
- [x] 138. Map TF.js `tf.matMul` to ONNX `MatMul`.
- [x] 139. Map TF.js `tf.square` to ONNX `Pow` (exponent 2).
- [x] 140. Map TF.js `tf.sqrt` to ONNX `Sqrt`.
- [x] 141. Map TF.js `tf.exp` to ONNX `Exp`.
- [x] 142. Map TF.js `tf.log` to ONNX `Log`.
- [x] 143. Map TF.js `tf.maximum` to ONNX `Max`.
- [x] 144. Map TF.js `tf.minimum` to ONNX `Min`.
- [x] 145. Map TF.js `tf.sum` to ONNX `ReduceSum`.
- [x] 146. Map TF.js `tf.mean` to ONNX `ReduceMean`.
- [x] 147. Map TF.js `tf.max` to ONNX `ReduceMax`.
- [x] 148. Map TF.js `tf.min` to ONNX `ReduceMin`.
- [x] 149. Map TF.js `tf.argMax` to ONNX `ArgMax`.
- [x] 150. Map TF.js `tf.argMin` to ONNX `ArgMin`.
- [x] 151. Map TF.js `tf.split` to ONNX `Split`.
- [x] 152. Map TF.js `tf.concat` to ONNX `Concat`.
- [x] 153. Map TF.js `tf.slice` to ONNX `Slice`.
- [x] 154. Map TF.js `tf.stridedSlice` to ONNX `Slice` (translating end masks).
- [x] 155. Map TF.js `tf.gather` to ONNX `Gather`.
- [x] 156. Map TF.js `tf.gatherNd` to ONNX `GatherND`.
- [x] 157. Map TF.js `tf.where` to ONNX `Where`.
- [x] 158. Map TF.js `tf.tensorScatterUpdate` to ONNX `ScatterND`.
- [x] 159. Map TF.js `tf.image.resizeBilinear` to ONNX `Resize`.
- [x] 160. Map TF.js `tf.image.resizeNearestNeighbor` to ONNX `Resize`.

### Phase 11: End-to-End Validation (Vision Architectures)

- [x] 161. Convert and validate TF.js `MobileNetV1`.
- [x] 162. Convert and validate TF.js `MobileNetV2`.
- [x] 163. Convert and validate TF.js `MobileNetV3`.
- [x] 164. Convert and validate Keras `ResNet50`.
- [x] 165. Convert and validate Keras `ResNet101`.
- [x] 166. Convert and validate Keras `InceptionV3`.
- [x] 167. Convert and validate Keras `Xception`.
- [x] 168. Convert and validate Keras `VGG16`.
- [x] 169. Convert and validate Keras `VGG19`.
- [x] 170. Convert and validate Keras `EfficientNetB0` through `B7`.
- [x] 171. Convert and validate Keras `DenseNet121`.
- [x] 172. Convert and validate Keras `NASNetMobile`.
- [x] 173. Convert and validate TF.js `PoseNet`.
- [x] 174. Convert and validate TF.js `BodyPix`.
- [x] 175. Verify 100% equivalent spatial output matrices against native TF.js execution (tolerance 1e-5).

### Phase 12: End-to-End Validation (NLP & Sequence Architectures)

- [x] 176. Convert and validate TF.js `Universal Sentence Encoder` (USE).
- [x] 177. Convert and validate Keras `Transformer` implementation (MultiHeadAttention).
- [x] 178. Convert and validate TF.js `Toxicity` text classifier.
- [x] 179. Convert and validate Keras `LSTM` character-level generator.
- [x] 180. Convert and validate Keras `GRU` sequence-to-sequence model.
- [x] 181. Verify dynamic sequence lengths compile cleanly to ONNX dynamic axes.
- [x] 182. Handle Keras Embedding weights loading properly into ONNX Gather initializers.
- [x] 183. Check precise parity of Bidirectional states outputs against TF.js.

### Phase 13: End-to-End Validation (Generative & Audio)

- [x] 184. Convert and validate Keras `DCGAN` generator.
- [x] 185. Convert and validate Keras `VAE` (Variational Autoencoder) decoding blocks.
- [x] 186. Convert and validate TF.js `SpeechCommands` audio classifier.
- [x] 187. Validate 1D Convolution outputs on raw audio sequences.
- [x] 188. Check 2D Convolution on Mel-spectrogram input formats.
- [x] 189. Validate UpSampling/Conv2DTranspose artifacts match TF.js completely.

### Phase 14: Subgraphs, Custom Layers & Control Flow

- [x] 190. Handle Keras `Lambda` layers. (Provide clear errors if un-translatable Python code is found, skip if JS equivalents exist).
- [x] 191. Attempt to trace JS closures in TF.js GraphModels and map them to ONNX subgraphs.
- [x] 192. Parse TF.js `ControlFlow` ops (`Switch`, `Merge`, `Enter`, `Exit`, `NextIteration`).
- [x] 193. Map TF.js `Loop` constructs to ONNX `Loop`.
- [x] 194. Map TF.js `Cond` constructs to ONNX `If`.
- [x] 195. Implement a registry for users to inject custom JS converters for their proprietary layers.
- [x] 196. Provide `registerConverter('MyCustomLayer', (node, builder) => { ... })` API.
- [x] 197. Handle sub-models (Keras models nested within Keras models) correctly by flattening the graph.
- [x] 198. Extract `keras_version` and `backend` information and embed into ONNX `producer_name`.

### Phase 15: Browser API, UI, and Packaging

- [x] 199. Expose TypeScript library API: `const onnxBytes = await keras2onnx(modelJson, weightsBin)`.
- [x] 200. Build a Node.js CLI: `onnx9000 keras convert my_model.h5 --output my_model.onnx`.
- [x] 201. Support CLI format auto-detection (inferring TF.js vs HDF5 from file signatures).
- [x] 202. Build the visual drag-and-drop web converter interface.
- [x] 203. Provide real-time conversion progress callbacks for UI updates.
- [x] 204. Handle memory-efficient ArrayBuffer transfers using JS `Transferable` objects.
- [x] 205. Validate final generated ONNX Protobuf structure using the internal `onnx9000` linting tool.
- [x] 206. Export the converter logic as an isolated NPM package `@onnx9000/tfjs-converter`.
- [x] 207. Create an automated migration script mapping standard `tfjs-converter` args to `onnx9000`.

### Phase 16: Optimizations & Graph Rewriting

- [x] 208. Implement TF.js explicit `FusedBatchNorm` un-fusing if targeting low ONNX opsets.
- [x] 209. Map TF.js `_FusedConv2D` explicitly to ONNX `Conv` + `Relu` or `Conv` + `Bias` + `Relu`.
- [x] 210. Map TF.js `_FusedMatMul` to ONNX explicitly.
- [x] 211. Remove TF.js explicit `Identity` chains injected by SavedModel builders.
- [x] 212. Resolve static subgraphs (e.g., `Shape` -> `Slice` -> `Concat`) into static initializers to minimize ONNX payload.
- [x] 213. Rewrite explicit channel transposes out of the graph by swapping weight dimension ordering on standard ops.
- [x] 214. Clean up `StopGradient` nodes (removing them entirely as ONNX is inference-only).

### Phase 17: Precision & Quantization

- [x] 215. Parse TF.js `float16` weights natively (handling DataView buffers correctly in JS).
- [x] 216. Parse TF.js `uint8` quantized weights natively.
- [x] 217. Read TF.js quantization scale/min/max metadata and embed into ONNX `DequantizeLinear`.
- [x] 218. Provide an option to cast all weights to `float16` during conversion to save space (`--fp16`).
- [x] 219. Provide an option to perform W8A16 or W4A16 quantization immediately during conversion.
- [x] 220. Ensure `int64` tensors in TF.js are gracefully downcast to `int32` for better WebGPU support down the line.

### Phase 18: Ecosystem Parity & Interoperability

- [x] 221. Establish CI pipeline matching official `tf2onnx` regression tests.
- [x] 222. Maintain exact equivalence with `tf2onnx` opset 13-19 standards.
- [x] 223. Convert HuggingFace standard Keras/TF models dynamically via Hub URLs.
- [x] 224. Support reading `.pb` (Protobuf) TensorFlow SavedModels via a WASM flatbuffer parser.
- [x] 225. Support parsing TensorFlow Hub URLs directly (`https://tfhub.dev/...`).
- [x] 226. Produce ONNX models that perfectly load into standard Python `onnxruntime` (not just `onnx9000`).
- [x] 227. Export a `metadata.json` sidecar preserving Keras training history, class labels, and dictionaries.

### Phase 19: Edge Cases, Quirks, and Telemetry

- [x] 228. Detect and warn on Keras `input_shape` missing dimensions (e.g., completely dynamic models without defined ranks).
- [x] 229. Handle `SpaceToBatchND` and `BatchToSpaceND` operations efficiently (often used in dilated convs).
- [x] 230. Map TF.js `NonMaxSuppressionV3/V4/V5` to ONNX `NonMaxSuppression`.
- [x] 231. Translate YOLO-specific custom darknet layers if represented in TF.js format.
- [x] 232. Handle unsupported opcodes by creating custom domains `ai.onnx.contrib.tfjs`.
- [x] 233. Provide clear error diagnostics displaying the exact node/layer that failed conversion.
- [x] 234. Avoid "Maximum Call Stack Size Exceeded" when traversing TF.js graphs with 10,000+ nodes.
- [x] 235. Track specific operator translation failures and report aggregate telemetry.

### Phase 20: Documentation & Final Delivery

- [x] 236. Create tutorial: "Migrating from TensorFlow.js to WebGPU ONNX in 5 minutes".
- [x] 237. Create tutorial: "Converting Keras H5 models directly in the Browser".
- [x] 238. Write detailed API specs for the TS conversion hooks.
- [x] 239. Include a compatibility matrix mapping Keras Layer versions to supported ONNX opsets.
- [x] 240. Publish an interactive CodeSandbox template integrating the converter.
- [x] 241. Add explicit support for `tf.keras.applications.*` extraction tests.
- [x] 242. Configure memory bounds checking on Web Worker processes to prevent Out Of Memory crashes.
- [x] 243. Add support for multiple ONNX domains natively.
- [x] 244. Implement graph cloning utilities for isolated subgraph testing.
- [x] 245. Track and propagate ONNX type constraints securely during the AST build.
- [x] 246. Ensure strict JSON sanitization for malicious TF.js manifests.
- [x] 247. Validate that dynamic batching (Axis 0) propagates correctly globally.
- [x] 248. Support explicit conversion targets (e.g., optimizing the resulting ONNX specifically for `webnn`).
- [x] 249. Embed the `onnx9000` logo and conversion timestamp inside the generated ONNX metadata.
- [x] 250. Map Keras `.keras` zip structure weights dynamically without unzipping fully to disk (streaming unzip).
- [x] 251. Validate conversion of `tf.einsum` correctly into explicit ONNX math.
- [x] 252. Map `tf.complex64` types (used in audio FFT models) cleanly if opset supports it, or split to real/imaginary floats.
- [x] 253. Translate boolean masking logic (`tf.boolean_mask`) to ONNX explicit `Where` and `NonZero`.
- [x] 254. Ensure `String` tensors in TF.js (e.g., Universal Sentence Encoder text inputs) map perfectly to ONNX STRING inputs.
- [x] 255. Map TF.js `StringSplit` and `StringToHashBucket` to explicit ONNX sequence structures or fallback JS ops.
- [x] 256. Handle `RaggedTensors` (common in modern TF NLP models) by translating to padded dense ONNX tensors dynamically.
- [x] 257. Verify that multiple input layers with varying types (e.g., Image + String) translate safely.
- [x] 258. Support `Keras 3.x` backend-agnostic topologies.
- [x] 259. Map custom loss functions inside `.h5` files into pure ONNX if `--export-training` is specified (future proofing).
- [x] 260. Release final v1.0 parity matching the Python `keras2onnx` capabilities exactly.
- [x] 261. Add testing for TF.js specific quantization (Float16) models.
- [x] 262. Support models with dynamic rank (uncommon but supported in TF.js).
- [x] 263. Properly handle TF.js `BroadcastTo` with ONNX `Expand`.
- [x] 264. Properly handle TF.js `TensorArray` operations (used in LSTMs).
- [x] 265. Implement memory limits for the ArrayBuffer loader.
- [x] 266. Enable progressive loading of weight shards to show loading bars.
- [x] 267. Map TF.js `Relu6` specifically to ONNX `Clip`.
- [x] 268. Handle Keras `SeparableConv1D` correctly (differing from 2D logic).
- [x] 269. Support extraction of Keras `Masking` layer.
- [x] 270. Handle TF.js `Unpack` and `Pack` operations.
- [x] 271. Handle TF.js `Fill` operations safely.
- [x] 272. Process custom initializers gracefully.
- [x] 273. Support `LeCunNormal`, `GlorotUniform` extraction accurately.
- [x] 274. Handle `tf.pad` modes (`CONSTANT`, `REFLECT`, `SYMMETRIC`).
- [x] 275. Map TF.js `SpaceToDepth` directly.
- [x] 276. Map TF.js `DepthToSpace` directly.
- [x] 277. Ensure Keras 3 `EinsumDense` is supported.
- [x] 278. Add conversion validation for Vision Transformers in TF.js.
- [x] 279. Add specific tests for TF.js Object Detection API exports.
- [x] 280. Handle `tf.round` explicitly.
- [x] 281. Handle `tf.floor` and `tf.ceil`.
- [x] 282. Convert `tf.clipByValue` to ONNX `Clip`.
- [x] 283. Map `tf.squaredDifference`.
- [x] 284. Map `tf.reciprocal`.
- [x] 285. Map `tf.sign`.
- [x] 286. Handle `tf.logicalNot`, `tf.logicalAnd`, `tf.logicalOr`.
- [x] 287. Translate `tf.oneHot` to ONNX `OneHot`.
- [x] 288. Translate `tf.cumsum` to ONNX `CumSum`.
- [x] 289. Ensure valid error when processing multi-device TF.js models.
- [x] 290. Parse training state configurations and strip them from inference payload.
- [x] 291. Translate `tf.linspace` and `tf.range`.
- [x] 292. Map `tf.diag` and `tf.diagPart`.
- [x] 293. Support complex matrices (eigen value/vector ops) fallback.
- [x] 294. Fully integrate Web Worker thread pooling for Transpose operations.
- [x] 295. Set up dedicated memory cleanup functions (preventing Blob leak).
- [x] 296. Publish benchmarking metrics comparing TF.js inference vs ONNX WebGPU inference for the same model.
- [x] 297. Support conversion to specialized target optimizations via CLI `--optimize`.
- [x] 298. Validate complete `tf2onnx` CLI parity.
- [x] 299. Create specific issue templates for failed model conversions.
- [x] 300. Maintain continuous deployment to `@onnx9000/keras` NPM.
