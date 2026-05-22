# ONNX35: SparseML (Web-Native Sparsity & Pruning Engine)

## Original Project Description

`SparseML` (developed by Neural Magic) is a toolkit that applies state-of-the-art sparsification techniques—such as unstructured pruning, N:M block-structured pruning, and quantization—to machine learning models. It uses declarative `.yaml` "recipes" to systematically remove redundant weights from deep neural networks. When combined with sparsification-aware execution engines (like Neural Magic's DeepSparse), these pruned ONNX models achieve massive speedups and memory reductions on commodity hardware without requiring expensive GPUs. The standard tool relies heavily on PyTorch and native C++ integrations for calibration, pruning, and export.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.sparse` implements the entire sparsification pipeline in **pure TypeScript and Python**, bringing Neural Magic's recipe-driven pruning directly to the browser.

- **Client-Side Pruning:** Users can drop a dense model and a `.yaml` recipe into a web page. The app processes the AST, applies the magnitude or block-pruning masks, and outputs a highly compressed `SparseTensorProto` ONNX model—all via Web Workers, with zero server interaction.
- **Web-Native Sparse Kernels:** Standard WebGPU and WebAssembly engines evaluate dense matrices. `onnx9000` compiles specialized sparse-matrix multiplication (SpMM) WGSL shaders and WASM SIMD loops, ensuring that the 2:4 or unstructured sparsity actually translates into _faster execution_ on the web, not just a smaller file size.
- **Zero-Dependency Execution:** Applies One-Shot (OBS) or magnitude pruning directly on the pure ONNX graph weights in memory, entirely bypassing the need to load the model into PyTorch to modify its parameters.

---

## Exhaustive Implementation Checklist

### Phase 1: Sparse Tensor Core & Data Formats

- [xx] 1. Implement `SparseTensor` base class extending `onnx9000.Tensor`.
- [xx] 2. Implement COO (Coordinate List) format parser in TS/Python.
- [xx] 3. Implement CSR (Compressed Sparse Row) format parser natively.
- [xx] 4. Implement CSC (Compressed Sparse Column) format parser.
- [xx] 5. Implement BSR (Block Sparse Row) format parser.
- [xx] 6. Convert Dense ONNX `TensorProto` to `SparseTensorProto` explicitly.
- [xx] 7. Convert `SparseTensorProto` to Dense `TensorProto` cleanly.
- [xx] 8. Extract `values` array for `SparseTensorProto`.
- [xx] 9. Extract `indices` array (1D/2D) for `SparseTensorProto`.
- [xx] 10. Support 1D tensor sparsity (Biases, Scales).
- [xx] 11. Support 2D matrix sparsity (Linear/Gemm).
- [xx] 12. Support 4D tensor sparsity (Conv kernels).
- [xx] 13. Detect maximum sparsity theoretically achievable based on epsilon values.
- [xx] 14. Support zero-copy views for sparse value arrays in JS (`TypedArray`).
- [xx] 15. Support converting standard HuggingFace sparse models to ONNX native sparse models.
- [xx] 16. Provide memory usage calculation (Dense vs Sparse byte comparison).
- [xx] 17. Enforce standard `SparseTensorProto` binary serialization.
- [xx] 18. Support mapping sparse inputs correctly to `safetensors` external references.
- [xx] 19. Manage JS `BigInt` indexing for massive sparse arrays safely.
- [xx] 20. Track compression ratio mathematically (`1.0 - (sparse_size / dense_size)`).

### Phase 2: Recipe Parser & Modifiers Engine (YAML)

- [xx] 21. Implement zero-dependency YAML parser natively in TS/Python.
- [xx] 22. Define `Modifier` base class for recipe execution.
- [xx] 23. Implement `ConstantPruningModifier` (Applying static masks).
- [xx] 24. Implement `MagnitudePruningModifier` (Global or layer-wise thresholds).
- [xx] 25. Implement `GlobalMagnitudePruningModifier`.
- [xx] 26. Implement `QuantizationModifier` (Injecting QAT/PTQ INT8 layers).
- [xx] 27. Implement `SparseQuantizationModifier` (Combining both).
- [xx] 28. Parse `init_sparsity` and `final_sparsity` parameters.
- [xx] 29. Parse `start_epoch` and `end_epoch` (ignoring epoch timing if applying One-Shot statically).
- [xx] 30. Parse `update_frequency` (mapping to static intervals if calibrating).
- [xx] 31. Support layer targeting using Regex patterns (e.g., `re:.*weight`).
- [xx] 32. Support exact layer name targeting `["conv1.weight", "conv2.weight"]`.
- [xx] 33. Parse `leave_unmasked` parameters (preventing specific nodes from pruning).
- [xx] 34. Support custom user-defined modifiers securely.
- [xx] 35. Evaluate recipes in topological order to prevent dependency masking conflicts.
- [xx] 36. Provide detailed recipe validation errors (e.g., Target layer not found).
- [xx] 37. Provide strict linting for Neural Magic specific `.yaml` configurations.
- [xx] 38. Export an applied recipe directly into the ONNX `metadata_props` as a string for tracking provenance.

### Phase 3: Unstructured Pruning Algorithms (Magnitude)

- [xx] 39. Implement Layer-wise Magnitude Pruning natively.
- [xx] 40. Calculate exact $L_1$ norms for individual weights.
- [xx] 41. Calculate exact $L_2$ norms if requested.
- [xx] 42. Apply Top-K masking to retain the largest magnitude weights per layer.
- [xx] 43. Implement Global Magnitude Pruning.
- [xx] 44. Gather all model weights into a single virtual 1D array for global threshold calculation.
- [xx] 45. Distribute global threshold mask safely back to original N-dimensional tensor shapes.
- [xx] 46. Support random pruning (baseline testing).
- [xx] 47. Handle uniform sparsity across all channels explicitly.
- [xx] 48. Implement sparsity distribution scaling (e.g., Erdos-Renyi-Kernel distributions).
- [xx] 49. Prevent completely zeroed channels (ensuring at least 1 weight survives per output dimension).
- [xx] 50. Freeze bias parameters explicitly during standard unstructured weight pruning.

### Phase 4: Structured Pruning Algorithms (N:M & Block)

- [xx] 51. Implement strict N:M pruning algorithm (e.g., Nvidia 2:4).
- [xx] 52. Reshape target matrices to `[K, 4]` or `[K, M]` structurally.
- [xx] 53. Apply `ArgMax` iteratively to retain the 2 largest elements per block.
- [xx] 54. Generate bitmasks corresponding to the N:M layout.
- [xx] 55. Validate 2:4 compliance strictly (raising errors if dims aren't multiples of 4).
- [xx] 56. Implement 4:8 structured pruning.
- [xx] 57. Implement block-sparse pruning (e.g., `[32, 32]` contiguous zero blocks).
- [xx] 58. Implement 1D channel pruning (eliminating entire filters in Convolutions).
- [xx] 59. Implement row pruning for GEMM/Linear layers.
- [xx] 60. Propagate channel eliminations topologically (rewiring downstream layer input sizes).
- [xx] 61. Drop output dimension slices natively if a filter is completely pruned.
- [xx] 62. Update downstream biases if channel pruning occurs.
- [xx] 63. Update downstream `BatchNormalization` constants natively if channel pruning occurs.
- [xx] 64. Resolve 2:4 sparse encoding metadata specifically for TensorRT / WebGPU injection.
- [xx] 65. Handle transposed linear layer dimensions securely during block processing.

### Phase 5: Optimal Brain Surgeon (OBS) & Advanced Sparsity

- [xx] 66. Implement One-Shot OBS (Optimal Brain Surgeon) approximations.
- [xx] 67. Provide Taylor expansion tracking for weight saliency (if calibration data is provided).
- [xx] 68. Calculate diagonal Hessian approximations purely in Python/JS.
- [xx] 69. Support Fisher Information Matrix approximations for parameter importance.
- [xx] 70. Implement Movement Pruning (simulating weight updates via gradient tracking if requested).
- [xx] 71. Implement gradual pruning schedules mapped to calibration loop steps.
- [xx] 72. Execute Hessian calculations inside Web Workers to prevent main-thread freezing.
- [xx] 73. Enable batch-chunked Hessian approximations to prevent RAM overflow on massive LLMs.
- [xx] 74. Map exact saliency scores to a temporary Graph metadata structure for visualization.
- [xx] 75. Allow explicit user manipulation of saliency scores via a visual interface.

### Phase 6: Sparse-Quantization (Combining INT8 + Sparsity)

- [xx] 76. Apply `QuantizationModifier` over a statically pruned graph.
- [xx] 77. Ignore `0.0` masked values explicitly during MinMax scale calibration.
- [xx] 78. Ignore `0.0` masked values explicitly during Entropy (KL) scale calibration.
- [xx] 79. Ensure zero-points align perfectly with the sparse `0` mask natively.
- [xx] 80. Compress Sparse INT8 tensors via bit-packing (storing 4-bit indices and 8-bit values).
- [xx] 81. Support asymmetric sparse-quantization cleanly.
- [xx] 82. Generate specific `SparseQLinearConv` topologies (if mapped to custom WebGPU ops).
- [xx] 83. Support generating ONNX `QuantizeLinear` nodes acting exclusively on `SparseTensorProto` inputs.
- [xx] 84. Flag potential numerical underflow where quantization forces dense weights to become sparsely zeroed unintentionally.
- [xx] 85. Provide specific W4A16 (4-bit weight) block sparsity generation routines.

### Phase 7: AST Injection & Masking Engine (`onnx9000.modifier` bridge)

- [xx] 86. Connect `onnx9000.sparse` to `onnx9000.modifier` graph mutator natively.
- [xx] 87. Extract all `Constant` nodes matching the regex definitions.
- [xx] 88. Execute masking (elementwise multiplication by a generated `0/1` tensor mask) securely in-memory.
- [xx] 89. Bake the masked tensor explicitly back into the ONNX AST.
- [xx] 90. Strip the dense representation immediately to trigger JS Garbage Collection.
- [xx] 91. Identify and collapse structurally 100% sparse `Constant` tensors into a scalar `0`.
- [xx] 92. Analyze topological dead ends created by 100% sparse layers.
- [xx] 93. Run standard Dead Code Elimination (DCE) automatically after a pruning pass.
- [xx] 94. Run standard Constant Folding automatically after a pruning pass.
- [xx] 95. Provide tracking: "Layer `Conv_42` went from 1.2M params to 120k params."

### Phase 8: WebGPU Sparse Kernels (WGSL)

- [xx] 96. Implement `SpMM` (Sparse Matrix-Dense Matrix Multiplication) in WGSL.
- [xx] 97. Support COO format ingestion in WGSL directly.
- [xx] 98. Support CSR format ingestion (Row pointers + Column indices + Values) in WGSL.
- [xx] 99. Optimize WGSL CSR traversal using `workgroup` shared memory.
- [xx] 100. Implement Block-Sparse MatMul in WGSL (optimized for N:M structured pruning).
- [xx] 101. Implement 2:4 structured sparse WebGPU shader via intrinsic bitwise lookups if possible.
- [xx] 102. Validate memory coalescing bounds for WGSL sparse indices.
- [xx] 103. Emit `SparseConv2D` specifically for channel-pruned convolutions.
- [xx] 104. Dispatch WebGPU compute shaders selectively based on `sparsity > 0.60` (falling back to dense MatMul if sparsity is low, as dense is faster).
- [xx] 105. Embed explicit indices and pointers natively into WebGPU `StorageBuffer` objects.
- [xx] 106. Pre-transpose Dense matrices in WebGPU to optimize SpMM memory access patterns.
- [xx] 107. Generate specialized WGSL shaders dynamically based on exact block-sparsity sizes.
- [xx] 108. Optimize WGSL `atomicAdd` if scatter-based SpMM approaches are utilized.
- [xx] 109. Support WebGPU FP16 (`shader-f16`) in all SpMM and SparseConv shaders.
- [xx] 110. Profile SpMM vs Dense latency dynamically on the client's GPU before locking in the execution schedule.

### Phase 9: WASM SIMD Sparse Kernels (CPU Execution)

- [xx] 111. Implement `SpMM` loops natively in C++ transpiled to WASM.
- [xx] 112. Utilize WASM SIMD128 (`v128`) for vectorizing non-zero value multiplications.
- [xx] 113. Implement CSR format traversal natively in WASM memory space.
- [xx] 114. Optimize inner loops by pre-fetching column indices into WASM registers.
- [xx] 115. Implement specialized 2:4 block-sparse CPU kernels.
- [xx] 116. Skip fully zeroed rows entirely via explicit branch hints (`__builtin_expect`).
- [xx] 117. Implement multi-threaded (`SharedArrayBuffer`) sparse matrix chunking natively.
- [xx] 118. Handle INT8 sparse evaluation natively in WASM via DP4A-style emulation loops.
- [xx] 119. Allocate sparse arrays cleanly within the `onnx9000` WASM static memory arena.
- [xx] 120. Verify WASM Sparse evaluation outperforms dense evaluation on >70% sparse networks.

### Phase 10: Calibration & Loss Simulation (In-Browser Fine-tuning)

- [xx] 121. Support `DataLoader` abstractions explicitly for sparse calibration runs.
- [xx] 122. Evaluate Mean Squared Error (MSE) degradation after applying a sparse mask.
- [xx] 123. Evaluate Cross-Entropy loss degradation natively in JS/Python.
- [xx] 124. Implement Mask Fine-Tuning: Iteratively un-masking high-gradient parameters to recover accuracy.
- [xx] 125. Process gradients natively via the `onnx9000.training` AOT autograd module.
- [xx] 126. Accumulate gradients over calibration batches to determine sensitive weights.
- [xx] 127. Support early stopping if the sparse model degrades below a defined target accuracy.
- [xx] 128. Provide visually updating accuracy charts (Chart.js/D3) during the browser-based calibration loop.
- [xx] 129. Manage memory gracefully by destroying intermediate activations during large batch calibrations.
- [xx] 130. Fallback to entirely static pruning if no calibration dataset is supplied.

### Phase 11: ONNX Serializer (SparseTensor Export)

- [xx] 131. Provide API: `export_sparse_model(model, "sparse.onnx")`.
- [xx] 132. Encode `SparseTensorProto` natively bypassing standard `TensorProto`.
- [xx] 133. Serialize `indices` perfectly per ONNX spec.
- [xx] 134. Serialize `values` perfectly per ONNX spec.
- [xx] 135. Manage correct Endianness during binary writing.
- [xx] 136. Maintain standard `ModelProto` structures without corrupting downstream parsers.
- [xx] 137. Export Sparse ONNX utilizing `external_data` (splitting `.bin`) for models > 2GB.
- [xx] 138. Apply ZIP compression optionally since sparse models compress exceptionally well via DEFLATE.
- [xx] 139. Embed the Sparsification `metadata_props` strictly mapped to standard Neural Magic keys.
- [xx] 140. Generate a deterministic byte output (same model + same recipe = same bytes).

### Phase 12: Specific Architecture Support (LLMs & Vision)

- [xx] 141. Apply Unstructured Pruning seamlessly to `BERT` attention heads.
- [xx] 142. Apply 2:4 Structured Pruning to `ResNet50` convolutional kernels.
- [xx] 143. Apply Block-sparsity to `LLaMA` feed-forward (`up_proj`, `down_proj`) layers.
- [xx] 144. Identify `QKV` attention packing and prune consistently across logical head boundaries.
- [xx] 145. Preserve standard positional embeddings (RoPE) and `token_embd` layers from pruning automatically.
- [xx] 146. Prune ViT (Vision Transformer) Patch Embeddings strictly along valid dimensional bounds.
- [xx] 147. Evaluate sparsity impact on `YOLO` detection heads.
- [xx] 148. Evaluate sparsity impact on `Whisper` encoder blocks.
- [xx] 149. Support specific HuggingFace `SparseML` models natively out of the box (e.g., `neuralmagic/llama2-7b-sparse`).
- [xx] 150. Translate natively formatted DeepSparse model graphs back to generic ONNX if requested.

### Phase 13: Sparsity Profiling & Memory Analysis

- [xx] 151. Profile explicit FLOPs saved due to zero-skipping.
- [xx] 152. Calculate "Theoretical Speedup" vs "Actual WebGPU Speedup".
- [xx] 153. Expose API: `onnx9000.sparse.profile(model)`.
- [xx] 154. Render layer-by-layer sparsity percentage in an ASCII table.
- [xx] 155. Provide JSON reports for CI/CD automation pipelines.
- [xx] 156. Analyze cache hit-rates mathematically based on the generated CSR structures.
- [xx] 157. Identify bottleneck dense layers that are dragging down overall sparse execution times.
- [xx] 158. Generate memory fragmentation statistics for the WASM arena post-pruning.
- [xx] 159. Calculate explicit disk-storage savings (Dense MB vs Sparse MB).
- [xx] 160. Expose interactive HTML Flamegraphs highlighting sparsified operations.

### Phase 14: Web UI (The Interactive Pruner)

- [xx] 161. Build static Web Components page "ONNX Web Pruner".
- [xx] 162. Implement drag-and-drop ingestion of `model.onnx` and `recipe.yaml`.
- [xx] 163. Display a 3D/2D visualization of the model topology.
- [xx] 164. Render a "Sparsity Slider" allowing users to dial in global sparsity from `0.0` to `0.99`.
- [xx] 165. Highlight layers dynamically in the UI (e.g., turning green as they reach target sparsity).
- [xx] 166. Provide a "Calibrate & Run" button executing the pruning in a background Web Worker.
- [xx] 167. Show real-time progress bars extracting tensors, applying masks, and compacting data.
- [xx] 168. Expose a "Download Sparse ONNX" button streaming the Blob to the filesystem.
- [xx] 169. Render interactive histograms of weight distributions (identifying magnitude cutoffs visually).
- [xx] 170. Ensure the UI functions 100% completely offline after initial load.

### Phase 15: Node.js & CLI Integration (`onnx9000 sparse`)

- [xx] 171. Implement CLI: `onnx9000 sparse prune model.onnx --recipe recipe.yaml -o sparse.onnx`.
- [xx] 172. Add `--sparsity 0.8` global override flag.
- [xx] 173. Add `--structured 2:4` flag to enforce block sparsity explicitly.
- [xx] 174. Support processing directories of models dynamically.
- [xx] 175. Allow inputting calibration datasets via `--data calibration.json`.
- [xx] 176. Extract the NPM package independently: `@onnx9000/sparse`.
- [xx] 177. Configure GitHub Actions workflows to auto-prune large models in the cloud safely.
- [xx] 178. Handle process exits cleanly on massive models throwing memory boundary warnings.
- [xx] 179. Set up `pino` or `winston` for structured terminal logging during pruning.
- [xx] 180. Validate CLI parity against standard Neural Magic `sparseml.onnx` scripts.

### Phase 16: Interoperability with other `onnx9000` Tools

- [xx] 181. Integration: `onnx9000.optimum` -> `onnx9000.sparse` -> `onnx9000.quantize`.
- [xx] 182. Inject `SparseTensorProto` natively into the `Netron` visualizer for rendering block structures.
- [xx] 183. Map sparse parameters flawlessly back into `onnx9000.coreml` export if targeting ANE (which occasionally supports structured sparsity).
- [xx] 184. Guarantee `onnx9000.array` (Eager API) can instantiate sparse tensors dynamically.
- [xx] 185. Ensure `onnx9000.iree` compiles sparse MLIR dialects natively based on the pruned structure.
- [xx] 186. Use `onnx-tool` specifically to assert the new sparse FLOP counts exactly match predictions.
- [xx] 187. Execute sparse ONNX directly via the `onnx9000.genai` LLM pipeline to achieve fast-token generation on CPUs.
- [xx] 188. Support generating `GGUF` binaries packed with sparse layouts if requested.
- [xx] 189. Provide direct AST mapping for `onnx9000.modifier` manual edits post-pruning.
- [xx] 190. Load Safetensors (`.safetensors`) natively, prune them in memory, and export to Sparse ONNX without saving intermediate dense protobufs.

### Phase 17: Deep Execution & Edge Cases

- [xx] 191. Validate numerical stability of extremely sparse matrices (0.99 sparsity) multiplying against dense activations.
- [xx] 192. Handle `NaN` propagation specifically in SpMM.
- [xx] 193. Throw warnings if a user attempts to sparsify a tiny model (e.g., 2-layer MLP) where CSR overhead outweighs dense execution.
- [xx] 194. Fallback from CSR back to Dense dynamically inside WebGPU if the sparsity drops mid-computation.
- [xx] 195. Implement specific memory bounds checks preventing integer overflow during CSR array generation.
- [xx] 196. Handle unaligned matrices correctly in 2:4 structured constraints (e.g., padding matrices to multiple of 4 internally).
- [xx] 197. Support mixed batch-size evaluation natively against sparse weights.
- [xx] 198. Protect against infinite loops during Taylor series Hessian approximations on flat gradients.
- [xx] 199. Manage memory mapped `.bin` files cleanly when overwriting dense with sparse blocks.
- [xx] 200. Execute precise tolerance matching (atol=1e-5) comparing dense vs sparse execution on identical calibration inputs.

### Phase 18: Security, Validation & File Processing

- [xx] 201. Verify `SparseTensorProto` schemas correctly implement the ONNX IR version 11+ requirements.
- [xx] 202. Reject corrupt YAML recipes containing arbitrary code execution markers.
- [xx] 203. Prevent prototype pollution via malformed JSON calibration objects.
- [xx] 204. Isolate the Web Worker processing context to prevent Cross-Site Scripting (XSS) via metadata.
- [xx] 205. Implement exact byte boundary validations for JS `ArrayBuffer` slicing during CSR extraction.
- [xx] 206. Ensure all generated models pass the internal `onnx.checker` polyfill cleanly.
- [xx] 207. Trap division by zero if entire channels are pruned and subsequently scaled by BatchNormalization.
- [xx] 208. Sanitize ONNX strings natively during metadata packing.
- [xx] 209. Track and enforce Javascript `Number.MAX_SAFE_INTEGER` for memory pointers.
- [xx] 210. Validate that sparse structures correctly maintain `__dlpack__` interop protocols where possible.

### Phase 19: Comprehensive Documentation

- [xx] 211. Write Tutorial: "Pruning ResNet50 in the Browser".
- [xx] 212. Write Tutorial: "Understanding 2:4 Block Sparsity and WebGPU".
- [xx] 213. Document the precise `SparseTensorProto` serialization sequence.
- [xx] 214. Create an architectural diagram showing how the AST masking pass operates.
- [xx] 215. Provide a compatibility matrix detailing which standard Neural Magic recipes are supported.
- [xx] 216. Document the SpMM WebGPU shader memory access patterns for advanced graphics developers.
- [xx] 217. Produce specific API guides for `onnx9000.sparse.Modifier`.
- [xx] 218. Detail the mathematical operations underlying the Hessian approximation.
- [xx] 219. Explain how to target different hardware backends (CPU SIMD vs WebGPU) from a sparse model.
- [xx] 220. Release benchmark comparisons (Dense vs Sparse latency) on standard Apple Silicon and Chrome V8.

### Phase 20: Delivery & Final Polish

- [xx] 221. Establish a test suite converting 50+ diverse Dense models to Sparse models automatically.
- [xx] 222. Expose specific hooks to visualize the exact non-zero weight distribution on an HTML Canvas element.
- [xx] 223. Output a detailed JSON "Sparsity Report Card" suitable for MLOps dashboards.
- [xx] 224. Handle models with exactly zero trainable parameters cleanly.
- [xx] 225. Process 0-D scalar limits safely inside the sparsity calculator.
- [xx] 226. Ensure the `onnx9000` CLI supports chained operations (`onnx9000 optimize --prune --quantize`).
- [xx] 227. Emit warnings if a generated sparse model is loaded into an older execution provider lacking sparse support.
- [xx] 228. Provide a "De-Sparsify" utility to inflate `SparseTensorProto` back into dense arrays for backwards compatibility.
- [xx] 229. Allow configuring the Web Worker thread count explicitly.
- [xx] 230. Test execution explicitly on Safari to manage its strict memory allocation quirks.
- [xx] 231. Add support for creating an interactive threshold tuning UI widget.
- [xx] 232. Verify memory release post-garbage collection inside V8 traces.
- [xx] 233. Handle explicit `float64` fallback cleanly (downcasting to `float32`).
- [xx] 234. Generate explicit memory layouts aligning with HuggingFace Hub expectations.
- [xx] 235. Translate `String` inputs accurately despite sparsity operations.
- [xx] 236. Add macros specifically to bypass sparsity on critical attention heads.
- [xx] 237. Evaluate static variables completely to avoid re-evaluating sparsity constraints dynamically.
- [xx] 238. Compile `CumSum` correctly under sparse configurations.
- [xx] 239. Validate multi-dimensional `GatherND` loop correctness with sparse weights.
- [xx] 240. Validate `ScatterND` memory updates appropriately.
- [xx] 241. Ensure `ConstantOfShape` generates valid sparse arrays.
- [xx] 242. Map `Softplus` correctly on sparsified input arrays.
- [xx] 243. Compile `Einsum` cleanly taking advantage of known zero bounds.
- [xx] 244. Implement memory overlap checking at generation time (ensuring in-place masking is safe).
- [xx] 245. Validate multi-model multiplexing natively (running 2 sparse models simultaneously).
- [xx] 246. Establish automated NPM publish pipelines for `@onnx9000/sparse`.
- [xx] 247. Provide `#ifdef` toggles in the C-generator to conditionally compile SpMM vs Dense based on thresholds.
- [xx] 248. Provide static performance metrics inline (e.g. `// Estimated Dense MACs: X, Sparse MACs: Y`).
- [xx] 249. Create a JSON schema definitions file for the supported YAML recipe structures.
- [xx] 250. Support `Einsum` unrolling directly into nested C loops for sparse CPU evaluation.
- [xx] 251. Validate execution parity with `deepsparse` reference C++ implementations.
- [xx] 252. Add a `verify_checksum()` utility for the generated Sparse ONNX binaries.
- [xx] 253. Track precise byte alignment requirements for WebGPU `Float16` buffers passed to SpMM.
- [xx] 254. Develop custom loaders for multi-file YAML recipe setups.
- [xx] 255. Support overriding the target WebGPU specification explicitly during WebGPU shader generation.
- [xx] 256. Handle `tf.js` specific graph structures transpiled into ONNX sparse structures.
- [xx] 257. Map Python `__call__` explicitly to standard Pyodide sparse dispatch.
- [xx] 258. Add specific CLI flags limiting output verbosity during heavy calibration runs.
- [xx] 259. Render graph connections in C source comments explicitly if exporting to `onnx2c`.
- [xx] 260. Output the memory planner sparse allocations dynamically in console logs.
- [xx] 261. Expose the AST sparse compiler via an isolated NPM module `@onnx9000/sparse-compiler`.
- [xx] 262. Support dynamic checking of WebNN sparse matrix multiplication APIs (if spec introduces them).
- [xx] 263. Establish a standard interface for custom block-sparse headers.
- [xx] 264. Support `Einsum` explicitly unrolled.
- [xx] 265. Ensure deterministic float formatting across all JS engines.
- [xx] 266. Provide array compression algorithms specifically for CSR format transmission.
- [xx] 267. Implement automated documentation generation for custom sparse ops.
- [xx] 268. Verify that sparse tensors do not leak memory during repeated session runs.
- [xx] 269. Provide a `to_sparse()` helper in the Python Eager API.
- [xx] 270. Support `LLaMA` sparse attention with specialized mask handling.
