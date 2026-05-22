# ONNX30: onnx-array-api (Web-Native NumPy/Eager API for ONNX)

## Original Project Description

`onnx-array-api` is a Python library that provides a NumPy-like, Eager-execution API for dynamically creating and evaluating ONNX graphs. Instead of writing verbose `onnx.helper` node definitions or tracing a PyTorch model, developers can write mathematical operations using standard array semantics (e.g., `z = x + y * 2`), and the library automatically constructs the corresponding ONNX graph and executes it via ONNX Runtime under the hood. It bridges the gap between static graph definition and eager, interactive numerical computing.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.array` provides this exact NumPy-like experience, but entirely within JavaScript/TypeScript and Pyodide, with zero dependency on the C++ ONNX Runtime.

- **Dual Language Support:** Provides both a Python API (for Pyodide/JupyterLite) and a native TypeScript API (for browser-based apps), sharing the exact same underlying WASM math kernels.
- **Lazy vs Eager Toggling:** Operations can either execute instantly via WebGPU/WASM (Eager mode) or build up a massive ONNX `GraphProto` in the background (Lazy mode) to be exported as a `.onnx` file later.
- **No Python Required for TS Developers:** JavaScript developers get a full NumPy/TensorFlow.js-like numerical library (`import * as np from 'onnx9000/array'`) that natively speaks ONNX Protobuf, making it trivial to author ONNX models strictly via JS math syntax.
- **JIT Compilation to WGSL:** In Lazy mode, complex mathematical expressions written in JS are JIT-compiled directly into fused WebGPU WGSL shaders rather than executing node-by-node.

---

## Exhaustive Implementation Checklist

### Phase 1: Core Array/Tensor Object (`onnx9000.array.Tensor`)

- [x] 1. Define the base `EagerTensor` class extending `onnx9000.Tensor`.
- [x] 2. Define the `LazyTensor` class (stores an AST node reference rather than raw data).
- [x] 3. Implement `Tensor` instantiation from JS Arrays (`const t = np.array([1, 2, 3])`).
- [x] 4. Implement instantiation from TypedArrays (`Float32Array`, `Int32Array`).
- [x] 5. Implement explicit `dtype` forcing during instantiation (`dtype='float16'`).
- [x] 6. Implement `Tensor.shape` property getter.
- [x] 7. Implement `Tensor.dtype` property getter.
- [x] 8. Implement `Tensor.ndim` property getter.
- [x] 9. Implement `Tensor.size` property getter.
- [x] 10. Implement `Tensor.numpy()` / `Tensor.data()` to extract raw values.
- [x] 11. Support printing tensors gracefully to the console (truncating large arrays).
- [x] 12. Implement `np.zeros(shape, dtype)`.
- [x] 13. Implement `np.ones(shape, dtype)`.
- [x] 14. Implement `np.empty(shape, dtype)`.
- [x] 15. Implement `np.full(shape, fill_value, dtype)`.
- [x] 16. Implement `np.eye(N, M, k, dtype)`.
- [x] 17. Implement `np.identity(n, dtype)`.
- [x] 18. Implement `np.arange(start, stop, step, dtype)`.
- [x] 19. Implement `np.linspace(start, stop, num, endpoint, dtype)`.
- [x] 20. Implement automatic context management (switching between Lazy builder mode and Eager mode).

### Phase 2: Basic Mathematical Operations (Eager & Lazy)

- [x] 21. Implement `add(a, b)` / `a.add(b)`.
- [x] 22. Implement `subtract(a, b)` / `a.sub(b)`.
- [x] 23. Implement `multiply(a, b)` / `a.mul(b)`.
- [x] 24. Implement `divide(a, b)` / `a.div(b)`.
- [x] 25. Implement `power(a, b)` / `a.pow(b)`.
- [x] 26. Implement `mod(a, b)`.
- [x] 27. Implement `absolute(a)` / `a.abs()`.
- [x] 28. Implement `negative(a)` / `a.neg()`.
- [x] 29. Implement `sign(a)`.
- [x] 30. Implement `exp(a)`.
- [x] 31. Implement `log(a)`.
- [x] 32. Implement `log10(a)`.
- [x] 33. Implement `log2(a)`.
- [x] 34. Implement `sqrt(a)`.
- [x] 35. Implement `square(a)`.
- [x] 36. Implement `cbrt(a)`.
- [x] 37. Implement `reciprocal(a)`.
- [x] 38. Support implicit scalar-to-tensor broadcasting in all math ops (e.g., `a.add(5)`).
- [x] 39. Support standard NumPy broadcasting rules (e.g., `[3, 1] + [1, 4] -> [3, 4]`).
- [x] 40. Ensure Lazy mode emits `Constant` nodes automatically for scalar arguments.

### Phase 3: Trigonometric Operations

- [x] 41. Implement `sin(a)`.
- [x] 42. Implement `cos(a)`.
- [x] 43. Implement `tan(a)`.
- [x] 44. Implement `arcsin(a)`.
- [x] 45. Implement `arccos(a)`.
- [x] 46. Implement `arctan(a)`.
- [x] 47. Implement `sinh(a)`.
- [x] 48. Implement `cosh(a)`.
- [x] 49. Implement `tanh(a)`.
- [x] 50. Implement `arcsinh(a)`.
- [x] 51. Implement `arccosh(a)`.
- [x] 52. Implement `arctanh(a)`.
- [x] 53. Implement `deg2rad(a)`.
- [x] 54. Implement `rad2deg(a)`.

### Phase 4: Matrix & Linear Algebra Operations

- [x] 55. Implement `matmul(a, b)`.
- [x] 56. Implement `dot(a, b)`.
- [x] 57. Implement `vdot(a, b)`.
- [x] 58. Implement `inner(a, b)`.
- [x] 59. Implement `outer(a, b)`.
- [x] 60. Implement `tensordot(a, b, axes)`.
- [x] 61. Implement `einsum(subscripts, ...operands)`.
- [x] 62. Implement `transpose(a, axes)`.
- [x] 63. Implement `a.T` shorthand for matrix transposition.
- [x] 64. Implement `swapaxes(a, axis1, axis2)`.
- [x] 65. Implement `trace(a, offset, axis1, axis2)`.
- [x] 66. Implement `linalg.norm(x, ord, axis, keepdims)`.
- [x] 67. Implement `linalg.det(a)`.
- [x] 68. Implement `linalg.inv(a)` (using WASM fallbacks if ONNX natively lacks it).
- [x] 69. Implement `linalg.solve(a, b)`.
- [x] 70. Map complex linear algebra natively to ONNX loops or WebGPU compute if standard ops are insufficient.

### Phase 5: Reduction Operations

- [x] 71. Implement `sum(a, axis, keepdims)`.
- [x] 72. Implement `prod(a, axis, keepdims)`.
- [x] 73. Implement `mean(a, axis, keepdims)`.
- [x] 74. Implement `std(a, axis, keepdims)`.
- [x] 75. Implement `var(a, axis, keepdims)`.
- [x] 76. Implement `min(a, axis, keepdims)`.
- [x] 77. Implement `max(a, axis, keepdims)`.
- [x] 78. Implement `argmin(a, axis, keepdims)`.
- [x] 79. Implement `argmax(a, axis, keepdims)`.
- [x] 80. Implement `ptp(a, axis)` (Peak to Peak - Max minus Min).
- [x] 81. Implement `all(a, axis, keepdims)`.
- [x] 82. Implement `any(a, axis, keepdims)`.
- [x] 83. Implement `cumsum(a, axis)`.
- [x] 84. Implement `cumprod(a, axis)`.

### Phase 6: Shape Manipulation & Array Operations

- [x] 85. Implement `reshape(a, newshape)`.
- [x] 86. Implement `a.reshape(newshape)`.
- [x] 87. Implement `ravel(a)` (flattening to 1D).
- [x] 88. Implement `squeeze(a, axis)`.
- [x] 89. Implement `expand_dims(a, axis)`.
- [x] 90. Implement `broadcast_to(array, shape)`.
- [x] 91. Implement `concatenate(arrays, axis)`.
- [x] 92. Implement `stack(arrays, axis)`.
- [x] 93. Implement `vstack(tup)`.
- [x] 94. Implement `hstack(tup)`.
- [x] 95. Implement `dstack(tup)`.
- [x] 96. Implement `split(ary, indices_or_sections, axis)`.
- [x] 97. Implement `array_split(ary, indices_or_sections, axis)`.
- [x] 98. Implement `tile(A, reps)`.
- [x] 99. Implement `repeat(a, repeats, axis)`.
- [x] 100. Implement `pad(array, pad_width, mode, constant_values)`.

### Phase 7: Logical & Relational Operations

- [x] 101. Implement `equal(x1, x2)` / `x1.eq(x2)`.
- [x] 102. Implement `not_equal(x1, x2)` / `x1.neq(x2)`.
- [x] 103. Implement `less(x1, x2)` / `x1.lt(x2)`.
- [x] 104. Implement `less_equal(x1, x2)` / `x1.lte(x2)`.
- [x] 105. Implement `greater(x1, x2)` / `x1.gt(x2)`.
- [x] 106. Implement `greater_equal(x1, x2)` / `x1.gte(x2)`.
- [x] 107. Implement `logical_and(x1, x2)`.
- [x] 108. Implement `logical_or(x1, x2)`.
- [x] 109. Implement `logical_not(x)`.
- [x] 110. Implement `logical_xor(x1, x2)`.
- [x] 111. Implement `allclose(a, b, rtol, atol)`.
- [x] 112. Implement `isclose(a, b, rtol, atol)`.
- [x] 113. Implement `isnan(x)`.
- [x] 114. Implement `isinf(x)`.
- [x] 115. Implement `where(condition, x, y)`.

### Phase 8: Sorting, Searching, and Indexing

- [x] 116. Implement `sort(a, axis)`.
- [x] 117. Implement `argsort(a, axis)`.
- [x] 118. Implement `nonzero(a)`.
- [x] 119. Implement `extract(condition, arr)`.
- [x] 120. Implement `take(a, indices, axis)` (mapping to ONNX `Gather`).
- [x] 121. Implement `take_along_axis(arr, indices, axis)` (mapping to ONNX `GatherElements`).
- [x] 122. Implement `put(a, ind, v, mode)`.
- [x] 123. Implement `put_along_axis(arr, indices, values, axis)`.
- [x] 124. Support basic slice syntax emulation in JS (`a.slice([start, stop, step])`).
- [x] 125. Support multidimensional slicing `a.slice([ [start1, stop1], [start2, stop2] ])`.

### Phase 9: Advanced Neural Network Ops (onnx9000.nn)

- [x] 126. Expose neural network ops seamlessly within the array API.
- [x] 127. Implement `nn.relu(x)`.
- [x] 128. Implement `nn.sigmoid(x)`.
- [x] 129. Implement `nn.softmax(x, axis)`.
- [x] 130. Implement `nn.log_softmax(x, axis)`.
- [x] 131. Implement `nn.gelu(x)`.
- [x] 132. Implement `nn.conv2d(x, w, b, strides, pads, dilations, groups)`.
- [x] 133. Implement `nn.max_pool2d(x, kernel_shape, strides, pads)`.
- [x] 134. Implement `nn.avg_pool2d(x, kernel_shape, strides, pads)`.
- [x] 135. Implement `nn.batch_norm(x, scale, B, mean, var, epsilon)`.
- [x] 136. Implement `nn.layer_norm(x, scale, B, axis, epsilon)`.
- [x] 137. Implement `nn.dropout(x, ratio)`.
- [x] 138. Implement `nn.linear(x, weight, bias)`.
- [x] 139. Map all `nn` ops dynamically to their direct ONNX operator equivalents during Lazy building.
- [x] 140. Expose standard loss functions natively (e.g., `nn.cross_entropy_loss`).

### Phase 10: Eager Execution Engine (WebGPU/WASM)

- [x] 141. Ensure Eager mode immediately evaluates the ONNX AST for a single operation.
- [x] 142. Compile tiny single-node ONNX graphs on the fly and execute via `onnx9000.runtime`.
- [x] 143. Cache compiled micro-graphs (e.g., a simple `Add` graph) to prevent recompilation overhead on rapid looping.
- [x] 144. Allow forced targeting for Eager mode (`np.set_device('webgpu')`).
- [x] 145. Implement zero-copy buffer sharing between consecutive Eager ops running on WebGPU.
- [x] 146. Track tensor reference counts to automatically free WebGPU memory for intermediate Eager tensors when no longer used.
- [x] 147. Support explicit `.dispose()` calls on Tensors for tight memory loops.
- [x] 148. Fallback to WASM Math automatically if a tensor is small enough (preventing GPU dispatch overhead).
- [x] 149. Expose `.cpu()` and `.gpu()` methods on the Tensor object to force memory transfers.
- [x] 150. Handle asynchronous execution naturally: math operations return Promises if WebGPU is active (`await a.add(b)`).

### Phase 11: Lazy Graph Builder (The Exporter)

- [x] 151. Implement `np.lazy_mode(true)` to switch global context.
- [x] 152. Implement `np.Input(name, shape, dtype)` to explicitly define graph ingress points.
- [x] 153. When in Lazy mode, math operations return `LazyTensor` (representing an AST Edge) instead of data.
- [x] 154. Implement AST node generation on math method calls (e.g., `a.add(b)` generates an ONNX `Add` node in the background).
- [x] 155. Track topological order inherently as the user writes TS/JS code.
- [x] 156. Implement `np.export_model(outputs, filename)` to serialize the AST to `.onnx`.
- [x] 157. Ensure constants created in Lazy mode are correctly embedded into the `.onnx` as Initializers.
- [x] 158. Auto-generate node names (`Add_1`, `MatMul_2`) if not explicitly provided by the user.
- [x] 159. Support explicit naming: `a.add(b, { name: "MyAddition" })`.
- [x] 160. Detect unused computational branches in Lazy mode and strip them automatically upon export.

### Phase 12: Graph Tracing & Python Integration

- [x] 161. Implement a JS-equivalent to PyTorch `make_fx` / Tracing.
- [x] 162. Allow passing a standard JS function `function myModel(x, y) { return x.add(y).mul(2); }` and auto-tracing it into an ONNX graph.
- [x] 163. Handle native JS control flow (`if/else`) during tracing by either unwrapping dynamically or emitting ONNX `If` nodes via specialized hooks.
- [x] 164. Export the TS API directly into Pyodide via JS-Py bindings.
- [x] 165. Ensure Python code `z = np.add(x, y)` correctly calls the TS `onnx9000.array.add` under the hood.
- [x] 166. Support Python operator overloading natively in Pyodide (`x + y` evaluates via the library).
- [x] 167. Implement `__getitem__` and `__setitem__` in Python mapped to the slicing APIs.
- [x] 168. Ensure output ONNX models are 100% compliant with standard Python `onnx-array-api` output.
- [x] 169. Support exporting sub-graphs explicitly from traced functions.
- [x] 170. Create decorators `@onnx_function` to enforce strict type checking before tracing.

### Phase 13: JIT Compilation (Lazy to WebGPU Shader Fusions)

- [x] 171. If Lazy mode is active but the user calls `.numpy()` / `.evaluate()`, trigger a Just-In-Time compile.
- [x] 172. Collapse chained elementwise operations (e.g., `(x * y) + z`) into a single, fused WGSL shader locally.
- [x] 173. Execute the JIT-compiled macro-kernel on WebGPU instantly and return the result.
- [x] 174. Discard the intermediate AST nodes once the macro-kernel is built (acting as an ultra-fast NumExpr equivalent for JS).
- [x] 175. Cache JIT shaders based on the AST hash to speed up loop evaluations.
- [x] 176. Provide explicit tuning APIs: `np.compile(myModel, { optimize: 'O3' })`.
- [x] 177. If targeting WebNN, JIT compile the AST block directly to a WebNN `MLGraph` and execute.
- [x] 178. Handle fallback gracefully: if an op cannot be fused, chunk the JIT block and execute standard micro-graphs.
- [x] 179. Benchmark JIT fused execution vs naive Eager execution.
- [x] 180. Provide verbose logging: `np.set_log_level('DEBUG')` to print generated WGSL shaders during JIT.

### Phase 14: NumPy Parity & Edge Cases

- [x] 181. Ensure `NaN` and `Infinity` handling strictly matches NumPy IEEE-754 semantics.
- [x] 182. Implement `np.nan_to_num(x)`.
- [x] 183. Implement `np.clip(a, a_min, a_max)` matching exactly.
- [x] 184. Implement `np.around(a, decimals)`.
- [x] 185. Implement `np.fix(a)`.
- [x] 186. Implement `np.i0(x)` (Modified Bessel function).
- [x] 187. Implement `np.sinc(x)`.
- [x] 188. Support `axis` parameter as tuples (e.g., `axis=(0, 2)`).
- [x] 189. Resolve negative axes exactly as NumPy does (counting from the back).
- [x] 190. Handle 0-D tensors (scalars) accurately, as ONNX handles them differently than older TF/NumPy versions.

### Phase 15: Quality Assurance & Testing

- [x] 191. Write unit tests comparing JS `onnx9000.array` outputs natively against a running Python NumPy instance.
- [x] 192. Ensure absolute tolerance (`atol`) and relative tolerance (`rtol`) limits are respected in test suites.
- [x] 193. Create test suite verifying Lazy mode creates valid ONNX AST nodes for all 150+ math operations.
- [x] 194. Execute the generated `.onnx` files through `onnxruntime-node` to ensure strict standard compliance.
- [x] 195. Fuzz the JIT compiler with randomly chained elementwise operations.
- [x] 196. Fuzz the shape broadcasting engine to ensure it mimics NumPy perfectly.
- [x] 197. Validate memory leak absence in Eager WebGPU mode over 10,000 loop iterations.
- [x] 198. Configure CI to run tests against Pyodide inside Headless Chrome.
- [x] 199. Publish test coverage reports for the `onnx9000.array` module specifically.
- [x] 200. Enforce strict TS typing, throwing compile-time errors if shapes/types mismatch in explicitly typed inputs.

### Phase 16: Interoperability with Ecosystem Tools

- [x] 201. Support ingesting tensors generated by `onnx9000.transformers` feature extractors natively.
- [x] 202. Allow using `onnx9000.array` within `onnx9000.modifier` custom JS node replacement scripts.
- [x] 203. Integrate seamlessly with `@tensorflow/tfjs` tensors (providing bi-directional `.fromTfjs()` / `.toTfjs()` converters).
- [x] 204. Integrate with Hugging Face `tokenizers` outputs natively.
- [x] 205. Enable exporting generated ONNX models straight to `onnx9000.coreml` for iOS usage.
- [x] 206. Export models straight to `onnx9000.iree` for AOT standalone JS execution.
- [x] 207. Support ingesting standard JSON arrays from REST APIs transparently.
- [x] 208. Implement `.toDataURL()` for rendering Image tensors (HWC, C=3/4) directly to HTML Canvas objects.
- [x] 209. Implement `.toAudioBuffer()` for writing sequences directly to Web Audio API.
- [x] 210. Implement standard CSV/TSV parsing natively into `Tensor` objects.

### Phase 17: String & Custom Data Types

- [x] 211. Support ONNX `STRING` data types in the Eager array API.
- [x] 212. Implement `np.char.add` (concatenating string tensors).
- [x] 213. Implement `np.char.equal` (string matching).
- [x] 214. Implement `np.char.replace`.
- [x] 215. Implement Regex extract mapping to ONNX `RegexFullMatch` if opset allows.
- [x] 216. Support custom complex numbers (`complex64`, `complex128`) by internally mapping to float arrays with trailing `[..., 2]` dimension.
- [x] 217. Handle BFloat16 (`bfloat16`) casting natively.
- [x] 218. Support quantized integer types natively (`uint8`, `int8`, `uint4`).
- [x] 219. Expose dynamic quantization helpers directly on the Tensor: `a.quantize_dynamic()`.
- [x] 220. Support boolean tensors cleanly, matching JS `true`/`false` mapping to Int8 `1`/`0`.

### Phase 18: Documentation & Developer Experience

- [x] 221. Build comprehensive API docs mapping `numpy.X` to `onnx9000.array.X`.
- [x] 222. Provide a "Rosetta Stone" mapping TF.js commands to `onnx9000` commands.
- [x] 223. Include JSDoc comments directly on the methods to provide inline VSCode hovering.
- [x] 224. Publish an interactive REPL on the documentation website to execute math live in the browser.
- [x] 225. Provide tutorial: "Building a Custom Neural Network from Scratch in TypeScript using `onnx9000.array`".
- [x] 226. Provide tutorial: "JIT Compiling WebGPU Shaders from JS Math".
- [x] 227. Release as an independent NPM package `@onnx9000/array`.
- [x] 228. Ensure tree-shaking works perfectly (importing `np.add` doesn't bundle the whole framework).
- [x] 229. Write warning logs when developers trigger slow-paths (e.g., executing un-fusable ops forcing multiple GPU readbacks).
- [x] 230. Build VSCode snippets for rapid model prototyping.

### Phase 19: Random Number Generation & Stateful APIs

- [x] 231. Implement `np.random.rand()`.
- [x] 232. Implement `np.random.randn()`.
- [x] 233. Implement `np.random.randint()`.
- [x] 234. Implement `np.random.uniform()`.
- [x] 235. Implement `np.random.normal()`.
- [x] 236. Implement `np.random.seed(seed)` to guarantee determinism across JS environments.
- [x] 237. Ensure Random ops map to valid ONNX `RandomNormal` / `RandomUniform` nodes during Lazy execution.
- [x] 238. Generate pseudo-random numbers efficiently via WASM algorithms (e.g., PCG or XorShift).
- [x] 239. Handle ONNX stateful generation if required by specific opsets.
- [x] 240. Implement stateful tracking for custom iteration loops built with the API.

### Phase 20: Final Polish and Release Readiness

- [x] 241. Validate nested Tracing: A traced function calling another traced function emits a clean, flat ONNX graph.
- [x] 242. Prevent name collisions globally when auto-generating node names for 10k+ nodes.
- [x] 243. Allow inline ONNX graph optimization during export (`np.export_model(..., { optimize: 'O3' })`).
- [x] 244. Implement `np.save` to serialize raw tensor data directly to `.npy` binary format.
- [x] 245. Implement `np.load` to read `.npy` or `.npz` files directly from disk/URL.
- [x] 246. Establish benchmarking suite measuring Eager dispatch overhead natively in V8 (Chrome).
- [x] 247. Establish benchmarking suite measuring Eager dispatch overhead in JavaScriptCore (Firefox).
- [x] 248. Provide clear "Not Implemented" exceptions for NumPy operations lacking any valid ONNX operator mapping.
- [x] 249. Optimize GC pauses during heavy lazy graph construction strings.
- [x] 250. Handle deeply nested tuples in `stack`/`concat` APIs.
- [x] 251. Build in array indexing `a.get(0, 2, 1)` and `a.set(0, 2, 1, value)`.
- [x] 252. Add a `np.vectorize` equivalent to map standard JS scalar math functions to ONNX parallel loops.
- [x] 253. Map `np.meshgrid`.
- [x] 254. Map `np.mgrid`.
- [x] 255. Translate ONNX custom domains gracefully inside Eager mode.
- [x] 256. Provide visual execution plan dumps using `console.table`.
- [x] 257. Hook into standard `window.performance` API for fine-grained browser profiling.
- [x] 258. Support memory profiling hooks `np.memory().gpu_allocated`.
- [x] 259. Validate output ONNX models are immediately runnable in iOS CoreML bindings.
- [x] 260. Implement `np.einsum_path` optimization utility.
- [x] 261. Handle large tensor creation seamlessly by deferring to WebGPU Buffer mapping.
- [x] 262. Check edge case zero-sized arrays.
- [x] 263. Map string constants into proper UTF-8 encoded binary arrays.
- [x] 264. Support generic sequence types in Python mapping to ONNX `SequenceProto`.
- [x] 265. Enforce memory constraints on Pyodide environments implicitly.
- [x] 266. Enable seamless JS <-> Pyodide memory sharing using `PyBuffer`.
- [x] 267. Handle multi-threading in Eager mode using Web Workers explicitly.
- [x] 268. Provide graceful WebGL fallbacks if WebGPU isn't available for Eager math.
- [x] 269. Output a "Supported Numpy Ops" compatibility matrix file automatically during build.
- [x] 270. Add support for creating sparse tensors directly via `np.sparse()`.
- [x] 271. Implement specific matrix factorizations (`np.linalg.svd`) via CPU-bound WASM if GPU implementation is unstable.
- [x] 272. Map specific `np.fft` routines to ONNX `DFT` (Discrete Fourier Transform) nodes if opset allows.
- [x] 273. Support `np.pad` complex padding modes (e.g., `wrap`, `maximum`).
- [x] 274. Implement advanced indexing using arrays of integers.
- [x] 275. Handle bitwise operations natively (`bitwise_and`, `bitwise_or`).
- [x] 276. Provide hooks for setting thread limits explicitly (`np.set_num_threads(4)`).
- [x] 277. Validate parity with original `onnx-array-api` v0.2 Python implementations.
- [x] 278. Add strict integration checking for ONNX models built exclusively in the browser and executed in C++.
- [x] 279. Create custom exception classes (`BroadcastError`, `TypeMismatchError`) mirroring standard numerical libraries.
- [x] 280. Establish automated NPM publish pipelines.
- [x] 281. Enable users to override ONNX domain versions (`np.set_opset(18)`).
- [x] 282. Add testing for extremely deep graph creation (tracing 1000+ operations in a loop).
- [x] 283. Create custom memory leak detection for the JIT compilation engine.
- [x] 284. Allow importing WebNN execution providers natively into the Eager dispatch loop.
- [x] 285. Support executing Eager math using Apple Neural Engine natively on macOS Safari.
- [x] 286. Handle ONNX node attributes dynamically updating based on Tensor sizes.
- [x] 287. Expose raw WebGPU CommandEncoders for developers wishing to interleave their own graphics logic.
- [x] 288. Manage memory layout conversion (NCHW vs NHWC) automatically depending on backend preference.
- [x] 289. Track precise model memory bounds during trace evaluation to fail fast on Out Of Memory.
- [x] 290. Finalize rigorous integration tests proving complete offline functionality.
- [x] 291. Develop `np.polyfit` routines.
- [x] 292. Support `np.histogram`.
- [x] 293. Map `np.digitize`.
- [x] 294. Enable custom tensor serialization formats.
- [x] 295. Execute deep lifecycle analysis of Eager objects to prevent GC lockups.
- [x] 296. Maintain strict exact parity against NumPy 1.26 functionality.
- [x] 297. Support `--disable-webgpu-fp16` for legacy compatibility testing.
- [x] 298. Validate precise execution under 1GB RAM bounds.
- [x] 299. Write comprehensive API documentation for the `onnx9000.array` namespace.
- [x] 300. Release v1.0 feature complete certification for `onnx9000.array`.
