# ONNX19: ONNX Runtime GenAI (WASM-First Generative Execution)

## Original Project Description

ONNX Runtime GenAI (ORT GenAI) is a highly specialized extension of the standard ONNX Runtime designed specifically for executing large generative AI models (like LLMs, Whisper, and Stable Diffusion). Standard graph execution is insufficient for generative models because they require complex control loops (autoregressive decoding), dynamic memory management across sequence generations (KV Caching), and specific search algorithms (Beam Search, Top-K/Top-P sampling). ORT GenAI provides a native C++ API and custom operations to handle these generative loops efficiently, avoiding the overhead of shuttling tokens back and forth between the host language (Python) and the execution runtime.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of a separate C++ library bridging into Python, `onnx9000`'s GenAI implementation is built natively as an extension module within the web-first monolith.

- **WebAssembly / WebGPU Native Loops:** The autoregressive generation loop is implemented directly in TypeScript/WASM, preventing the catastrophic latency of crossing the JS-to-WASM boundary for every single generated token.
- **Integrated KV Cache Management:** Memory management for Key-Value caches is handled by the `onnx9000` memory planner directly in the WebGPU VRAM or WASM linear memory, utilizing ring buffers to handle infinite context window generation.
- **Unified Pipeline:** Eliminates the need for a secondary library. `onnx9000.genai` is a first-class citizen, wrapping the core execution engine to provide high-level APIs like `model.generate()`.
- **Browser-Side Tokenization:** Incorporates lightweight WASM tokenizers directly into the generation pipeline, allowing the browser to accept raw text and output raw text without server-side preprocessing.

---

## Exhaustive Implementation Checklist

### Phase 1: Core GenAI Pipeline & State Management

- [x] 1. Define `GeneratorParams` configuration object.
- [x] 2. Define `ModelParams` configuration object.
- [x] 3. Implement base `Model` class for GenAI wrappers.
- [x] 4. Implement base `Generator` class for stateful decoding.
- [x] 5. Implement `State` object to hold execution graph and KV cache.
- [x] 6. Create `Tensor` utility extensions specific to sequence lengths.
- [x] 7. Implement dynamic shape allocation strategies for growing sequences.
- [x] 8. Implement a pre-fill phase executor (processing the initial prompt).
- [x] 9. Implement a decode phase executor (generating token by token).
- [x] 10. Support seamless transition between pre-fill and decode phases.
- [x] 11. Implement cross-layer KV cache synchronization.
- [x] 12. Support for continuous batching (adding requests mid-generation).
- [x] 13. Support for sequence batching (multiple independent sequences).
- [x] 14. Implement paged attention memory management (conceptually in WASM/WGSL).
- [x] 15. Handle context window overflow (sliding window cache ejection).
- [x] 16. Implement asynchronous generation stepping (`await generator.compute_logits()`).
- [x] 17. Implement synchronous generation stepping for blocking environments.
- [x] 18. Support early stopping conditions (EOS token reached).
- [x] 19. Support max length stopping conditions.
- [x] 20. Implement a unified `generate()` high-level API.
- [x] 21. Provide callback hooks for streaming output (yielding tokens).
- [x] 22. Implement memory reuse across generation requests.
- [x] 23. Handle dynamic model loading (loading weights asynchronously during pre-fill).
- [x] 24. Implement graceful abort/cancellation of a generation loop.
- [x] 25. Support sub-graph partitioning for multi-GPU/WebGPU chunking.

### Phase 2: Generation Algorithms (Search & Sampling)

- [x] 26. Define `SearchOptions` configuration struct.
- [x] 27. Implement Greedy Search algorithm.
- [x] 28. Implement Beam Search algorithm.
- [x] 29. Manage beam search state (beam scores, beam tokens, beam histories).
- [x] 30. Implement beam search pruning and sorting.
- [x] 31. Support `num_beams` parameter.
- [x] 32. Support `num_return_sequences` parameter.
- [x] 33. Implement multinomial sampling.
- [x] 34. Implement Top-K sampling filter.
- [x] 35. Implement Top-P (Nucleus) sampling filter.
- [x] 36. Implement Min-P sampling filter.
- [x] 37. Implement Temperature scaling applied to logits.
- [x] 38. Support `repetition_penalty` filter.
- [x] 39. Support `presence_penalty` filter.
- [x] 40. Support `frequency_penalty` filter.
- [x] 41. Support `length_penalty` filter (primarily for beam search).
- [x] 42. Implement `no_repeat_ngram_size` filter.
- [x] 43. Support forced BOS (Beginning of Sequence) token injection.
- [x] 44. Support forced EOS (End of Sequence) token generation.
- [x] 45. Implement custom logit bias injection (boosting/penalizing specific tokens).
- [x] 46. Support custom bad words list (banning specific token sequences).
- [x] 47. Support custom allowed words list (restricting vocabulary).
- [x] 48. Implement diverse beam search (grouping beams to ensure variety).
- [x] 49. Support typical decoding sampling.
- [x] 50. Implement a modular logit processor pipeline.
- [x] 51. Create a WASM-optimized logit sorting/filtering kernel (crucial for speed).
- [x] 52. Create a WebGPU-optimized Top-K/Top-P extraction shader.
- [x] 53. Implement random seed control for deterministic sampling.
- [x] 54. Provide probability distributions per token in the output.
- [x] 55. Implement contrastive search algorithm.

### Phase 3: Tokenization & Text Processing

- [x] 56. Define base `Tokenizer` interface.
- [x] 57. Implement `TokenizerStream` for real-time decoding.
- [x] 58. Implement Byte-Pair Encoding (BPE) algorithm in WASM.
- [x] 59. Implement WordPiece tokenization algorithm.
- [x] 60. Implement Unigram tokenization algorithm.
- [x] 61. Support loading HuggingFace `tokenizer.json` formats.
- [x] 62. Support loading SentencePiece `.model` binaries natively.
- [x] 63. Implement byte-level BPE pre-tokenization.
- [x] 64. Implement basic whitespace pre-tokenization.
- [x] 65. Implement punctuation splitting pre-tokenization.
- [x] 66. Implement Unicode normalization (NFC, NFD, NFKC, NFKD).
- [x] 67. Support added tokens (special tokens mapping).
- [x] 68. Handle unknown `<unk>` token replacements.
- [x] 69. Implement `encode()` method (text to token IDs).
- [x] 70. Implement `decode()` method (token IDs to text).
- [x] 71. Implement batched encoding.
- [x] 72. Implement batched decoding.
- [x] 73. Provide token ID to string lookup utilities.
- [x] 74. Handle whitespace stripping/preservation rules cleanly.
- [x] 75. Implement a robust Trie structure for fast token matching in WASM.
- [x] 76. Handle UTF-8 decoding boundaries safely in streaming mode.
- [x] 77. Integrate tokenizer instantiation via `Model.create_tokenizer()`.
- [x] 78. Support specific tokenizer dialects (Llama, GPT-2, T5, Bert).
- [x] 79. Implement chat template rendering (applying Jinja templates).
- [x] 80. Fallback JS tokenizers if WASM module fails to load.

### Phase 4: KV Cache & Attention Architectures

- [x] 81. Implement a generic `KVCache` management class.
- [x] 82. Support standard Multi-Head Attention (MHA) caching.
- [x] 83. Support Grouped-Query Attention (GQA) caching structures.
- [x] 84. Support Multi-Query Attention (MQA) caching structures.
- [x] 85. Implement continuous memory allocation for caches.
- [x] 86. Implement fragmented (paged) memory allocation for caches.
- [x] 87. Support past key/value graph inputs.
- [x] 88. Support present key/value graph outputs.
- [x] 89. Manage in-place KV cache updates (mutating past_key_values directly).
- [x] 90. Implement rotary positional embeddings (RoPE) scaling.
- [x] 91. Support dynamic RoPE calculation based on sequence length.
- [x] 92. Support ALiBi (Attention with Linear Biases) positional encodings.
- [x] 93. Implement cross-attention caching (for Encoder-Decoder models).
- [x] 94. Optimize cache memory layouts (e.g., interleaving K and V).
- [x] 95. Implement cache clearing/reset APIs.
- [x] 96. Support offloading KV cache to CPU memory when WebGPU VRAM is full.
- [x] 97. Implement cache quantization (storing K/V as int8 or fp8).
- [x] 98. Support sliding window attention limits (e.g., Mistral).
- [x] 99. Handle varying batch sizes between pre-fill and decode steps.
- [x] 100. Implement specialized WebGPU shaders for fast KV cache concatenation.

### Phase 5: Model-Specific Optimizations & Architectures

- [x] 101. Support **Llama** architecture variants (Llama 2, Llama 3).
- [x] 102. Support **Mistral** architecture variants.
- [x] 103. Support **Gemma** architecture variants.
- [x] 104. Support **Phi** architecture variants (Phi-2, Phi-3).
- [x] 105. Support **Qwen** architecture variants.
- [x] 106. Support **GPT-NeoX** architectures.
- [x] 107. Support **OPT** architectures.
- [x] 108. Support **T5** (Encoder-Decoder) architectures.
- [x] 109. Support **BART** architectures.
- [x] 110. Support **Whisper** (Speech-to-Text) architectures.
- [x] 111. Build specialized graph modifiers to detect and optimize Llama attention.
- [x] 112. Implement fused FlashAttention-like kernels for WebGPU.
- [x] 113. Implement fused FlashAttention-like kernels for WASM (SIMD).
- [x] 114. Support weight-only quantization kernels (Int4/Int8 weights, FP32/FP16 compute).
- [x] 115. Support AWQ (Activation-aware Weight Quantization) execution.
- [x] 116. Support GPTQ execution.
- [x] 117. Handle custom vocabulary sizes dynamically.
- [x] 118. Implement MoE (Mixture of Experts) expert routing natively.
- [x] 119. Handle dynamic expert loading for MoE models in the browser.
- [x] 120. Optimize feed-forward network (FFN) fusions (SwiGLU, GeGLU).
- [x] 121. Support multi-modal inputs (passing image embeddings to LLMs).
- [x] 122. Implement vision encoder pipelines (e.g., CLIP) alongside GenAI.
- [x] 123. Support LoRA (Low-Rank Adaptation) adapter loading.
- [x] 124. Enable dynamic swapping of LoRA adapters during generation.
- [x] 125. Support speculative decoding (using a draft model to accelerate target model).

### Phase 6: API Mappings & Web Integration

- [x] 126. Create `onnx9000.genai.Model` Python API.
- [x] 127. Create `onnx9000.genai.GeneratorParams` Python API.
- [x] 128. Create `onnx9000.genai.Tokenizer` Python API.
- [x] 129. Create TypeScript bindings: `onnx9000-genai.ts`.
- [x] 130. Export `GeneratorParams` TS interface.
- [x] 131. Export `Model` TS interface.
- [x] 132. Export `Tokenizer` TS interface.
- [x] 133. Implement a Web Worker dedicated to the GenAI execution loop.
- [x] 134. Create a messaging protocol between main thread and GenAI worker.
- [x] 135. Expose an `AsyncGenerator` for TS streaming output (`for await (const token of ...)`).
- [x] 136. Support passing existing WebGPU device instances to the GenAI model.
- [x] 137. Implement memory progress callbacks (for downloading large LLM weights).
- [x] 138. Handle indexedDB caching of downloaded `.onnx` and `.safetensors` files.
- [x] 139. Implement automated hardware capability detection (selecting WASM vs WebGPU).
- [x] 140. Expose profiling data (tokens/sec, time-to-first-token) via the API.
- [x] 141. Ensure garbage collection of generator state when streams are closed.
- [x] 142. Support standard HuggingFace `generation_config.json` loading.
- [x] 143. Build an OpenAI-compatible REST API wrapper utilizing `onnx9000.genai` under the hood.
- [x] 144. Create a local web server utility serving the OpenAI-compatible endpoints.
- [x] 145. Implement cross-origin resource sharing (CORS) configurations for local serving.

### Phase 7: Generative Builders & Export Tooling

- [x] 146. Create `onnx9000.genai.builder` module to prepare standard models for GenAI.
- [x] 147. Implement a PyTorch to ONNX exporter specifically tuned for GenAI graph structures.
- [x] 148. Automate the insertion of KV cache inputs/outputs during export.
- [x] 149. Automate the conversion of static sequence lengths to dynamic axes.
- [x] 150. Implement a graph pass to remove unwanted past-state initializers.
- [x] 151. Build a CLI command: `onnx9000 genai build <model_id> --target webgpu`.
- [x] 152. Build a CLI command: `onnx9000 genai chat <model_path>`.
- [x] 153. Implement automatic folder structuring (creating `model.onnx`, `tokenizer.json`, etc.).
- [x] 154. Support splitting large models into chunks (`model-001.onnx`, `model-002.onnx`).
- [x] 155. Generate a manifest file describing the model chunk layout.
- [x] 156. Implement weight externalization during export to minimize proto size.
- [x] 157. Validate exported model structures against GenAI requirements.
- [x] 158. Provide an automated quantization step during the build process (e.g., INT4 block quantization).
- [x] 159. Support exporting with embedded tokenizers (storing tokenizer config inside ONNX metadata).
- [x] 160. Create integration scripts for downloading directly from HuggingFace Hub.

### Phase 8: Advanced Inference Techniques

- [x] 161. Implement prompt caching (saving KV states of frequent system prompts to disk/IDB).
- [x] 162. Implement batched prompt processing (processing prefix trees efficiently).
- [x] 163. Support grammar-guided generation (Constrained Decoding via BNF/EBNF).
- [x] 164. Support JSON schema-guided generation.
- [x] 165. Implement regex-guided generation.
- [x] 166. Handle stopping criteria based on complex string matching.
- [x] 167. Implement lookahead decoding techniques.
- [x] 168. Implement Medusa/EAGLE head support (generating multiple tokens per step).
- [x] 169. Support watermarking of generated text (e.g., Kirchenbauer et al.).
- [x] 170. Implement prefix matching for fast retrieval-augmented generation (RAG) updates.

### Phase 9: UI Components & Demos

- [x] 171. Build a barebones HTML/JS demo demonstrating browser-local Llama execution.
- [x] 172. Create a Web Components function: `useGenAI(modelUrl)`.
- [x] 173. Create a Vanilla utility: `useGenAI(modelUrl)`.
- [x] 174. Implement a terminal UI (TUI) chat interface for the CLI.
- [x] 175. Create a WebGL/Canvas visualizer showing token probabilities in real-time.
- [x] 176. Implement a drag-and-drop interface for loading local ONNX files.
- [x] 177. Provide a progressive web app (PWA) wrapper for offline GenAI usage.
- [x] 178. Create an example showing Whisper audio transcription feeding into an LLM.
- [x] 179. Create an example demonstrating streaming JSON extraction from unstructured text.
- [x] 180. Document best practices for memory management in mobile browsers.

### Phase 10: Performance, Testing, and Compliance

- [x] 181. Create unit tests for all logit processors.
- [x] 182. Create unit tests for beam search logic.
- [x] 183. Create unit tests for KV cache indexing math.
- [x] 184. Implement fuzzing for the BPE tokenizer logic.
- [x] 185. Benchmark Time-To-First-Token (TTFT) against standard ONNX Runtime.
- [x] 186. Benchmark Tokens-Per-Second (TPS) across various prompt lengths.
- [x] 187. Profile memory consumption during max-context generation.
- [x] 188. Ensure exact numerical parity (or within acceptable tolerance) with HuggingFace Transformers outputs.
- [x] 189. Create a regression test suite using known prompts and expected outputs.
- [x] 190. Verify correct handling of zero-length prompts.
- [x] 191. Verify correct handling of prompts exceeding the maximum context window.
- [x] 192. Ensure correct batch padding behavior in sequence batching.
- [x] 193. Implement logging for generation statistics (prompt tokens, completion tokens, times).
- [x] 194. Document the process for supporting a new model architecture.
- [x] 195. Create automated memory leak detection tests for the generator loop.
- [x] 196. Validate WebGPU shader precision constraints on different OS/Driver combinations.
- [x] 197. Ensure WASM fallback matches WebGPU output deterministically.
- [x] 198. Establish CI pipeline specifically for GenAI heavy integration tests.
- [x] 199. Write comprehensive API documentation for the `onnx9000.genai` namespace.
- [x] 200. Achieve 100% test coverage for the core generation loop state machine.

### Phase 11: Text-to-Image / Multi-modal GenAI

- [x] 201. Define `ImageGeneratorParams`.
- [x] 202. Implement UNet/DiT inference loop for diffusion models.
- [x] 203. Implement VAE (Variational Autoencoder) decoding step.
- [x] 204. Support DDIM scheduler.
- [x] 205. Support Euler Ancestral scheduler.
- [x] 206. Support PNDM scheduler.
- [x] 207. Support LCM (Latent Consistency Model) schedulers.
- [x] 208. Implement classifier-free guidance (CFG) logic.
- [x] 209. Support negative prompts handling.
- [x] 210. Implement latent noise generation with controlled seeds.
- [x] 211. Manage the multi-model pipeline (Text Encoder -> UNet -> VAE).
- [x] 212. Support Stable Diffusion v1.5 architectures.
- [x] 213. Support Stable Diffusion XL (SDXL) architectures.
- [x] 214. Support image-to-image generation (adding noise to base image).
- [x] 215. Support inpainting (handling mask inputs in the UNet loop).
- [x] 216. Implement ControlNet support alongside the UNet.
- [x] 217. Expose progressive image generation hooks (yielding partial images).
- [x] 218. Support exporting the VAE output directly to an HTML Canvas `ImageData` object.
- [x] 219. Handle dynamic resolution scaling.
- [x] 220. Implement memory optimizations for the diffusion loop to prevent WebGPU crashes.

### Phase 12: Audio GenAI

- [x] 221. Support Text-to-Speech (TTS) architectures (e.g., VITS).
- [x] 222. Support Bark architecture.
- [x] 223. Support MusicGen architecture.
- [x] 224. Implement streaming audio output (yielding PCM chunks).
- [x] 225. Handle mel-spectrogram generation loops.
- [x] 226. Integrate with Web Audio API for direct playback.
- [x] 227. Implement vocoder decoding logic.
- [x] 228. Handle multi-speaker embeddings.
- [x] 229. Ensure continuous audio generation without clicking artifacts.
- [x] 230. Provide Python and JS APIs for saving generated audio to `.wav`.

### Phase 13: Edge Case Handling & Stability

- [x] 231. Handle extremely large vocabularies (>100k tokens) without memory bloat.
- [x] 232. Manage Out-Of-Memory (OOM) WebGPU errors gracefully, falling back or shrinking cache.
- [x] 233. Handle NaN/Inf propagation during generation (resetting or skipping).
- [x] 234. Support aborting a generation request based on an external `AbortSignal`.
- [x] 235. Validate inputs against expected model shapes to prevent silent failures.
- [x] 236. Ensure thread safety in Python multi-threading environments for the generator.
- [x] 237. Ensure Web Worker isolation and lifecycle management in the browser.
- [x] 238. Provide robust error messages for malformed chat templates.
- [x] 239. Handle unexpected end-of-stream during model file downloading.
- [x] 240. Implement a safe mode that disables advanced sampling if incompatibilities arise.

### Phase 14: Ecosystem Integration

- [x] 241. Provide integration examples with LangChain.js.
- [x] 242. Provide integration examples with LlamaIndex.TS.
- [x] 243. Create a unified pipeline representation for sharing GenAI models on standard model hubs.
- [x] 244. Implement a conversion script mapping standard `GGUF` files to `onnx9000` GenAI packages.
- [x] 245. Support consuming metadata directly from HuggingFace `config.json`.
- [x] 246. Provide typing definitions compatible with major TS frameworks (Next.js, Nuxt).
- [x] 247. Create a Discord/Slack bot template using the local GenAI engine.
- [x] 248. Integrate with local vector databases for completely offline RAG applications.
- [x] 249. Publish benchmark comparisons against standard `llama.cpp` and `onnxruntime-genai`.
- [x] 250. Release final `v1.0` feature parity certification for GenAI capabilities.

### Phase 15: Deep WASM/WebGPU Optimizations

- [x] 251. Implement WebGPU buffer mapping to avoid redundant CPU-GPU copies during logit retrieval.
- [x] 252. Optimize WASM indirect function calls in the generation loop.
- [x] 253. Utilize WebGPU `compute` subgroups for fast logit reduction (max/sum) if available.
- [x] 254. Implement ring-buffer logic purely in WGSL to manage KV cache without host intervention.
- [x] 255. Support asynchronous WebGPU pipeline compilation during pre-fill phase to hide latency.
- [x] 256. Implement custom memory allocators in WASM specifically tuned for tensor lifecycle in GenAI.
- [x] 257. Profile and minimize JS garbage collection pauses during streaming.
- [x] 258. Support 16-bit float (fp16) WebGPU extensions globally for GenAI graphs.
- [x] 259. Implement pre-fetching of next-layer weights in memory-constrained environments.
- [x] 260. Develop custom shader generation strategies specifically for MoE routing logic.

### Phase 16: Extended Features

- [x] 261. Support drafting models for speculative decoding (running a small model to predict multiple tokens).
- [x] 262. Verify drafted tokens with the target model efficiently in a single pass.
- [x] 263. Implement self-consistency decoding (generating multiple paths and choosing the majority).
- [x] 264. Support explicit continuous batching API (adding sequences to an active batch queue).
- [x] 265. Implement a priority queue for continuous batching requests.
- [x] 266. Expose intermediate hidden states during generation for visualization or analysis.
- [x] 267. Implement prompt compression algorithms.
- [x] 268. Support chunked pre-filling (processing very long prompts in blocks to maintain UI responsiveness).
- [x] 269. Allow dynamic adjustment of generation parameters (e.g., temperature) mid-generation.
- [x] 270. Handle multi-turn conversation caching inherently within the `State` object.

### Phase 17: Multi-GPU / Distributed Execution

- [x] 271. Implement basic tensor parallelism (splitting weights across multiple WebGPU devices/contexts).
- [x] 272. Handle inter-device synchronization for multi-GPU setups.
- [x] 273. Support pipeline parallelism (allocating layers sequentially to different workers/devices).
- [x] 274. Create a master coordinator script for multi-worker distributed browser execution.
- [x] 275. Handle node failure/dropouts in a browser-based distributed generation setup.
- [x] 276. Implement communication primitives (AllReduce, AllGather) using WebRTC or BroadcastChannel.
- [x] 277. Profile communication overhead vs compute gains in distributed browser environments.
- [x] 278. Build a demonstration of collaborative inference (multiple users generating a single response).
- [x] 279. Support distributing the KV cache across multiple devices.
- [x] 280. Establish security protocols for sharing tensor data between browser contexts.

### Phase 18: Quality Assurance & Tooling

- [x] 281. Build a specialized debugger UI stepping through the generation loop token by token.
- [x] 282. Visualize attention maps generated during decoding.
- [x] 283. Visualize beam search trees dynamically.
- [x] 284. Implement a linter for custom sampling configurations.
- [x] 285. Generate detailed trace logs (compatible with Chrome Tracing) for the GenAI pipeline.
- [x] 286. Create a suite of "broken" models to ensure the runtime fails gracefully.
- [x] 287. Maintain a known-issues database mapped to specific hardware driver bugs (e.g., specific Android WebGPU issues).
- [x] 288. Automate testing of specific tokenizer corner cases (emoji, complex unicode).
- [x] 289. Provide a script to compare `onnx9000` logit outputs directly with Python PyTorch tensors.
- [x] 290. Implement a feature toggle system to disable experimental GenAI optimizations.

### Phase 19: Security & Safety

- [x] 291. Implement prompt injection detection heuristics.
- [x] 292. Integrate with content safety filters (e.g., Llama Guard) seamlessly in the pipeline.
- [x] 293. Ensure secure execution boundaries for Web Workers handling third-party models.
- [x] 294. Prevent malicious `.onnx` files from exploiting the GenAI memory allocator.
- [x] 295. Sanitize chat templates to prevent arbitrary code execution via template injection.
- [x] 296. Implement resource limits to prevent denial-of-service via infinite generation loops.
- [x] 297. Support encrypted model execution (decrypting weights dynamically in WASM memory).
- [x] 298. Validate digital signatures of downloaded GenAI packages.
- [x] 299. Ensure no sensitive KV cache data leaks between independent generation requests.
- [x] 300. Maintain strict Content Security Policy (CSP) compliance for web deployments.
