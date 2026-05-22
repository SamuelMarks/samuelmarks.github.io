# ONNX20: Transformers.js (WASM-Native Auto-Pipelines)

## Original Project Description

Transformers.js is a wildly popular JavaScript port of Hugging Face's `transformers` Python library. It enables developers to run pre-trained models (text, vision, audio, multimodal) directly in the browser or Node.js. It achieves this by combining `onnxruntime-web` for tensor execution with pure-JavaScript implementations of tokenizers, feature extractors, and data processors. It abstracts away the complexity of model execution by providing the `pipeline()` API, allowing users to perform tasks like sentiment analysis, image classification, or speech recognition with just a few lines of code.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

Instead of acting as a JavaScript wrapper around a massive compiled C++ runtime (`onnxruntime-web`) and relying on slow pure-JS data processors, `onnx9000` integrates the Transformers ecosystem natively into its AOT/WASM core.

- **WASM-Accelerated Processors:** Image resizing, Mel-spectrogram generation, and BPE tokenization are implemented as highly optimized WASM modules rather than pure JS, preventing UI thread blocking and offering near-native data preparation speeds.
- **Zero-Overhead Inference:** Uses `onnx9000`'s lightweight runtime or AOT-compiled WebGPU shaders instead of a 2MB-5MB generic execution provider.
- **Unified AutoClasses:** The Python and TypeScript/Browser codebases share the same architectural logic via the monolith, meaning a model supported in Python `onnx9000` is instantly available in the browser via `onnx9000.transformers`.
- **WebGPU First:** All Vision and Audio processing tensors seamlessly share memory spaces with the execution backend (WebGPU), eliminating expensive CPU-to-GPU memory copying during pipeline execution.

---

## Exhaustive Implementation Checklist

### Phase 1: Pipeline API & Task Orchestration

- [x] 1. Implement the base `Pipeline` class.
- [x] 2. Implement the `pipeline(task, model, ...)` factory function.
- [x] 3. Support `feature-extraction` pipeline (getting hidden states).
- [x] 4. Support `text-classification` pipeline (e.g., sentiment analysis).
- [x] 5. Support `token-classification` pipeline (e.g., NER, POS tagging).
- [x] 6. Support `question-answering` pipeline.
- [x] 7. Support `zero-shot-classification` pipeline.
- [x] 8. Support `translation` pipeline.
- [x] 9. Support `summarization` pipeline.
- [x] 10. Support `text-generation` pipeline (integrating with ONNX19 GenAI).
- [x] 11. Support `text2text-generation` pipeline.
- [x] 12. Support `fill-mask` pipeline.
- [x] 13. Support `image-classification` pipeline.
- [x] 14. Support `object-detection` pipeline.
- [x] 15. Support `zero-shot-image-classification` pipeline.
- [x] 16. Support `image-segmentation` pipeline.
- [x] 17. Support `depth-estimation` pipeline.
- [x] 18. Support `image-to-image` pipeline.
- [x] 19. Support `audio-classification` pipeline.
- [x] 20. Support `automatic-speech-recognition` (ASR) pipeline.
- [x] 21. Support `text-to-speech` (TTS) pipeline.
- [x] 22. Support `document-question-answering` pipeline.
- [x] 23. Support `visual-question-answering` pipeline.
- [x] 24. Support `image-feature-extraction` pipeline.
- [x] 25. Support pipeline batching (`[input1, input2]`).
- [x] 26. Implement `top_k` argument parsing in classification pipelines.
- [x] 27. Implement thresholding arguments in detection pipelines.
- [x] 28. Support generic `device` flag (mapping to WebGPU/WASM).
- [x] 29. Support `dtype` casting in pipelines (fp32, fp16, int8).
- [x] 30. Implement progressive callbacks in pipelines (for streaming or download progress).
- [x] 31. Implement pipeline pooling (keeping models hot in memory).
- [x] 32. Allow custom pre_process step overriding in pipelines.
- [x] 33. Allow custom post_process step overriding in pipelines.
- [x] 34. Allow forward step overriding in pipelines.
- [x] 35. Ensure structured error throwing for unsupported pipeline/model combos.

### Phase 2: Tokenizer Engine (Full HF Compatibility)

- [x] 36. Define `PreTrainedTokenizer` base class.
- [x] 37. Define `PreTrainedTokenizerFast` base class (WASM backed).
- [x] 38. Support loading `tokenizer_config.json`.
- [x] 39. Support loading `tokenizer.json` (the fast tokenizer format).
- [x] 40. Implement WASM BPE (Byte-Pair Encoding) implementation.
- [x] 41. Implement WASM WordPiece implementation.
- [x] 42. Implement WASM Unigram implementation.
- [x] 43. Handle `padding="max_length"` keyword argument.
- [x] 44. Handle `padding="longest"` keyword argument.
- [x] 45. Handle `padding=False` keyword argument.
- [x] 46. Handle `truncation=True` keyword argument.
- [x] 47. Handle `truncation="only_first"` keyword argument.
- [x] 48. Handle `truncation="only_second"` keyword argument.
- [x] 49. Handle `truncation="longest_first"` keyword argument.
- [x] 50. Handle `max_length` keyword argument.
- [x] 51. Handle `stride` keyword argument for overlapping contexts.
- [x] 52. Handle `return_tensors` ("np", "pt", "tf", "ort", "webgpu").
- [x] 53. Handle `return_attention_mask` keyword argument.
- [x] 54. Handle `return_token_type_ids` keyword argument.
- [x] 55. Handle `return_overflowing_tokens` keyword argument.
- [x] 56. Handle `return_special_tokens_mask` keyword argument.
- [x] 57. Handle `return_offsets_mapping` keyword argument.
- [x] 58. Implement word to token ID mapping (`word_ids()`).
- [x] 59. Implement character to token ID mapping (`char_to_token()`).
- [x] 60. Implement token to character mapping (`token_to_chars()`).
- [x] 61. Support text pairs (Sentence A, Sentence B).
- [x] 62. Implement special token addition logic.
- [x] 63. Handle `bos_token`, `eos_token`, `unk_token`, `sep_token`, `pad_token`, `cls_token`, `mask_token`.
- [x] 64. Process complex `AddedToken` configurations (lstrip, rstrip, single_word).
- [x] 65. Implement `decode()` and `batch_decode()`.
- [x] 66. Support `skip_special_tokens` in decoding.
- [x] 67. Support `clean_up_tokenization_spaces` in decoding.
- [x] 68. Implement regex-based pre-tokenizers in WASM.
- [x] 69. Implement byte-level pre-tokenizers.
- [x] 70. Implement Metaspace pre-tokenizers.
- [x] 71. Implement punctuation splitting pre-tokenizers.
- [x] 72. Implement decoders (ByteLevel, WordPiece, Metaspace).
- [x] 73. Provide a fallback JS implementation for non-WASM environments.
- [x] 74. Implement chat templates using a lightweight JS Jinja engine.
- [x] 75. Validate inputs (strings, lists of strings, nested lists).

### Phase 3: Vision Processors & Image Handling

- [x] 76. Define `BaseImageProcessor` interface.
- [x] 77. Create `onnx9000.Image` object wrapper (handling Canvas/ImageData/Blob/URL).
- [x] 78. Support loading images directly from URLs natively.
- [x] 79. Support loading images from base64 strings.
- [x] 80. Implement `do_resize` logic.
- [x] 81. Implement WASM-accelerated bilinear interpolation resizing.
- [x] 82. Implement WASM-accelerated bicubic interpolation resizing.
- [x] 83. Implement WASM-accelerated nearest-neighbor interpolation resizing.
- [x] 84. Implement `do_center_crop` logic.
- [x] 85. Implement `do_random_crop` logic.
- [x] 86. Implement `do_pad` logic (constant, reflect, edge padding).
- [x] 87. Implement `do_rescale` (e.g., multiplying by 1/255).
- [x] 88. Implement `do_normalize` (subtracting mean, dividing by std).
- [x] 89. Support custom `image_mean` and `image_std` parameters.
- [x] 90. Handle layout conversion (HWC to CHW format).
- [x] 91. Implement `ImageProcessor` batching (lists of images).
- [x] 92. Support `return_tensors` specifically for WebGPU image uploads.
- [x] 93. Create specialized `ViTImageProcessor`.
- [x] 94. Create specialized `CLIPImageProcessor`.
- [x] 95. Create specialized `DeiTImageProcessor`.
- [x] 96. Create specialized `DetrImageProcessor`.
- [x] 97. Create specialized `YolosImageProcessor`.
- [x] 98. Implement bounding box drawing utilities on HTML Canvas.
- [x] 99. Implement segmentation mask drawing utilities on HTML Canvas.
- [x] 100. Write WebGPU shaders for on-device image normalization to bypass CPU.
- [x] 101. Write WebGPU shaders for on-device image resizing.
- [x] 102. Support Exif orientation correction before processing.
- [x] 103. Handle RGBA to RGB conversion.
- [x] 104. Handle Grayscale to RGB conversion.
- [x] 105. Optimize raw pixel array copying into WASM heap.

### Phase 4: Audio Processors (Feature Extractors)

- [x] 106. Define `SequenceFeatureExtractor` base class.
- [x] 107. Create `onnx9000.Audio` object wrapper.
- [x] 108. Support loading audio from URLs.
- [x] 109. Support loading audio from Blob/File objects.
- [x] 110. Integrate `AudioContext` for in-browser audio decoding.
- [x] 111. Implement WASM-accelerated 1D audio resampling.
- [x] 112. Implement `do_pad` for audio (zero padding, reflection).
- [x] 113. Implement `do_truncate` for audio sequence lengths.
- [x] 114. Support `return_attention_mask` for padded audio sequences.
- [x] 115. Implement Short-Time Fourier Transform (STFT) in WASM.
- [x] 116. Implement Windowing functions (Hann, Hamming, Mel) in WASM.
- [x] 117. Implement Mel-filterbank matrix generation.
- [x] 118. Implement Mel-spectrogram computation pipeline (STFT -> Power -> Mel).
- [x] 119. Implement log10 application for log-mel spectrograms.
- [x] 120. Create specialized `WhisperFeatureExtractor`.
- [x] 121. Create specialized `Wav2Vec2FeatureExtractor`.
- [x] 122. Create specialized `SpeechT5FeatureExtractor`.
- [x] 123. Implement raw waveform chunking (for long-form audio processing).
- [x] 124. Handle multi-channel audio (downmixing to mono).
- [x] 125. Normalize audio amplitude arrays (zero mean, unit variance).
- [x] 126. Support Voice Activity Detection (VAD) pre-processing (optional extension).
- [x] 127. Implement Web Audio API Worklet for streaming feature extraction.
- [x] 128. Optimize memory usage during large Mel-spectrogram generation.
- [x] 129. Implement audio output formatters (Float32Array to WAV blob).
- [x] 130. Ensure floating point determinism across JS, WASM, and WebGPU for audio ops.

### Phase 5: Auto Classes & Hub Integration

- [x] 131. Implement `AutoConfig.from_pretrained()`.
- [x] 132. Implement `AutoTokenizer.from_pretrained()`.
- [x] 133. Implement `AutoFeatureExtractor.from_pretrained()`.
- [x] 134. Implement `AutoProcessor.from_pretrained()`.
- [x] 135. Implement `AutoModel.from_pretrained()`.
- [x] 136. Implement `AutoModelForSequenceClassification`.
- [x] 137. Implement `AutoModelForTokenClassification`.
- [x] 138. Implement `AutoModelForQuestionAnswering`.
- [x] 139. Implement `AutoModelForCausalLM`.
- [x] 140. Implement `AutoModelForMaskedLM`.
- [x] 141. Implement `AutoModelForSeq2SeqLM`.
- [x] 142. Implement `AutoModelForImageClassification`.
- [x] 143. Implement `AutoModelForObjectDetection`.
- [x] 144. Implement `AutoModelForSpeechSeq2Seq`.
- [x] 145. Implement fetching from `hf.co` Hub REST API.
- [x] 146. Support custom Hub endpoints/mirrors.
- [x] 147. Implement API Key authentication for private models.
- [x] 148. Support `revision` flag (fetching specific branches/commits).
- [x] 149. Support resolving ONNX filenames (`model.onnx`, `model_quantized.onnx`).
- [x] 150. Implement IndexedDB caching via CacheStorage API for models.
- [x] 151. Implement ETag checking to prevent redundant model downloads.
- [x] 152. Implement concurrent multipart file downloading for large models.
- [x] 153. Implement fallback caching strategies for Node.js (`fs`).
- [x] 154. Support reading models from local directory paths.
- [x] 155. Provide an API to clear/manage the downloaded model cache.

### Phase 6: Core Model Execution Wrappers

- [x] 156. Define `PreTrainedModel` base class.
- [x] 157. Connect `PreTrainedModel` to the `onnx9000` execution backend.
- [x] 158. Implement model initialization logic (loading weights into WebGPU/WASM).
- [x] 159. Parse `config.json` to configure model input/output layers dynamically.
- [x] 160. Manage `session_options` (threads, execution providers).
- [x] 161. Implement the `__call__` / `forward` method abstracting the inference session.
- [x] 162. Handle dynamic batch size resolution prior to graph execution.
- [x] 163. Map model-specific input names (e.g., `input_ids`, `pixel_values`).
- [x] 164. Map model-specific output names (e.g., `logits`, `last_hidden_state`).
- [x] 165. Support external data format files (`model.onnx_data`).
- [x] 166. Implement automatic input casting (e.g., BigInt64 to Int32 for WASM).
- [x] 167. Handle `attention_mask` application internally if required by specific ops.
- [x] 168. Attach the `GenerationMixin` for text-generative models.
- [x] 169. Provide explicit memory disposal methods (`model.dispose()`).
- [x] 170. Create debugging mode to trace input/output shapes per execution step.

### Phase 7: Post-Processing & Output Generation

- [x] 171. Implement generic `post_process` hooks.
- [x] 172. Implement post-processing for Text Classification (applying Softmax, indexing `id2label`).
- [x] 173. Implement post-processing for Token Classification (aggregating sub-words, aligning offsets).
- [x] 174. Implement post-processing for Question Answering (finding max start/end logits, span extraction).
- [x] 175. Implement post-processing for Zero-Shot Classification (NLI entailment/contradiction mapping).
- [x] 176. Implement post-processing for Image Classification (Softmax -> Top K).
- [x] 177. Implement post-processing for Object Detection.
- [x] 178. Build Non-Maximum Suppression (NMS) in WASM.
- [x] 179. Build bounding box denormalization (cx,cy,w,h to xmin,ymin,xmax,ymax).
- [x] 180. Implement post-processing for Semantic Segmentation (argmax over spatial dims).
- [x] 181. Support chunked output decoding for ASR (Whisper timestamps processing).
- [x] 182. Construct `ModelOutput` classes (similar to HF dictionaries).
- [x] 183. Support raw output returning (`return_tensors=True` on pipelines).
- [x] 184. Support streaming generation responses (Generators/AsyncIterators).

### Phase 8: NLP Architecture Support (Validation)

- [x] 185. Validate end-to-end `BERT` pipeline.
- [x] 186. Validate end-to-end `RoBERTa` pipeline.
- [x] 187. Validate end-to-end `DistilBERT` pipeline.
- [x] 188. Validate end-to-end `ALBERT` pipeline.
- [x] 189. Validate end-to-end `DeBERTa` pipeline.
- [x] 190. Validate end-to-end `MobileBERT` pipeline.
- [x] 191. Validate end-to-end `T5` pipeline.
- [x] 192. Validate end-to-end `BART` pipeline.
- [x] 193. Validate end-to-end `MarianMT` pipeline.
- [x] 194. Validate end-to-end `GPT-2` pipeline.
- [x] 195. Validate end-to-end `LLaMA` pipeline (integrating GenAI capabilities).
- [x] 196. Validate end-to-end `Mistral` pipeline.
- [x] 197. Validate end-to-end `Gemma` pipeline.
- [x] 198. Validate end-to-end `Phi` pipeline.
- [x] 199. Handle missing token type IDs cleanly for architectures that ignore them.
- [x] 200. Ensure position ID injection works for models without internal generators.

### Phase 9: Vision & Audio Architecture Support (Validation)

- [x] 201. Validate end-to-end `ViT` (Vision Transformer) pipeline.
- [x] 202. Validate end-to-end `ResNet` pipeline.
- [x] 203. Validate end-to-end `Swin` Transformer pipeline.
- [x] 204. Validate end-to-end `MobileNetV2` pipeline.
- [x] 205. Validate end-to-end `ConvNeXT` pipeline.
- [x] 206. Validate end-to-end `DETR` pipeline.
- [x] 207. Validate end-to-end `YOLOS` pipeline.
- [x] 208. Validate end-to-end `SegFormer` pipeline.
- [x] 209. Validate end-to-end `CLIP` pipeline (Image + Text).
- [x] 210. Validate end-to-end `OwlViT` pipeline.
- [x] 211. Validate end-to-end `BLIP` pipeline.
- [x] 212. Validate end-to-end `TrOCR` pipeline.
- [x] 213. Validate end-to-end `Whisper` pipeline (ASR).
- [x] 214. Validate end-to-end `Wav2Vec2` pipeline (ASR).
- [x] 215. Validate end-to-end `SpeechT5` pipeline (TTS).
- [x] 216. Validate end-to-end `Hubert` pipeline.
- [x] 217. Validate end-to-end `Clap` pipeline.

### Phase 10: Utility, Math & Tensor Interop

- [x] 218. Implement `softmax(tensor, axis)` utility.
- [x] 219. Implement `log_softmax(tensor, axis)` utility.
- [x] 220. Implement `sigmoid(tensor)` utility.
- [x] 221. Implement `get_top_k(tensor, k)` utility.
- [x] 222. Implement `cosine_similarity(a, b)` utility.
- [x] 223. Implement `dot_product(a, b)` utility.
- [x] 224. Ensure utilities auto-dispatch to WASM/WebGPU for large tensors.
- [x] 225. Expose tensor shape manipulation (`view`, `reshape`, `transpose`).
- [x] 226. Provide bi-directional conversion: `onnx9000.Tensor` <-> `Float32Array`.
- [x] 227. Provide bi-directional conversion: `onnx9000.Tensor` <-> standard JSON arrays.
- [x] 228. Handle multi-dimensional array slicing syntaxes in TS.
- [x] 229. Support strided array access logic in JS wrappers.
- [x] 230. Implement `Math.erf` polyfills if necessary.

### Phase 11: Export Tooling & Python Parity

- [x] 231. Ensure Python API `onnx9000.transformers.pipeline()` matches JS API perfectly.
- [x] 232. Implement auto-conversion script (`onnx9000 transformers export <model_id>`).
- [x] 233. Generate `.onnx` files targeting optimal WebGPU topologies during export.
- [x] 234. Generate optimized `tokenizer.json` files.
- [x] 235. Extract and format `preprocessor_config.json`.
- [x] 236. Extract and format `generation_config.json`.
- [x] 237. Bundle pipeline configurations into `onnx9000-pipeline.json` for rapid loading.
- [x] 238. Provide INT8 dynamic quantization during export.
- [x] 239. Provide FP16 casting during export.
- [x] 240. Publish an equivalent to `optimum-cli` natively within `onnx9000`.

### Phase 12: Worker & Web-Native Optimizations

- [x] 241. Implement `WorkerPipeline` wrapper to execute pipelines entirely in a Web Worker.
- [x] 242. Support Zero-Copy transfer of `Float32Array` buffers between main thread and workers.
- [x] 243. Create message passing interface for streaming worker text generation.
- [x] 244. Implement `SharedArrayBuffer` support for multi-threading if CORS/COOP allows.
- [x] 245. Expose memory limit configurations (e.g., throwing error instead of crashing browser).
- [x] 246. Support off-thread image decoding using `createImageBitmap`.
- [x] 247. Prevent main thread blocking during large model compilation (WebGPU async pipeline creation).
- [x] 248. Support Service Workers to preload pipelines for completely offline PWA experiences.
- [x] 249. Integrate `requestIdleCallback` for non-blocking background model initialization.
- [x] 250. Provide detailed performance tracing API (Network vs Compilation vs Inference time).

### Phase 13: Edge Case Handling

- [x] 251. Handle inputs exceeding maximum sequence length gracefully (auto-truncation).
- [x] 252. Manage WebGPU context loss and restore without application crash.
- [x] 253. Handle completely empty text inputs.
- [x] 254. Handle empty/zero-dimension images.
- [x] 255. Catch and log unhandled exceptions securely without leaking internal paths.
- [x] 256. Handle missing properties in older `config.json` revisions.
- [x] 257. Provide graceful fallbacks for models without `generation_config`.
- [x] 258. Support environments without IndexedDB (e.g., Incognito Mode).
- [x] 259. Support environments without `fetch` API (Node.js fallback).
- [x] 260. Manage circular dependencies in pipeline module loading.

### Phase 14: Quality Assurance & Testing

- [x] 261. Achieve 100% API compatibility with Hugging Face's `transformers.js` v2/v3 syntax.
- [x] 262. Create CI tests comparing Python HF outputs with TS `onnx9000` outputs.
- [x] 263. Establish a daily test suite running against the top 100 HF models.
- [x] 264. Unit test every tokenizer configuration option independently.
- [x] 265. Unit test WASM image resizing against standard Pillow/OpenCV outputs.
- [x] 266. Unit test WASM STFT/Mel outputs against librosa outputs.
- [x] 267. Track memory leaks using Chrome DevTools automated puppeteer tests.
- [x] 268. Maintain benchmarking dashboards comparing `onnx9000` vs `onnxruntime-web`.
- [x] 269. Enforce strict TypeScript typing for all public APIs and Config objects.
- [x] 270. Create interactive notebook tutorials (Jupyter/Observable) demonstrating usage.

### Phase 15: Developer Experience & Ecosystem

- [x] 271. Provide Vanilla HTML/JS boilerplate template using `onnx9000.transformers`.
- [x] 272. Provide Vanilla HTML/JS boilerplate template.
- [x] 273. Provide Chrome Extension boilerplate utilizing background scripts.
- [x] 274. Create comprehensive documentation for migrating from `transformers.js`.
- [x] 275. Support importing from CDNs (unpkg, jsdelivr) as an ES Module.
- [x] 276. Build an interactive web playground (like HF Spaces) exclusively running `onnx9000`.
- [x] 277. Implement a CLI tool to start a local REST API mimicking Hugging Face Inference Endpoints.
- [x] 278. Implement a Node.js C++ addon bridge as a fallback for ultra-heavy models.
- [x] 279. Support direct integration with `LangChain.js` tools and embeddings.
- [x] 280. Integrate with the `Gradio` Python library.

### Phase 16: Extended Pipeline Features

- [x] 281. Support returning probabilities for all classes in classification pipelines.
- [x] 282. Add sentiment scores mapping (1 star to 5 star conversions).
- [x] 283. Support multi-label classification post-processing (sigmoid instead of softmax).
- [x] 284. Allow providing a custom `id2label` dictionary at runtime.
- [x] 285. Implement context aggregation in Question Answering for long documents.
- [x] 286. Handle batch generation padding dynamically.
- [x] 287. Implement image tiling for high-resolution object detection.
- [x] 288. Add vocal isolation/stemming capabilities to audio pipelines.
- [x] 289. Add face detection specific utilities (wrapping generic object detection).
- [x] 290. Support semantic search utilities (cosine similarity wrappers over feature extraction).

### Phase 17: Security & Reliability

- [x] 291. Validate all model tensors to ensure bounds checking.
- [x] 292. Implement a safe-loading mode that refuses to execute models with custom code.
- [x] 293. Sandbox Web Workers executing untrusted user models.
- [x] 294. Secure cache storage against cross-site scripting (XSS) extraction.
- [x] 295. Implement resource-limit quotas for auto-downloading models.
- [x] 296. Enforce strict Content Security Policy (CSP) guidelines in generated boilerplate.
- [x] 297. Support offline-only mode (throwing errors instead of reaching out to network).
- [x] 298. Validate digital signatures of official ONNX model binaries.
- [x] 299. Prevent prototype pollution in configuration parsers.
- [x] 300. Release v1.0 feature complete certification.
