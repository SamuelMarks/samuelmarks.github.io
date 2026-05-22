# ONNX Runtime Extensions Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `onnxruntime-extensions` within the `onnx9000` ecosystem.
The original project embeds heavy C++ and Rust libraries (Hugging Face `tokenizers`, `OpenCV`, `librosa`, etc.) as custom operators to allow ONNX models to handle raw text, images, and audio directly.
Our `onnx9000` reimplementation uses a pure-Python and pure-JavaScript strategy. We provide zero-dependency implementations of Byte-Pair Encoding (BPE), SentencePiece, audio feature extraction (STFT/MelSpectrogram), and image decoding natively. In the browser, this means leveraging native WebCodecs, WebAudio APIs, and optimized RegExp instead of downloading massive compiled binaries, enabling end-to-end models (Text -> Predictions or Audio -> Predictions) natively on the web.

## Exhaustive Parity Checklist

### 1. NLP Text Tokenization & Processing (Python & JS Parity) (45+ items)

- [xx] Implement `ai.onnx.contrib.StringSplit`
- [xx] Implement `ai.onnx.contrib.StringJoin`
- [xx] Implement `ai.onnx.contrib.StringLower`
- [xx] Implement `ai.onnx.contrib.StringUpper`
- [xx] Implement `ai.onnx.contrib.StringRegexReplace` (Using native Python `re` / JS `RegExp`)
- [xx] Implement `ai.onnx.contrib.StringRegexSplitWithOffsets`
- [xx] Implement `ai.onnx.contrib.StringLength`
- [xx] Implement `ai.onnx.contrib.StringConcat`
- [xx] Implement `ai.onnx.contrib.StringMapping` (Dictionary lookup)
- [xx] Implement zero-dependency `GPT2Tokenizer` (BPE algorithm)
- [xx] Implement zero-dependency `RobertaTokenizer` (BPE algorithm)
- [xx] Implement zero-dependency `BertTokenizer` (WordPiece algorithm)
- [xx] Implement zero-dependency `SentencepieceTokenizer` (Unigram / BPE)
- [xx] Implement `ai.onnx.contrib.BlingFireSentenceBreaker`
- [xx] Implement `ai.onnx.contrib.BpeDecoder`
- [xx] Implement `ai.onnx.contrib.BpeEncoder`
- [xx] Implement `ai.onnx.contrib.BasicTokenizer` (Punctuation splitting, lowercasing)
- [xx] Implement `ai.onnx.contrib.WordpieceTokenizer`
- [xx] Implement `ai.onnx.contrib.SentencepieceTokenizer`
- [xx] Implement `ai.onnx.contrib.SentencepieceDecoder`
- [xx] Map Hugging Face `tokenizer.json` directly to ONNX CustomOp attributes automatically
- [xx] Parse `vocab.txt` directly into `WordpieceTokenizer` ONNX attributes
- [xx] Parse `merges.txt` directly into `BpeEncoder` ONNX attributes
- [xx] Parse `tokenizer.model` (SentencePiece Protobuf) purely in Python without the `sentencepiece` pip package
- [xx] Implement fast prefix-tree (Trie) token matching purely in Python/JS for WordPiece
- [xx] Handle explicit `[UNK]` (Unknown) token fallbacks natively
- [xx] Handle explicit `[CLS]`, `[SEP]`, `[PAD]`, `[MASK]` special tokens natively
- [xx] Support generating `input_ids` directly from raw strings
- [xx] Support generating `attention_mask` natively inside the custom op
- [xx] Support generating `token_type_ids` natively inside the custom op
- [xx] Implement subword regularization (BPE dropout) if supported
- [xx] Implement Byte-Level BPE (mapping raw bytes to tokens, critical for GPT-2/Llama)
- [xx] Ensure Unicode normalization (NFC, NFD, NFKC, NFKD) matches `unicodedata` in Python
- [xx] Emulate Python's `unicodedata.normalize` strictly in pure JavaScript
- [xx] Emulate Python's `re` module specific negative lookbehinds natively in JS if required
- [xx] Provide dynamic decoding (Tokens -> String) matching `tokenizers` outputs exactly
- [xx] Handle spacing prefix logic (` ` or `##`) strictly per model requirements
- [xx] Optimize BPE merging loop to execute in < 10ms per sentence in pure JS
- [xx] Support batch string tokenization (`['text1', 'text2']`) natively
- [xx] Pad sequences dynamically based on batch maximum length natively
- [xx] Expose native `max_length` truncation rules explicitly
- [xx] Expose native `truncation_strategy` (e.g. `longest_first`) explicitly
- [xx] Emit precise `offset_mapping` tensors for Named Entity Recognition (NER) alignment
- [xx] Catch heavily malformed UTF-8 gracefully without crashing the graph
- [xx] Ensure BPE dictionaries can be passed dynamically as inputs OR baked as attributes

### 2. Audio Processing & Speech Feature Extraction (45+ items)

- [xx] Implement `ai.onnx.contrib.AudioDecoder` (Using native OS libraries / WebAudio API)
- [xx] Implement `ai.onnx.contrib.AudioDecoder` using `ffmpeg-python` or `librosa` natively as a fallback
- [xx] Implement `ai.onnx.contrib.AudioDecoder` in JS using native HTML5 `<audio>` / WebCodecs
- [xx] Parse WAV headers purely in Python (no dependencies)
- [xx] Parse WAV headers purely in JavaScript
- [xx] Support decoding MP3 (if host OS or Browser supports it)
- [xx] Support decoding FLAC
- [xx] Implement `ai.onnx.contrib.Resample` (Audio rate conversion)
- [xx] Implement fast 1D linear interpolation for resampling natively
- [xx] Implement Sinc interpolation (high quality) for resampling natively
- [xx] Implement `ai.onnx.contrib.Window` (Hann, Hamming, Blackman, etc.)
- [xx] Implement `ai.onnx.contrib.STFT` (Short-Time Fourier Transform) natively in Python
- [xx] Implement `STFT` natively in JavaScript using TypedArrays
- [xx] Ensure exact numerical parity with `torch.stft(return_complex=True)`
- [xx] Support `n_fft`, `hop_length`, `win_length` parameters natively
- [xx] Implement `ai.onnx.contrib.MelSpectrogram` natively
- [xx] Generate Mel filterbanks entirely in math (zero-dependency) mirroring `librosa.filters.mel`
- [xx] Implement `f_min` and `f_max` bounding for Mel filterbanks
- [xx] Support `htk` vs `slaney` Mel scale formulas
- [xx] Implement `ai.onnx.contrib.Mfcc` (Mel-Frequency Cepstral Coefficients)
- [xx] Implement Discrete Cosine Transform (DCT) Type II for MFCC purely in math
- [xx] Implement `ai.onnx.contrib.Spectrogram` (Magnitude/Power spectrum)
- [xx] Implement `ai.onnx.contrib.LogMelSpectrogram`
- [xx] Implement Amplitude to DB scale conversion natively (`10 * log10(S / ref)`)
- [xx] Implement exact `top_db` clipping parity with `librosa.power_to_db`
- [xx] Implement Whisper specific Log-Mel Spectrogram (exactly 80/128 Mel bins)
- [xx] Extract exactly the required 30-second Whisper padding dynamically
- [xx] Implement `ai.onnx.contrib.InverseSpectrogram` (Griffin-Lim approximation)
- [xx] Implement `ai.onnx.contrib.InverseSTFT` (iSTFT) purely in math
- [xx] Handle complex number multiplication manually in JS TypedArrays for iSTFT
- [xx] Overlap-add (OLA) logic implemented strictly in 1D array loops
- [xx] Test audio decoding performance in WebWorker (avoiding main thread UI freezing)
- [xx] Ensure `STFT` frame padding matches `center=True` PyTorch padding identically
- [xx] Ensure `STFT` frame padding matches `center=False` PyTorch padding identically
- [xx] Expose native `Float32Array` representations directly from WebAudio `OfflineAudioContext`
- [xx] Resample explicitly using `OfflineAudioContext` sampleRate properties in the browser
- [xx] Bypass strict WebAudio CORS restrictions using explicit ArrayBuffer loading
- [xx] Extract multiple channels (Stereo -> Mono downmixing) natively
- [xx] Provide a fallback for environments lacking WebAudio (Node.js) using pure JS math
- [xx] Implement `ai.onnx.contrib.Fbank` (Kaldi-compatible filterbanks)
- [xx] Compare `STFT` outputs to `librosa` reference test suite (atol=1e-4)
- [xx] Compare `MelSpectrogram` outputs to PyTorch `torchaudio` reference test suite
- [xx] Implement custom Fast Fourier Transform (FFT) radix-2 algorithm in pure JS
- [xx] Implement FFT radix-2 natively in pure Python
- [xx] Validate audio length exact bounds after resampling to prevent out-of-bounds indexing

### 3. Vision Preprocessing & Image Decoders (40+ items)

- [xx] Implement `ai.onnx.contrib.ImageDecoder` natively in Python via `Pillow` (if available)
- [xx] Implement `ImageDecoder` natively in JS via `Image` object / `createImageBitmap`
- [xx] Support decoding JPEG directly to Tensor `[H, W, C]`
- [xx] Support decoding PNG directly to Tensor `[H, W, C]`
- [xx] Support decoding WEBP directly to Tensor `[H, W, C]`
- [xx] Map Canvas `ImageData` instantly to ONNX `Float32Array` in JS
- [xx] Map `ImageDecoder` output directly to `[B, C, H, W]` if requested
- [xx] Implement `ai.onnx.contrib.Resize` (Nearest, Bilinear, Bicubic) natively
- [xx] Ensure `Resize` aligns with OpenCV `cv2.resize` exactly (pixel grid alignment)
- [xx] Ensure `Resize` aligns with Pillow `Image.resize` exactly
- [xx] Implement `ai.onnx.contrib.CenterCrop`
- [xx] Implement `ai.onnx.contrib.Pad` (Image specific edge/reflect padding)
- [xx] Implement `ai.onnx.contrib.Normalize` (Subtract Mean, Divide by Std) natively
- [xx] Support dynamic normalization statistics (e.g. ImageNet `mean=[0.485, 0.456, 0.406]`)
- [xx] Implement `ai.onnx.contrib.FormatConversion` (RGB to BGR)
- [xx] Implement `ai.onnx.contrib.FormatConversion` (RGB to Grayscale)
- [xx] Implement `ai.onnx.contrib.FormatConversion` (RGB to YUV)
- [xx] Implement `ai.onnx.contrib.FormatConversion` (YUV to RGB)
- [xx] Implement `ai.onnx.contrib.DrawBoundingBoxes` (For object detection visualization)
- [xx] Implement text rendering in `DrawBoundingBoxes` natively using JS `<canvas>` context
- [xx] Implement Python text rendering via `Pillow.ImageDraw`
- [xx] Extract Video Frames natively in JS using HTML5 `<video>` element capturing
- [xx] Implement `ai.onnx.contrib.ExtractVideoFrames` in Python using `av` or `OpenCV` fallback
- [xx] Read Video natively via `WebCodecs` API for high-performance zero-copy WebGPU upload
- [xx] Provide exact scaling logic for `Resize` with `keep_aspect_ratio` and padding
- [xx] Implement `ai.onnx.contrib.GaussianBlur` natively in pure math (2D convolution)
- [xx] Implement `ai.onnx.contrib.MedianBlur`
- [xx] Implement `ai.onnx.contrib.CannyEdgeDetector` purely in math (Sobel + NMS + Hysteresis)
- [xx] Map PyTorch `torchvision.transforms.ToTensor()` exact scaling rules (`/ 255.0`)
- [xx] Map PyTorch `torchvision.transforms.Normalize()` exact behavior
- [xx] Map PIL image reading orientation automatically (EXIF rotation correction)
- [xx] Decode base64 embedded images automatically in `ImageDecoder`
- [xx] Handle explicit alpha channel dropping (RGBA -> RGB) natively
- [xx] Handle explicit alpha channel blending on white backgrounds
- [xx] Expose WebGL based hardware-accelerated image resizing in the browser
- [xx] Provide pure Python fallback image resizing if `Pillow` is not installed
- [xx] Evaluate strict pixel-perfect validation against `cv2.resize` output arrays
- [xx] Test massive image resizing (4K -> 224x224) natively in WASM without memory leaks
- [xx] Extract image EXIF metadata explicitly if requested
- [xx] Extract Multi-page TIFFs or animated GIFs natively if supported

### 4. Mathematical Custom Ops & Specific Topologies (35+ items)

- [xx] Implement `ai.onnx.contrib.Inverse` (Matrix Inversion) natively via Gaussian Elimination / LU Decomposition
- [xx] Implement `Inverse` natively in JS for arbitrary square matrices
- [xx] Implement `ai.onnx.contrib.Gelu` (if exact standard ONNX Gelu is missing)
- [xx] Implement `ai.onnx.contrib.FastGelu`
- [xx] Implement `ai.onnx.contrib.BiasGelu`
- [xx] Implement `ai.onnx.contrib.QuickGelu`
- [xx] Implement `ai.onnx.contrib.LayerNormalization`
- [xx] Implement `ai.onnx.contrib.SkipLayerNormalization`
- [xx] Implement `ai.onnx.contrib.EmbedLayerNormalization`
- [xx] Implement `ai.onnx.contrib.Attention` (Standard multi-head dot product attention)
- [xx] Implement `Attention` natively optimized for WebGPU (fusing QKV natively)
- [xx] Implement `ai.onnx.contrib.DecoderAttention`
- [xx] Implement `ai.onnx.contrib.RotaryPositionalEmbedding` (RoPE)
- [xx] Implement `ai.onnx.contrib.GroupNorm`
- [xx] Implement `ai.onnx.contrib.DynamicQuantizeLinear`
- [xx] Implement `ai.onnx.contrib.DynamicQuantizeMatMul`
- [xx] Implement `ai.onnx.contrib.QAttention`
- [xx] Implement `ai.onnx.contrib.QGemm`
- [xx] Implement `ai.onnx.contrib.QLinearAdd`
- [xx] Implement `ai.onnx.contrib.QLinearAveragePool`
- [xx] Implement `ai.onnx.contrib.QLinearConcat`
- [xx] Implement `ai.onnx.contrib.QLinearGlobalAveragePool`
- [xx] Implement `ai.onnx.contrib.QLinearLeakyRelu`
- [xx] Implement `ai.onnx.contrib.QLinearMul`
- [xx] Implement `ai.onnx.contrib.QLinearReduceMean`
- [xx] Implement `ai.onnx.contrib.QLinearSigmoid`
- [xx] Implement `ai.onnx.contrib.NhwcConv`
- [xx] Implement `ai.onnx.contrib.NhwcMaxPool`
- [xx] Implement `ai.onnx.contrib.Bifurcation`
- [xx] Implement `ai.onnx.contrib.GridSample` (Bilinear interpolation natively)
- [xx] Map `torch.nn.functional.grid_sample` exact edge cases (`align_corners=True/False`)
- [xx] Implement `ai.onnx.contrib.Unique`
- [xx] Implement `ai.onnx.contrib.TopK`
- [xx] Support custom CUDA kernels via PyCUDA / Numba if executed strictly in Python
- [xx] Support custom WGSL injections for any `ai.onnx.contrib.*` math op in the browser

### 5. Generative AI & Decoding Architectures (30+ items)

- [xx] Implement `ai.onnx.contrib.BeamSearch` (Standard auto-regressive decoding loop)
- [xx] Implement `ai.onnx.contrib.GreedySearch`
- [xx] Implement `ai.onnx.contrib.Sampling` (Top-K / Top-P logic natively)
- [xx] Implement Native Python Generator `yield` for text streaming from `BeamSearch` / `GreedySearch`
- [xx] Implement Native JS Async Generator `yield` for text streaming to the UI
- [xx] Support explicit `temperature` scaling inside the CustomOp
- [xx] Support explicit `repetition_penalty` scaling
- [xx] Support explicit `length_penalty` inside BeamSearch
- [xx] Handle specific KV-Cache (Key-Value Cache) past states natively across iterations
- [xx] Update KV-Cache tensors explicitly in memory without dynamic array reallocations
- [xx] Process `eos_token_id` (End of Sequence) natively to terminate the generation loop early
- [xx] Process `pad_token_id` natively
- [xx] Ensure BeamSearch scores track log-probabilities identically to HuggingFace `generate()`
- [xx] Implement `ai.onnx.contrib.WhisperBeamSearch` (Specific Audio-to-Text alignment logic)
- [xx] Implement `ai.onnx.contrib.WhisperGreedySearch`
- [xx] Extract Whisper timestamps explicitly based on output tokens
- [xx] Extract Whisper language detection natively
- [xx] Implement standard `N-Gram` blocking directly inside the generation loop
- [xx] Manage `past_key_values` dynamically inside the Pyodide / JS environment securely
- [xx] Provide an abstraction to inject arbitrary HuggingFace models (e.g. Llama, Mistral) into the CustomOp
- [xx] Evaluate specific WebGPU memory leaks during prolonged `BeamSearch` loops
- [xx] Flush GPU command queues periodically during auto-regressive loops to prevent browser hangs
- [xx] Implement `ai.onnx.contrib.Trilu`
- [xx] Extract Attention matrices natively during Generation for visualization
- [xx] Compile Generative CustomOps dynamically into pure ONNX `Loop` + `If` subgraphs if requested
- [xx] Guarantee mathematical equivalence with PyTorch `model.generate(...)` (atol=1e-5)
- [xx] Map `LogitsProcessor` concepts natively to ONNX Tensors (masking operations before Softmax)
- [xx] Support `min_length` constraints implicitly
- [xx] Support `bad_words_ids` constraints implicitly
- [xx] Catch maximum sequence length bounds logically to prevent OOM

### 6. Post-Processing & Standard Outputs (25+ items)

- [xx] Implement `ai.onnx.contrib.NonMaxSuppression` (NMS) natively for object detection
- [xx] Support Batched NMS dynamically
- [xx] Support Multi-Class NMS natively
- [xx] Implement standard Intersection-Over-Union (IoU) calculation purely in math
- [xx] Sort bounding box confidences in pure Python/JS natively
- [xx] Emulate `torchvision.ops.nms` mathematically
- [xx] Implement `ai.onnx.contrib.MatrixNMS` (Mask R-CNN specific)
- [xx] Implement `ai.onnx.contrib.YoloBox` (YoloV3/V4 anchor extraction)
- [xx] Implement `ai.onnx.contrib.PriorBox` (SSD anchor generation)
- [xx] Implement `ai.onnx.contrib.DetectionOutput` (Combining anchors + NMS + scores)
- [xx] Return JSON-compliant structured arrays directly from JS Native CustomOps for UI rendering
- [xx] Support drawing text labels and boxes natively on Canvas
- [xx] Implement `ai.onnx.contrib.SoftmaxCrossEntropyLoss` (Inference/Evaluation mode extraction)
- [xx] Support thresholding specific probability boundaries
- [xx] Extract Polygon / Segmentation contours from specific UNet custom outputs natively
- [xx] Provide explicit Post-Processing utility libraries (e.g. `onnx9000.vision.postprocess_yolo`)
- [xx] Evaluate exact NMS boundary pixel matches against OpenCV `cv2.dnn.NMSBoxes`
- [xx] Optimize Javascript TypedArray sorting (custom quicksort for Float32 views) for NMS
- [xx] Expose Native WebAssembly NMS fallback for massive dense box arrays (>100,000 boxes)
- [xx] Test numerical stability of Bounding Box coordinates (clamping implicitly to image dimensions)
- [xx] Implement specific `ai.onnx.contrib.SoftNMS` (Gaussian / Linear penalty functions)
- [xx] Compare `SoftNMS` output to original Python implementations exactly
- [xx] Strip specific CustomOps dynamically to yield raw ONNX outputs if the UI prefers manual processing
- [xx] Wrap CustomOp outputs directly into `ort.Tensor` abstractions securely
- [xx] Map all CustomOps natively into the Execution Provider boundary (e.g., executing in WebGPU or CPU)

### 7. Exporting, Serializing, and Integration (25+ items)

- [xx] Serialize `Tokenizer` CustomOps directly to `.onnx` Graph
- [xx] Dump BPE Dictionary (`merges.txt`, `vocab.txt`) directly into `AttributeProto` natively
- [xx] Embed Image Preprocessing CustomOps directly into an existing ONNX Graph
- [xx] Build End-to-End ONNX graph (String -> Tokenizer -> Model -> Detokenizer -> String)
- [xx] Build End-to-End ONNX graph (Audio Bytes -> Whisper -> String)
- [xx] Build End-to-End ONNX graph (Image Bytes -> ResNet -> Softmax -> String Label)
- [xx] Prevent serialization of absolute local file paths into the CustomOp Attributes
- [xx] Expose CLI utility: `onnx9000 extend model.onnx --add-tokenizer tokenizer.json`
- [xx] Expose CLI utility: `onnx9000 extend model.onnx --add-image-preprocess`
- [xx] Embed CustomOp domain string `ai.onnx.contrib` appropriately into the `ModelProto` `opset_import`
- [xx] Validate `ai.onnx.contrib` operator schemas strictly before execution
- [xx] Support reading models created by official `onnxruntime-extensions` seamlessly
- [xx] Catch explicitly missing `ai.onnx.contrib` definitions and list them clearly
- [xx] Provide user API `onnx9000.register_custom_op()` natively in Python
- [xx] Provide user API `onnx9000.register_custom_op()` natively in TypeScript/JS
- [xx] Map Python custom functions to ONNX CustomOp names
- [xx] Map JS custom functions to ONNX CustomOp names
- [xx] Validate custom function I/O counts dynamically before execution
- [xx] Check exact `dtype` alignments in CustomOp outputs natively
- [xx] Check exact `shape` alignments in CustomOp outputs natively
- [xx] Profile specific CustomOp execution latencies natively inside `Netron` visualizer
- [xx] Test cross-language execution: Python serializes a CustomOp Graph, JS parses and executes it natively
- [xx] Guarantee identical results across pure Python vs pure JS execution
- [xx] Support generating `bfloat16` metadata implicitly across CustomOps
- [xx] Expose an interactive test UI validating all supported Extensions visually in the browser

### 8. Testing & Validation (Edge Cases) (30+ items)

- [xx] Unit Test: Tokenize "Hello World" using GPT-2 BPE CustomOp
- [xx] Unit Test: Tokenize "Hello World" using BERT WordPiece CustomOp
- [xx] Unit Test: Tokenize "Hello World" using SentencePiece Unigram CustomOp
- [xx] Unit Test: Detokenize specific `input_ids` back to strings perfectly
- [xx] Unit Test: Handle explicit Chinese / Japanese characters natively in Tokenizers (Unicode normalization)
- [xx] Unit Test: Validate Whisper `LogMelSpectrogram` exact numerical output on 16kHz audio
- [xx] Unit Test: Evaluate `Resize` on a 4K image down to `224x224` (Bilinear)
- [xx] Unit Test: Execute YOLO `NonMaxSuppression` across 10,000 bounding boxes in under 50ms
- [xx] Unit Test: Evaluate `BeamSearch` output lengths and probabilities
- [xx] Stress Test: Tokenize a 10,000 word document natively in WASM/JS
- [xx] Handle empty strings (`""`) elegantly in all string processing ops
- [xx] Handle explicit single-character sequences
- [xx] Test CustomOp interactions with dynamic batch sizes `[N, ...]`
- [xx] Evaluate JS RegExp engine discrepancies versus Python `re` engine natively
- [xx] Validate fallback Image Decoding on corrupted JPEGs
- [xx] Verify WebAudio fallback when sample rate conversion ratio is irrational (e.g. 44.1kHz -> 16kHz)
- [xx] Check boundary conditions for `Pad` when padding sizes exceed actual image dimensions
- [xx] Test BPE merges when byte-fallback occurs explicitly (Llama tokens)
- [xx] Profile memory usage of the BPE Trie structure natively in Python/JS
- [xx] Verify Garbage Collection releases the loaded BPE Vocabularies completely when Session is destroyed
- [xx] Test `GreedySearch` exact token predictions against Hugging Face `transformers` reference
- [xx] Validate `StringSplit` delimiter parsing edge cases (e.g., repeating delimiters)
- [xx] Ensure `StringRegexReplace` processes backreferences cleanly (``)
- [xx] Validate WebCodecs VideoFrame to Float32Array zero-copy speeds (sub-5ms)
- [xx] Handle CORS restrictions safely if an extension requests external network loading
- [xx] Test nested CustomOps (e.g., `StringLower` -> `StringSplit` -> `Tokenizer`)
- [xx] Execute `pytest` across all `ai.onnx.contrib` op schemas natively
- [xx] Output human-readable error messages for unrecognized or missing CustomOp domains
- [xx] Provide explicit developer warnings if a JS execution polyfill runs significantly slower than WASM
- [xx] Validate `Float16` typed array representations natively for all CustomOps that support mixed precision

### 9. Advanced Tokenization & NLP Edge Cases (20+ items)

- [xx] Implement `ByteLevel` mapping specific to GPT-2 / RoBERTa (encoding 256 bytes to unicode)
- [xx] Implement `Metaspace` decoding specific to SentencePiece (replacing `_` with space)
- [xx] Support explicit `add_prefix_space` logic dynamically in tokenizers
- [xx] Support explicit `trim_offsets` logic for offset mappings
- [xx] Implement `ai.onnx.contrib.BertTokenizerDecoder` natively
- [xx] Map explicit `clean_up_tokenization_spaces` logic natively
- [xx] Implement `BpeDecoder` exact string replacement loops efficiently in JS
- [xx] Map `vocab.json` and `merges.txt` pairs cleanly from a single loaded directory or HF Hub URL
- [xx] Parse explicitly `.model` protobuf files if `sentencepiece` was originally used natively
- [xx] Emulate `Unigram` language modeling algorithms purely in Python (no C++ bindings)
- [xx] Emulate `Unigram` language modeling algorithms purely in Javascript
- [xx] Support handling explicit `control_tokens` (e.g. `<|endoftext|>`, `<|im_start|>`)
- [xx] Prevent special tokens from being split during BPE tokenization (`split_special_tokens=False`)
- [xx] Extract padding dynamically based on batch maximums
- [xx] Support left-padding specifically for decoder-only architectures (e.g. LLaMA/GPT)
- [xx] Support right-padding specifically for encoder architectures (e.g. BERT)
- [xx] Generate exactly matching `token_type_ids` for sentence pairs (`text_a`, `text_b`)
- [xx] Validate maximum length truncations properly drop the end of the sequence (or start if requested)
- [xx] Handle explicit `add_special_tokens=False` configurations
- [xx] Provide dictionary caching to speed up repeated identical tokenizations natively

### 10. Exhaustive Cross-Platform Fallbacks & Polyfills (15+ items)

- [xx] Fallback ImageDecoder to JS `CanvasRenderingContext2D` if `createImageBitmap` fails or is unsupported
- [xx] Fallback VideoDecoder to JS `Canvas` drawing `HTMLVideoElement` if `WebCodecs` is unavailable
- [xx] Fallback AudioDecoder to pure JS math-based WAV decoder if `WebAudio` is unavailable (Node.js)
- [xx] Fallback STFT complex multiplications to pure JS if WASM SIMD is unsupported
- [xx] Expose native Python `Image.open` fallbacks if `cv2` or `av` are missing
- [xx] Handle Javascript `TextDecoder`/`TextEncoder` limitations natively on older browsers
- [xx] Provide a polyfill for JS `String.prototype.matchAll` if executing in legacy environments
- [xx] Emulate 64-bit integer arithmetic securely for sequence numbering (preventing JS precision loss)
- [xx] Check explicit HTTP Content-Types when fetching images/audio to route to correct decoder
- [xx] Route `application/octet-stream` explicitly based on file extension sniffing
- [xx] Prevent Cross-Origin Resource Sharing (CORS) blocks on `<canvas>` pixel extraction (`tainted canvas`)
- [xx] Enforce explicitly configured `crossOrigin="anonymous"` attributes when loading remote media
- [xx] Support Node.js `Buffer` mappings cleanly to `Blob` objects for unified decoding APIs
- [xx] Inject custom Polyfill imports selectively during the build step (Tree-shaking)
- [xx] Test the fallback mechanisms on completely headless testing environments (e.g. GitHub Actions)
