# onnx-safetensors Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `safetensors` within the `onnx9000` ecosystem.
The original `safetensors` library (by Hugging Face) relies on Rust bindings to provide secure, fast, and zero-copy tensor serialization.
Our `onnx9000` reimplementation provides a 100% pure-Python and pure-JavaScript (WASM/WebGPU) parser. It reads `.safetensors` files directly, mapping them securely to ONNX `TensorProto` data structures. By leveraging native POSIX `mmap` in Python and `ArrayBuffer` / `fetch` Range requests in the browser, we can progressively stream and instantly load multi-gigabyte LLM weights into memory-constrained environments with true zero-copy performance and absolute security (no `pickle` vulnerabilities).

## Exhaustive Parity Checklist

### 1. Pure-Python/JS Core Parsing Engine (40+ items)

- [x] Implement zero-dependency `.safetensors` header parser in Python
- [x] Implement zero-dependency `.safetensors` header parser in JavaScript (TypeScript)
- [x] Read 8-byte little-endian unsigned integer (header size `N`)
- [x] Read standard JSON header payload of exact length `N`
- [x] Validate JSON encoding strictly as UTF-8
- [x] Extract tensor metadata: `dtype`
- [x] Extract tensor metadata: `shape`
- [x] Extract tensor metadata: `data_offsets` `[begin, end]`
- [x] Implement O(1) dictionary lookups for tensor names
- [x] Verify `begin` and `end` offsets fit within the bounds of the file size
- [x] Verify `end - begin` exactly matches the calculated byte size of `shape` \* `dtype_size`
- [x] Support implicit 8-byte alignment padding validation
- [x] Ignore internal JSON whitespace formatting variants seamlessly
- [x] Reject duplicate tensor names in the JSON header
- [x] Reject overlapping tensor data regions within the file buffer
- [x] Reject tensor data regions that precede the end of the JSON header
- [x] Extract global `__metadata__` dictionary natively
- [x] Provide lazy-evaluation generator yielding tensor slices on demand
- [x] Close file handles gracefully upon parser garbage collection
- [x] Expose API to list all tensors without reading binary payload (`.keys()`)
- [x] Support parsing strictly from an in-memory byte buffer (e.g., downloaded RAM)
- [x] Support parsing from a generic `io.BytesIO` / `io.BufferedReader` stream
- [x] Provide strict boundary error messages for corrupted headers
- [x] Ensure nested or invalid JSON types in `__metadata__` do not crash the parser
- [x] Implement fast path for extracting single tensors by name
- [x] Implement bulk extraction for multiple tensors
- [x] Support indexing into tensor lists natively `tensors['weight']`
- [x] Expose total byte footprint calculator for memory planning
- [x] Emulate `safetensors.safe_open` API signature for PyTorch ecosystem compatibility
- [x] Enable framework-agnostic tensor abstraction (`onnx9000.Tensor` object mapping)
- [x] Provide an explicit validation-only mode (`check_safetensors(file)`)
- [x] Reject files exceeding JS `Number.MAX_SAFE_INTEGER` boundaries for offsets safely
- [x] Parse Hugging Face sharded formats (`model-00001-of-00002.safetensors`)
- [x] Aggregate sharded JSON indexes (`model.safetensors.index.json`) logically
- [x] Implement global dictionary view spanning multiple sharded `.safetensors` files
- [x] Detect and warn about unreferenced data (bytes in the file not mapped to any tensor)
- [x] Reject unaligned tensor offsets (must be 8-byte aligned per standard)

### 2. Zero-Copy Memory & Mmap Implementation (30+ items)

- [x] Implement POSIX `mmap` for instant disk-to-memory mapping natively in Python
- [x] Implement Windows `mmap` equivalent (`CreateFileMapping`) safely
- [x] Extract NumPy `ndarray` views directly from the `mmap` buffer (zero-copy)
- [x] Ensure NumPy arrays generated are marked as read-only (`writeable=False`)
- [x] Prevent Python Garbage Collector from closing `mmap` while views are alive
- [x] Support explicit `madvise` hints (`MADV_WILLNEED`, `MADV_RANDOM`) for OS page caching
- [x] Implement Pyodide `FS` virtual filesystem zero-copy extraction
- [x] Map ArrayBuffers in WebAssembly explicitly from the Safetensors buffer
- [x] Support zero-copy injection of weights directly into WebGPU `GPUBuffer`
- [x] Implement `memoryview` based slicing for environments without NumPy
- [x] Avoid intermediate byte string allocations (`file.read()`) during standard extraction
- [x] Guarantee memory layout contiguity (C-contiguous) for all extracted tensors
- [x] Extract multi-dimensional slices lazily (e.g., `tensor[0:10, :]` via numpy strides)
- [x] Handle explicit file-descriptor leaking protections (Context Managers)
- [x] Support `mmap` across multiple processes via OS shared memory semantics
- [x] Provide memory-pinned buffer extraction (CUDA Pinned Memory emulation) if requested
- [x] Verify page alignment optimizations natively
- [x] Prevent swapping to disk dynamically on specific critical tensors via `mlock`
- [x] Handle multi-gigabyte models on 32-bit WASM gracefully (falling back to chunked arrays)
- [x] Emulate 64-bit memory addressing for WASM-64 future compatibility
- [x] Share Safetensors `mmap` memory maps across Python threads cleanly

### 3. Web-Specific Progressive Streaming & HTTP (40+ items)

- [x] Implement HTTP `Range` request wrapper for native JS `fetch`
- [x] Download ONLY the JSON header using initial 8-byte + N-byte HTTP requests
- [x] Stream specific tensor payloads dynamically using `Range: bytes=begin-end`
- [x] Assemble distributed LLM weights exclusively on demand (e.g., layer by layer loading)
- [x] Integrate with WebGPU chunked pipeline execution (stream layer, execute, discard layer)
- [x] Support HTTP `Keep-Alive` explicitly for thousands of rapid range requests
- [x] Implement ServiceWorker caching for downloaded tensor chunks (`CacheStorage`)
- [x] Implement IndexedDB persistence for entire `.safetensors` files in the browser
- [x] Provide visual progress bar hooks for `Content-Length` streams
- [x] Parse `Accept-Ranges: bytes` headers to fallback gracefully if streaming is blocked
- [x] Support WebSockets / WebRTC for P2P tensor weight distribution in browser
- [x] Stream `.safetensors` directly from Hugging Face Hub URLs seamlessly
- [x] Implement auto-retry logic with exponential backoff for interrupted Range requests
- [x] Decrypt HTTP chunked encodings securely
- [x] Expose native `ReadableStream` interfaces for streaming JSON parsers
- [x] Prevent browser Out-Of-Memory by actively destroying `Uint8Array` views after WGSL upload
- [x] Manage parallel chunk downloads (e.g., fetching 4 layers simultaneously)
- [x] Throttle parallel requests to prevent browser network stack blocking
- [x] Support generating standard `ai.onnx` models entirely client-side using downloaded weights
- [x] Enable cross-origin resource sharing (CORS) header pre-flight validations explicitly
- [x] Handle 416 Range Not Satisfiable errors gracefully
- [x] Fallback to full file download if server ignores Range headers

### 4. Data Type, Endianness & Tensor Alignment (40+ items)

- [x] Parse `F64` -> `Float64` / `float64`
- [x] Parse `F32` -> `Float32` / `float32`
- [x] Parse `F16` -> `Float16` / `float16`
- [x] Parse `BF16` -> `BFloat16` / `bfloat16`
- [x] Parse `I64` -> `Int64` / `int64`
- [x] Parse `I32` -> `Int32` / `int32`
- [x] Parse `I16` -> `Int16` / `int16`
- [x] Parse `I8` -> `Int8` / `int8`
- [x] Parse `U64` -> `UInt64` / `uint64`
- [x] Parse `U32` -> `UInt32` / `uint32`
- [x] Parse `U16` -> `UInt16` / `uint16`
- [x] Parse `U8` -> `UInt8` / `uint8`
- [x] Parse `BOOL` -> `Bool` / `bool`
- [x] Implement fallback byte-swapping if host architecture is Big-Endian
- [x] Ensure strict Little-Endian decoding for all types natively
- [x] Parse complex types (`C64`, `C128`) if standard evolves, or map to F32/F64 pairs
- [x] Reject unrecognized or proprietary data types securely
- [x] Emulate `bfloat16` natively in standard JavaScript (`Float32Array` masking)
- [x] Validate standard HuggingFace specific type strings (e.g., `F16` vs `FLOAT16`)
- [x] Implement explicit downcasting hooks (e.g., load `F32` but immediately return `F16` view)
- [x] Implement explicit quantization hooks (load `F16`, convert to INT8 array dynamically)
- [x] Extract tensor dimensionality explicitly (1D, 2D, 3D, 4D, ND)
- [x] Throw error on 0-dimensional scalars if not correctly encoded as `shape: []`
- [x] Handle massive dimensions safely (`shape: [1, 32, 128000, 128]`)
- [x] Verify `int64` bounds securely (preventing Python `OverflowError` during slicing)
- [x] Decode sub-byte quantization (e.g., NF4, INT4) explicitly via byte unpacking strategies
- [x] Unpack specific AWQ / GPTQ packed `safetensors` layouts correctly

### 5. ONNX Graph Integration & Surgery (30+ items)

- [x] Convert `.safetensors` mappings directly into ONNX `Initializer` tensors
- [x] Replace standard `.bin` external data natively with `.safetensors` lookups
- [x] Intercept ONNX parsing to pull constants exclusively from `.safetensors` indices
- [x] Strip raw byte arrays from `ModelProto` and dump to `.safetensors` dynamically
- [x] Export ONNX model to a `.onnx` (topology only) and `.safetensors` (weights only) pair
- [x] Inject `GraphSurgeon` parameters directly from loaded safetensors dictionaries
- [x] Rewrite ONNX `Constant` nodes seamlessly into `safetensors` memory views
- [x] Support initializing an `onnx9000` Python compiled model from a `.safetensors` file
- [x] Validate ONNX topological shapes against `safetensors` extracted shapes at runtime
- [x] Warn on shape mismatches between ONNX ValueInfo and Safetensor arrays
- [x] Warn on dtype mismatches between ONNX ValueInfo and Safetensor arrays
- [x] Emulate standard ONNX Runtime `SessionOptions` external data configurations
- [x] Pack multi-layer ONNX transformers heavily using shared `.safetensors` allocations
- [x] Flatten nested Safetensors attributes into Graph inputs securely
- [x] Validate `ai.onnx` operators evaluate correctly against the extracted views

### 6. Security, Auditing & Validation (The "Safe" in Safetensors) (30+ items)

- [x] Prevent Arbitrary Code Execution (0-day vulnerabilities vs Python `pickle`)
- [x] Enforce strict sandboxing of file parsing logic
- [x] Validate `shape` arrays contain no negative dimensions
- [x] Validate `shape` arrays contain no impossibly large dimensions (e.g., > 2^50)
- [x] Prevent Path Traversal vulnerabilities in sharded `index.json` path loading
- [x] Prevent XML External Entity (XXE) or JSON injection in header parsing
- [x] Reject `.safetensors` files where `end` offset is smaller than `begin` offset
- [x] Ensure byte offsets strictly respect the total file size reported by the OS
- [x] Catch dynamic schema mutation attacks (JSON tampering)
- [x] Validate `__metadata__` strings do not contain executable script tags (XSS protection for browsers)
- [x] Fuzz-test parser against heavily corrupted JSON headers
- [x] Fuzz-test parser against heavily corrupted binary offsets
- [x] Provide cryptographic hashing verification (`SHA256`) of the file buffer optionally
- [x] Verify metadata signatures (if standardized securely by HuggingFace)
- [x] Ensure parser runs securely within Cloudflare Workers isolates

### 7. Distributed Server & High-Performance IO (30+ items)

- [x] Deploy Safetensors `mmap` views natively in Ray Clusters for zero-copy IPC
- [x] Serialize `onnx9000.Tensor` wrappers across gRPC efficiently using Safetensors formats
- [x] Implement lazy-loading for Celery distributed background workers
- [x] Use `sendfile` or `splice` Linux syscalls explicitly for networking weights from disk
- [x] Optimize AWS S3 `boto3` integration with HTTP Range requests natively
- [x] Optimize Azure Blob Storage `get_blob_client` with Range offsets
- [x] Optimize GCP Cloud Storage chunked loading directly into memory
- [x] Maximize Page Cache utilization on NVMe arrays (reading 70B parameters < 2 seconds)
- [x] Provide distributed sharding algorithms natively (e.g., loading only layer 1-10 on Node A)
- [x] Expose native `MPI` rank loading filters (Node $i$ only loads Safetensor array $i$)
- [x] Pipeline parallelism loading strategies (Stream Layer N+1 while computing Layer N)
- [x] Tensor parallelism loading strategies (Load slice `[:, 0:Dim/2]` directly from disk)

### 8. Serialization, Exporting & Creation (40+ items)

- [x] Implement `.safetensors` writing logic purely in Python
- [x] Implement `.safetensors` writing logic purely in Javascript (Node.js/Browser)
- [x] Accept generic Python dictionaries (`dict[str, np.ndarray]`) for serialization
- [x] Accept ONNX `TensorProto` arrays for serialization
- [x] Generate strict UTF-8 JSON headers
- [x] Calculate correct 8-byte padded alignments dynamically
- [x] Write header size unsigned 64-bit integer
- [x] Append binary buffers efficiently using `writev` / vectorized I/O
- [x] Prevent memory explosion during serialization by streaming arrays sequentially
- [x] Support generating `__metadata__` standard dictionary fields
- [x] Support appending format version identifiers natively
- [x] Support `safetensors.save_file` API parity
- [x] Support `safetensors.save` (return raw bytes) API parity
- [x] Warn against duplicate keys during generation
- [x] Handle `bfloat16` generation securely
- [x] Export massive 100GB+ arrays by creating chunked sharded sets automatically
- [x] Generate the corresponding `model.safetensors.index.json` natively
- [x] Validate generated files immediately via loopback reading
- [x] Validate generated files are byte-for-byte identical to Rust reference implementation
- [x] Stream serialization natively via `yield` buffers (chunked HTTP uploads)

### 9. Edge Cases, Framework Interop & Testing (30+ items)

- [x] Unit Test: 0-byte tensor saving/loading (`shape=[]`)
- [x] Unit Test: 1D, 2D, 3D, 4D standard Float32 arrays
- [x] Unit Test: Endianness conversion tests natively
- [x] Unit Test: JSON header > 10MB test (massive vocabulary models)
- [x] Unit Test: Extremely high precision dimensions (e.g. `2^31 - 1`)
- [x] Interop: Support loading PyTorch Safetensors correctly mapped to ONNX conventions
- [x] Interop: Support loading TensorFlow Safetensors correctly mapped
- [x] Interop: Support loading Flax/JAX Safetensors natively
- [x] Remap Flax hierarchical keys (`layers.0.attention.kernel`) to standard `.weight` suffixes dynamically
- [x] Provide strict dictionary equivalence testing (`np.testing.assert_allclose`)
- [x] Test memory leak protections (asserting `mmap` references drop to 0)
- [x] Validate Javascript WebAssembly Out-of-Bounds protections
- [x] Expose benchmarking scripts comparing pure Python vs `pickle` vs `rust-safetensors`

### 10. Explicit JavaScript / WASM Typed Array Mappings (30+ items)

- [x] Map Safetensors `F64` directly to JS `Float64Array` without duplication
- [x] Map Safetensors `F32` directly to JS `Float32Array` without duplication
- [x] Map Safetensors `I32` directly to JS `Int32Array` without duplication
- [x] Map Safetensors `I16` directly to JS `Int16Array` without duplication
- [x] Map Safetensors `I8` directly to JS `Int8Array` without duplication
- [x] Map Safetensors `U32` directly to JS `Uint32Array` without duplication
- [x] Map Safetensors `U16` directly to JS `Uint16Array` without duplication
- [x] Map Safetensors `U8` directly to JS `Uint8Array` without duplication
- [x] Map Safetensors `I64` safely to JS `BigInt64Array` natively
- [x] Map Safetensors `U64` safely to JS `BigUint64Array` natively
- [x] Emulate `F16` in JS using `Uint16Array` views (providing decoding hooks to `Float32`)
- [x] Emulate `BF16` in JS using `Uint16Array` views (providing left-shift decoding hooks)
- [x] Ensure JS TypedArrays are generated using `buffer.slice()` only if explicitly requested (memory copy)
- [x] Default JS TypedArrays to `new Float32Array(buffer, byteOffset, length)` (zero-copy)
- [x] Validate JS byte offsets are multiples of the TypedArray element sizes (padding checks)
- [x] Provide unaligned buffer fallback (copying unaligned data to aligned arrays if zero-copy fails)
- [x] Expose `SharedArrayBuffer` mapping for multi-threaded WebWorker access
- [x] Pass Safetensors pointers directly into Pyodide WASM memory (`Module.HEAPU8`)
- [x] Prevent JS Garbage Collector from sweeping the underlying ArrayBuffer prematurely
- [x] Support Node.js `Buffer` natively without invoking browser-only APIs
- [x] Support Node.js `fs.openSync` and `fs.readSync` for manual chunk reading
- [x] Expose a Javascript generator `async function* load_tensors(file)`
- [x] Parse UTF-8 JSON headers natively using JS `TextDecoder` (handling streaming bytes)

### 11. Comprehensive Error Handling & Exceptions (30+ items)

- [x] Raise `SafetensorsHeaderTooLargeError` if header size `N` > 100MB
- [x] Raise `SafetensorsInvalidHeaderError` if UTF-8 JSON decoding fails
- [x] Raise `SafetensorsInvalidJSONError` if JSON parses but is structurally invalid
- [x] Raise `SafetensorsDuplicateKeyError` if tensor names overlap
- [x] Raise `SafetensorsInvalidOffsetError` if `begin` > `end`
- [x] Raise `SafetensorsOutOfBoundsError` if `end` > `file_size`
- [x] Raise `SafetensorsOverlapError` if data regions mathematically intersect
- [x] Raise `SafetensorsAlignmentError` if `begin` is not 8-byte aligned
- [x] Raise `SafetensorsInvalidDtypeError` if the `dtype` string is unrecognized
- [x] Raise `SafetensorsShapeMismatchError` if `(end - begin) != volume(shape) * dtype_size`
- [x] Raise `SafetensorsFileEmptyError` if file size is 0
- [x] Raise `SafetensorsFileTooSmallError` if file size < 8 bytes
- [x] Catch `OSError` / `IOError` cleanly during `mmap` initialization
- [x] Catch `MemoryError` dynamically if system RAM cannot map the file (32-bit limits)
- [x] Catch `RangeError` in Javascript if TypedArray bounds are exceeded
- [x] Catch `TypeError` in Python if passing non-string keys to the dictionary interface
- [x] Raise `SafetensorsWriteError` if serialization disk space is exhausted
- [x] Provide explicit error boundaries for cross-platform file locking (Windows vs Linux)
- [x] Provide explicit warnings if `__metadata__` is missing standard HuggingFace keys
- [x] Gracefully catch and report JSON deeply nested recursion limits
- [x] Ensure all custom exceptions subclass a base `SafetensorsError` for easy `try/except` handling

### 12. Hugging Face Hub Integration & Ecosystem (25+ items)

- [x] Support direct parsing of `hf://` protocol URIs natively
- [x] Authenticate HTTP Range requests using `HF_TOKEN` environment variables implicitly
- [x] Parse Hugging Face Hub `model.safetensors.index.json` natively
- [x] Resolve sharded file paths relative to the Hub repository structure
- [x] Cache downloaded chunks dynamically to `~/.cache/huggingface/hub/` automatically
- [x] Emulate `huggingface_hub` `cached_download` paths if the library is not installed
- [x] Validate Hub ETag headers before resuming interrupted Range requests
- [x] Warn on Hub Rate Limiting (HTTP 429) dynamically
- [x] Back-off dynamically based on Hub `Retry-After` headers
- [x] Verify Hub downloaded `.safetensors` against `sha256` hashes provided in the repository
- [x] Support fetching weights directly from specific commits/branches natively (`revision=...`)
- [x] Expose progress tracking callbacks compatible with standard `tqdm` bars
- [x] Auto-detect if a model repository defaults to `.bin` (PyTorch) vs `.safetensors` and prioritize `.safetensors`
- [x] Support PyTorch `load_state_dict` direct emulation (returning dict of PyTorch-compatible tensors)

### 13. Deep Framework Weight Layout Mappings (20+ items)

- [x] Emulate PyTorch `Conv1d` weight layouts seamlessly (translating ONNX shapes if necessary)
- [x] Emulate PyTorch `Conv2d` weight layouts seamlessly (`[O, I, H, W]`)
- [x] Emulate PyTorch `Linear` weight layouts seamlessly (`[O, I]`)
- [x] Emulate TensorFlow `Conv2D` weight layouts seamlessly (`[H, W, I, O]`)
- [x] Emulate TensorFlow `Dense` weight layouts seamlessly (`[I, O]`)
- [x] Emulate Flax `Dense` weight layouts seamlessly (`[I, O]`)
- [x] Emulate Flax `Conv` weight layouts seamlessly (`[H, W, I, O]`)
- [x] Provide dynamic transposition hooks: `tensor.transpose_on_load()`
- [x] Support `Safetensor` weights that natively bake-in `GroupNormalization` scales/biases
- [x] Support `Safetensor` weights mapped to `LayerNormalization` arrays
- [x] Resolve QKV (Query/Key/Value) weight concatenation differences across PyTorch and TF natively
- [x] Split loaded QKV tensors automatically if ONNX topological inputs expect separated Q, K, V
- [x] Concatenate separated Q, K, V tensors automatically if ONNX topology expects a packed QKV

### 14. Performance Profiling & Advanced Benchmarking (20+ items)

- [x] Benchmark: Peak memory usage loading 7B parameter model (should be ~0 RAM overhead via mmap)
- [x] Benchmark: Total time to extract 10,000 specific small tensors from a massive file
- [x] Benchmark: Total time to stream a single 1GB layer over HTTP (measuring overhead)
- [x] Profile OS Page Cache hit-rates natively (if tooling allows)
- [x] Profile Garbage Collection pressure in V8/Node.js after loading a 2GB model
- [x] Unit Test: Concurrency (Read the same `.safetensors` file from 16 Threads simultaneously)
- [x] Unit Test: Multiprocessing (Read the same `.safetensors` file from 4 Processes simultaneously)
- [x] Unit Test: Write a 1GB `.safetensors` file and verify throughput > 500MB/s
- [x] Unit Test: Load a completely sparse (all zeros) tensor array flawlessly
- [x] Validate Python `memoryview` slice extraction latency (<1ms per slice)
- [x] Monitor and test HTTP Keep-Alive connection limits natively to prevent socket exhaustion

### 15. WASM Specific Memory Alignment & Buffers (20+ items)

- [x] Explicitly pad `U8` JS extracts to 8-byte boundaries if passing to WebAssembly
- [x] Explicitly pad `I8` JS extracts to 8-byte boundaries if passing to WebAssembly
- [x] Pad `F16` JS extracts to 8-byte boundaries natively before Emscripten ingestion
- [x] Implement Emscripten `_malloc` wrapper in JS to pre-allocate exact payload sizes safely
- [x] Handle Javascript `DataView` limits if `buffer.byteLength` > 2GB (Chrome/V8 limits)
- [x] Ensure `Float32Array` views strictly align on 4-byte boundaries (JS spec)
- [x] Ensure `Float64Array` views strictly align on 8-byte boundaries (JS spec)
- [x] Copy unaligned `Float32` data explicitly (via `Uint8Array`) to aligned buffers if needed
- [x] Fallback gracefully when `SharedArrayBuffer` is blocked by Cross-Origin-Opener-Policy (COOP) headers
- [x] Leverage WebCodecs `VideoFrame` (hack) for GPU zero-copy upload if WebGL doesn't natively support TypedArrays
- [x] Expose WASM linear memory bounds dynamically (`wasmMemory.buffer.byteLength`)
- [x] Catch WASM `OOM` errors internally when Safetensors payload exceeds available Pages
- [x] Trigger explicit JS GC (`gc()`) dynamically after offloading large tensors to WebGPU buffers

### 16. Advanced Dict Manipulation & Utility API (20+ items)

- [x] Support PyTorch `state_dict()` semantic patching dynamically
- [x] Rename keys natively during loading (`safetensors.load_file(..., prefix="model.")`)
- [x] Filter keys natively using Regex (`safetensors.load_file(..., pattern=".*weight$")`)
- [x] Merge two `.safetensors` files in-memory natively (`dict.update()`)
- [x] Merge a `.bin` PyTorch checkpoint with a `.safetensors` dictionary visually
- [x] Serialize merged dictionaries efficiently back to disk
- [x] Extract single specific key natively: `safetensors.get_tensor("file.safetensors", "key")`
- [x] Expose `save_file` parameter to explicitly overwrite existing files vs appending
- [x] Output raw JSON dictionary explicitly: `safetensors.get_metadata("file.safetensors")`
- [x] Verify file integrity without loading tensors: `safetensors.check_file_validity()`
- [x] Provide utility to convert PyTorch `.bin` (Pickle) directory to `.safetensors` directory automatically
- [x] Provide utility to convert TensorFlow `SavedModel` variables directly to `.safetensors`

### 17. Model-Specific Parsing Architectures & Edge Cases (15+ items)

- [x] Parse explicitly LLaMA format Safetensors layouts (`layers.0.self_attn.q_proj.weight`)
- [x] Parse explicitly BERT format Safetensors layouts (`bert.encoder.layer.0.attention.self.query.weight`)
- [x] Parse explicitly Whisper format Safetensors layouts (encoder and decoder sub-dictionaries)
- [x] Parse explicitly Stable Diffusion formats (U-Net, VAE, TextEncoder sharded safetensors)
- [x] Parse SDXL massive UNet `.safetensors` dynamically
- [x] Correctly handle empty `__metadata__` dictionaries (e.g. `{}`)
- [x] Correctly handle `__metadata__` with explicit format strings (`"format": "pt"`)
- [x] Extract HuggingFace specific quantization metadata (e.g. `bitsandbytes` scale parameters hidden in JSON)
- [x] Verify `int4` block scaling arrays are mapped correctly relative to the primary weight

### 18. Final Precision, Testing, and Compliance Verification (15+ items)

- [x] Profile HTTP Request times dynamically for 100 concurrent Range requests
- [x] Measure total elapsed time parsing exactly 10,000 JSON keys natively
- [x] Unit Test: Verify `bfloat16` mathematical conversions round accurately in JS
- [x] Unit Test: Verify 1D `Int8` arrays load perfectly without padding issues
- [x] Output `__metadata__` length natively before parsing tensors (for early exit checks)
- [x] Log precise byte ranges applied during HTTP chunks
- [x] Catch exactly `404 Not Found` for missing shards immediately
- [x] Catch exactly `403 Forbidden` for private HuggingFace Repos without `HF_TOKEN`
- [x] Guarantee no usage of `eval()` or `Function()` in JS parser natively
- [x] Guarantee no usage of `eval()` or `exec()` in Python parser natively
- [x] Export TypeScript `.d.ts` module securely exposing all API functions
