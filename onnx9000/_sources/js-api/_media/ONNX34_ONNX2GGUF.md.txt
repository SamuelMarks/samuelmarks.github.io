# ONNX34: onnx2gguf (Web-Native GGUF Compiler & Llama.cpp Bridge)

## Original Project Description

The `ggml` ecosystem (famously powering `llama.cpp`) uses the **GGUF** (GPT-Generated Unified Format) binary format. GGUF is heavily optimized for fast loading and memory-mapping (mmap) of large language models (LLMs) on CPUs and GPUs. Converting models into GGUF traditionally requires heavy Python scripts (`convert.py`) that depend on PyTorch, SentencePiece, and Hugging Face Transformers to parse `.safetensors`, `.bin`, or `.onnx` files, extract tokenizers, and map tensors into the strict GGUF naming and metadata standards.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.onnx2gguf` bridges the standard ONNX ecosystem to the `llama.cpp` ecosystem using a **100% pure TypeScript and Python binary compiler**.

- **Zero-Dependency Binary Emission:** It parses ONNX and Safetensors natively, bypassing PyTorch and SentencePiece completely, to emit GGUF files directly in memory or via streaming disk I/O.
- **Browser-Based GGUF Compilation:** Users can drag-and-drop an ONNX model and its associated `tokenizer.json` into a webpage. The tool instantly packs them into a valid `.gguf` file locally, providing the ultimate privacy-preserving model converter.
- **Native Sub-byte Quantization:** Standard GGUF relies on `llama.cpp` tools for K-quants (e.g., `Q4_K_M`). `onnx9000` implements the GGML quantization math directly in WASM/Python, allowing users to quantize an ONNX model directly to a `Q4_0` or `Q8_0` GGUF payload in a single pass.
- **Bidirectional Support:** Unlike standard scripts, `onnx9000` can also _read_ `.gguf` files natively, mapping GGML's custom tensors back up into standard ONNX operations to execute `llama.cpp` models directly inside the WebGPU-accelerated `onnx9000` runtime.

---

## Exhaustive Implementation Checklist

### Phase 1: GGUF Format Binary Serialization Engine

- [x] 1. Implement zero-dependency GGUF Builder in TypeScript/JS.
- [x] 2. Implement zero-dependency GGUF Builder in Python.
- [x] 3. Emit GGUF Magic Bytes (`0x46554747` / "GGUF").
- [x] 4. Emit GGUF Version (Strictly targeting Version 3).
- [x] 5. Handle strict Little-Endian serialization natively across all platforms.
- [x] 6. Write `tensor_count` (uint64).
- [x] 7. Write `metadata_kv_count` (uint64).
- [x] 8. Implement `write_string` matching GGUF length-prefixed format (uint64 length + UTF-8 bytes).
- [x] 9. Support implicit padding alignment (defaulting to 32-byte alignment for tensors).
- [x] 10. Implement metadata writing for `UINT8`.
- [x] 11. Implement metadata writing for `INT8`.
- [x] 12. Implement metadata writing for `UINT16`.
- [x] 13. Implement metadata writing for `INT16`.
- [x] 14. Implement metadata writing for `UINT32`.
- [x] 15. Implement metadata writing for `INT32`.
- [x] 16. Implement metadata writing for `FLOAT32`.
- [x] 17. Implement metadata writing for `BOOL`.
- [x] 18. Implement metadata writing for `STRING`.
- [x] 19. Implement metadata writing for `ARRAY` types.
- [x] 20. Implement metadata writing for `UINT64`.
- [x] 21. Implement metadata writing for `INT64`.
- [x] 22. Implement metadata writing for `FLOAT64`.
- [x] 23. Implement dynamic KV dictionary schema mapping inside the Builder.
- [x] 24. Write `tensor_info` blocks (Name, Dimensions, GGML Type, Offset).
- [x] 25. Calculate explicit tensor binary offsets dynamically during header generation.
- [x] 26. Guarantee absolute structural compliance with the `ggml` standard C-struct parsers.
- [x] 27. Provide an API to stream binary tensor arrays directly to the output buffer/file.
- [x] 28. Throw strict validation errors if a string exceeds standard allocation limits.
- [x] 29. Fuzz-test the GGUF writer against corrupted metadata dictionaries.
- [x] 30. Support Javascript `BigInt` securely for all `uint64` file size offsets.

### Phase 2: General & Standard Metadata Mapping

- [x] 31. Set `general.architecture` automatically based on ONNX graph topology.
- [x] 32. Set `general.name` mapping to `ModelProto.graph.name`.
- [x] 33. Set `general.author` mapping to `ModelProto.producer_name`.
- [x] 34. Set `general.version` mapping to `ModelProto.model_version`.
- [x] 35. Set `general.file_type` automatically based on the detected quantization level.
- [x] 36. Set `general.quantization_version` (typically 2).
- [x] 37. Set `general.alignment` explicitly in the KV store (typically 32).
- [x] 38. Support user overrides for any `general.*` KV pair.
- [x] 39. Provide a fallback `general.architecture = "unknown"` if the ONNX graph doesn't match standard LLM architectures.
- [x] 40. Extract and sanitize ONNX `doc_string` into `general.description`.

### Phase 3: LLaMA Architecture Metadata Mapping

- [x] 41. Identify LLaMA architecture topologies automatically from ONNX IR.
- [x] 42. Extract and set `llama.context_length`.
- [x] 43. Extract and set `llama.embedding_length`.
- [x] 44. Extract and set `llama.block_count`.
- [x] 45. Extract and set `llama.feed_forward_length`.
- [x] 46. Extract and set `llama.attention.head_count`.
- [x] 47. Extract and set `llama.attention.head_count_kv` (for GQA/MQA).
- [x] 48. Extract and set `llama.attention.layer_norm_rms_epsilon`.
- [x] 49. Extract and set `llama.rope.dimension_count`.
- [x] 50. Extract and set `llama.rope.freq_base`.
- [x] 51. Extract and set `llama.vocab_size`.
- [x] 52. Handle dynamic detection of Grouped Query Attention (GQA) ratios.
- [x] 53. Determine and map SwiGLU vs GeGLU activation patterns natively into the metadata.
- [x] 54. Check for standard vs transposed weight formats prior to generating LLAMA metadata.
- [x] 55. Validate `block_count` matches the actual number of Transformer layer copies found in the ONNX graph.

### Phase 4: Other LLM Architectures Metadata Mapping

- [x] 56. Identify and support **Mistral** architecture.
- [x] 57. Extract Mistral sliding window parameters (`mistral.attention.sliding_window`).
- [x] 58. Identify and support **Phi-2 / Phi-3** architecture.
- [x] 59. Identify and support **Qwen2** architecture.
- [x] 60. Identify and support **Gemma** architecture.
- [x] 61. Extract Gemma specific layer norm scale parameters.
- [x] 62. Identify and support **StarCoder** architecture.
- [x] 63. Identify and support **Falcon** architecture.
- [x] 64. Identify and support **Bloom** architecture.
- [x] 65. Identify and support **Mixtral** (MoE) architecture.
- [x] 66. Extract `expert_count` and `expert_used_count` for MoE architectures.
- [x] 67. Identify and support **StableLM** architecture.
- [x] 68. Identify and support **Command-R** architecture.
- [x] 69. Support exporting classic **BERT** architectures to GGUF (rare, but supported by some GGML forks).
- [x] 70. Throw descriptive errors if an unsupported topology is forced into a strict architecture mapping.

### Phase 5: Tokenizer Extraction & GGUF Embedding

- [x] 71. Extract tokenizer natively from ONNX `ai.onnx.contrib` (HuggingFace tokenizers).
- [x] 72. Extract tokenizer by directly parsing an external `tokenizer.json` file.
- [x] 73. Extract tokenizer by directly parsing an external `tokenizer.model` (SentencePiece) file.
- [x] 74. Map tokenizer model type to `tokenizer.ggml.model` (`llama`, `gpt2`, `bert`).
- [x] 75. Extract and format the complete vocabulary into `tokenizer.ggml.tokens` (ARRAY of STRINGs).
- [x] 76. Extract and format the scores into `tokenizer.ggml.scores` (ARRAY of FLOAT32).
- [x] 77. Extract and format token types into `tokenizer.ggml.token_type` (ARRAY of INT32).
- [x] 78. Extract and format merges for BPE tokenizers into `tokenizer.ggml.merges` (ARRAY of STRINGs).
- [x] 79. Determine and set `tokenizer.ggml.bos_token_id`.
- [x] 80. Determine and set `tokenizer.ggml.eos_token_id`.
- [x] 81. Determine and set `tokenizer.ggml.unknown_token_id`.
- [x] 82. Determine and set `tokenizer.ggml.padding_token_id`.
- [x] 83. Determine and set `tokenizer.ggml.separator_token_id` (if applicable).
- [x] 84. Determine and set `tokenizer.ggml.add_bos_token` (boolean).
- [x] 85. Determine and set `tokenizer.ggml.add_eos_token` (boolean).
- [x] 86. Support mapping specific Chat Templates into `tokenizer.chat_template` (STRING).
- [x] 87. Extract byte-fallback configurations specifically for Llama models.
- [x] 88. Encode vocabulary strings safely, preserving whitespace and control characters exactly as GGML expects.
- [x] 89. Generate a dummy tokenizer if no tokenizer metadata is provided (to prevent `llama.cpp` crash, though inference will be raw tokens).
- [x] 90. Validate the length of `tokens` array matches `vocab_size` exactly.

### Phase 6: ONNX to GGUF Tensor Naming Translation

- [x] 91. Build a Regex-based tensor renaming engine.
- [x] 92. Map ONNX/HF `model.embed_tokens.weight` -> GGUF `token_embd.weight`.
- [x] 93. Map ONNX/HF `model.layers.N.input_layernorm.weight` -> GGUF `blk.N.attn_norm.weight`.
- [x] 94. Map ONNX/HF `model.layers.N.self_attn.q_proj.weight` -> GGUF `blk.N.attn_q.weight`.
- [x] 95. Map ONNX/HF `model.layers.N.self_attn.k_proj.weight` -> GGUF `blk.N.attn_k.weight`.
- [x] 96. Map ONNX/HF `model.layers.N.self_attn.v_proj.weight` -> GGUF `blk.N.attn_v.weight`.
- [x] 97. Map ONNX/HF `model.layers.N.self_attn.o_proj.weight` -> GGUF `blk.N.attn_output.weight`.
- [x] 98. Support fused QKV mapping -> GGUF `blk.N.attn_qkv.weight`.
- [x] 99. Map ONNX/HF `model.layers.N.post_attention_layernorm.weight` -> GGUF `blk.N.ffn_norm.weight`.
- [x] 100. Map ONNX/HF `model.layers.N.mlp.gate_proj.weight` -> GGUF `blk.N.ffn_gate.weight`.
- [x] 101. Map ONNX/HF `model.layers.N.mlp.down_proj.weight` -> GGUF `blk.N.ffn_down.weight`.
- [x] 102. Map ONNX/HF `model.layers.N.mlp.up_proj.weight` -> GGUF `blk.N.ffn_up.weight`.
- [x] 103. Map ONNX/HF `model.norm.weight` -> GGUF `output_norm.weight`.
- [x] 104. Map ONNX/HF `lm_head.weight` -> GGUF `output.weight`.
- [x] 105. Handle bias mappings (e.g., `blk.N.attn_q.bias`) if the architecture uses biases.
- [x] 106. Handle `ffn_gate_up.weight` for fused MLP layers natively.
- [x] 107. Handle MoE explicit layer mapping (`blk.N.ffn_gate_inp.weight`).
- [x] 108. Support overriding the mapping dictionaries via a user-provided JSON configuration.
- [x] 109. Maintain an internal registry of standard mappings for all 15+ supported LLM architectures.
- [x] 110. Fail cleanly and log unmatched tensors if an ONNX model contains unknown weights.

### Phase 7: Tensor Memory Transposition & Layout Adjustments

- [x] 111. GGML expects standard Row-Major arrays. Ensure ONNX weights match.
- [x] 112. Identify 1D tensors (Biases, LayerNorm scales) and write them directly.
- [x] 113. Identify 2D tensors (Linear/MatMul weights) and write them directly.
- [x] 114. Handle Transposition specifically if the ONNX graph uses `[In, Out]` instead of `[Out, In]` for MatMuls.
- [x] 115. Ensure `token_embd.weight` shape strictly matches `[vocab_size, embedding_length]`.
- [x] 116. Ensure `output.weight` shape strictly matches `[vocab_size, embedding_length]`.
- [x] 117. Intercept and permute Attention weights (Q, K) if the ONNX model did not cleanly isolate the RoPE head dimensions.
- [x] 118. Handle multi-dimensional (3D/4D) tensors if exporting non-LLM architectures to GGUF.
- [x] 119. Pad tensors implicitly if GGML quantization block requirements mandate specific dimension multiples.
- [x] 120. Provide a zero-copy fast-path when weights require no transposition or quantization.

### Phase 8: GGUF Quantization Engine (GGML Data Types)

- [x] 121. Support generating `GGML_TYPE_F32` (0).
- [x] 122. Support generating `GGML_TYPE_F16` (1).
- [x] 123. Implement C-equivalent `Float32` to `Float16` downcasting loop natively in JS/Python.
- [x] 124. Support generating `GGML_TYPE_Q4_0` (2).
- [x] 125. Implement `Q4_0` quantization math (block size 32, scale, 4-bit nibbles).
- [x] 126. Support generating `GGML_TYPE_Q4_1` (3).
- [x] 127. Implement `Q4_1` quantization math (block size 32, scale, min_val, 4-bit nibbles).
- [x] 128. Support generating `GGML_TYPE_Q8_0` (8).
- [x] 129. Implement `Q8_0` quantization math (block size 32, scale, 8-bit values).
- [x] 130. Extract standard ONNX `QuantizeLinear` nodes and convert them dynamically to the requested GGML format.
- [x] 131. Bypass quantizing 1D tensors (Biases, Norm scales) natively (GGML strictly keeps these as F32).
- [x] 132. Bypass quantizing `token_embd.weight` if `--leave-embeddings-f32` is flagged.
- [x] 133. Evaluate WASM SIMD acceleration for the `Q4_0` quantization loop to prevent browser freezing on 10GB files.
- [x] 134. Track quantization errors (MSE) dynamically to warn users if `Q4_0` degrades the model severely.
- [x] 135. Map `bfloat16` ONNX inputs safely to `GGML_TYPE_F32` or quantize them, as native BF16 support in GGML can be spotty.
- [x] 136. Support mapping `INT8` ONNX weights to `GGML_TYPE_Q8_0` with zero conversion overhead if scales align.
- [x] 137. Output precise block-alignment validation errors if tensor dimensions are not multiples of 32.
- [x] 138. Validate `Q8_0` structural format exactly matches the C-struct `block_q8_0` definition.
- [x] 139. Validate `Q4_0` structural format exactly matches the C-struct `block_q4_0` definition.
- [x] 140. Generate quantization histograms if detailed logs are requested.

### Phase 9: Reverse Translation (GGUF -> ONNX)

- [x] 141. Implement zero-dependency GGUF Reader in TypeScript/JS.
- [x] 142. Implement zero-dependency GGUF Reader in Python.
- [x] 143. Parse GGUF v2 and v3 headers smoothly.
- [x] 144. Extract metadata dictionary into a standard Python/JS dictionary object.
- [x] 145. Expose `safetensors`-like lazy evaluation API (`gguf.get_tensor("blk.0.attn_q.weight")`).
- [x] 146. Reverse map GGUF standard names back to ONNX/HF naming conventions.
- [x] 147. Reverse map `GGML_TYPE_F32` -> ONNX `FLOAT32`.
- [x] 148. Reverse map `GGML_TYPE_F16` -> ONNX `FLOAT16`.
- [x] 149. Handle `Q4_0` reverse mapping: Extract blocks, apply scales, and emit ONNX `Float32` arrays dynamically.
- [x] 150. Emulate native WebGPU/WASM `Q4_0` decompression shaders directly inside the `onnx9000` execution engine.
- [x] 151. Reconstruct the ONNX AST (Nodes and Edges) specifically based on the parsed `general.architecture`.
- [x] 152. Synthesize `MatMul`, `Add`, `LayerNormalization` nodes correctly based on LLaMA architecture flags.
- [x] 153. Inject RoPE embedding generation directly into the reconstructed ONNX graph.
- [x] 154. Convert `tokenizer.ggml.tokens` back into standard ONNX `String` tensors or a `.json` file for HF.
- [x] 155. Provide CLI command: `onnx9000 gguf2onnx model.gguf -o model.onnx`.
- [x] 156. Safely handle multi-file GGUF splits (`model-00001-of-00002.gguf`) during reading.
- [x] 157. Export reconstructed GGUF weights directly into `onnx9000.safetensors` external data format.
- [x] 158. Reconstruct Attention Mask logic securely during reverse-translation.
- [x] 159. Output a fully compliant, executable `.onnx` graph capable of running in standard ONNX Runtime.
- [x] 160. Test the GGUF reader successfully mapping an `INT8` model to an ONNX `QuantizeLinear` -> `MatMul` graph.

### Phase 10: CLI Tooling (`onnx9000 onnx2gguf`)

- [x] 161. Implement CLI: `onnx9000 onnx2gguf model.onnx -o model.gguf`.
- [x] 162. Support `--tokenizer <path>` flag to point explicitly to a `tokenizer.json` or `tokenizer.model`.
- [x] 163. Support `--outtype <type>` (e.g., `f32`, `f16`, `q8_0`, `q4_0`).
- [x] 164. Support `--architecture <arch>` override flag (e.g., `--architecture llama`).
- [x] 165. Support `--split-max-size` flag (e.g., `4GB`) to automatically generate GGUF shards.
- [x] 166. Handle directory inputs (e.g., reading a directory containing `model.safetensors` and `config.json`).
- [x] 167. Implement progress bars specifically tracking block-quantization percentage.
- [x] 168. Expose `--dry-run` flag to print the generated Metadata and Tensor maps without writing binary data.
- [x] 169. Provide memory footprint checks warning the user before initiating a 70B parameter conversion.
- [x] 170. Validate full CLI parity against `llama.cpp/convert_hf_to_gguf.py`.

### Phase 11: The Web UI (Client-Side Exporter)

- [x] 171. Build a static Web Components Web UI for `onnx2gguf`.
- [x] 172. Implement drag-and-drop for `model.onnx` or multiple `.safetensors` files simultaneously.
- [x] 173. Implement drag-and-drop for `tokenizer.json` and `config.json`.
- [x] 174. Display an interactive table previewing the extracted KV Metadata.
- [x] 175. Allow the user to explicitly edit KV metadata values in the browser before export.
- [x] 176. Implement visual dropdowns for the Quantization Target (`f16`, `q8_0`, `q4_0`).
- [x] 177. Use Web Workers to process the multi-gigabyte conversion without freezing the UI.
- [x] 178. Utilize Streams API to pipe the output `.gguf` directly to the local filesystem (preventing RAM limit crashes).
- [x] 179. Show real-time encoding speed (e.g., `Encoding Q4_0: 150 MB/s`).
- [x] 180. Provide specific browser compatibility warnings if running on devices with <8GB RAM.

### Phase 12: HuggingFace Hub Integration

- [x] 181. Support fetching model configurations natively from the HF Hub API.
- [x] 182. Download `config.json` and `tokenizer.json` dynamically if they aren't provided locally.
- [x] 183. Identify `.safetensors.index.json` manifests on the Hub and stream-convert shards seamlessly.
- [x] 184. Append HF repository origin strings to the `general.source.url` KV metadata.
- [x] 185. Handle HF authentication tokens for private repositories.
- [x] 186. Output standard `README.md` text formatted for a new GGUF model repository.
- [x] 187. Publish a Node.js utility script capable of fetching, converting, and re-uploading to the Hub automatically.
- [x] 188. Catch 404s and 403s cleanly during remote configuration fetching.
- [x] 189. Cache fetched weight chunks into IndexedDB/Local disk temporarily during processing.
- [x] 190. Match `transformers` internal dictionary structures seamlessly when bypassing standard ONNX inputs.

### Phase 13: Edge Cases, Security & Validation

- [x] 191. Prevent integer overflows in Javascript when aggregating massive tensor offsets (`BigInt` enforcement).
- [x] 192. Handle extremely long tensor names (validating against reasonable limits).
- [x] 193. Trap Out-Of-Memory (OOM) errors and report them as "Hardware Constrained" rather than vague JS stack traces.
- [x] 194. Handle 0-D (Scalar) tensors strictly if the architecture requires them.
- [x] 195. Verify that `llama.cpp` itself can cleanly read and execute the generated `.gguf` file without warnings.
- [x] 196. Sanitize tokenizer strings containing invalid or malformed UTF-8 characters.
- [x] 197. Validate `alignment` rules strictly. Every tensor data block MUST start at an address divisible by `general.alignment`.
- [x] 198. Protect against path traversal attacks if loading shards based on manifest JSON strings.
- [x] 199. Strip all sensitive local system path strings from the metadata payload.
- [x] 200. Execute validation against Endianness discrepancies if running the Node.js CLI on Big-Endian architectures.

### Phase 14: LLaMA.cpp Specific Quirks & Hacks

- [x] 201. Support `llama.attention.key_length` overrides explicitly.
- [x] 202. Support `llama.attention.value_length` overrides explicitly.
- [x] 203. Set `llama.tensor_data_layout` if spatial conventions differ.
- [x] 204. Manage custom `expert_weights_scale` vectors specifically for MoE.
- [x] 205. Implement implicit permuting of `attn_q` and `attn_k` weights specifically for RoPE application in `llama.cpp`.
- [x] 206. Identify and reverse permutation of RoPE weights if reversing GGUF -> ONNX.
- [x] 207. Extract `layer_norm_eps` globally or locally depending on the specific model dialect.
- [x] 208. Emit dummy `rope.freq_base` if the ONNX model calculates RoPE dynamically but `llama.cpp` requires static definition.
- [x] 209. Map standard SentencePiece control tokens directly to the `tokenizer.ggml.tokens` array indices.
- [x] 210. Map `AddedTokens` natively to the end of the `tokenizer.ggml.tokens` array.

### Phase 15: Advanced Quantization (K-Quants & Custom Types)

- [x] 211. Provide structural support for `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` formats in the parser.
- [x] 212. Implement `Q4_K` quantization loop native in JS/Python (highly complex block structures).
- [x] 213. Implement `Q6_K` quantization loop.
- [x] 214. Validate Super-Block and Sub-Block alignments natively.
- [x] 215. Extract `IQ2_XXS` and `IQ2_XS` configurations if the parser is used as a reader.
- [x] 216. Allow users to define a mixed-quantization strategy (e.g., `Q8_0` for Attention, `Q4_0` for FFN).
- [x] 217. Emit the corresponding `general.file_type` for mixed `K` quants dynamically.
- [x] 218. Utilize Web Workers to parallelize block quantization (chunking the weight matrices).
- [x] 219. Prevent quantizing the final `output.weight` layer by default, as doing so severely damages perplexity.
- [x] 220. Output the quantization error MSE strictly for the user's review.

### Phase 16: Interoperability with other `onnx9000` Tools

- [x] 221. Integration: `onnx9000.optimum` -> `onnx9000.onnx2gguf` pipeline.
- [x] 222. Convert HuggingFace PyTorch models to ONNX to GGUF in one fluid CLI command.
- [x] 223. Support injecting `onnx9000.modifier` AST changes directly into the GGUF compiler stream.
- [x] 224. Expose the `gguf` reading functionality natively to the `onnx9000.Netron` visualizer.
- [x] 225. Expose `safetensors` fast-streaming logic to bypass memory limits during GGUF packing.
- [x] 226. Use `onnx9000.onnx-tool` to estimate the final GGUF file size before compiling.
- [x] 227. Compare original ONNX MACs vs GGUF intended MACs (informational).
- [x] 228. Provide a simple `np.load_gguf("model.gguf")` API mapping directly to `onnx9000.array`.
- [x] 229. Allow executing the GGUF file natively inside the `onnx9000.genai` LLM pipeline.
- [x] 230. Support evaluating raw `.gguf` files via WebGPU directly inside the browser.

### Phase 17: Code Quality, Optimization & CI

- [x] 231. Establish automated CI pipeline converting TinyLlama to GGUF.
- [x] 232. Run the generated GGUF through an actual compiled `llama.cpp` binary in GitHub Actions to verify functionality.
- [x] 233. Measure memory allocations during the TypeScript Web Worker process (keeping it below 1GB).
- [x] 234. Expose strict Typescript interfaces (`GGUFMetadata`, `GGUFTensorInfo`) for external API consumers.
- [x] 235. Compile the JS codebase using ESBuild/Vite targeting modern ESM modules.
- [x] 236. Eliminate any Node.js native `fs` imports from the core converter library to ensure strict isomorphic compliance.
- [x] 237. Emulate `fs` using `File` and `Blob` APIs correctly.
- [x] 238. Export a standalone UMD package for easy `<script>` tag inclusion.
- [x] 239. Test across Windows, Linux, and macOS environments natively.
- [x] 240. Ensure all generated code passes rigorous linting and formatting.

### Phase 18: Specific Metadata & Model Edge Cases

- [x] 241. Extract `falcon.attention.use_alibi` parameter correctly.
- [x] 242. Extract `starcoder.attention.head_count_kv` correctly.
- [x] 243. Handle models with tied embeddings (where `output.weight` is mathematically linked to `token_embd.weight`).
- [x] 244. Skip generating `output.weight` explicitly if tied embeddings are detected to save disk space.
- [x] 245. Validate multi-query attention (MQA) correctly maps to $KV\_Heads = 1$.
- [x] 246. Handle custom padding token configurations that mismatch BOS/EOS.
- [x] 247. Encode specific `phi2` normalization constants correctly.
- [x] 248. Resolve specific `qwen2` RoPE parameters natively.
- [x] 249. Map classic Transformer relative positional encodings if found.
- [x] 250. Support extracting vocabulary from legacy HuggingFace `vocab.json` instead of just `tokenizer.json`.

### Phase 19: Comprehensive Documentation

- [x] 251. Write Tutorial: "Converting ONNX to GGUF in the Browser".
- [x] 252. Write Tutorial: "Reading a GGUF model natively in Node.js".
- [x] 253. Document the complete mapping schema of `ONNX -> GGUF`.
- [x] 254. Provide a compatibility matrix of supported LLM architectures.
- [x] 255. Detail the memory limitations and browser configuration (e.g., enabling WASM-64) required for massive models.
- [x] 256. Provide clear code snippets for integrating `@onnx9000/onnx2gguf` into a Web application.
- [x] 257. Document all supported CLI flags and arguments.
- [x] 258. Create an architectural diagram showing how data flows from `.safetensors` through the transpiler to `.gguf`.
- [x] 259. Publish a specific "Troubleshooting" guide for shape mismatch errors.
- [x] 260. Include a benchmark report proving quantization parity with `llama.cpp`.

### Phase 20: Final Deliverables & Web-Native Polish

- [x] 261. Expose `get_metadata(file)` API for incredibly fast extraction without parsing the multi-GB payload.
- [x] 262. Support dynamic fetching of GGUF metadata via HTTP Range requests natively.
- [x] 263. Establish a unified error handling class (`GGUFParseError`, `GGUFCompileError`).
- [x] 264. Guarantee absolutely identical hash sums if the tool is run twice on the same model.
- [x] 265. Provide visual feedback (spinners/bars) during long I/O operations.
- [x] 266. Enable "Append" mode, allowing users to inject new KV metadata into an existing `.gguf` file without rewriting the multi-GB tensor block.
- [x] 267. Cleanly catch user attempts to load incompatible model types (e.g., attempting to compile a ResNet as a LLaMA).
- [x] 268. Provide a simple `onnx9000 info model.gguf` CLI command to dump the metadata to the console.
- [x] 269. Map `general.license` strings dynamically from Hub configs.
- [x] 270. Extract hardware-specific execution hints if embedded in ONNX properties.
- [x] 271. Verify that `ArrayBuffer` allocation sizes strictly respect the necessary 8-byte boundaries globally.
- [x] 272. Add custom support for decoding string values spanning multiple chunks safely.
- [x] 273. Support `tokenizer.ggml.pre` string configurations explicitly (e.g., `llama3`, `deepseek`).
- [x] 274. Handle multiple chat templates if stored as arrays in `tokenizer.json`.
- [x] 275. Ensure backward compatibility with `llama.cpp` binaries compiled in mid-2023.
- [x] 276. Develop specific fallbacks for un-quantizable dense layers.
- [x] 277. Render visual warnings if quantization scales evaluate to `NaN` or `Infinity`.
- [x] 278. Automatically drop intermediate caches immediately after block packing.
- [x] 279. Expose raw internal byte arrays if a user requests them natively.
- [x] 280. Handle specific MoE (Mixture of Experts) expert routing maps cleanly.
- [x] 281. Test performance degradation when Web Workers are unavailable (synchronous mode).
- [x] 282. Add a `verify_checksum()` utility verifying the internal structural integrity of the GGUF arrays.
- [x] 283. Support generating the older `GGML` or `GGJT` magic bytes if specifically requested via legacy flags.
- [x] 284. Allow developers to register custom metadata serializers.
- [x] 285. Support parsing and embedding custom LoRA weights directly into the GGUF payload.
- [x] 286. Handle ONNX Sequence outputs logically inside the GGUF translation boundary.
- [x] 287. Expose an interactive JSON viewer specifically for GGUF metadata inside the web tool.
- [x] 288. Manage specific Safari-only bugs regarding `Blob` and `File` object lifecycle limits.
- [x] 289. Add a specific warning if the user attempts to run a `Q4_0` quantization on a model with `bfloat16` inputs.
- [x] 290. Support conversion from `.h5` natively via `onnx9000.keras` linking prior to GGUF packing.
- [x] 291. Validate exact mathematical equivalence of `Exp` and `Log` limits during Quantization routines.
- [x] 292. Add support for importing TensorFlow `SavedModel` directly to GGUF (via ONNX multi-hop).
- [x] 293. Build specific handling for `Float8` standards if GGUF v4 introduces them.
- [x] 294. Enable deep network traces in the JS Console if the `--verbose` flag is passed to the web component.
- [x] 295. Configure explicit warnings for missing `bos_token_id`.
- [x] 296. Validate 32-bit offset limits securely, throwing standard exceptions before memory corruption occurs.
- [x] 297. Support executing the output `.gguf` file using `onnx9000`'s native WebGPU GenAI loop.
- [x] 298. Establish automated workflows to deploy the converter to a CDN (`unpkg.com`).
- [x] 299. Write comprehensive API documentation mapping TS generation targets.
- [x] 300. Release v1.0 feature complete certification for `onnx9000.onnx2gguf` bridging the ONNX and LLAMA.CPP communities.
