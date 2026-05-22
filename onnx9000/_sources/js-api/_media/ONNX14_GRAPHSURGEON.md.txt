# ONNX GraphSurgeon Replication & Parity Tracker

## Description

This document tracks the complete reimplementation of `ONNX GraphSurgeon` within the `onnx9000` ecosystem.
Unlike the original project, our implementation is tightly integrated into a pure-Python, zero-dependency Intermediate Representation (IR). It avoids the massive C++ Protocol Buffers backend entirely.
By operating purely on Python dictionaries, lists, and lightweight dataclasses, this GraphSurgeon can modify, prune, fold, and optimize 1GB+ graphs instantaneously within a browser (WASM/Pyodide) or a distributed serverless environment with zero installation overhead.

## Exhaustive Parity Checklist

### 1. Core IR Abstractions & Graph Topology (50+ items)

- [x] Implement `Graph` core abstraction
- [x] Implement `Node` core abstraction
- [x] Implement `Tensor` base abstraction
- [x] Implement `Variable` (dynamic tensor) abstraction
- [x] Implement `Constant` (static tensor) abstraction
- [x] Support Graph `name` manipulation
- [x] Support Graph `opset_imports` registry and manipulation
- [x] Support Graph `doc_string` extraction/modification
- [x] Support Graph `inputs` list modification (adding/removing global inputs)
- [x] Support Graph `outputs` list modification (adding/removing global outputs)
- [x] Support Node `op` (operator type) modification
- [x] Support Node `name` modification
- [x] Support Node `attrs` (attributes) dictionary-like access
- [x] Support Node `inputs` list manipulation (binding/unbinding Tensors)
- [x] Support Node `outputs` list manipulation (binding/unbinding Tensors)
- [x] Implement Node `i()` utility for quick input tensor retrieval
- [x] Implement Node `o()` utility for quick output tensor retrieval
- [x] Support Tensor `name` modification
- [x] Support Tensor `dtype` manipulation
- [x] Support Tensor `shape` manipulation
- [x] Support Constant `values` manipulation (via zero-copy NumPy/Pyodide array views)
- [x] Implement Graph `tensors()` lazy evaluation dictionary generator
- [x] Implement Graph `nodes` sequential list abstraction
- [x] Automatically track Node dependencies (Producers/Consumers)
- [x] Implement Tensor `inputs` (producer node) tracking
- [x] Implement Tensor `outputs` (consumer nodes) tracking
- [x] Ensure `Node` hashes uniquely by identity for fast dictionary lookups
- [x] Ensure `Tensor` hashes uniquely by identity
- [x] Support cyclic graph detection during topology traversal
- [x] Implement pure-Python structural equality checking (`__eq__`) for Graphs
- [x] Implement structural equality checking for Nodes
- [x] Implement structural equality checking for Attributes
- [x] Implement shape broadcasting rules within `Constant` mathematical evaluation
- [x] Implement `Attribute` type inference (Int, Float, String, Tensor, Graph, Lists)
- [x] Support sub-graphs nested inside Node attributes (e.g., for `If`, `Loop`)
- [x] Recursively traverse nested sub-graphs during Graph operations
- [x] Support pure-Python deep-copying of the entire Graph (`copy()`)
- [x] Support deep-copying of isolated Nodes
- [x] Support deep-copying of isolated Tensors
- [x] Support extracting subgraph isolating all dependencies of a target Tensor
- [x] Implement memory-efficient string deduplication for Tensor names
- [x] Handle duplicate Node names natively (auto-rename / uniquify)
- [x] Handle duplicate Tensor names natively (auto-rename / uniquify)
- [x] Implement Graph `producer_map` caching for fast traversal
- [x] Implement Graph `consumer_map` caching for fast traversal

### 2. Node & Graph Traversal Operations (30+ items)

- [x] Implement Graph `toposort()` (Topological Sort)
- [x] Implement Graph `cleanup()` (Dead Code Elimination)
- [x] Implement Graph `fold_constants()` (Constant Folding)
- [x] Implement Graph `simplify()` (Alias to fold + cleanup + toposort)
- [x] Implement Node `prev_nodes()` utility
- [x] Implement Node `next_nodes()` utility
- [x] Implement Tensor `is_empty()` check
- [x] Implement Tensor `is_dynamic()` check (contains -1 or symbolic dims)
- [x] Implement Graph `walk()` Depth-First Search (DFS)
- [x] Implement Graph `walk()` Breadth-First Search (BFS)
- [x] Walk generator: yield only Nodes
- [x] Walk generator: yield only Tensors
- [x] Walk generator: yield only Constants
- [x] Walk generator: yield only Variables
- [x] Support bidirectional walking (forward from inputs, backward from outputs)
- [x] Find node by exact `name`
- [x] Find nodes by `op` type
- [x] Find nodes matching a Regex on `name`
- [x] Find nodes matching a Regex on `op` type
- [x] Find tensor by exact `name`
- [x] Find tensors matching a Regex on `name`
- [x] Isolate all nodes of a specific `opset` domain
- [x] Find path between two nodes (shortest path)
- [x] Find all paths between two nodes
- [x] Detect disconnected graph components
- [x] Extract connected component subgraph
- [x] Analyze critical path (longest latency path heuristic)
- [x] Profiling: Count MACs/FLOPs statically based on shapes
- [x] Profiling: Count static memory footprint of all `Constant`s
- [x] Profiling: Estimate activation memory footprint

### 3. Surgical Modifications & Topology Editing (40+ items)

- [x] `Graph.append_node()`: Add node to end of graph
- [x] `Graph.insert_node()`: Insert node at specific topology index
- [x] `Graph.remove_node()`: Delete node and sever tensor links
- [x] `Graph.replace_node()`: Swap node, rewiring inputs/outputs
- [x] `Graph.disconnect_node()`: Unbind node but keep in memory
- [x] `Tensor.clear_inputs()`: Disconnect from producer
- [x] `Tensor.clear_outputs()`: Disconnect from all consumers
- [x] `Node.disconnect_input()`: Sever specific input link
- [x] `Node.disconnect_output()`: Sever specific output link
- [x] `Node.replace_input()`: Swap input tensor seamlessly
- [x] `Node.replace_output()`: Swap output tensor seamlessly
- [x] Support injecting a single Node into an existing Tensor edge
- [x] Support bypassing a Node (connect its inputs directly to its consumers)
- [x] Convert `Variable` to `Constant` (Baking in values)
- [x] Convert `Constant` to `Variable` (Extracting values to runtime inputs)
- [x] Auto-generate missing Tensor names during surgery
- [x] Auto-generate missing Node names during surgery
- [x] Support fusing two sequential Nodes manually
- [x] Support splitting one Node into multiple Nodes manually
- [x] Append an entirely separate Graph into the current Graph
- [x] Prepend an entirely separate Graph into the current Graph
- [x] Register new global Input to Graph
- [x] Register new global Output to Graph
- [x] Remove global Input (if unused or replaced)
- [x] Remove global Output (if unused or replaced)
- [x] Reorder global Inputs safely
- [x] Reorder global Outputs safely
- [x] Upgrade Node Opsets individually (e.g., Cast opset 9 to opset 13)
- [x] Downgrade Node Opsets individually
- [x] Rename Op types across the entire graph natively
- [x] Rename domains across the entire graph natively
- [x] Inject `Identity` nodes for debugging probes
- [x] Remove all `Identity` nodes (Identity Elimination)
- [x] Promote internal Tensor to global Output (for intermediate debugging)
- [x] Demote global Output to internal Tensor
- [x] Promote Constant to global Input (Parameter)
- [x] Isolate Subgraph by specifying entry Tensors and exit Tensors
- [x] Duplicate Subgraph (templating)
- [x] Inject custom CustomOp / Plugin nodes safely

### 4. Advanced Pattern Matching & Replacement (40+ items)

- [x] Implement declarative subgraph pattern matcher
- [x] Match node by `op` type
- [x] Match node by attribute value / condition
- [x] Match node by input tensor shape condition
- [x] Match node by input tensor dtype condition
- [x] Match node by output tensor shape condition
- [x] Match sequential chains (e.g., A -> B -> C)
- [x] Match branched structures (e.g., A -> B, A -> C, B+C -> D)
- [x] Wildcard matching for indeterminate nodes in a pattern
- [x] Optional node matching (match A -> [B] -> C)
- [x] Unordered input matching (match Add(A, B) exactly like Add(B, A))
- [x] Implement `replace_pattern()` automatic subgraph substitution
- [x] Support mapping matched attributes to replacement subgraph
- [x] Support mapping matched input tensors to replacement subgraph
- [x] Match explicitly constant subgraphs
- [x] Match purely dynamic subgraphs
- [x] Validate subgraph replacement preserves topological integrity
- [x] Ensure nested subgraphs (If/Loop) are matched recursively
- [x] Prevent overlapping pattern matches during parallel replacement
- [x] Log and trace all pattern matches and replacements
- [x] Register custom matcher callbacks via Python functions

### 5. Standard Graph Optimizations (60+ items)

- [x] Built-in pass: Constant Folding (Math ops)
- [x] Built-in pass: Constant Folding (Shape/Slice ops)
- [x] Built-in pass: Constant Folding (Reshape/Transpose)
- [x] Built-in pass: Dead Code Elimination (DCE)
- [x] Built-in pass: Identity Elimination
- [x] Built-in pass: Dropout Elimination
- [x] Built-in pass: Cast Elimination (redundant casts)
- [x] Built-in pass: Transpose Sinking (push transposes past elementwise ops)
- [x] Built-in pass: Layout Conversion (NCHW <-> NHWC)
- [x] Built-in pass: Shape Inference (statically propagating shapes)
- [x] Built-in pass: Symbolic Shape Inference
- [x] Built-in pass: DType Inference (propagating types)
- [x] Fusion pass: `Conv` + `BatchNormalization` -> `Conv`
- [x] Fusion pass: `Conv` + `Add` -> `Conv` (Bias)
- [x] Fusion pass: `Conv` + `Mul` -> `Conv`
- [x] Fusion pass: `MatMul` + `Add` -> `Gemm`
- [x] Fusion pass: `Gemm` + `Relu` -> `Gemm`
- [x] Fusion pass: `Conv` + `Relu` -> `Conv`
- [x] Fusion pass: `Conv` + `Clip` -> `Conv`
- [x] Fusion pass: Sequential `Reshape` ops -> Single `Reshape`
- [x] Fusion pass: Sequential `Transpose` ops -> Single `Transpose` (or Identity)
- [x] Fusion pass: Sequential `Slice` ops -> Single `Slice`
- [x] Fusion pass: `Squeeze` + `Unsqueeze` cancellations
- [x] Fusion pass: `Split` + `Concat` cancellations
- [x] Fusion pass: `Pad` + `Slice` cancellations
- [x] Fusion pass: GELU exact pattern (Erf) -> `Gelu` (if supported)
- [x] Fusion pass: GELU tanh pattern -> `Gelu`
- [x] Fusion pass: LayerNormalization pattern -> `LayerNormalization`
- [x] Fusion pass: Attention pattern -> `MultiHeadAttention`
- [x] Fusion pass: Rotary Positional Embedding (RoPE) pattern
- [x] Fusion pass: GroupNormalization pattern
- [x] Optimization: Strip `doc_string` globally to save memory
- [x] Optimization: Strip tensor names to save memory (minification)
- [x] Optimization: Deduplicate identical `Constant` tensors globally
- [x] Optimization: Pack individual `Constant`s into a single contiguous binary blob
- [x] Optimization: Downcast `Float64` to `Float32` globally
- [x] Optimization: Downcast `Float32` to `Float16` globally
- [x] Optimization: Downcast `Int64` to `Int32` globally (where safe)
- [x] Quantization pass: Static INT8 (Dynamic fake-quantize replacement)
- [x] Quantization pass: Weight-only INT8 packing
- [x] Quantization pass: Weight-only INT4 packing

### 6. Zero-Dependency & Lightweight Runtime Integrations (30+ items)

- [x] Parse `ModelProto` fully natively in Python (no `import onnx`)
- [x] Parse `GraphProto` fully natively in Python
- [x] Parse `NodeProto` fully natively in Python
- [x] Parse `TensorProto` fully natively in Python
- [x] Parse `AttributeProto` fully natively in Python
- [x] Parse `ValueInfoProto` fully natively in Python
- [x] Export directly to raw byte payloads (zero-copy serialization)
- [x] Provide lazy-loading for >2GB `TensorProto` external data
- [x] Support WASM `ArrayBuffer` directly as Constant backing storage
- [x] Support Pyodide memory-view zero-copy bridging
- [x] Drag-and-drop parsing inside standard Chrome/Safari browser environments
- [x] Sub-50ms Graph Surgeon initialization latency
- [x] Interactive Web UI hook: export graph state to JSON for Visualizer (Netron parity)
- [x] Stream modified subgraphs instantly to WebGPU compiler
- [x] Serverless ready: Run GraphSurgeon safely inside AWS Lambda memory limits
- [x] Cloudflare Worker ready: Strip all parsing logic to under 2MB
- [x] CLI fully operational without `protobuf` C++ or `onnx` pip package installed
- [x] Ray/Celery distributed compatibility (Serializable IR structures)
- [x] Auto-chunking of large constant arrays for HTTP streaming

### 7. Explicit Opset & Validation Tools (30+ items)

- [x] Implement Graph topological validation checker
- [x] Validate no cyclical dependencies exist natively
- [x] Validate all internal Tensors have producers
- [x] Validate no conflicting Tensor names exist
- [x] Validate Node attributes match strict ONNX Opset specifications
- [x] Automatically upgrade an entire Graph to Opset 15
- [x] Automatically upgrade an entire Graph to Opset 16
- [x] Automatically upgrade an entire Graph to Opset 17
- [x] Automatically upgrade an entire Graph to Opset 18
- [x] Automatically upgrade an entire Graph to Opset 19
- [x] Automatically upgrade an entire Graph to Opset 20
- [x] Automatically upgrade an entire Graph to Opset 21
- [x] Strict type checking pass (ensure Float32 doesn't flow into Int64 requirements)
- [x] Strict shape dimension checking pass (tensor algebra validation)
- [x] Validate `If` subgraphs have identical output typings
- [x] Validate `Loop` subgraphs maintain state iteration typings
- [x] Compare two GraphSurgeon graphs for strict equivalence
- [x] Compare two GraphSurgeon graphs for semantic equivalence (ignoring names)
- [x] Print human-readable ASCII summary of the Graph structure
- [x] Generate standard ONNX protobuf `.onnx` output payloads
- [x] Generate human-readable `.txt` output payloads
- [x] Export constants safely to external data chunks (`.bin`)
- [x] Import constants safely from external data chunks

### 8. TensorRT Pattern Injection & Surgery (30+ items)

- [x] Inject `QuantizeLinear` dynamically based on heuristic thresholds
- [x] Inject `DequantizeLinear` dynamically based on heuristic thresholds
- [x] Fuse `QuantizeLinear` + `DequantizeLinear` into `FakeQuantize`
- [x] Unfuse `FakeQuantize` into `QuantizeLinear` + `DequantizeLinear`
- [x] Support TensorRT-specific custom plugins (e.g., `TRT_PluginV2`)
- [x] Inject custom TensorRT `GridSample` nodes safely
- [x] Inject custom TensorRT `NMS` nodes safely
- [x] Inject custom TensorRT `RoIAlign` nodes safely
- [x] Convert `NonMaxSuppression` (ONNX) to TensorRT `BatchedNMSDynamic_TRT`
- [x] Convert `Resize` (ONNX) to TensorRT `ResizeNearest` or `ResizeLinear`
- [x] Convert `TopK` (ONNX) to TensorRT `TopK_TRT`
- [x] Convert `ScatterND` (ONNX) to TensorRT `ScatterND_TRT`
- [x] Extract dynamic shape profiles (min/opt/max) into graph metadata
- [x] Extract sequence length bounds into graph metadata
- [x] Map TensorRT explicit precision bounds to ONNX `dtype` overrides
- [x] Enforce explicit FP16 boundaries for TRT engine optimization
- [x] Enforce explicit INT8 boundaries for TRT engine optimization
- [x] Inject TRT calibration nodes dynamically
- [x] Support generating TRT-friendly grouped convolutions
- [x] Support generating TRT-friendly separated convolutions

### 9. Detailed Tensor Manipulations & Mathematics (30+ items)

- [x] Transpose constant tensors natively in-memory (Numpy/Pyodide)
- [x] Reshape constant tensors natively in-memory
- [x] Broadcast constant tensors natively in-memory
- [x] Slice constant tensors natively in-memory
- [x] Concatenate constant tensors natively in-memory
- [x] Cast constant tensors (e.g., Float32 to Float16) in-memory
- [x] Quantize constant tensors to INT8 natively in-memory
- [x] Unpack packed quantized INT4 weights to INT8 or Float16 in-memory
- [x] Evaluate mathematical expression graphs (e.g., Constant + Add + Constant)
- [x] Evaluate boolean expression graphs (e.g., Constant == Constant)
- [x] Evaluate logical expression graphs (e.g., Constant AND Constant)
- [x] Evaluate shape-related graphs natively (`Shape`, `Size`, `Gather`)
- [x] Evaluate index-related graphs natively (`NonZero`)
- [x] Support memory-mapping (mmap) large constant values during manipulation
- [x] Support lazy evaluation of constant values to preserve RAM
- [x] Extract single scalar values from 0D constants safely
- [x] Embed multiple smaller constants into a single large constant (Tensor packing)
- [x] Expand single large constants into multiple smaller constants (Tensor unpacking)
- [x] Support sparse tensor dense expansion inside GraphSurgeon
- [x] Support dense tensor to sparse tensor compression inside GraphSurgeon

### 10. Graph Debugging & Inspection Utilities (20+ items)

- [x] Print Node input/output topology map logically
- [x] Print all Graph Constants sorted by memory size (to find bloat)
- [x] Print all Node Op types sorted by frequency (profiling)
- [x] Trace all Ops interacting with a specific Tensor implicitly
- [x] Trace the computational origin (inputs) of a specific Tensor
- [x] Trace the computational destiny (outputs) of a specific Tensor
- [x] Dump specific sub-graph to raw JSON for Netron integration
- [x] Visualize sub-graph dynamically in Browser Canvas via hooks
- [x] Validate Node attributes for missing required fields
- [x] Validate Node attributes for invalid string/tensor encodings
- [x] Compare floating point constants safely (`numpy.allclose` logic)
- [x] Compare integer constants safely
- [x] Warn on implicit broadcasting (e.g., Add(1x3x224x224, 3))
- [x] Warn on dimension mismatches (`numpy` shape validation)
- [x] Identify isolated nodes (dead code) interactively

### 11. Custom Node Injection & Callbacks (20+ items)

- [x] Register new Custom Operator `schema` to the Graph
- [x] Inject Custom Operator with arbitrarily named inputs
- [x] Inject Custom Operator with arbitrarily named outputs
- [x] Inject Custom Operator with custom Dictionary attributes
- [x] Inject Custom Operator with custom Tensor attributes
- [x] Automatically track Custom Operator domains during export
- [x] Safely delete Custom Operators during surgery
- [x] Support user-defined callback for Node creation (Hook logic)
- [x] Support user-defined callback for Node deletion (Hook logic)
- [x] Support user-defined callback for Graph compilation (Hook logic)
- [x] Support user-defined callback for Topology sort changes
- [x] Safely wrap unrecognized ONNX domains into Custom Ops natively
- [x] Unwrap Custom Ops into sub-graphs (if an expander is provided)
- [x] Validate Custom Ops against user-provided type inference logic
- [x] Provide dynamic type inference for Custom Ops via Python lambdas

### 12. Surgical Testing & Quality Assurance (20+ items)

- [x] Unit Test: Topologically sort a 10,000 node graph in < 50ms
- [x] Unit Test: Deep copy a 1GB graph safely
- [x] Unit Test: Extract a dense BERT subgraph based strictly on attention outputs
- [x] Unit Test: Replace all `Gelu` custom nodes with Erf standard nodes
- [x] Unit Test: Strip 50+ unconnected identity/shape ops sequentially
- [x] Unit Test: Insert a probe `Identity` node after every `Conv` output
- [x] Unit Test: Fold chained `Reshape` -> `Transpose` -> `Reshape` logic correctly
- [x] Unit Test: Validate cyclic detection throws appropriate exceptions
- [x] Unit Test: Ensure `Variable` to `Constant` promotion preserves memory
- [x] Unit Test: Evaluate `numpy.allclose` logic across massive constants
- [x] Unit Test: Parse and dump `opset=13` to `opset=17` correctly
- [x] Unit Test: Inject custom WebGPU hooks directly into `ai.onnx` domains
- [x] Unit Test: Convert a multi-input Node to single-input sequentially
- [x] Unit Test: Handle duplicate node insertions gracefully
- [x] Unit Test: Prevent invalid node deletion gracefully
