# Netron Replication & Parity Tracker

## Description

This document tracks the complete reimplementation and architectural enhancement of `Netron` within the `onnx9000` ecosystem.
While the original `Netron` is a fantastic static visualizer (Electron/Web based), it struggles with massive models (like 10GB+ LLMs) due to DOM-based SVG rendering limitations. Our `onnx9000` reimplementation leverages a unified WASM-accelerated pure-JavaScript/Python parser combined with a highly optimized WebGL/Canvas rendering pipeline.
Furthermore, because it is deeply integrated with `onnx9000`'s IR and `GraphSurgeon`, it transcends being a simple visualizer: it becomes a **live, interactive graph editor and profiler** running entirely in the browser at 60FPS, without installing heavy desktop applications.

## Exhaustive Parity Checklist

### 1. Core Architecture & Parsing Engine (40+ items)

- [x] Implement zero-dependency WASM/JS ONNX `ModelProto` parser
- [x] Implement lazy-loading for large `.onnx` files (chunked streaming)
- [x] Support memory-mapped reading of massive multi-GB models without crashing the browser
- [x] Parse `GraphProto` into an optimized internal visualization DAG structure
- [x] Parse `NodeProto` into visualization nodes
- [x] Parse `TensorProto` (Constants/Initializers) with lazy data extraction
- [x] Parse `ValueInfoProto` (Inputs/Outputs/Value Info) for shape/type inference
- [x] Parse `AttributeProto` natively (Floats, Ints, Strings, Tensors, Graphs)
- [x] Implement WebWorker based off-main-thread parsing to keep UI responsive
- [x] Support handling external data files (`.bin` weights) locally via File API
- [x] Handle corrupt or truncated ONNX files gracefully with partial visualization
- [x] Support parsing strictly ONNX standard topologies
- [x] Support parsing `ai.onnx.ml` domain topologies
- [x] Support parsing TensorFlow `GraphDef` (`.pb`) via `onnx9000` transpilation
- [x] Support parsing TensorFlow `SavedModel` via transpilation
- [x] Support parsing TensorFlow Lite (`.tflite`) via transpilation
- [x] Support parsing Keras (`.h5`, `.keras`) via transpilation
- [x] Support parsing PyTorch (`.pt`, `.pth`) via transpilation (TorchScript extraction)
- [x] Support parsing CoreML (`.mlmodel`) via transpilation
- [x] Support parsing Scikit-Learn (Pickle) via `skl2onnx` transpilation
- [x] Support parsing XGBoost / LightGBM / CatBoost via `onnxmltools` transpilation
- [x] Support parsing PaddlePaddle (`.pdmodel`) via transpilation
- [x] Extract Graph metadata (Producer, Version, Description, DocString)
- [x] Extract Model Opset imports and domain configurations
- [x] Traverse deeply nested subgraphs (`If`, `Loop`) natively in the parser
- [x] Resolve missing `ValueInfo` shapes by running static shape inference
- [x] Generate unique IDs for all parsed entities for rapid DOM/Canvas binding
- [x] Deduplicate shared Initializers to reduce memory footprint
- [x] Compress repeated identical subgraphs visually (e.g., Transformer blocks)
- [x] Auto-detect model formats based on file signatures / magic bytes
- [x] Expose TypeScript/JS bindings for embedding parser in other Web Components apps
- [x] Support parsing models hosted on remote HTTP/HTTPS servers
- [x] Support parsing models directly from GitHub raw URLs
- [x] Extract and format tensor data into readable matrices (1D, 2D, 3D slices)
- [x] Provide endianness-safe parsing of raw binary tensor payloads
- [x] Support `bfloat16` and `float16` binary decoding in JS
- [x] Support decoding strings and string arrays
- [x] Identify and handle quantized parameter formats gracefully
- [x] Support custom op domain fallback (render as generic nodes)
- [x] Provide strict error boundary logs for unsupported attributes

### 2. Graph Layout & WebGL Rendering Engine (40+ items)

- [x] Implement robust Directed Acyclic Graph (DAG) layout algorithm (Dagre alternative)
- [x] Optimize layout engine for graphs with >100,000 nodes
- [x] Execute graph layout calculations in WebWorkers (Offscreen)
- [x] Support strictly vertical (Top-to-Bottom) flow layout
- [x] Support strictly horizontal (Left-to-Right) flow layout
- [x] Implement HTML5 Canvas / WebGL rendering backend for 60FPS pan/zoom
- [x] Fallback to SVG rendering for legacy environments or exact exports
- [x] Draw generic computation nodes
- [x] Draw Graph global inputs with distinct styling
- [x] Draw Graph global outputs with distinct styling
- [x] Draw Constant / Initializer nodes with distinct styling -[x] Draw nested subgraphs as expandable/collapsible macro-nodes
- [x] Draw tensor edges (connections) as smooth bezier curves
- [x] Draw tensor edges as orthogonal straight lines
- [x] Draw cyclic edges gracefully without infinite loops (e.g., RNN states)
- [x] Implement level-of-detail (LOD) rendering (hide text when zoomed out)
- [x] Render data types (`float32`, `int64`) as edge labels
- [x] Render tensor shapes (`[1, 3, 224, 224]`) as edge labels
- [x] Render dynamic symbolic dimensions (`[batch, sequence, 768]`) seamlessly -[x] Implement collision detection to prevent edge-label overlap -[x] Highlight producer-to-consumer paths clearly -[x] Support grouping nodes by namespace (e.g., `layer1.conv.weight`) -[x] Collapse distinct namespaces into single blocks automatically
- [x] Highlight node status (e.g., Error, Selected, Hovered)
- [x] Implement smooth scrolling and trackpad pinch-to-zoom -[x] Implement minimap (bird's-eye view) for massive graphs
- [x] Render grid background (optional)
- [x] Support rendering multiple isolated graphs in the same workspace
- [x] Calculate bounding boxes accurately for custom node names -[x] Optimize edge routing for dense "skip-connection" architectures (ResNet) -[x] Render quantized/dequantized chains compactly
- [x] Ensure sub-millisecond node hit-testing (clicking/hovering)
- [x] Support WebGL instance rendering for identical nodes (performance)
- [x] Prevent UI freezing when expanding a massive nested subgraph
- [x] Support hardware-accelerated CSS transforms for pan/zoom -[x] Render custom operator shapes/icons if metadata provides them
- [x] Support auto-centering on loaded models
- [x] Support auto-centering on specific searched nodes -[x] Render node execution providers (if profiled) dynamically
- [x] Render specific edge colors based on tensor dtype (e.g., Float=Blue, Int=Green)

### 3. Node Interaction & Property Sidebar (30+ items)

- [x] Select single node -> Open Sidebar Properties
- [x] Sidebar: Display Node Op Type and Domain
- [x] Sidebar: Display Node exact Name
- [x] Sidebar: Display Node Documentation/DocString
- [x] Sidebar: Render all Attributes cleanly
- [x] Sidebar: Render `float` attributes
- [x] Sidebar: Render `int` attributes
- [x] Sidebar: Render `string` attributes
- [x] Sidebar: Render `tensor` attributes (with "Click to View" matrix expansions)
- [x] Sidebar: Render `graph` attributes (with "Click to Open Subgraph" links)
- [x] Sidebar: List all explicit Inputs
- [x] Sidebar: List all explicit Outputs
- [x] Sidebar: List Input connections (links to producer nodes)
- [x] Sidebar: List Output connections (links to consumer nodes)
- [x] Sidebar: Render missing / optional inputs gracefully
- [x] Sidebar: Display raw tensor shapes and types for each connection -[x] Select edge -> Highlight entire tensor pathway
- [x] Select Graph Input -> Display global parameter details
- [x] Select Graph Output -> Display global projection details -[x] Provide deep-linking (URL anchors) to specific Node IDs -[x] Support multi-node selection -[x] Sidebar: Show common properties for multi-node selection -[x] Provide "View Raw Protobuf" for developers inspecting individual nodes -[x] Extract and format standard ONNX op documentation instantly
- [x] Support viewing `Tensor` data as raw numbers (matrix format)
- [x] Support viewing `Tensor` data as a flattened array -[x] Support viewing Image `Tensor` data as an actual RGB/Grayscale image -[x] Provide pagination for massive tensor arrays (>1000 elements)
- [x] Extract min, max, mean, and variance dynamically for selected Initializers -[x] Plot histogram of weight distributions inside the Sidebar

### 4. Search, Filtering, and Navigation (25+ items)

- [x] Implement fast fuzzy-search by Node Name
- [x] Implement search by Node Operator Type (e.g., find all `Conv`)
- [x] Implement search by Tensor Name
- [x] Implement search by Attribute Name or Value
- [x] Highlight all search results simultaneously on the Canvas
- [x] Step through search results (Next / Previous) smoothly panning the camera -[x] Filter out (hide) unselected nodes temporarily -[x] Implement "Find Source" (traverse upstream to graph inputs) -[x] Implement "Find Sinks" (traverse downstream to graph outputs) -[x] Implement "Select Subgraph" (isolate a region based on boundary nodes) -[x] Support Regex in the search bar -[x] Highlight shortest path between Node A and Node B -[x] Highlight all paths between Node A and Node B -[x] Filter out specific Op Types visually (e.g., "Hide all Identity nodes") -[x] Filter out Constant/Initializer nodes to simplify topology viewing -[x] Expand all nested subgraphs command -[x] Collapse all nested subgraphs command -[x] Quick-jump to specific Opset Domain configurations -[x] Isolate matching nodes into a separate floating view -[x] Save/Load custom view states (camera position, selected nodes) -[x] Display number of hidden/filtered nodes dynamically -[x] Display real-time node count, edge count, and parameter count -[x] Jump to Node by its internal index -[x] Search by execution provider/device (if metadata exists) -[x] Filter by tensor data type (e.g., "Show all float64 tensors")

### 5. Export, Sharing, and Tooling (20+ items)

-[x] Export visualization to SVG -[x] Export visualization to high-res PNG -[x] Export visualization to high-res JPEG -[x] Export isolated subgraph to SVG/PNG -[x] Save modified graph back to `.onnx` (if interactive mode enabled) -[x] Save graph structural metadata to JSON -[x] Export exact node properties to Markdown/Text -[x] Generate reproducible script (Python/onnx9000) that constructs the viewed graph -[x] Export tensor weight data to `.npy` (NumPy) natively -[x] Export tensor weight data to `.csv` -[x] Share current view (URL generation with embedded state) -[x] Integrate instantly with Jupyter Notebooks (IFrame/Widget) -[x] Integrate seamlessly as a VSCode Extension (Webview) -[x] Serve visualization locally via CLI (`onnx9000 serve model.onnx`) -[x] Support "Drag-and-Drop" of files directly onto the canvas -[x] Provide comprehensive hotkeys (Zoom, Pan, Search, Expand) -[x] Support macOS native touch-bar / trackpad gestures -[x] Implement offline-first Progressive Web App (PWA) capabilities -[x] Pre-cache ONNX documentation for offline viewing -[x] Allow loading user-defined custom op documentation JSONs

### 6. Visual Styling, Theming & Accessibility (20+ items)

-[x] Implement default Light Theme -[x] Implement default Dark Theme -[x] Support OS-level `prefers-color-scheme` auto-switching -[x] High contrast mode for accessibility -[x] Color-blind safe color palettes for Node domains -[x] Customizable node coloring rules (e.g., Color all `Relu` nodes Red) -[x] Color nodes based on namespace hierarchy -[x] Color nodes based on execution time (Heatmap mode) -[x] Color tensors based on memory size (Heatmap mode) -[x] Color tensors based on quantization state (INT8 vs FP32) -[x] Toggle edge thickness based on tensor volume (dimensions) -[x] Clean sans-serif typography (Inter / Roboto) -[x] Crisp edge-rendering on High-DPI (Retina) displays -[x] CSS-customizable UI elements -[x] Localized UI (English, Chinese, Japanese, etc.) -[x] Provide tooltips on hover for compact mode -[x] Make all SVG paths keyboard-navigable -[x] Screen reader support for node properties -[x] Ensure WCAG AA contrast compliance -[x] Support custom user-provided CSS stylesheets

### 7. Interactive Editing (GraphSurgeon Integration) (40+ items)

-[x] Switch from "View Mode" to "Edit Mode" -[x] Click and delete Node directly from the Canvas -[x] Click and delete Edge directly from the Canvas -[x] Drag to connect an Output Tensor to a new Input Node -[x] Drag an Edge to rewire it to a different Node -[x] Insert new Node from an Operator Palette (Searchable list of ONNX ops) -[x] Auto-complete missing inputs when injecting a new Node -[x] Edit Node `name` directly in Sidebar -[x] Edit Node `op_type` directly in Sidebar -[x] Edit/Add/Remove Node Attributes (Float, Int, String) in Sidebar -[x] Edit global Graph `name` and metadata -[x] Promote internal Tensor to Graph Output (Right-click -> Export) -[x] Demote Graph Output to internal Tensor -[x] Convert `Constant` to Graph Input (`Variable`) visually -[x] Convert Graph Input to `Constant` by injecting raw matrix data -[x] Trigger purely visual Constant Folding (collapse math subgraphs) -[x] Trigger purely visual Dead Code Elimination (prune graph) -[x] Trigger purely visual Shape Inference (update all edges instantly) -[x] Highlight structurally invalid topologies (e.g., cyclic loops in non-RNNs) -[x] Highlight type mismatches (e.g., feeding Float into Int node) -[x] Highlight shape mismatches (e.g., MatMul dim conflicts) -[x] Provide Undo / Redo stack for all surgical operations -[x] Isolate Node: Delete everything except the selected Node's path -[x] Duplicate Node -[x] Copy Subgraph -[x] Paste Subgraph -[x] Auto-layout after every surgical change (smooth animation) -[x] "Extract Subgraph" feature to save a highlighted section as a new `.onnx` -[x] "Merge Graph" feature to append another `.onnx` file visually -[x] Modify `opset` versions directly from UI -[x] Downgrade specific ops natively via UI command -[x] Inject custom TensorRT / execution provider plugins natively via UI -[x] Right-click -> "Wrap in If Node" -[x] Right-click -> "Wrap in Loop Node" -[x] Drag external `.bin` weights directly onto an Input to convert to Constant -[x] Support editing raw Tensor Data via interactive Matrix Grid editor -[x] Support generating random normal/uniform data for Inputs -[x] Diff Mode: Visually compare two `.onnx` files, highlighting added/removed/changed nodes -[x] Diff Mode: Highlight exact weight differences (deltas) in Constants -[x] "Bake" all current changes to a perfectly valid `ai.onnx` Graph Proto

### 8. Live Profiling & Analysis Overlays (30+ items)

-[x] Overlay Mode: Show Tensor static memory sizes (MB/KB) directly on edges -[x] Overlay Mode: Show Node computational cost (MACs/FLOPs) directly on nodes -[x] Overlay Mode: Show Node parameter count -[x] Summarize total model FLOPs dynamically in the toolbar -[x] Summarize total model Memory footprint dynamically -[x] Profile Mode: Connect to local Python JIT to execute graph visually -[x] Profile Mode: Overlay actual execution time (ms) per Node via heatmap -[x] Profile Mode: Overlay actual peak VRAM per Node -[x] Profile Mode: Highlight bottleneck paths dynamically -[x] Live inference: Inject data, click "Run", view outputs in Sidebar -[x] Inject Image data visually via File Upload -[x] Inject Text/String data visually -[x] Visualize runtime tensor statistics (Min, Max, Mean, Sparsity) per execution -[x] Support stepping through execution (Node-by-Node visual debugger) -[x] Pause execution on `NaN` or `Inf` generation dynamically -[x] Graph metric: Depth of the deepest path -[x] Graph metric: Count of specific operators (e.g., "15 Convs, 4 MatMuls") -[x] Support custom cost-model plugins for hardware specific latency estimates -[x] Highlight memory fragmentation issues (if memory arena is simulated) -[x] Visualize WebGPU shader dispatch boundaries dynamically -[x] Visualize WASM SIMD boundaries dynamically -[x] Display symbolic dimension resolution traces -[x] Overlay Node fusion opportunities (e.g., "Can be fused with Conv") -[x] Generate comprehensive HTML report summarizing all overlays -[x] Support timeline trace import (Chrome Tracing JSON) mapping to graph nodes -[x] Map TensorRT Engine profiles back to original ONNX visual nodes -[x] Map CoreML Engine profiles back to original ONNX visual nodes -[x] Analyze attention mask sparsity dynamically -[x] Identify redundant reshape/transpose operations visually -[x] Analyze dynamic batch size (`N`) propagation safety globally

### 9. Extended Multi-Format Parser Parity (30+ items)

-[x] Parse Darknet (`.cfg`, `.weights`) topologies via pure Python translation -[x] Parse Caffe (`.prototxt`, `.caffemodel`) topologies via pure Python translation -[x] Parse Caffe2 (`predict_net.pb`) topologies via pure Python translation -[x] Parse MXNet (`.json`, `.params`) topologies via pure Python translation -[x] Parse NCNN (`.param`, `.bin`) topologies via pure Python translation -[x] Parse MNN (`.mnn`) topologies via pure Python translation -[x] Parse OpenVINO IR (`.xml`, `.bin`) topologies via pure Python translation -[x] Parse RKNN (`.rknn`) topologies via pure Python translation -[x] Parse TensorFlow.js (`model.json`, `.bin`) topologies natively in JS -[x] Parse TensorFlow Lite FlatBuffer directly into the unified DAG -[x] Parse UFF (`.uff`) natively -[x] Parse TensorRT (`.plan`, `.trt`) engine definitions natively -[x] Parse ML.NET (`.zip`) topologies natively -[x] Parse MindSpore Lite (`.ms`) topologies natively -[x] Parse ONNX external data formats accurately (e.g. `raw_data`, `data_location`) -[x] Support massive Safetensors (`.safetensors`) weights mapping to visual nodes -[x] Validate NCNN int8 precision configurations visually -[x] Validate MNN fp16 precision configurations visually -[x] Validate OpenVINO precision settings visually -[x] Validate CoreML `mlprogram` versus legacy `neuralnetwork` architectures visually -[x] Support legacy `mlmodelc` (compiled CoreML) visualization -[x] Support generic `tar.gz` and `zip` model package extraction locally -[x] Extract and format JSON embedded metadata gracefully -[x] Translate unsupported framework ops into generic "CustomOp" nodes -[x] Expose native `FlatBuffers` JS bindings for ultra-fast TFLite parsing -[x] Expose native `Protobuf` JS bindings for ultra-fast ONNX parsing -[x] Implement zero-copy buffer views across the JS/WASM boundary -[x] Prevent 32-bit index overflow in massive model parameter files -[x] Support displaying raw bytes for unrecognized binary attributes

### 10. Deep UX / Interactive Tooling (30+ items)

-[x] Support drag-selecting multiple nodes (marquee tool) -[x] Right-click context menu: "Collapse Subgraph" -[x] Right-click context menu: "Expand Subgraph" -[x] Right-click context menu: "Group Selected Nodes" -[x] Right-click context menu: "Ungroup Nodes" -[x] Right-click context menu: "Copy Node JSON" -[x] Right-click context menu: "Copy Node ID" -[x] Double-click edge: Auto-zoom to producer and consumer -[x] Double-click canvas: Reset Zoom and Pan -[x] Hover edge: Display interactive tooltip with exact `[shape]` and `dtype` -[x] Hover node: Display interactive tooltip with Op description -[x] Command Palette (`Ctrl+K`): Fast action triggering (Search, Export, Layout) -[x] Minimap: Click to teleport to specific graph region -[x] Minimap: Drag viewing rectangle to pan the main canvas -[x] Minimap: Highlight search results as tiny dots -[x] Breadcrumb navigation: `Graph > If_1 > Subgraph > Conv_2` -[x] Property Sidebar: Searchable attributes list -[x] Property Sidebar: Copy individual attribute values to clipboard -[x] "Zen Mode": Hide all toolbars, minimaps, and sidebars (Canvas only) -[x] "Code Mode": Split screen between Visual DAG and raw `.onnx` textual representation -[x] Support multiple open tabs (View multiple models simultaneously) -[x] Compare Tabs: Side-by-side synchronized panning and zooming -[x] Custom URL parameters (e.g. `?url=model.onnx&node=Conv_1`) -[x] History panel: View all recent searches and selections -[x] Loading Progress Bar: Smoothly track `XMLHttpRequest` / Fetch API bytes -[x] Display exact rendering FPS counter (Performance mode) -[x] Display Javascript Heap Size dynamically (Performance mode) -[x] Auto-suspend rendering loop when graph is idle (Save battery) -[x] Throttled rendering for massive graphs (Maintain UI responsiveness) -[x] Configurable edge routing algorithms (Orthogonal vs Spline vs Direct)
