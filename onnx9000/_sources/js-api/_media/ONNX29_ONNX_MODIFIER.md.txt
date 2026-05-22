# ONNX29: onnx-modifier (Web-Native Graph Editor & Visualizer)

## Original Project Description

`onnx-modifier` is a lightweight, community-built Python tool and Flask-based web UI that allows developers to visually edit ONNX models. Users can load a model, visualize its graph, delete nodes, rename inputs/outputs, change batch sizes, and modify node attributes directly without needing to write complex Python scripts using the low-level `onnx.helper` API. It is incredibly useful for debugging, pruning broken nodes before export, or fixing shape mismatches in models downloaded from the internet.

## How `onnx9000` Deviates (The WASM-First Monolith Approach)

`onnx9000.modifier` completely drops the Python backend (Flask/Protobuf) and operates as a **100% serverless, client-side web application**.

- **Zero-Install Editing:** A user visits a static HTML page, drops an `.onnx` file, edits it using a rich interactive Canvas/WebGL UI, and clicks "Download". Nothing is uploaded to a server.
- **Integrated Graph Validation:** Leveraging the `onnx9000` AST engine, any structural changes (like deleting a node) instantly trigger a client-side shape and type inference pass, visually highlighting broken edges in red _before_ the user exports the model.
- **WASM-Backed Subgraph Execution:** A user can select a specific node or subgraph in the UI and click "Run Here", prompting `onnx9000` to execute just that portion of the graph instantly using WebGPU or WASM to verify the numerical impact of their edit.

---

## Exhaustive Implementation Checklist

### Phase 1: Core AST & In-Memory Graph Mutator

- [x] 1. Implement `GraphMutator` class wrapping the `onnx9000` AST.
- [x] 2. Support `addNode(opType, inputs, outputs, attributes, name)`.
- [x] 3. Support `removeNode(nodeName)`.
- [x] 4. Support `removeNode(nodeIndex)`.
- [x] 5. Implement automatic edge healing when deleting a pass-through node (connecting its input to its output).
- [x] 6. Support `renameNode(oldName, newName)`.
- [x] 7. Support `replaceNode(oldNodeName, newNodeDef)`.
- [x] 8. Support `changeNodeOpType(nodeName, newOpType)`.
- [x] 9. Support `renameInput(oldName, newName)`.
- [x] 10. Support `renameOutput(oldName, newName)`.
- [x] 11. Propagate input/output name changes globally across all downstream/upstream nodes.
- [x] 12. Support `addInput(name, type, shape)`.
- [x] 13. Support `removeInput(name)`.
- [x] 14. Support `addOutput(name, type, shape)`.
- [x] 15. Support `removeOutput(name)`.
- [x] 16. Support `addInitializer(name, type, shape, dataBuffer)`.
- [x] 17. Support `removeInitializer(name)`.
- [x] 18. Support `updateInitializer(name, newDataBuffer)`.
- [x] 19. Support converting an Input into an Initializer (baking a constant).
- [x] 20. Support converting an Initializer into an Input (making a constant dynamic).
- [x] 21. Support `setNodeAttribute(nodeName, attrName, attrValue, attrType)`.
- [x] 22. Support `removeNodeAttribute(nodeName, attrName)`.
- [x] 23. Implement a transactional rollback system (Undo/Redo stack for graph edits).
- [x] 24. Implement topological re-sorting after graph mutations.
- [x] 25. Support updating model metadata (`producer_name`, `version`, `doc_string`).

### Phase 2: Static Analysis & Validation Engine

- [x] 26. Implement a fast, synchronous `verify()` method tracking graph validity.
- [x] 27. Detect dangling nodes (nodes whose outputs are never consumed and aren't graph outputs).
- [x] 28. Detect unresolved inputs (nodes expecting an input that doesn't exist).
- [x] 29. Detect cyclic dependencies introduced by bad edits.
- [x] 30. Detect type mismatches (e.g., feeding an INT64 into a Conv2D node expecting FLOAT32).
- [x] 31. Implement local shape inference: given modified inputs, recalculate a node's output shape.
- [x] 32. Cascade shape inference changes throughout the entire downstream graph.
- [x] 33. Highlight dimension mismatches (e.g., MatMul expecting `[M, K] x [K, N]`, but got `[M, X]`).
- [x] 34. Support explicit shape overriding (forcing an intermediate tensor to be a specific shape to bypass strict inference bugs).
- [x] 35. Implement dead code elimination specifically triggered by user UI interactions (e.g., "Clean Graph" button).

### Phase 3: The Visualization Engine (Canvas/WebGL)

- [x] 36. Build a custom WebGL/Canvas 2D renderer for drawing the directed acyclic graph (DAG).
- [x] 37. Implement the Sugiyama layout algorithm (or integrate Dagre.js) to auto-arrange nodes into visual layers.
- [x] 38. Support vertical rendering mode (Top-to-Bottom).
- [x] 39. Support horizontal rendering mode (Left-to-Right).
- [x] 40. Implement infinite pan and zoom capabilities (mouse wheel, trackpad, pinch-to-zoom).
- [x] 41. Implement a minimap/radar view for navigating massive graphs (10k+ nodes).
- [x] 42. Render Initializers distinctly from dynamic Inputs (e.g., different shapes or colors).
- [x] 43. Render Constants distinctly.
- [x] 44. Render standard Operations (e.g., rounded rectangles).
- [x] 45. Render Graph Outputs distinctly.
- [x] 46. Display the `op_type` prominently on every node.
- [x] 47. Display the tensor shape and type directly on the connecting edges.
- [x] 48. Implement edge routing (orthogonal routing or curved splines) to minimize line crossing confusion.
- [x] 49. Support edge highlighting (hovering an edge highlights the producer and consumer nodes).
- [x] 50. Implement visual grouping (drawing bounding boxes around Subgraphs or NameScopes).

### Phase 4: UI/UX Components & Interactions

- [x] 51. Build the main layout: Graph View (center), Properties Panel (right), Structure Tree (left).
- [x] 52. Implement node selection logic (single click).
- [x] 53. Implement multi-node selection (Shift+Click or Drag Box).
- [x] 54. Display selected node properties in the Right Panel (Name, OpType, Attributes, Inputs, Outputs).
- [x] 55. Display selected edge properties (Name, Type, Shape, Producer, Consumers).
- [x] 56. Create inline editable text fields for node names in the Properties Panel.
- [x] 57. Create dropdown menus for editing enum-based attributes (e.g., `auto_pad`).
- [x] 58. Create array editors for editing list attributes (e.g., `pads: [1, 1, 1, 1]`).
- [x] 59. Implement a "Delete" button (and mapping to the `Del` / `Backspace` keyboard key).
- [x] 60. Implement a context menu (Right Click on Node -> "Delete", "Disconnect", "Duplicate").
- [x] 61. Implement edge dragging: clicking an output port on Node A and dragging a line to an input port on Node B.
- [x] 62. Implement edge deletion: clicking an edge and pressing Delete.
- [x] 63. Implement an "Add Node" modal with a searchable list of all valid ONNX operators.
- [x] 64. Automatically populate the attributes form based on the selected `op_type`'s ONNX specification schema.
- [x] 65. Support collapsing/expanding complex nodes (e.g., hiding the body of an `If` or `Loop` node to save screen space).

### Phase 5: Batch Modification & Advanced Editing

- [x] 66. Implement "Change Batch Size" utility (dynamically updates Dimension 0 across all Inputs and intermediate shapes).
- [x] 67. Implement "Make Dynamic" utility (changes static batch size `1` to `-1` / `?` or string variable `batch_size`).
- [x] 68. Implement "Strip Initializers" utility (removes all weights, saving only the structural `.onnx` for sharing topology).
- [x] 69. Implement "Extract Subgraph" utility (select N nodes -> right click -> "Save as new Model").
- [x] 70. Implement "Insert Identity" utility (drops an Identity node on an edge for debugging breakpoints).
- [x] 71. Implement "Change Opset Version" utility (attempts to auto-upgrade or auto-downgrade the model schema).
- [x] 72. Provide regex-based batch renaming (e.g., renaming all nodes starting with `old_prefix/` to `new_prefix/`).
- [x] 73. Support injecting Cast nodes automatically if the user connects incompatible types via the UI.
- [x] 74. Implement a "Find Node by Name" search bar.
- [x] 75. Implement a "Find Node by Type" search filter (e.g., highlighting all `Conv` layers).

### Phase 6: Initializer / Weight Editing

- [x] 76. Implement an Initializer Inspector in the UI (showing Min, Max, Mean, Variance of the weight tensor).
- [x] 77. Render small 2D weights (e.g., 3x3 Conv kernels) as visual pixel grids (heatmaps) in the properties panel.
- [x] 78. Support explicitly editing scalar initializer values via a text input field.
- [x] 79. Support zeroing out an initializer (setting all values to 0).
- [x] 80. Support injecting random noise into an initializer (for fuzzing or privacy obfuscation).
- [x] 81. Support downloading a specific initializer as a `.bin` or `.npy` file.
- [x] 82. Support uploading and replacing an initializer's data from a local `.bin` file.
- [x] 83. Support precision casting specifically for initializers via the UI (e.g., clicking "Convert to FP16").
- [x] 84. Track exact byte sizes of individual initializers to help users identify the heaviest layers in their model.
- [x] 85. Provide a "Prune" button that applies a magnitude threshold to an initializer (setting values < threshold to 0).

### Phase 7: Interactive Graph Execution & Debugging (The "Run Here" Feature)

- [x] 86. Integrate the core `onnx9000` execution runtime into the modifier UI.
- [x] 87. Support "Set as Temporary Output": User clicks an intermediate edge, and the graph compiles to yield that specific tensor.
- [x] 88. Implement an Input Data Generator (creating random dummy data matching the input shapes).
- [x] 89. Allow users to manually input values for small inputs (e.g., JSON arrays) or upload images for Image inputs.
- [x] 90. Execute the graph natively in the browser via WebAssembly/WebGPU based on the dummy/user data.
- [x] 91. Display the execution output tensor visually (e.g., as an image if shape is HWC, or a JSON array if 1D).
- [x] 92. Support "Run Subgraph": User selects Node A, B, and C. The UI extracts them, generates dummy inputs for Node A, runs the subgraph, and shows Node C's output.
- [x] 93. Profile execution: display the exact execution time (in ms) taken by the currently selected node/subgraph.
- [x] 94. Implement a step-by-step debugger: "Step Next", executing the graph one node at a time and visualizing the tensor flowing through the edges.
- [x] 95. Implement breakpoint pausing in the visual debugger.

### Phase 8: Data Privacy & Security

- [x] 96. Ensure the entire Web Components application is served as a static bundle (no server backend).
- [x] 97. Provide a standalone HTML file option (single-file export containing UI, logic, and wasm encoded in base64) for offline editing.
- [x] 98. Ensure massive models (>2GB) utilize standard `File` slicing APIs and do not crash the browser's maximum heap limit.
- [x] 99. Restrict execution features if the model triggers potential infinite loops (timeout circuit breakers).
- [x] 100. Disallow prototype pollution or malicious script injection if the ONNX metadata contains `<script>` tags.

### Phase 9: Model Export & Serialization

- [x] 101. Implement `export()` method combining the AST and mutated Initializers back into a standard ONNX Protobuf string.
- [x] 102. Validate the final protobuf against the ONNX spec schema explicitly before allowing the download.
- [x] 103. Generate a standard browser download (`blob` URL) for the resulting `.onnx` file.
- [x] 104. Support exporting large models using the `external_data` ONNX format (zipping the `.onnx` and `.onnx_data` together in the browser).
- [x] 105. Generate an "Edit Log" sidecar file (a JSON array recording every mutation made during the session) so edits can be replayed scripturally later.
- [x] 106. Provide a "Copy to Clipboard" button that generates the exact Python `onnx.helper` code required to recreate the user's manual edits.
- [x] 107. Generate a graph summary text file (Layer counts, Parameter counts, Total MACs).
- [x] 108. Enable downloading the graph visualization as a high-resolution SVG or PNG file.
- [x] 109. Support converting the visual graph layout into standard Graphviz `.dot` format.
- [x] 110. Guarantee binary determinism: opening an ONNX file and clicking "Save" without edits produces a byte-identical file.

### Phase 10: Specific ONNX Operator Custom Editors

- [x] 111. Create a specialized UI form for editing `Conv` (Fields: Strides, Pads, Dilations, Groups).
- [x] 112. Create a specialized UI form for editing `Gemm` (Toggles for transA, transB, alpha, beta).
- [x] 113. Create a specialized UI form for editing `Split` (Array editor for the `split` attribute).
- [x] 114. Create a specialized UI form for editing `Resize` (Dropdowns for coordinate_transformation_mode, mode, nearest_mode).
- [x] 115. Create a specialized UI form for editing `Squeeze` / `Unsqueeze` (Axes array editor).
- [x] 116. Create a specialized UI form for editing `Cast` (Dropdown mapping integer enum values to human-readable strings like 'FLOAT', 'INT8').
- [x] 117. Create a specialized UI form for editing `Constant` (showing the raw tensor value inline).
- [x] 118. Create a specialized UI interface for viewing inside `If` subgraphs (Graph within a Graph).
- [x] 119. Create a specialized UI interface for viewing inside `Loop` subgraphs.
- [x] 120. Provide tooltips hovering over specific attribute names that fetch documentation directly from the official ONNX spec definitions.

### Phase 11: CLI Interface (onnx9000.modifier CLI)

- [x] 121. Expose `onnx9000 edit <model.onnx>` command to start a local development server hosting the UI and serving the target file automatically.
- [x] 122. Expose `onnx9000 prune <model.onnx> --nodes "node_1,node_2"` for headless CLI modification.
- [x] 123. Expose `onnx9000 rename-input <model.onnx> --old "X" --new "Y"` via CLI.
- [x] 124. Expose `onnx9000 change-batch <model.onnx> --size 8` via CLI.
- [x] 125. Support headless JSON mutation scripts: `onnx9000 mutate <model.onnx> --script edits.json`.

### Phase 12: Interoperability with Pyodide

- [x] 126. Expose the `GraphMutator` API to a Pyodide web-worker.
- [x] 127. Allow users to write arbitrary Python scripts in a Monaco Editor panel within the UI.
- [x] 128. Run the Python script securely against the active graph (e.g., `for node in graph.nodes: if node.op_type == 'Relu': graph.remove(node)`).
- [x] 129. Map Pyodide state changes back to the UI visualizer instantly.
- [x] 130. Support standard `import onnx` within the Pyodide environment to allow users to use familiar legacy scripts.

### Phase 13: Accessibility and Theming

- [x] 131. Implement Dark Mode UI.
- [x] 132. Implement Light Mode UI.
- [x] 133. Provide a colorblind-friendly palette for the node typings (e.g., differentiating Conv vs Matmul clearly without relying solely on red/green).
- [x] 134. Support complete keyboard navigation (tabbing through nodes, pressing enter to open properties).
- [x] 135. Support screen-reader announcements when nodes are selected or deleted.

### Phase 14: Automated Graph Fixers (One-Click Macros)

- [x] 136. Implement "Fix Mixed Precision": Automatically identifies Cast anomalies and normalizes the graph to FP32 or FP16.
- [x] 137. Implement "Remove Training Nodes": Automatically strips `Dropout`, `Gradient`, and `YieldOp` nodes from the graph.
- [x] 138. Implement "Fold Constants": Runs the `onnx9000` constant folding optimizer and updates the visual layout.
- [x] 139. Implement "Extract Weights": Converts all large Constants/Inputs into external Initializers automatically.
- [x] 140. Implement "Sanitize Names": Replaces complex/illegal node and edge names with sequential alphanumeric IDs.

### Phase 15: Quality Assurance & Parity Testing

- [x] 141. Write unit tests for node deletion preserving topological order.
- [x] 142. Write unit tests for batch size mutation propagating correctly through `Reshape` constants.
- [x] 143. Test the UI rendering limit (ensuring WebGL handles a 50,000-node graph at >30 FPS).
- [x] 144. Validate the exported `.onnx` byte structure against the native Python `onnx-modifier` outputs.
- [x] 145. Automate browser testing using Playwright to simulate dropping nodes and verifying edge counts.

### Phase 16: File Handling Edge Cases

- [x] 146. Process ONNX models utilizing opset versions < 7 gracefully.
- [x] 147. Parse and warn about unrecognized custom domain operations cleanly (rendering them as gray unknown nodes).
- [x] 148. Handle corrupted `.onnx` files by providing a partial load (showing whatever AST was successfully parsed before the crash).
- [x] 149. Optimize memory: Release the original ArrayBuffer payload immediately after parsing the AST to free up JS Heap.
- [x] 150. Use SharedArrayBuffer to offload the Protobuf parsing sequence to a background Web Worker.

### Phase 17: Integration with Other `onnx9000` Tools

- [x] 151. Add an "Optimize" button that bridges directly to `onnx9000.optimum` graph rewriting routines (O1, O2, O3).
- [x] 152. Add a "Quantize" button that bridges directly to `onnx9000.optimum` dynamic quantization, updating the visual graph with the new `DynamicQuantizeLinear` nodes.
- [x] 153. Allow direct hand-off from `onnx9000.keras` (User converts a `.h5` file and is immediately presented with the modifier UI).
- [x] 154. Provide "Export to CoreML" integration right from the modifier UI.
- [x] 155. Provide "Compile to IREE/WebNN" integration from the modifier UI.

### Phase 18: Collaboration & Cloud Extensions (Optional Layer)

- [x] 156. Support generating a shareable Base64 URI fragment if the model is < 5MB.
- [x] 157. Create a "Copy Link" feature that stores the graph temporarily in IndexedDB and allows opening it across different browser tabs.
- [x] 158. Implement CRDT (Conflict-free Replicated Data Type) structures for multi-user real-time graph editing (like Figma for ONNX).
- [x] 159. Support WebSocket connections for synchronizing node drags/moves between users.
- [x] 160. Create a unified `onnx9000.modifier` Web Component that external developers can embed in their own MLOps dashboards.

### Phase 19: Tooling for Generative AI specifically

- [x] 161. Implement a specialized view for LLM repeated layers (e.g., visually stacking `TransformerBlock_0` through `TransformerBlock_31` to save screen space).
- [x] 162. Provide an automated macro to inject explicit KV-cache `past_key_values` inputs into a static LLM graph.
- [x] 163. Provide an automated macro to convert static Rotary Positional Embeddings (RoPE) limits into dynamic limits.
- [x] 164. Support visualizing 3D/4D tensors specific to vision models (e.g., showing a spatial heatmap of initializers if the shape matches standard convolution patterns).
- [x] 165. Parse Hugging Face `config.json` alongside the ONNX file to provide human-readable names for intermediate states.

### Phase 20: Polish & Documentation

- [x] 166. Publish an interactive web demo at `modifier.onnx9000.dev`.
- [x] 167. Create tooltip guides explaining ONNX specific concepts (e.g., what an Initializer is vs an Input) for beginners.
- [x] 168. Ensure strict conformance to WAI-ARIA accessibility standards for web apps.
- [x] 169. Provide comprehensive tutorials: "Fixing Shape Mismatches", "Pruning a Model for Web Deploy".
- [x] 170. Release v1.0 feature parity certification against the original Python `onnx-modifier`.
- [x] 171. Add visual indication of node execution time bottlenecks (e.g., coloring nodes red if they take > 10ms during "Run Here" profiling).
- [x] 172. Add visual indication of node memory footprint bottlenecks.
- [x] 173. Support `onnx_data` sidecar file modification (updating external weights without pulling them into the main proto).
- [x] 174. Validate WebGL context loss recovery.
- [x] 175. Enable importing ONNX models dynamically via HuggingFace Hub repository URLs in the UI.
- [x] 176. Implement specific validation for `If` graph structures to prevent UI freezing on recursion.
- [x] 177. Provide layout caching so moving a node retains its position on reload.
- [x] 178. Handle copy/pasting nodes within the graph (duplication).
- [x] 179. Handle copy/pasting subgraphs entirely.
- [x] 180. Validate edge routing algorithm efficiency on graphs > 10,000 edges.
- [x] 181. Add support for creating subgraph abstractions visually (selecting nodes and grouping them).
- [x] 182. Implement graph-level attribute viewing (for ONNX models that store metadata in attributes).
- [x] 183. Manage the display of extremely long string constants efficiently in the properties panel.
- [x] 184. Implement a strict "Read Only" mode for embedding the visualizer without editing capabilities.
- [x] 185. Support parsing and visual rendering of ONNX sparse tensor formats.
- [x] 186. Optimize the undo/redo stack memory footprint (saving deltas instead of full graph copies).
- [x] 187. Ensure correct parsing of ONNX TensorProto `raw_data` formats (Float16 arrays, BFloat16 arrays).
- [x] 188. Support importing JSON representation of the ONNX graph directly.
- [x] 189. Add a feature to "De-duplicate" constants visually in the graph representation.
- [x] 190. Provide a specific export configuration that forces Opsets to match WebNN compliance.
- [x] 191. Create custom warning badges on nodes that are known to be slow or unsupported in WebGPU.
- [x] 192. Integrate with browser File System Access API for seamless local save overwrites.
- [x] 193. Render text representations of mathematical expressions for basic math nodes (`Add`, `Mul`).
- [x] 194. Handle rendering of models with empty graphs gracefully.
- [x] 195. Verify that `value_info` metadata is correctly updated when modifying internal graph shapes.
- [x] 196. Implement an edge filtering mechanism (e.g., hiding control flow edges to reduce visual noise).
- [x] 197. Support custom node coloring based on regular expressions (e.g., `Color all 'LayerNorm' nodes blue`).
- [x] 198. Export the WebGL renderer as an independent generic DAG viewer component.
- [x] 199. Publish the web asset to a standard Docker container for enterprise internal usage.
- [x] 200. Execute load tests verifying the browser doesn't crash on standard 2GB LLM `.onnx` files.
- [x] 201. Add support for visual node snapping to a grid.
- [x] 202. Provide "Align Left/Right/Center" utilities for selected groups of nodes.
- [x] 203. Create a feature to automatically format node names based on depth (e.g., `layer_1/conv`, `layer_2/conv`).
- [x] 204. Validate `Cast` node conversions accurately represent bounds correctly.
- [x] 205. Implement a fallback Canvas 2D renderer if WebGL initialization fails in the browser.
- [x] 206. Configure explicit memory thresholds before warning the user that a save might fail.
- [x] 207. Support viewing the raw protobuf JSON alongside the visual editor.
- [x] 208. Add right-click -> "Extract to Python script" generating `onnx.helper.make_node` snippets.
- [x] 209. Enable "Replace with Constant" macro (runs a node and bakes its result as a constant).
- [x] 210. Implement edge hover preview of tensor shape sizes.
- [x] 211. Provide visual cues for `INT8` quantized topologies.
- [x] 212. Provide visual cues for `W4A16` packed weight topologies.
- [x] 213. Add specific UI warnings for nodes using `double` (float64) precision.
- [x] 214. Create an automated test checking `changeBatchSize` function across all common layer types.
- [x] 215. Validate `removeInput` does not orphan required parameters for strict ONNX nodes.
- [x] 216. Ensure `addOutput` automatically infers the correct shape from the requested edge.
- [x] 217. Test `updateInitializer` strictly enforces array buffer length matches type specifications.
- [x] 218. Add support for creating new ONNX subgraphs entirely from scratch (starting with blank canvas).
- [x] 219. Include pre-built snippets (e.g., dragging and dropping a standard ResBlock from a menu).
- [x] 220. Support exporting the node statistics directly to CSV/Excel.
- [x] 221. Implement deep linking (e.g., sharing a URL that automatically opens a model from a remote URL and centers on node `Conv_14`).
- [x] 222. Ensure local IndexedDB cache is properly cleared on page reload to avoid stale state.
- [x] 223. Support parsing and rendering `TrainingInfoProto` if present in the ONNX file.
- [x] 224. Allow editing of `dim_param` string variables for dynamic axes.
- [x] 225. Add a "Validate Opset" macro checking compatibility with opset 13-21.
- [x] 226. Ensure the UI gracefully handles rapid sequential clicks (debouncing expensive layout recalculations).
- [x] 227. Test the application across Chrome, Firefox, and Safari engines.
- [x] 228. Provide a detailed "Help" modal mapping keyboard shortcuts.
- [x] 229. Expose a global `window.onnxModifier` object for developer console hacking.
- [x] 230. Validate output generation with `external_data` enabled when handling models >2GB.
- [x] 231. Handle edge cases involving empty initializers (`raw_data` length 0).
- [x] 232. Support `Float8E4M3FN` rendering specifically for newer ONNX opsets.
- [x] 233. Optimize serialization sequence (minimizing GC pauses during download generation).
- [x] 234. Include custom logic to edit ONNX `Sequence` type attributes.
- [x] 235. Include custom logic to edit ONNX `Map` type attributes.
- [x] 236. Allow explicit defining of `SparseTensorProto` constants.
- [x] 237. Create UI component for tracking multiple sub-graphs within one overarching model.
- [x] 238. Verify that multi-level nested `If` subgraphs render correctly without logical entanglement.
- [x] 239. Enable "Save State" to store the current editing session locally without downloading.
- [x] 240. Allow comparison of two separate ONNX files loaded simultaneously side-by-side.
- [x] 241. Provide visual indicators mapping identical weights (detecting deduplication).
- [x] 242. Build visual tool tracking gradient paths (if training nodes exist).
- [x] 243. Set up memory profiling to alert users if their edits will cause a memory spike on inference.
- [x] 244. Handle the "Drop ONNX file here" UX explicitly via standard browser events.
- [x] 245. Validate file drop on all operating systems.
- [x] 246. Establish automated testing of the `onnx9000 edit` CLI command via local playwright instances.
- [x] 247. Track exact node deletion counts and log them in console for user reference.
- [x] 248. Provide an option to "Auto-Fix" missing initializers by injecting dummy Zero arrays.
- [x] 249. Display warning banners when the model uses deprecated Opset versions.
- [x] 250. Finalize rigorous integration tests proving the editor works completely offline.
- [x] 251. Handle `Optional` input edge rendering gracefully (using dashed lines).
- [x] 252. Add a `Graph Level Attributes` property panel.
- [x] 253. Render Graph Doc Strings using Markdown in the UI.
- [x] 254. Support editing the graph `ir_version`.
- [x] 255. Support visual editing of the `opset_import` list.
- [x] 256. Handle adding multiple opset domains gracefully.
- [x] 257. Verify correct parsing and UI representation of `FunctionProto` custom functions.
- [x] 258. Support editing `FunctionProto` bodies recursively.
- [x] 259. Manage complex naming collisions when duplicating nodes.
- [x] 260. Render large constant tensors via paginated tables in the properties panel to prevent UI freezing.
- [x] 261. Add "Export Node to JSON" feature.
- [x] 262. Support copying a node's attributes to another node of the same type.
- [x] 263. Establish a user-feedback collection mechanism inside the UI.
- [x] 264. Support translating standard PyTorch exported graphs into cleaner visual representations (collapsing noise).
- [x] 265. Allow pinning specific nodes to fixed coordinates on the canvas.
- [x] 266. Enable drawing of custom text annotations directly onto the canvas.
- [x] 267. Export annotations alongside the model in a metadata dictionary.
- [x] 268. Detect and display warning for model inputs lacking defined shape properties.
- [x] 269. Render a progress bar tracking the Sugiyama layout algorithm for large graphs.
- [x] 270. Add an abort button for the layout algorithm if it takes too long.
- [x] 271. Fallback to a fast grid layout if Sugiyama fails or times out.
- [x] 272. Implement custom logic for editing `ScatterND` specific tensor updates.
- [x] 273. Support visualization of ONNX Sequence inputs/outputs.
- [x] 274. Implement UI hooks mapping to `onnx9000` execution profiling metrics directly.
- [x] 275. Show memory bandwidth utilization estimates per node.
- [x] 276. Ensure proper cleanup of the WebGL context on component unmount.
- [x] 277. Render graphs using standard WebGL Instanced Drawing for massive performance gains.
- [x] 278. Add anti-aliasing configurations for the canvas viewer.
- [x] 279. Support saving layout preferences to `localStorage`.
- [x] 280. Add multi-language localization support for the Modifier UI.
- [x] 281. Build the production asset pipeline via Vite or Webpack strictly.
- [x] 282. Expose specific hooks for unit testing DOM interactions.
- [x] 283. Implement touch event handling for iPad/Mobile Safari rendering.
- [x] 284. Allow editing of tensor attributes visually via Hex editor.
- [x] 285. Confirm that models generated by `onnx-modifier` execute flawlessly in standard ONNX Runtime C++.
- [x] 286. Handle ONNX proto version conflicts gracefully.
- [x] 287. Implement a "Validate Graph" manual trigger button.
- [x] 288. Publish full NPM package `@onnx9000/modifier`.
- [x] 289. Add a strict mode that prevents generating non-standard ONNX architectures.
- [x] 290. Provide a detailed summary of changes prompt before exporting.
- [x] 291. Add visual node highlighting based on inference path tracking.
- [x] 292. Add custom layout padding configurations.
- [x] 293. Handle invalid JSON attribute inputs safely.
- [x] 294. Enable custom shape inference logic hooking.
- [x] 295. Set up continuous integration analyzing WebGL framerates.
- [x] 296. Verify correct parsing of the `metadata_props` mapping.
- [x] 297. Render `String` nodes with truncated inline text representations.
- [x] 298. Validate complete disconnection behavior without crashing shape inference.
- [x] 299. Create standard Github Issue forms explicitly for the Modifier tool.
- [x] 300. Maintain rigorous parity checks against new versions of `onnx-modifier`.
