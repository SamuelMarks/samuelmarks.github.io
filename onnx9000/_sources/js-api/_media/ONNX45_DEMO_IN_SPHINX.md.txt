# ONNX45: Demo in Sphinx (Vanilla JS, WASM Lazy-Load)

## Problem Description

We are building a highly interactive, rich web frontend fully integrated into our Sphinx documentation. The goal is to allow users to explore model conversions directly within the browser, demonstrating the power of our tools without requiring local installation.

**Key Requirements:**

- **Zero Framework Dependency:** Strictly Vanilla JS. No Angular, React, Vue, Svelte, or other heavy UI frameworks. Native DOM APIs and Web Components where appropriate.
- **Lazy Loaded WASM:** WASM binaries are heavy. The initial load must be lightweight. A central "Load WASM live demo" button overlay must be clicked to fetch, initialize, and run the WASM-based compilers/converters.
- **Strict Quality Gates:**
  - **100% Documentation Coverage:** Every function, class, and module must be documented (JSDoc).
  - **100% Unit Test Coverage:** Every logical branch must be covered.
  - **Comprehensive E2E Tests:** Playwright specs for all critical user flows.

**Architecture & Layout:**
The UI consists of a 3-pane layout (Left, Right, and Bottom):

1. **Left-Hand Side (LHS - Source):**
   - **Source Dropdown:** Select from Keras, onnxscript / spox, TensorFlow, Caffe, MXNet, PaddlePaddle, Scikit-Learn, LightGBM, XGBoost, CatBoost, SparkML.
   - **Directory Tree:** Displays example files and directory structure for the selected source.
   - **Editor:** Monaco Editor instance showing the active source file.

2. **Right-Hand Side (RHS - Target):**
   - **Target Dropdown:** Select from .onnx, MLIR, C++, Apple CoreML, Caffe, Keras, MXNet, TensorFlow, CNTK, PyTorch.
   - **Directory Tree:** Displays the generated artifacts/files.
   - **Viewer/Editor:** Monaco Editor instance showing the active compiled/converted target file.

3. **Bottom Pane (Output & Visualization):**
   - **Tabs System:** At least two tabs.
   - **Console Tab:** Captures and displays stdout/stderr logs from the WASM conversion process, mimicking a terminal.
   - **ONNX Visualization Tab:** Renders a visual graph representation of the generated ONNX model (using a lightweight vanilla JS renderer or an embedded netron-like view).

## Implementation Steps

### Phase 1: Project Setup & Vanilla JS Build Pipeline

- [x] Set up dedicated `apps/sphinx-demo-ui` directory for the frontend.
- [x] Initialize `package.json` without React/Vue/Angular dependencies.
- [x] Configure Vite or ESBuild for bundling Vanilla JS assets.
- [x] Configure TypeScript with strict typing (`tsconfig.json`).
- [x] Set up ESLint with rules forbidding framework imports.
- [x] Set up Prettier for consistent code formatting.
- [x] Configure Jest or Vitest for unit testing.
- [x] Configure Istanbul/c8 to enforce 100% line, branch, and function coverage.
- [x] Set up TypeDoc or similar for 100% JSDoc generation.
- [x] Create build scripts for `dev`, `build`, `test`, `coverage`, and `docs`.
- [x] Integrate the frontend build step into the main repository `Makefile` or `turbo.json`.
- [x] Set up a mock server for local development without Sphinx.
- [x] Define CSS architecture (CSS Variables, modular CSS or BEM methodology).
- [x] Create base reset and typography CSS files.
- [x] Establish a Vanilla JS event bus (PubSub) for cross-component communication.
- [x] Create base `Component` class for DOM lifecycle management (mount, unmount, render).
- [x] Set up a centralized State Store (vanilla JS proxy or observable pattern).
- [x] Write unit tests for the event bus with 100% coverage.
- [x] Write unit tests for the State Store with 100% coverage.
- [x] Document (JSDoc) all base architecture modules.

### Phase 2: Sphinx Extension & Integration

- [x] Create `docs/sphinx_ext_demo_ui.py` Sphinx extension.
- [x] Define a custom Sphinx directive `.. interactive-demo::`.
- [x] Configure the extension to inject the bundled CSS assets into the Sphinx page.
- [x] Configure the extension to inject the bundled JS assets (defer load).
- [x] Ensure the directive outputs a target `div` with correct IDs and data attributes.
- [x] Implement parameter passing from RST to the JS app via `data-` attributes.
- [x] Write Python unit tests for the Sphinx directive HTML generation.
- [x] Write Python unit tests for asset injection logic.
- [x] Create a dummy RST page to test the directive locally.
- [x] Configure Sphinx `conf.py` to run the Vite/ESbuild pipeline on `make html`.
- [x] Handle dark/light mode synchronization between Sphinx theme (e.g., Furo) and the Demo UI.
- [x] Test responsive behavior of the injected container within the Sphinx layout.
- [x] Ensure Monaco Editor web workers are properly resolved when hosted via Sphinx.
- [x] Ensure WASM assets are correctly copied to the Sphinx `_static` build directory.
- [x] Add Python docstrings to `sphinx_ext_demo_ui.py` achieving 100% coverage.

### Phase 3: Layout & Split Panes

- [x] Implement main grid layout CSS (CSS Grid).
- [x] Create `SplitPane` vanilla JS component.
- [x] Implement drag-to-resize logic for horizontal split (LHS | RHS).
- [x] Implement drag-to-resize logic for vertical split (Top | Bottom Pane).
- [x] Add boundary constraints to prevent panes from collapsing entirely.
- [x] Add double-click to reset pane sizes.
- [x] Persist pane sizes in `localStorage`.
- [x] Write CSS for pane dividers/handles (hover states, grab cursors).
- [x] Create `LHSContainer` component and mount it.
- [x] Create `RHSContainer` component and mount it.
- [x] Create `BottomContainer` component and mount it.
- [x] Implement responsive layout fallback for mobile (stack panes vertically).
- [x] Write unit tests for the drag-to-resize mathematical logic.
- [x] Write unit tests for `localStorage` persistence logic.
- [x] Write DOM tests verifying the SplitPane component mounts correctly.
- [x] Write DOM tests verifying resize events emit correctly.
- [x] Document all Layout components and utility functions.
- [x] Set up Playwright E2E spec: `layout-resizing.spec.ts`.
- [x] E2E: Verify dragging horizontal splitter changes LHS/RHS widths.
- [x] E2E: Verify dragging vertical splitter changes top/bottom heights.

### Phase 4: WASM Lazy Loading & Overlay

- [x] Create the `WasmOverlay` UI component (dimmed background, centered content).
- [x] Style the 'Load WASM live demo' primary action button.
- [x] Add a loading spinner and progress text elements (hidden by default).
- [x] Implement the `WasmManager` singleton to handle fetch and initialization.
- [x] Add logic to show the overlay on initial render if WASM is not loaded.
- [x] Bind the 'Load WASM' button click to the `WasmManager` load method.
- [x] Implement `fetch` with `ReadableStream` to track WASM download progress.
- [x] Update progress text/bar during WASM download.
- [x] Implement WASM module instantiation logic.
- [x] Handle WASM load errors (network failure, instantiation error) with user-friendly error messages.
- [x] Dispatch `WASM_LOADED` event on the event bus upon success.
- [x] Hide the overlay with a CSS transition when `WASM_LOADED` is fired.
- [x] Write unit tests mocking `fetch` to verify progress bar math.
- [x] Write unit tests verifying error state rendering.
- [x] Write unit tests for `WasmManager` state transitions.
- [x] Ensure 100% coverage on `WasmManager` and `WasmOverlay`.
- [x] Document WASM loading utilities and components.
- [x] Create Playwright E2E spec: `wasm-loading.spec.ts`.
- [x] E2E: Verify UI is blocked until WASM is loaded.
- [x] E2E: Verify clicking load fetches WASM and removes overlay.

### Phase 5: Shared UI Components (File Tree & Monaco)

- [x] Create reusable `Dropdown` vanilla JS component.
- [x] Implement keyboard navigation (arrows, enter, escape) for `Dropdown`.
- [x] Create reusable `FileTree` vanilla JS component.
- [x] Implement folder expand/collapse logic in `FileTree`.
- [x] Implement file selection logic (highlight active file) in `FileTree`.
- [x] Implement CSS for nested tree levels and file/folder icons.
- [x] Integrate Monaco Editor base setup (loader script or ESM bundle).
- [x] Create `Editor` component wrapping Monaco.
- [x] Handle Monaco theme switching (light/dark).
- [x] Handle window resize events to call Monaco's `layout()` method.
- [x] Implement a file caching layer to preserve editor state/scroll when switching files.
- [x] Write unit tests for `Dropdown` component and keyboard navigation.
- [x] Write unit tests for `FileTree` data structure mapping and selection logic.
- [x] Write unit tests for Editor wrapper initialization.
- [x] Achieve 100% unit test coverage for all shared UI components.
- [x] Add 100% JSDoc coverage to shared UI components.
- [x] Create Playwright E2E spec: `shared-components.spec.ts`.
- [x] E2E: Verify dropdown opens, selects item, and closes.
- [x] E2E: Verify file tree expands/collapses.
- [x] E2E: Verify Monaco editor renders text.

### Phase 6: LHS (Source) Implementation

- [x] Define abstract syntax tree / mock example data for Keras.
- [x] Implement source state logic for selecting Keras.
- [x] Create unit tests validating data loading for Keras.
- [x] Ensure E2E script verifies Keras populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for onnxscript/spox.
- [x] Implement source state logic for selecting onnxscript/spox.
- [x] Create unit tests validating data loading for onnxscript/spox.
- [x] Ensure E2E script verifies onnxscript/spox populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for TensorFlow.
- [x] Implement source state logic for selecting TensorFlow.
- [x] Create unit tests validating data loading for TensorFlow.
- [x] Ensure E2E script verifies TensorFlow populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for Caffe.
- [x] Implement source state logic for selecting Caffe.
- [x] Create unit tests validating data loading for Caffe.
- [x] Ensure E2E script verifies Caffe populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for MXNet.
- [x] Implement source state logic for selecting MXNet.
- [x] Create unit tests validating data loading for MXNet.
- [x] Ensure E2E script verifies MXNet populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for PaddlePaddle.
- [x] Implement source state logic for selecting PaddlePaddle.
- [x] Create unit tests validating data loading for PaddlePaddle.
- [x] Ensure E2E script verifies PaddlePaddle populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for Scikit-Learn.
- [x] Implement source state logic for selecting Scikit-Learn.
- [x] Create unit tests validating data loading for Scikit-Learn.
- [x] Ensure E2E script verifies Scikit-Learn populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for LightGBM.
- [x] Implement source state logic for selecting LightGBM.
- [x] Create unit tests validating data loading for LightGBM.
- [x] Ensure E2E script verifies LightGBM populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for XGBoost.
- [x] Implement source state logic for selecting XGBoost.
- [x] Create unit tests validating data loading for XGBoost.
- [x] Ensure E2E script verifies XGBoost populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for CatBoost.
- [x] Implement source state logic for selecting CatBoost.
- [x] Create unit tests validating data loading for CatBoost.
- [x] Ensure E2E script verifies CatBoost populates LHS tree and editor.
- [x] Define abstract syntax tree / mock example data for SparkML.
- [x] Implement source state logic for selecting SparkML.
- [x] Create unit tests validating data loading for SparkML.
- [x] Ensure E2E script verifies SparkML populates LHS tree and editor.
- [x] Mount LHS `Dropdown` with all source frameworks.
- [x] Bind LHS `Dropdown` change event to update global state.
- [x] Mount LHS `FileTree` and bind to current source state.
- [x] Mount LHS `Editor` and bind to active file selection from LHS `FileTree`.
- [x] Implement logic to display placeholder text when no file is selected.
- [x] Write unit tests verifying LHS state updates cascade to Tree and Editor.
- [x] Achieve 100% coverage on LHS orchestrator components.
- [x] Document LHS orchestrator and state handlers.
- [x] Create Playwright E2E spec: `lhs-interactions.spec.ts`.
- [x] E2E: Verify changing LHS dropdown updates the file tree.
- [x] E2E: Verify clicking an LHS file updates the LHS Monaco editor.

### Phase 7: RHS (Target) Implementation

- [x] Define output mock file structure for .onnx.
- [x] Implement target state logic for selecting .onnx.
- [x] Create unit tests validating target generation/mapping for .onnx.
- [x] Ensure E2E script verifies .onnx populates RHS tree and editor.
- [x] Define output mock file structure for MLIR.
- [x] Implement target state logic for selecting MLIR.
- [x] Create unit tests validating target generation/mapping for MLIR.
- [x] Ensure E2E script verifies MLIR populates RHS tree and editor.
- [x] Define output mock file structure for C++.
- [x] Implement target state logic for selecting C++.
- [x] Create unit tests validating target generation/mapping for C++.
- [x] Ensure E2E script verifies C++ populates RHS tree and editor.
- [x] Define output mock file structure for Apple CoreML.
- [x] Implement target state logic for selecting Apple CoreML.
- [x] Create unit tests validating target generation/mapping for Apple CoreML.
- [x] Ensure E2E script verifies Apple CoreML populates RHS tree and editor.
- [x] Define output mock file structure for Caffe.
- [x] Implement target state logic for selecting Caffe.
- [x] Create unit tests validating target generation/mapping for Caffe.
- [x] Ensure E2E script verifies Caffe populates RHS tree and editor.
- [x] Define output mock file structure for Keras.
- [x] Implement target state logic for selecting Keras.
- [x] Create unit tests validating target generation/mapping for Keras.
- [x] Ensure E2E script verifies Keras populates RHS tree and editor.
- [x] Define output mock file structure for MXNet.
- [x] Implement target state logic for selecting MXNet.
- [x] Create unit tests validating target generation/mapping for MXNet.
- [x] Ensure E2E script verifies MXNet populates RHS tree and editor.
- [x] Define output mock file structure for TensorFlow.
- [x] Implement target state logic for selecting TensorFlow.
- [x] Create unit tests validating target generation/mapping for TensorFlow.
- [x] Ensure E2E script verifies TensorFlow populates RHS tree and editor.
- [x] Define output mock file structure for CNTK.
- [x] Implement target state logic for selecting CNTK.
- [x] Create unit tests validating target generation/mapping for CNTK.
- [x] Ensure E2E script verifies CNTK populates RHS tree and editor.
- [x] Define output mock file structure for PyTorch.
- [x] Implement target state logic for selecting PyTorch.
- [x] Create unit tests validating target generation/mapping for PyTorch.
- [x] Ensure E2E script verifies PyTorch populates RHS tree and editor.
- [x] Mount RHS `Dropdown` with all target frameworks.
- [x] Bind RHS `Dropdown` change event to initiate WASM conversion pipeline.
- [x] Mount RHS `FileTree` to display output artifacts.
- [x] Mount RHS `Editor` and bind to active file from RHS `FileTree`.
- [x] Set RHS Monaco Editor to read-only mode.
- [x] Implement loading overlay on RHS during active WASM conversion.
- [x] Write unit tests verifying RHS state updates based on WASM output.
- [x] Achieve 100% coverage on RHS orchestrator components.
- [x] Document RHS orchestrator and state handlers.
- [x] Create Playwright E2E spec: `rhs-interactions.spec.ts`.
- [x] E2E: Verify changing RHS dropdown triggers the conversion flow.
- [x] E2E: Verify RHS file tree reflects target artifacts.
- [x] E2E: Verify clicking an RHS file displays read-only content in RHS editor.

### Phase 8: Bottom Pane Tabs & Architecture

- [x] Create `Tabs` vanilla JS component.
- [x] Implement tab switching logic (hide/show panels).
- [x] Implement CSS for active/inactive tabs.
- [x] Add keyboard navigation for Tabs (left/right arrows).
- [x] Mount `Tabs` component in the `BottomContainer`.
- [x] Initialize 'Console' tab panel.
- [x] Initialize 'ONNX Visualization' tab panel.
- [x] Write unit tests for Tab state management.
- [x] Write DOM tests for Tab rendering and aria-attributes (accessibility).
- [x] Achieve 100% test coverage for Tabs component.
- [x] Document Tabs component.
- [x] Create Playwright E2E spec: `bottom-pane-tabs.spec.ts`.
- [x] E2E: Verify clicking tabs toggles visibility of underlying panels.
- [x] E2E: Verify keyboard navigation switches tabs.

### Phase 9: Console Implementation

- [x] Create `Console` UI component (a scrolling `div` with monospace text).
- [x] Implement a `Logger` utility to intercept `console.log`, `console.warn`, `console.error`.
- [x] Bind `Logger` utility to the WASM execution context.
- [x] Format log lines with timestamps and severity colors.
- [x] Implement auto-scroll to bottom on new log entry.
- [x] Add a 'Clear Console' button.
- [x] Handle massive log volumes (implement a maximum line limit and virtual scrolling if necessary).
- [x] Write unit tests for the `Logger` utility (interception and formatting).
- [x] Write DOM tests for auto-scrolling logic.
- [x] Write tests for the 'Clear Console' functionality.
- [x] Achieve 100% coverage for Console component and Logger.
- [x] Document Console component and Logger.
- [x] Create Playwright E2E spec: `console-pane.spec.ts`.
- [x] E2E: Verify WASM execution outputs text to the console.
- [x] E2E: Verify 'Clear' empties the console.

### Phase 10: ONNX Visualization Tab

- [x] Evaluate and select a lightweight, vanilla JS compatible graph renderer (e.g., d3-graphviz, cytoscape.js, or minimal netron embed).
- [x] Implement `OnnxVisualizer` component wrapper.
- [x] Create an adapter to parse `.onnx` binary/protobuf outputs from WASM into a graph model.
- [x] Render nodes (Operators) and edges (Tensors).
- [x] Implement zooming and panning functionality in the visualizer.
- [x] Implement click-to-inspect node properties.
- [x] Handle empty states (when target is not `.onnx`).
- [x] Write unit tests for the ONNX-to-Graph data adapter.
- [x] Write DOM tests for zooming/panning controls.
- [x] Achieve 100% coverage for the `OnnxVisualizer` data mapping logic.
- [x] Document `OnnxVisualizer` and adapter.
- [x] Create Playwright E2E spec: `onnx-viz-pane.spec.ts`.
- [x] E2E: Verify selecting `.onnx` target renders a graph.
- [x] E2E: Verify clicking a node opens a properties tooltip.

### Phase 11: Integration, Audits, and Final Polish

- [x] Integrate the complete flow: Change LHS -> Trigger WASM -> Update RHS -> Update Console -> Update Viz.
- [x] Implement debouncing on LHS editor input to auto-trigger WASM conversion gracefully.
- [x] Handle cyclical dependencies and clean up event listeners on unmount (prevent memory leaks).
- [x] Run `npm run coverage` and identify any missed branches in LHS logic.
- [x] Add missing unit tests for LHS logic to hit 100%.
- [x] Run coverage and identify missed branches in RHS logic.
- [x] Add missing unit tests for RHS logic to hit 100%.
- [x] Run coverage and identify missed branches in Utilities/State Store.
- [x] Add missing unit tests for Utilities to hit 100%.
- [x] Run TypeDoc/JSDoc generation and identify any undocumented functions.
- [x] Add missing documentation to reach 100% doc coverage.
- [x] Review E2E test flakiness and add necessary `waitFor` assertions.
- [x] Execute the full Playwright E2E suite against a local Sphinx build.
- [x] Optimize CSS bundle size (purge unused styles).
- [x] Optimize JS bundle size (minify, tree-shake).
- [x] Test the demo UI across major browsers (Chrome, Firefox, Safari).
- [x] Verify accessibility (ARIA roles, keyboard nav, contrast ratios) using lighthouse.
- [x] Create a final PR template checklist specifically for this UI feature.
- [x] Ensure CI/CD workflows are updated to block PRs if coverage drops below 100%.
- [x] Add `test-ui` and `test-ui-e2e` jobs to `.github/workflows/test-js.yml`.
- [x] Add `docs-build` step in CI to verify the Sphinx directive compiles without errors.
- [x] Verify the 'load WASM' latency on a slow network connection via DevTools throttling.
- [x] Document the architecture and adding new frameworks in `ONNX45_DEMO_IN_SPHINX.md` (Self-referential check).
- [x] Final team review and sign-off.
- [x] Implement caching layer for parsed ASTs and WASM outputs.
- [x] Write unit tests for the caching layer eviction policy (100% coverage).
- [x] Add `localStorage` serialization for current editor states.
- [x] Write unit tests for `localStorage` parse errors.
- [x] Create E2E spec: verify page reload restores editor state.
- [x] E2E spec: verify page reload restores selected LHS source.
- [x] E2E spec: verify page reload restores selected RHS target.
- [x] Create a fallback text viewer for non-text outputs on RHS.
- [x] Add unit tests for binary-to-hex formatting fallback.
- [x] Ensure 100% JSDoc coverage on formatters and parsers.
- [x] Final manual exploratory testing session on mobile devices.
- [x] Final manual exploratory testing session on tablets.
- [x] Final security audit on user input handling in the editor.
- [x] Ensure `DOMPurify` (or equivalent) sanitizes any generated HTML from WASM.
- [x] Write unit tests for XSS prevention in the console and visualizer.
- [x] Add final Sphinx integration E2E spec to ensure directive parses options.
- [x] Validate cross-origin isolation (COOP/COEP) headers for `SharedArrayBuffer` support.
- [x] Write tests ensuring graceful degradation if `SharedArrayBuffer` is blocked.
- [x] Merge PR and publish release announcement.
