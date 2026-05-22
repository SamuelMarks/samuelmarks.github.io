# ONNX45: Extended Demo in Sphinx (Multi-Step Pipelines)

## Problem Description

Following the successful implementation of the 1-Step Source-to-Target pipeline (Phase 1-11), we need to expose the deep capabilities of the `onnx9000` ecosystem. This includes optimization, quantization, MLIR lowering, C/C++ transpilation, and live in-browser execution using WASM and WebNN.

To avoid cluttering the UI, we are introducing a **Progressive Pipeline UI**. Users can "Promote to Source" (shifting the RHS target back to the LHS source) to chain conversions indefinitely. Additionally, a new "Run Inference" execution layer will provide interactive testing and profiling of the generated artifacts directly in the browser.

**Key Requirements:**

- **Zero Framework Dependency:** Strictly Vanilla JS.
- **100% Quality Gates:** 100% JSDoc, 100% Unit Test Coverage, Comprehensive Playwright E2E.
- **Non-Blocking Architecture:** All heavy compilation and execution tasks MUST run in Web Workers.

## Implementation Steps

### Phase 12: Core Pipeline Architecture & State Management

- [x] Define `PipelineNode` vanilla JS class for chaining transformations.
- [x] Extend global state store to support a `PipelineHistory` array.
- [x] Implement `undo()` method in state store to revert pipeline steps.
- [x] Implement `redo()` method in state store.
- [x] Write unit tests verifying `PipelineHistory` immutability.
- [x] Write unit tests verifying `undo()` correctly restores previous Source/Target states.
- [x] Write unit tests verifying `redo()` functionality.
- [x] Define `PipelineValidator` utility to check compatibility between target outputs and new source inputs.
- [x] Add compatibility matrix: `.onnx` -> `onnx-simplifier`, `Olive`, `onnx-mlir`, `onnx2c`, `ORT Web`.
- [x] Add compatibility matrix: `MLIR` -> `IREE Compiler`.
- [x] Add compatibility matrix: `C/C++` -> `Emscripten`.
- [x] Write unit tests for `PipelineValidator` covering all valid matrices.
- [x] Write unit tests for `PipelineValidator` covering invalid transitions.
- [x] Update `EventBus` to handle `PIPELINE_STEP_ADDED` and `PIPELINE_STEP_REMOVED` events.
- [x] Add 100% JSDoc to `PipelineNode`, `PipelineHistory`, and `PipelineValidator`.
- [x] Create UI component for "Pipeline Breadcrumbs" (e.g., PyTorch > ONNX > MLIR > VMFB).
- [x] Mount breadcrumbs component in the top header of the UI.
- [x] Bind breadcrumbs to `PipelineHistory` state changes.
- [x] Implement click-to-revert logic on breadcrumb items.
- [x] Write DOM tests verifying breadcrumb rendering based on state.
- [x] Write DOM tests verifying click-to-revert dispatches correct state changes.
- [x] Add CSS for breadcrumb active, hover, and disabled states.
- [x] Add Playwright E2E spec: `pipeline-breadcrumbs.spec.ts`.
- [x] E2E: Verify completing a conversion adds a breadcrumb.
- [x] E2E: Verify clicking a previous breadcrumb restores the LHS/RHS UI state.
- [x] Ensure 100% coverage on new architecture core logic.
- [x] Document the extended state machine in `ARCHITECTURE.md`.
- [x] Set up telemetry mock to track pipeline depth (for internal analytics).
- [x] Write unit tests for the telemetry mock.
- [x] Review Phase 12 against Vanilla JS constraints.

### Phase 13: "Promote to Source" (Shift Left) Mechanics

- [x] Create `PromoteButton` vanilla JS component with `[ < Promote to Source ]` label.
- [x] Mount `PromoteButton` in the RHS Target header.
- [x] Implement CSS for `PromoteButton` (primary action style, disabled state).
- [x] Add logic to disable `PromoteButton` if RHS is empty or currently processing.
- [x] Bind `PromoteButton` click to dispatch `PROMOTE_TARGET_TO_SOURCE` event.
- [x] Implement state handler for `PROMOTE_TARGET_TO_SOURCE`.
- [x] Logic: Map RHS active file content to LHS Editor state.
- [x] Logic: Update LHS Dropdown to reflect the new source type (e.g., `.onnx`).
- [x] Logic: Clear RHS Target state and Editor content.
- [x] Logic: Push current state to `PipelineHistory`.
- [x] Add CSS transition for "shift left" visual effect on Monaco Editors.
- [x] Write unit tests for `PROMOTE_TARGET_TO_SOURCE` state transitions.
- [x] Write unit tests verifying RHS clears on promotion.
- [x] Write DOM tests verifying `PromoteButton` disables correctly during WASM processing.
- [x] Write DOM tests verifying `PromoteButton` enables upon successful conversion.
- [x] Add toast notification system for successful promotion ("Artifact promoted to Source").
- [x] Create `ToastNotification` vanilla JS component.
- [x] Implement toast timeout and CSS fade-out logic.
- [x] Write DOM tests for `ToastNotification` component.
- [x] Add 100% JSDoc to `PromoteButton` and `ToastNotification`.
- [x] Create Playwright E2E spec: `promote-to-source.spec.ts`.
- [x] E2E: Verify clicking "Promote to Source" moves content from RHS to LHS.
- [x] E2E: Verify LHS framework dropdown updates automatically based on promoted file type.
- [x] E2E: Verify RHS is cleared after promotion.
- [x] E2E: Verify toast notification appears on promotion.
- [x] E2E: Verify "Promote to Source" is disabled when RHS is empty.
- [x] Add keyboard shortcut (e.g., `Ctrl+Shift+Left`) to trigger promotion.
- [x] Write unit tests for the keyboard shortcut listener.
- [x] Ensure 100% line/branch coverage for Phase 13 components.
- [x] Document promotion workflow in user-facing Sphinx guides.

### Phase 14: Web Worker & WASM Execution Orchestration

- [x] Define `WorkerManager` singleton for managing Web Worker pools.
- [x] Implement worker initialization logic from bundled WASM wrappers.
- [x] Define standardized `WorkerMessage` protocol (ID, Type, Payload, Error).
- [x] Implement `postMessage` wrapper with Promise-based responses.
- [x] Add timeout logic to worker requests to prevent deadlocks.
- [x] Implement `terminate()` method to forcefully stop a runaway worker.
- [x] Add UI cancel button to interrupt long-running conversions.
- [x] Bind cancel button to `WorkerManager.terminate()`.
- [x] Handle worker `onerror` and `onmessageerror` events.
- [x] Map worker errors to user-friendly Console tab output.
- [x] Set up `SharedArrayBuffer` support where available for zero-copy memory transfers.
- [x] Add feature detection fallback to standard `postMessage` copy if SAB is restricted (COOP/COEP).
- [x] Write unit tests for `WorkerManager` instantiating and pooling workers.
- [x] Write unit tests mocking `Worker` to test Promise resolution/rejection.
- [x] Write unit tests verifying timeout logic throws correctly.
- [x] Write unit tests for worker termination and pool recycling.
- [x] Write unit tests for feature detection fallback logic.
- [x] Add 100% JSDoc to all Worker management utilities.
- [x] Implement `WasmExecutionStream` to pipe stdout/stderr from worker to Console UI via chunks.
- [x] Write unit tests for `WasmExecutionStream` chunk buffering.
- [x] Add Playwright E2E spec: `web-worker-execution.spec.ts`.
- [x] E2E: Verify long-running WASM task can be cancelled via UI.
- [x] E2E: Verify cancelled task resets UI state gracefully.
- [x] E2E: Verify stdout streams to console in real-time without freezing main thread.
- [x] Ensure `WorkerManager` cleans up resources on `window.beforeunload`.
- [x] Profile memory leaks in worker instantiation using DevTools.
- [x] Add memory limit safeguards to reject excessively large inputs.
- [x] Write unit tests for memory limit rejection logic.
- [x] Ensure 100% test coverage for Phase 14 modules.
- [x] Document worker architecture in `ONNX_WEBGPU_SUPPORT.md`.

### Phase 15: Pipeline A - Optimization & Quantization (Olive)

- [x] Add `Olive` / `onnx-simplifier` to Target Dropdown options (requires `.onnx` source).
- [x] Implement `OliveOptimizer` worker wrapper.
- [x] Define state logic for selecting "Optimize (Olive)".
- [x] Expose Olive configuration UI (sliders for quantization level, toggle for INT8/FP16).
- [x] Create `OliveConfigPanel` vanilla JS component.
- [x] Mount `OliveConfigPanel` in RHS when Olive is selected.
- [x] Write DOM tests for `OliveConfigPanel` inputs and state binding.
- [x] Integrate WASM build of `onnx-simplifier`.
- [x] Execute `onnx-simplifier` step in Web Worker.
- [x] Handle success/failure of simplification step.
- [x] Integrate WASM build of `Olive` (mocked or actual, depending on engine limits).
- [x] Implement dynamic graph diffing in `OnnxVisualizer`.
- [x] Highlight removed/fused nodes in red/green within the Visualizer tab.
- [x] Extract model size (bytes) before and after optimization.
- [x] Display compression ratio metric in the UI.
- [x] Write unit tests verifying size metric calculations.
- [x] Write unit tests for Olive config state serialization.
- [x] Write unit tests for optimization worker payload formatting.
- [x] Add 100% JSDoc to Pipeline A specific components.
- [x] Create Playwright E2E spec: `pipeline-a-optimization.spec.ts`.
- [x] E2E: Verify selecting Olive shows config panel.
- [x] E2E: Verify running optimization produces a smaller `.onnx` target.
- [x] E2E: Verify visualizer shows side-by-side or highlighted graph differences.
- [x] E2E: Verify promoting optimized ONNX sets it as new LHS source.
- [x] Implement static shape inference toggle in Olive config.
- [x] Implement transformer-specific optimizations (fusion) toggle.
- [x] Write unit tests for static shape inference payload flag.
- [x] Ensure 100% coverage on Optimization logic.
- [x] Add optimization tutorial to Sphinx docs.
- [x] Verify Olive UI layout remains responsive on mobile.

### Phase 16: Pipeline A - Execution & Profiling (ORT Web/WebNN)

- [x] Create `RunInferenceButton` in RHS Target header (visible only for runnable artifacts).
- [x] Add `Execution Profiler` tab to Bottom Pane.
- [x] Implement `ORTWebRunner` wrapper class.
- [x] Implement `WebNNPolyfillRunner` wrapper class.
- [x] Create UI dropdown to select Execution Provider (WASM, WebGL, WebGPU, WebNN).
- [x] Write unit tests for EP selection and fallback logic.
- [x] Handle WebGPU/WebNN feature detection and disable unsupported dropdown options.
- [x] Integrate ORT Web session instantiation in a Web Worker.
- [x] Implement execution timer using `performance.now()`.
- [x] Calculate Time To First Token (TTFT) for text-generation models.
- [x] Calculate Tokens Per Second (TPS) metrics.
- [x] Create `MetricsDashboard` component to display TPS, TTFT, and total latency.
- [x] Mount `MetricsDashboard` in the Execution Profiler tab.
- [x] Plot latency variance using a lightweight vanilla JS canvas chart.
- [x] Track memory utilization during inference using Web Performance API (if available).
- [x] Write unit tests for TTFT and TPS calculation math.
- [x] Write DOM tests for `MetricsDashboard` rendering.
- [x] Write unit tests for canvas charting boundaries and data normalization.
- [x] Handle asynchronous inference loops for streaming outputs.
- [x] Dispatch `INFERENCE_CHUNK_RECEIVED` events to update UI progressively.
- [x] Write unit tests for inference event streaming.
- [x] Add 100% JSDoc to Execution Pipeline components.
- [x] Create Playwright E2E spec: `pipeline-a-execution.spec.ts`.
- [x] E2E: Verify "Run Inference" button is visible for `.onnx` files.
- [x] E2E: Verify executing runs without crashing and updates Console.
- [x] E2E: Verify Execution Profiler tab populates with latency data.
- [x] E2E: Verify streaming output updates UI continuously.
- [x] Ensure strict cleanup of ORT sessions (`session.release()`) to prevent memory leaks.
- [x] Add `finally` blocks in worker to guarantee memory cleanup on error.
- [x] Run coverage report and hit 100% on execution wrappers.

### Phase 17: Pipeline B - MLIR & IREE Compiler Flow

- [x] Add `MLIR` to Target Dropdown options.
- [x] Implement `OnnxMlirCompiler` worker wrapper.
- [x] Add UI dropdown for MLIR emission type (ONNX Dialect, Linalg, LLVM IR).
- [x] Create `MlirAstViewer` tab in Bottom Pane.
- [x] Implement basic syntax highlighting for MLIR format in Monaco.
- [x] Parse `.onnx` to MLIR (ONNX Dialect) via Web Worker.
- [x] Map stdout errors from `onnx-mlir` to the code editor (gutter error markers).
- [x] Write unit tests for mapping CLI error lines to editor line numbers.
- [x] Write DOM tests verifying `MlirAstViewer` tab displays generated MLIR.
- [x] Add `IREE Compiler` to Target Dropdown options (requires `MLIR` source).
- [x] Implement `IreeCompiler` worker wrapper.
- [x] Create `IreeConfigPanel` component.
- [x] Add targeting options (WASM-CPU, WebGPU, Vulkan) to `IreeConfigPanel`.
- [x] Execute `iree-compile` passing MLIR to generate `.vmfb` (Virtual Machine FlatBuffer).
- [x] Display a binary hex viewer or read-only info screen for the `.vmfb` output in RHS Editor.
- [x] Write unit tests for `IreeConfigPanel` state management.
- [x] Write unit tests for VMFB binary parser/fallback UI.
- [x] Add 100% JSDoc to MLIR/IREE compiler wrappers.
- [x] Create Playwright E2E spec: `pipeline-b-mlir.spec.ts`.
- [x] E2E: Verify `.onnx` can be converted to MLIR.
- [x] E2E: Verify MLIR syntax highlighting applies in Monaco.
- [x] E2E: Verify promoting MLIR and targeting IREE generates a `.vmfb` artifact.
- [x] E2E: Verify compiler warnings appear in Console tab.
- [x] Add caching for intermediate MLIR dialects to speed up UI switching.
- [x] Write unit tests for MLIR compiler cache hit/miss logic.
- [x] Ensure 100% coverage on Phase 17 components.
- [x] Document the MLIR lowering tutorial in `TUTORIAL_MLIR_LOWERING.md`.
- [x] Add missing Pytest checks for the backend tools supplying these WASM binaries.

### Phase 18: Pipeline B - IREE WASM VM Execution

- [x] Add "Run Inference" action for `.vmfb` artifacts.
- [x] Implement `IreeVmRunner` worker wrapper.
- [x] Load IREE WASM VM runtime engine.
- [x] Define memory mapping logic for passing inputs to IREE VM.
- [x] Handle execution state: Initialize, Load Module, Invoke Function.
- [x] Provide UI dropdown to select the target exported function (e.g., `main`, `predict`).
- [x] Read exported function signatures from `.vmfb` header.
- [x] Write unit tests for VMFB header signature parsing.
- [x] Map execution results back from IREE memory space to JS typed arrays.
- [x] Format output tensors into human-readable tables in the UI.
- [x] Compare IREE output with ORT output (if both exist in history) to verify numerical accuracy.
- [x] Display numerical diff (e.g., max absolute error) in the Profiler Tab.
- [x] Write unit tests for numerical comparison utility.
- [x] Write DOM tests for table formatting of output arrays.
- [x] Write unit tests handling VM traps/crashes gracefully.
- [x] Add 100% JSDoc to IREE VM execution logic.
- [x] Create Playwright E2E spec: `pipeline-b-iree-execution.spec.ts`.
- [x] E2E: Verify executing VMFB runs in IREE VM and outputs a tensor.
- [x] E2E: Verify output tensor is formatted as a table.
- [x] E2E: Verify numerical diff compares against baseline ONNX execution.
- [x] Setup memory profiling specific to IREE VM heap usage.
- [x] Plot IREE heap usage on the `MetricsDashboard`.
- [x] Ensure VM engine instances are correctly disposed after execution.
- [x] Add strict memory leak tests using mocked `FinalizationRegistry`.
- [x] Reach 100% coverage on `IreeVmRunner`.

### Phase 19: Pipeline C - Edge C/C++ Compilation (ONNX2C)

- [x] Add `C/C++ (onnx2c)` to Target Dropdown options.
- [x] Implement `Onnx2CCompiler` worker wrapper.
- [x] Generate pure C code representation of the `.onnx` graph.
- [x] Provide syntax highlighting for C/C++ in the RHS Monaco editor.
- [x] Write unit tests for `Onnx2CCompiler` argument mapping.
- [x] Add `Emscripten/WASM` to Target Dropdown (requires `C/C++` source).
- [x] Implement in-browser Emscripten mock (or minimal clang WASM) to compile the C code.
- [x] Create `MemMockDashboard` tab to analyze static memory allocation in the generated C code.
- [x] Parse `malloc` and static array declarations from the C output.
- [x] Calculate total theoretical footprint (KB/MB) and display it.
- [x] Write unit tests for C AST parsing and memory footprint math.
- [x] Add 100% JSDoc to Pipeline C tools.
- [x] Execute compiled standalone WASM directly (Zero-dependency execution).
- [x] Compare execution latency between native ORT Web and standalone ONNX2C WASM.
- [x] Display latency comparison in Profiler Tab.
- [x] Write unit tests verifying latency comparison payload logic.
- [x] Create Playwright E2E spec: `pipeline-c-onnx2c.spec.ts`.
- [x] E2E: Verify `.onnx` converts to C code.
- [x] E2E: Verify C code displays with correct highlighting.
- [x] E2E: Verify memory footprint calculation displays correctly.
- [x] E2E: Verify compiling to Emscripten WASM and executing works.
- [x] Add "Download C Source" button to the UI.
- [x] Write DOM tests for Download button blob generation.
- [x] Ensure 100% test coverage for Phase 19.
- [x] Document ONNX2C usage in `ONNX33_ONNX2C.md`.

### Phase 20: Execution UI & Tensor Input Modals

- [x] Create `TensorInputModal` vanilla JS component.
- [x] Detect required input shapes from the active model artifact.
- [x] Render dynamic form fields based on input shapes (e.g., `[1, 3, 224, 224]`).
- [x] Add "Generate Random Data" button to fill inputs with normalized floats.
- [x] Write unit tests for dynamic shape form generation.
- [x] Write unit tests for random data generation (normal distribution).
- [x] Add "Upload Image" option for `N, 3, H, W` inputs.
- [x] Implement hidden `<input type="file">` and HTML5 Canvas to resize/crop images.
- [x] Extract RGB channels from Canvas and format as Float32Array.
- [x] Write unit tests for Image-to-Tensor array conversion math.
- [x] Write DOM tests for Image upload and Canvas rendering.
- [x] Add "Text Input" option for `INT64` sequence inputs (requires a tokenizer).
- [x] Integrate a lightweight JS tokenizer fallback for basic text models.
- [x] Write unit tests for Text-to-TokenArray logic.
- [x] Create `InferenceResultModal` component.
- [x] Render Softmax classification outputs as a progress-bar chart.
- [x] Render bounding box outputs as an overlay on the uploaded image.
- [x] Write DOM tests for bounding box CSS absolute positioning math.
- [x] Add 100% JSDoc to all Modals and Input processors.
- [x] Create Playwright E2E spec: `execution-input-output.spec.ts`.
- [x] E2E: Verify opening execution modal dynamically maps input shapes.
- [x] E2E: Verify "Generate Random Data" populates all fields.
- [x] E2E: Verify uploading an image populates the image preview.
- [x] E2E: Verify results are rendered as a classification chart.
- [x] Handle extremely large tensor inputs (e.g., audio files) gracefully via chunking.
- [x] Ensure strict cleanup of Image/Canvas data to prevent memory bloat.
- [x] Validate 100% test coverage for Phase 20 input/output handling.

### Phase 21: E2E Testing, Audits, & Integration

- [x] Run complete end-to-end `npm run test` validating 100% coverage on all new files.
- [x] Add missing branch coverage tests for edge cases in execution logic.
- [x] Add missing branch coverage tests for state history array bounds.
- [x] Ensure `eslint.config.js` passes cleanly with NO framework violations.
- [x] Execute `typedoc` and verify no missing JSDoc warnings.
- [x] Verify accessibility of multi-step pipeline (tab index, ARIA roles on new buttons).
- [x] Run Lighthouse audit, verify 90+ score for Performance and Accessibility.
- [x] Test entire UI flow on mobile Chrome (Android) verifying modals fit screens.
- [x] Test entire UI flow on mobile Safari (iOS) verifying WASM execution works.
- [x] Audit application bundle size (JS and CSS). Implement lazy-loading for heavy charting libs if needed.
- [x] Integrate extended Sphinx documentation confirming directives compile correctly.
- [x] Verify dark mode switches correctly invert colors for MLIR viewer and Tensor modals.
- [x] Check cross-browser functionality of Web Workers (Firefox, Safari, Edge, Chrome).
- [x] Run full Playwright test suite against the built Sphinx production output.
- [x] Identify and fix any flaky E2E tests utilizing `waitForFunction`.
- [x] Update `ROADMAP.md` indicating Phase 2 implementation complete.
- [x] Update `README.md` to highlight interactive multi-pipeline capabilities.
- [x] Perform security audit on uploaded file blobs and Web Worker message passing (prevent XSS/RCE).
- [x] Create comprehensive recorded demo GIF of Pipeline A, B, and C.
- [x] Add recorded GIFs to the project repository docs.
- [x] Merge PR and publish v2 release announcement.
- [x] Monitor crash logs or user feedback on extended capabilities.
- [x] Plan Phase 3 optimizations (WebGPU backend integrations).
- [x] Final team sign-off.
