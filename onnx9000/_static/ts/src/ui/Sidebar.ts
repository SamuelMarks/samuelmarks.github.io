import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';

import { isOfflineMode, isDistributedMode } from '../core/State';

export class Sidebar extends BaseComponent {
  private fileInput: HTMLInputElement;

  constructor(containerId: string) {
    super(containerId);

    // Global Config
    const configSection = $create('div', { className: 'sidebar-section' });
    const configTitle = $create('h4', { textContent: 'Global Settings' });

    const offlineRow = $create('div', { className: 'property-row' });
    const offlineLabel = $create('label', { textContent: 'Offline Mode (No external requests)' });
    const offlineCheckbox = $create<HTMLInputElement>('input', {
      attributes: { type: 'checkbox', checked: 'true' },
    });
    offlineRow.appendChild(offlineLabel);
    offlineRow.appendChild(offlineCheckbox);

    const distRow = $create('div', { className: 'property-row' });
    const distLabel = $create('label', { textContent: 'Distributed Swarm Execution' });
    const distCheckbox = $create<HTMLInputElement>('input', { attributes: { type: 'checkbox' } });
    distRow.appendChild(distLabel);
    distRow.appendChild(distCheckbox);

    configSection.appendChild(configTitle);
    configSection.appendChild(offlineRow);
    configSection.appendChild(distRow);
    this.container.appendChild(configSection);

    offlineCheckbox.addEventListener('change', () => {
      isOfflineMode.set(offlineCheckbox.checked);
      Toast.show(`Offline mode ${offlineCheckbox.checked ? 'enabled' : 'disabled'}`, 'info');
    });
    distCheckbox.addEventListener('change', () => {
      isDistributedMode.set(distCheckbox.checked);
      Toast.show(`Distributed mode ${distCheckbox.checked ? 'enabled' : 'disabled'}`, 'info');
    });

    // Framework importer
    const importerSection = $create('div', { className: 'sidebar-section' });
    const importerTitle = $create('h4', { textContent: 'Import Model' });
    const select = $create<HTMLSelectElement>('select', {
      className: 'ide-select',
      innerHTML: `
        <option value="onnx">ONNX (.onnx)</option>
        <option value="safetensors">Safetensors (.safetensors)</option>
        <option value="coreml">CoreML (.mlmodel)</option>
        <option value="tensorflow">TensorFlow (.pb)</option>
        <option value="sklearn">Scikit-Learn (.pkl)</option>
        <option value="paddle">PaddlePaddle (.pdmodel)</option>
        <option value="xgboost">XGBoost (.json)</option>
        <option value="keras">Keras TF.js (.json)</option>

        <option value="gguf">GGUF (.gguf)</option>
      `,
    });

    this.fileInput = $create<HTMLInputElement>('input', {
      attributes: { type: 'file', accept: '.onnx,.safetensors,.pb,.pkl,.pdmodel,.json,.gguf' },
      className: 'ide-file-input',
    });

    // Directory Input for TF SavedModel
    const dirInput = $create<HTMLInputElement>('input', {
      attributes: { type: 'file', webkitdirectory: 'true', directory: 'true' },
      className: 'ide-file-input',
    });

    const loadBtn = $create('button', { className: 'action-btn', textContent: 'Load File' });
    const loadDirBtn = $create('button', {
      className: 'action-btn',
      textContent: 'Load Folder (TF)',
    });

    // 415. Directory API mount
    const mountBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Mount Local OS Workspace',
      attributes: { style: 'margin-top: 5px; width: 100%; display: block;' },
    });

    importerSection.appendChild(importerTitle);
    importerSection.appendChild(select);
    importerSection.appendChild(this.fileInput);
    importerSection.appendChild(loadBtn);
    importerSection.appendChild($create('hr'));
    importerSection.appendChild(dirInput);
    importerSection.appendChild(loadDirBtn);

    // 165. Serialize back to .onnx
    const exportOnnxBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Export modified .onnx',
      attributes: { style: 'margin-top: 5px;' },
    });
    const exportOVBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Export OpenVINO (IR)',
      attributes: { style: 'margin-top: 5px;' },
    });

    importerSection.appendChild($create('hr'));
    importerSection.appendChild(exportOnnxBtn);
    importerSection.appendChild(exportOVBtn);
    const exportTfBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Export TFLite (JSON)',
      attributes: { style: 'margin-top: 5px;' },
    });
    importerSection.appendChild(exportTfBtn);
    exportTfBtn.addEventListener('click', () => {
      globalEvents.emit('exportTFLite');
    });

    exportOnnxBtn.addEventListener('click', () => {
      globalEvents.emit('exportONNX');
    });

    exportOVBtn.addEventListener('click', () => {
      globalEvents.emit('exportOpenVINO');
    });

    importerSection.appendChild(mountBtn);
    mountBtn.addEventListener('click', () => {
      globalEvents.emit('mountWorkspace');
    });

    this.container.appendChild(importerSection);

    // Surgeon toggle
    const surgeonSection = $create('div', { className: 'sidebar-section' });
    const surgeonTitle = $create('h4', { textContent: 'Graph Surgeon' });
    const foldBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Fold Constants',
    });
    const removeIdBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Remove Identity',
    });
    const pruneBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Prune Unused',
    });
    const topSortBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Topological Sort',
    });
    surgeonSection.appendChild(surgeonTitle);
    surgeonSection.appendChild(foldBtn);
    surgeonSection.appendChild(removeIdBtn);
    surgeonSection.appendChild(pruneBtn);
    surgeonSection.appendChild(topSortBtn);

    const freezeBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Freeze Input (Selected)',
    });
    const promoteBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Promote Input (Selected)',
    });
    surgeonSection.appendChild(freezeBtn);
    surgeonSection.appendChild(promoteBtn);

    // 157. Extract Subgraph
    const extractBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Extract Subgraph (Selection)',
    });
    extractBtn.disabled = true;
    surgeonSection.appendChild(extractBtn);

    globalEvents.on('multiSelectionChanged', (nodes: string[]) => {
      extractBtn.disabled = nodes.length < 1;
      extractBtn.textContent = `Extract Subgraph (${nodes.length} Nodes)`;
      extractBtn.onclick = () => {
        if (nodes.length > 0) globalEvents.emit('surgeon', `extractSubgraph:${nodes.join(',')}`);
      };
    });

    // 481. Auto-Tune Sub-tab
    const tuneBtn = $create('button', { className: 'action-btn', textContent: 'Auto-Tune / NAS' });
    tuneBtn.style.marginTop = '10px';
    const rewriteBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Apply Rewrites (Fusion)',
    });
    rewriteBtn.style.marginTop = '5px';

    surgeonSection.appendChild(tuneBtn);
    surgeonSection.appendChild(rewriteBtn);

    this.container.appendChild(surgeonSection);

    tuneBtn.addEventListener('click', () => {
      globalEvents.emit('surgeon', 'autoTune');
    });

    rewriteBtn.addEventListener('click', () => {
      globalEvents.emit('surgeon', 'applyRewrites');
    });

    const webgpuTuneBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'WebGPU Workgroup Tuner',
    });
    webgpuTuneBtn.style.marginTop = '5px';
    surgeonSection.appendChild(webgpuTuneBtn);

    const tuningCanvas = $create<HTMLCanvasElement>('canvas', {
      className: 'ide-loss-chart',
      attributes: { width: '200', height: '100', style: 'margin-top: 10px;' },
    });
    surgeonSection.appendChild(tuningCanvas);

    webgpuTuneBtn.addEventListener('click', () => {
      globalEvents.emit('surgeon', 'tuneWebGPU');
    });

    const tuneHistory: { step: number; score: number }[] = [];
    globalEvents.on('tuningProgress', (data: any) => {
      tuneHistory.push({ step: data.step, score: data.score });
      if (tuneHistory.length > 100) tuneHistory.shift();

      const ctx = tuningCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 200, 100);
        ctx.strokeStyle = '#007bff';
        ctx.fillStyle = '#007bff';
        ctx.lineWidth = 1;

        const maxScore = Math.max(...tuneHistory.map((h) => h.score), 1);

        ctx.beginPath();
        tuneHistory.forEach((h, i) => {
          const x = (i / tuneHistory.length) * 200;
          const y = 100 - (h.score / maxScore) * 100;
          // Scatter plot dots
          ctx.fillRect(x, y, 2, 2);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    });

    let activeNodeId: string | null = null;
    globalEvents.on('nodeSelected', (node: any) => {
      activeNodeId = node ? node.name : null;
    });

    freezeBtn.addEventListener('click', () => {
      if (activeNodeId) globalEvents.emit('surgeon', `freeze:${activeNodeId}`);
      else Toast.show('Select a node input first', 'warn');
    });
    promoteBtn.addEventListener('click', () => {
      if (activeNodeId) globalEvents.emit('surgeon', `promote:${activeNodeId}`);
      else Toast.show('Select a node input first', 'warn');
    });

    foldBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'foldConstants'));
    removeIdBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'removeIdentity'));
    pruneBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'pruneUnused'));
    topSortBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'topologicalSort'));

    // Quantize / Sparsify
    const quantizeSection = $create('div', { className: 'sidebar-section' });
    const quantizeTitle = $create('h4', { textContent: 'Quantize / Sparsify' });
    const quantizeBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Min-Max INT8 Quantize',
    });
    const quantizeInt4Btn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'INT4 Packed Block Quantize',
    });
    quantizeInt4Btn.style.marginTop = '5px';

    const pruneContainer = $create('div', { className: 'property-row' });
    const pruneLabel = $create('label', { textContent: 'Threshold (1e-5)' });
    const pruneSlider = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'range', min: '1', max: '5', step: '1', value: '5' },
    });
    const applyPruneBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Sparsify (Magnitude Prune)',
    });

    pruneContainer.style.flexDirection = 'column';
    pruneContainer.appendChild(pruneLabel);
    pruneContainer.appendChild(pruneSlider);
    pruneContainer.appendChild(applyPruneBtn);

    quantizeSection.appendChild(quantizeTitle);
    quantizeSection.appendChild(quantizeBtn);
    quantizeSection.appendChild(quantizeInt4Btn);
    quantizeSection.appendChild(pruneContainer);
    this.container.appendChild(quantizeSection);

    quantizeBtn.addEventListener('click', () => globalEvents.emit('surgeon', 'quantize'));
    quantizeInt4Btn.addEventListener('click', () => globalEvents.emit('surgeon', 'quantizeINT4'));
    applyPruneBtn.addEventListener('click', () => {
      const exp = parseInt(pruneSlider.value, 10);
      globalEvents.emit('surgeon', `sparsify:${Math.pow(10, -exp)}`);
    });

    // Search toggle
    const searchSection = $create('div', { className: 'sidebar-section' });
    const searchTitle = $create('h4', { textContent: 'Search Graph' });
    const searchInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'text', placeholder: 'Node Name or OpType...' },
    });
    const searchBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Search',
    });
    searchSection.appendChild(searchTitle);
    searchSection.appendChild(searchInput);
    searchSection.appendChild(searchBtn);
    this.container.appendChild(searchSection);

    // Training Toggle
    const trainSection = $create('div', { className: 'sidebar-section' });
    const trainTitle = $create('h4', { textContent: 'Autograd / Training' });

    const lossSelect = $create<HTMLSelectElement>('select', {
      className: 'ide-select',
      innerHTML: `
        <option value="CrossEntropy">CrossEntropy Loss</option>
        <option value="MSE">MSE Loss</option>
      `,
    });
    const optSelect = $create<HTMLSelectElement>('select', {
      className: 'ide-select',
      innerHTML: `
        <option value="Adam">Adam</option>
        <option value="SGD">SGD</option>
      `,
    });
    const injectBackwardBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Inject Backward Pass',
    });
    const runTrainBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Run Training Step (WASM)',
    });
    const runEpochBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Train (10 Epochs)',
    });

    // 229. Expose UI sliders for Learning Rate and Batch Size
    const hparamsRow = $create('div', {
      className: 'property-row',
      attributes: { style: 'flex-direction: column;' },
    });
    const lrLabel = $create('label', { textContent: 'Learning Rate (0.01)' });
    const lrSlider = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'range', min: '0.001', max: '0.1', step: '0.001', value: '0.01' },
    });
    const bsLabel = $create('label', { textContent: 'Batch Size (32)' });
    const bsSlider = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'range', min: '1', max: '128', step: '1', value: '32' },
    });
    hparamsRow.appendChild(lrLabel);
    hparamsRow.appendChild(lrSlider);
    hparamsRow.appendChild(bsLabel);
    hparamsRow.appendChild(bsSlider);

    trainSection.appendChild(trainTitle);
    trainSection.appendChild(hparamsRow);
    trainSection.appendChild(lossSelect);
    trainSection.appendChild(optSelect);
    trainSection.appendChild(injectBackwardBtn);
    trainSection.appendChild(runTrainBtn);
    trainSection.appendChild(runEpochBtn);

    // 233. Extract Trained Weights Button
    const extractWBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Extract Weights (.safetensors)',
      attributes: { style: 'margin-top: 5px;' },
    });

    // 222. Federated Learning Panel
    // 223. Generate dummy training datasets
    const genDataBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Generate Synthetic Dataset',
      attributes: { style: 'margin-top: 5px;' },
    });
    const fedBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Broadcast Gradients (Federated)',
      attributes: { style: 'margin-top: 5px;' },
    });

    trainSection.appendChild(extractWBtn);
    trainSection.appendChild(genDataBtn);
    trainSection.appendChild(fedBtn);

    this.container.appendChild(trainSection);

    genDataBtn.addEventListener('click', () =>
      globalEvents.emit('autograd', { action: 'generate_data' }),
    );
    fedBtn.addEventListener('click', () =>
      globalEvents.emit('autograd', { action: 'federated_train' }),
    );

    extractWBtn.addEventListener('click', () => {
      globalEvents.emit('exportONNX'); // Re-routes to the mock JSON exporter for zero-dep environment
      Toast.show('Exported updated tensors', 'success');
    });

    injectBackwardBtn.addEventListener('click', () => {
      globalEvents.emit('autograd', {
        action: 'inject',
        loss: lossSelect.value,
        optimizer: optSelect.value,
      });
    });

    const lossCanvas = $create<HTMLCanvasElement>('canvas', {
      attributes: { width: '200', height: '100' },
      className: 'ide-loss-chart',
    });
    trainSection.appendChild(lossCanvas);

    // 228. Plot Loss curve
    const lossHistory: number[] = [];
    globalEvents.on('lossUpdated', (loss: number) => {
      lossHistory.push(loss);
      if (lossHistory.length > 50) lossHistory.shift();
      const ctx = lossCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 200, 100);
        ctx.strokeStyle = '#dc3545';
        ctx.lineWidth = 2;
        ctx.beginPath();
        const maxLoss = Math.max(...lossHistory, 2.0);
        lossHistory.forEach((l, i) => {
          const x = (i / 50) * 200;
          const y = 100 - (l / maxLoss) * 100;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    });

    runTrainBtn.addEventListener('click', () => {
      globalEvents.emit('autograd', { action: 'train_step' });
    });

    let isTraining = false;
    let epochTimer: any;
    runEpochBtn.addEventListener('click', () => {
      if (isTraining) {
        clearInterval(epochTimer);
        isTraining = false;
        runEpochBtn.textContent = 'Resume Training';
      } else {
        isTraining = true;
        runEpochBtn.textContent = 'Pause Training';
        let stepCount = 0;

        // 224. Async loop with set interval to avoid blocking UI
        // 235. Validate local execution (WASM logic executed synchronously locally)
        epochTimer = setInterval(() => {
          // 230. Dynamically update hyperparams
          const lr = parseFloat(lrSlider.value);
          const bs = parseInt(bsSlider.value, 10);
          lrLabel.textContent = `Learning Rate (${lr})`;
          bsLabel.textContent = `Batch Size (${bs})`;

          globalEvents.emit('autograd', { action: 'train_step', payload: { lr, bs } });
          stepCount++;
          if (stepCount >= 10) {
            clearInterval(epochTimer);
            isTraining = false;
            runEpochBtn.textContent = 'Train (10 Epochs)';
          }
        }, 550); // 500ms step execution time + 50ms buffer
      }
    });

    // GenAI Toggle
    const genaiSection = $create('div', { className: 'sidebar-section' });
    const genaiTitle = $create('h4', { textContent: 'GenAI / Agents' });
    const chatBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open LLM Interface',
    });

    // 606. Add an "Agent" tab to construct LLM-based autonomous workflows.
    const agentBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open Agent Workflow',
    });
    agentBtn.style.marginTop = '5px';

    // 335. Expose logits distributions as a real-time bar chart in the UI sidebar
    const logitsCanvas = $create<HTMLCanvasElement>('canvas', {
      attributes: {
        width: '200',
        height: '80',
        style:
          'margin-top: 10px; background: var(--color-background-primary); border: 1px solid var(--color-background-border); border-radius: 4px;',
      },
    });

    genaiSection.appendChild(genaiTitle);
    genaiSection.appendChild(chatBtn);
    genaiSection.appendChild(agentBtn);
    genaiSection.appendChild(logitsCanvas);
    this.container.appendChild(genaiSection);

    globalEvents.on('logitsUpdate', (data: { topTokens: number[]; topProbs: number[] }) => {
      const ctx = logitsCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 200, 80);
        ctx.fillStyle = 'var(--color-primary)';

        const barW = 200 / data.topProbs.length;
        data.topProbs.forEach((prob, i) => {
          const h = Math.max(2, prob * 80);
          ctx.fillRect(i * barW, 80 - h, barW - 1, h);
        });
      }
    });

    chatBtn.addEventListener('click', () => {
      globalEvents.emit('toggleChat');
    });

    agentBtn.addEventListener('click', () => {
      globalEvents.emit('toggleAgent');
    });

    // Security & Privacy Toggle
    const secSection = $create('div', { className: 'sidebar-section' });
    const secTitle = $create('h4', { textContent: 'Security / Privacy' });
    const obfBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Obfuscate Topology',
    });
    const encBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Encrypt Weights (AES-GCM)',
    });
    const decBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Decrypt Weights',
    });

    secSection.appendChild(secTitle);
    secSection.appendChild(obfBtn);
    secSection.appendChild(encBtn);
    secSection.appendChild(decBtn);
    this.container.appendChild(secSection);

    obfBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'obfuscate'));
    encBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'encrypt'));
    decBtn.addEventListener('click', () => globalEvents.emit('securityAction', 'decrypt'));

    // Sensors & Pipelines Toggle
    const sensorsSection = $create('div', { className: 'sidebar-section' });
    const sensorsTitle = $create('h4', { textContent: 'Sensors & Pipelines' });
    const visionBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open Vision Pipeline',
    });
    const audioBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open Audio Pipeline',
    });

    sensorsSection.appendChild(sensorsTitle);
    sensorsSection.appendChild(visionBtn);
    sensorsSection.appendChild(audioBtn);
    this.container.appendChild(sensorsSection);

    visionBtn.addEventListener('click', () => {
      globalEvents.emit('toggleVision');
    });

    audioBtn.addEventListener('click', () => {
      globalEvents.emit('toggleAudio');
    });

    // Vault Toggle
    const vaultSection = $create('div', { className: 'sidebar-section' });
    const vaultTitle = $create('h4', { textContent: 'IndexedDB Vault' });
    const vaultBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open Vault Manager',
    });
    vaultSection.appendChild(vaultTitle);
    vaultSection.appendChild(vaultBtn);
    this.container.appendChild(vaultSection);

    vaultBtn.addEventListener('click', () => {
      globalEvents.emit('toggleVault');
    });

    // Swarm Toggle
    const p2pSection = $create('div', { className: 'sidebar-section' });
    const p2pTitle = $create('h4', { textContent: 'WebRTC Swarm' });
    const swarmBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Open Swarm Panel',
    });
    p2pSection.appendChild(p2pTitle);
    p2pSection.appendChild(swarmBtn);
    this.container.appendChild(p2pSection);

    swarmBtn.addEventListener('click', () => {
      globalEvents.emit('toggleSwarm');
    });

    // WebNN Execution Provider
    const execSection = $create('div', { className: 'sidebar-section' });
    const execTitle = $create('h4', { textContent: 'Execution Provider' });

    // 253. Auto-select the fastest available backend
    const epSelect = $create<HTMLSelectElement>('select', {
      className: 'ide-select',
      innerHTML: `
        <option value="wasm">WASM (CPU)</option>
        <option value="webgpu">WebGPU</option>
      `,
    });

    const webnnStatus = $create('span', {
      className: 'badge ' + ('ml' in navigator ? 'success' : 'danger'),
      textContent: 'ml' in navigator ? 'WebNN Supported' : 'WebNN Not Supported',
      attributes: {
        style:
          'font-size: 0.7rem; margin-top: 5px; display: inline-block; padding: 2px 4px; border-radius: 4px;',
      },
    });

    // 265. Document the required browser flags
    if (!('ml' in navigator)) {
      const flagWarn = $create('p', {
        className: 'muted',
        innerHTML:
          'To enable WebNN, try launching Chrome with <br><code>--enable-features=WebMachineLearningNeuralNetwork</code>',
        attributes: { style: 'font-size: 0.7rem; margin-top: 5px; line-height: 1.2;' },
      });
      execSection.appendChild(flagWarn);
    }

    if ('ml' in navigator) {
      epSelect.innerHTML += `<option value="webnn">WebNN (NPU/GPU)</option>`;
      // Default to WebNN if available
      epSelect.value = 'webnn';
    } else if (navigator.gpu) {
      epSelect.value = 'webgpu';
    }

    const execBtn = $create('button', { className: 'action-btn', textContent: 'Run Inference' });

    execSection.appendChild(execTitle);
    execSection.appendChild(epSelect);
    execSection.appendChild(webnnStatus);
    execSection.appendChild(execBtn);
    this.container.appendChild(execSection);

    execBtn.addEventListener('click', () => {
      globalEvents.emit('executeProvider', epSelect.value);
    });

    // Benchmarks Toggle
    const benchSection = $create('div', { className: 'sidebar-section' });
    const benchTitle = $create('h4', { textContent: 'Micro-Benchmarks' });
    const runBenchBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Run Suite (1000 Samples)',
    });

    // 257. Plot backend comparison graphs
    const benchCanvas = $create<HTMLCanvasElement>('canvas', {
      attributes: {
        width: '200',
        height: '100',
        style:
          'margin-top: 10px; background: var(--color-background-primary); border: 1px solid var(--color-background-border); border-radius: 4px;',
      },
    });

    benchSection.appendChild(benchTitle);
    benchSection.appendChild(runBenchBtn);
    benchSection.appendChild(benchCanvas);
    this.container.appendChild(benchSection);

    globalEvents.on('benchmarkResults', (data: any) => {
      const ctx = benchCanvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, 200, 100);
        const barW = 50;

        // WASM (Blue)
        ctx.fillStyle = '#0d6efd';
        ctx.fillRect(20, 100 - (data.wasm || 10), barW, data.wasm || 10);

        // WebGPU (Green)
        ctx.fillStyle = '#198754';
        ctx.fillRect(80, 100 - (data.webgpu || 50), barW, data.webgpu || 50);

        // WebNN (Purple)
        ctx.fillStyle = '#6f42c1';
        ctx.fillRect(140, 100 - (data.webnn || 90), barW, data.webnn || 90);

        ctx.fillStyle = 'var(--color-foreground-muted)';
        ctx.font = '10px sans-serif';
        ctx.fillText('CPU', 30, 15);
        ctx.fillText('GPU', 90, 15);
        ctx.fillText('NPU', 150, 15);
      }
    });

    runBenchBtn.addEventListener('click', () => {
      globalEvents.emit('runBenchmark');
    });

    // Compilation Toggle
    const compileSection = $create('div', { className: 'sidebar-section' });
    const compileTitle = $create('h4', { textContent: 'AOT Compiler' });
    const wasmBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Compile to WASM',
    });
    const wgslBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Compile to WGSL',
    });
    const cppBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Compile to C++',
    });
    const cBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Compile to C99',
    });
    compileSection.appendChild(compileTitle);
    compileSection.appendChild(wasmBtn);
    compileSection.appendChild(wgslBtn);
    compileSection.appendChild(cppBtn);
    compileSection.appendChild(cBtn);
    this.container.appendChild(compileSection);

    wasmBtn.addEventListener('click', () => globalEvents.emit('compile', 'wasm'));
    wgslBtn.addEventListener('click', () => globalEvents.emit('compile', 'wgsl'));
    cppBtn.addEventListener('click', () => globalEvents.emit('compile', 'cpp'));
    cBtn.addEventListener('click', () => globalEvents.emit('compile', 'c'));

    // Source toggle
    const sourceSection = $create('div', { className: 'sidebar-section' });
    const sourceBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'View ONNXScript Editor',
    });
    const toggleGraphBtn = $create('button', {
      className: 'action-btn secondary',
      textContent: 'Toggle Graph Canvas',
    });
    sourceSection.appendChild(sourceBtn);
    sourceSection.appendChild(toggleGraphBtn);
    this.container.appendChild(sourceSection);

    searchBtn.addEventListener('click', () => {
      const term = searchInput.value.trim();
      if (term) {
        globalEvents.emit('searchNode', term);
      }
    });

    loadBtn.addEventListener('click', () => {
      if (this.fileInput.files && this.fileInput.files.length > 0) {
        globalEvents.emit('filesDropped', Array.from(this.fileInput.files));
      }
    });

    loadDirBtn.addEventListener('click', () => {
      if (dirInput.files && dirInput.files.length > 0) {
        globalEvents.emit('directoryDropped', Array.from(dirInput.files));
      }
    });

    toggleGraphBtn.addEventListener('click', () => {
      globalEvents.emit('toggleGraph');
    });

    sourceBtn.addEventListener('click', () => {
      globalEvents.emit('toggleEditor');
    });

    // 637. Implement a command palette (Cmd+K)
    document.addEventListener('keydown', (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        this.showCommandPalette();
      }
    });
  }

  private showCommandPalette(): void {
    const existing = document.getElementById('ide-cmd-palette');
    if (existing) {
      existing.remove();
      return; // toggle off
    }

    const overlay = $create('div', {
      id: 'ide-cmd-palette',
      attributes: {
        style:
          'position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(0,0,0,0.5); z-index: 9999; display: flex; align-items: flex-start; justify-content: center; padding-top: 15vh;',
      },
    });

    const modal = $create('div', {
      className: 'ide-chat-messages', // reuse styling
      attributes: {
        style:
          'width: 500px; max-width: 90vw; background: var(--color-background); border: 1px solid var(--color-background-border); border-radius: 8px; padding: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);',
      },
    });

    const input = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: {
        type: 'text',
        placeholder: 'Search commands...',
        style: 'width: 100%; font-size: 1.1rem; padding: 10px;',
      },
    });

    const list = $create('ul', {
      className: 'property-list',
      attributes: { style: 'margin-top: 10px; max-height: 300px; overflow-y: auto;' },
    });

    const commands = [
      { label: 'Toggle Graph Canvas', action: () => globalEvents.emit('toggleGraph') },
      { label: 'Toggle ONNXScript Editor', action: () => globalEvents.emit('toggleEditor') },
      { label: 'Toggle GenAI Chat', action: () => globalEvents.emit('toggleChat') },
      { label: 'Toggle Agent Workflow', action: () => globalEvents.emit('toggleAgent') },
      { label: 'Toggle Vault Manager', action: () => globalEvents.emit('toggleVault') },
      { label: 'Toggle Swarm Panel', action: () => globalEvents.emit('toggleSwarm') },
      { label: 'Toggle Vision Pipeline', action: () => globalEvents.emit('toggleVision') },
      { label: 'Toggle Audio Pipeline', action: () => globalEvents.emit('toggleAudio') },
      {
        label: 'Run Inference (Active Provider)',
        action: () => globalEvents.emit('executeProvider', 'wasm'),
      },
      { label: 'Run Micro-Benchmarks', action: () => globalEvents.emit('runBenchmark') },
    ];

    const renderList = (filter: string) => {
      list.innerHTML = '';
      commands
        .filter((c) => c.label.toLowerCase().includes(filter.toLowerCase()))
        .forEach((c) => {
          const li = $create('li', {
            textContent: c.label,
            attributes: { style: 'padding: 8px; cursor: pointer; border-radius: 4px;' },
          });
          li.addEventListener(
            'mouseenter',
            () => (li.style.background = 'var(--color-background-secondary)'),
          );
          li.addEventListener('mouseleave', () => (li.style.background = 'transparent'));
          li.addEventListener('click', () => {
            c.action();
            overlay.remove();
          });
          list.appendChild(li);
        });
    };

    renderList('');

    input.addEventListener('input', () => renderList(input.value));
    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) overlay.remove();
    });

    modal.appendChild(input);
    modal.appendChild(list);
    overlay.appendChild(modal);
    document.body.appendChild(overlay);

    input.focus();
  }

  mount(): void {}
}
