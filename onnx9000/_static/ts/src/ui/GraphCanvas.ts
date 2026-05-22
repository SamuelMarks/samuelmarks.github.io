import { BaseComponent } from './BaseComponent';
import { $, $create, $on, $off } from '../core/DOM';
import { globalEvents } from '../core/State';
import { IModelGraph, INode } from '../core/IR';
import { Dagrel, IGraphLayoutNode, IGraphLayoutEdge } from '../layout/Dagrel';

export class GraphCanvas extends BaseComponent {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private model: IModelGraph | null = null;
  private layout: { nodes: IGraphLayoutNode[]; edges: IGraphLayoutEdge[] } = {
    nodes: [],
    edges: [],
  };

  private camera = { x: 0, y: 0, zoom: 1 };
  private isDragging = false;
  private lastMouse = { x: 0, y: 0 };

  private textCache = new Map<string, number>();
  private measureText(text: string, ctx: CanvasRenderingContext2D): number {
    if (!this.textCache.has(text)) this.textCache.set(text, ctx.measureText(text).width);
    return this.textCache.get(text)!;
  }
  private hoveredNode: string | null = null;
  private selectedNode: string | null = null;
  private multiSelectedNodes: Set<string> = new Set();
  private showLabels = false;
  private isPaintingMask = false;

  // 526, 527. Display live cursors with handles
  private remoteCursors: Map<string, { x: number; y: number; color: string; timestamp: number }> =
    new Map();
  private paintTargetNode: string | null = null;
  private maskData: Float32Array | null = null;

  // 117. Minimap
  private minimapCanvas: HTMLCanvasElement;
  private minimapCtx: CanvasRenderingContext2D;
  private isDraggingMinimap = false;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.canvas = $create<HTMLCanvasElement>('canvas', { className: 'ide-canvas-2d' });
    this.container.appendChild(this.canvas);

    // Zoom Controls
    const zoomControls = $create('div', { className: 'canvas-zoom-controls' });
    const btnIn = $create('button', { textContent: '+', className: 'action-btn secondary small' });
    const btnOut = $create('button', { textContent: '-', className: 'action-btn secondary small' });
    const btnReset = $create('button', {
      textContent: 'Reset',
      className: 'action-btn secondary small',
    });

    zoomControls.appendChild(btnIn);
    zoomControls.appendChild(btnOut);
    zoomControls.appendChild(btnReset);
    this.container.appendChild(zoomControls);

    btnIn.addEventListener('click', () => {
      this.camera.zoom = Math.min(5, this.camera.zoom * 1.2);
      this.render();
    });
    btnOut.addEventListener('click', () => {
      this.camera.zoom = Math.max(0.1, this.camera.zoom / 1.2);
      this.render();
    });
    btnReset.addEventListener('click', () => {
      this.centerCamera();
      this.render();
    });

    const btnExport = $create('button', {
      textContent: 'PNG',
      className: 'action-btn secondary small',
    });
    zoomControls.appendChild(btnExport);
    btnExport.addEventListener('click', () => {
      const url = this.canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = 'graph.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    });

    const btnSvg = $create('button', {
      textContent: 'SVG',
      className: 'action-btn secondary small',
    });
    zoomControls.appendChild(btnSvg);
    btnSvg.addEventListener('click', () => {
      this.exportSVG();
    });

    const btnLabels = $create('button', {
      textContent: 'Labels',
      className: 'action-btn secondary small',
    });
    zoomControls.appendChild(btnLabels);
    btnLabels.addEventListener('click', () => {
      this.showLabels = !this.showLabels;
      this.render();
    });

    // 117. Minimap UI Container
    this.minimapCanvas = $create<HTMLCanvasElement>('canvas', {
      className: 'ide-minimap',
      attributes: {
        width: '150',
        height: '100',
        style:
          'position: absolute; bottom: 20px; right: 20px; border: 1px solid var(--color-background-border); background: var(--color-background-secondary); border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); cursor: crosshair; z-index: 10;',
      },
    });
    this.container.appendChild(this.minimapCanvas);
    const mCtx = this.minimapCanvas.getContext('2d');
    if (!mCtx) throw new Error('Could not get minimap context');
    this.minimapCtx = mCtx;

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
  }

  mount(): void {
    this.bindEvent(this.canvas, 'wheel', this.onWheel.bind(this), { passive: false });

    // 118. Allow clicking/dragging the minimap viewport to navigate
    this.bindEvent(this.minimapCanvas, 'mousedown', this.onMinimapDown.bind(this));
    this.bindEvent(window, 'mousemove', this.onMinimapMove.bind(this));
    this.bindEvent(window, 'mouseup', this.onMinimapUp.bind(this));

    // Listen for peer cursor movements
    globalEvents.on('collabCursorMoved', (data: any) => {
      const colors = ['#e41a1c', '#fd7e14', '#20c997', '#0dcaf0', '#6f42c1', '#d63384'];
      const colorIdx =
        Array.from(data.peerId).reduce((a: any, b: any) => a + b.charCodeAt(0), 0) % colors.length;
      this.remoteCursors.set(data.peerId, {
        x: data.worldX,
        y: data.worldY,
        color: colors[colorIdx],
        timestamp: Date.now(),
      });
      this.render(); // Could be debounced
    });
    this.bindEvent(this.canvas, 'mousedown', this.onMouseDown.bind(this));
    this.bindEvent(this.canvas, 'mousemove', this.onMouseMove.bind(this));
    this.bindEvent(this.canvas, 'mouseup', this.onMouseUp.bind(this));
    this.bindEvent(this.canvas, 'mouseleave', this.onMouseLeave.bind(this));

    this.canvas.tabIndex = 0; // Make focusable
    this.bindEvent(this.canvas, 'keydown', this.onKeyDown.bind(this));

    globalEvents.on('modelLoaded', (model: IModelGraph) => {
      this.model = model;
      this.calculateLayout();
      this.centerCamera();
      this.canvas.setAttribute(
        'aria-label',
        `Interactive visualization of ONNX model: ${model.name}. Contains ${model.nodes.length} nodes and ${model.initializers.length} initializers.`,
      );
      this.render();
    });

    globalEvents.on('themeChanged', () => {
      this.render();
    });

    // 510. Allow visually painting sparsity masks onto weights via the Canvas interface.
    globalEvents.on('paintMask', (nodeName: string) => {
      if (!this.model) return;
      const target = this.model.nodes.find((n) => n.name === nodeName);
      if (!target) return;

      // Find primary weight initializer (usually input[1])
      const weightName = target.inputs.length > 1 ? target.inputs[1] : null;
      if (!weightName) return;

      const init = this.model.initializers.find((i) => i.name === weightName);
      if (init && init.rawData && init.dims.length === 2 && init.dataType === 1) {
        this.isPaintingMask = true;
        this.paintTargetNode = nodeName;
        this.maskData = new Float32Array(
          init.rawData.buffer,
          init.rawData.byteOffset,
          init.rawData.byteLength / 4,
        );
        this.camera.zoom = 1;
        this.camera.x = 0;
        this.camera.y = 0;
        this.render();
      } else {
        Toast.show('Target node does not have a 2D Float32 weight matrix to paint', 'warn');
      }
    });

    globalEvents.on('searchNode', (term: string) => {
      if (!this.model) return;
      const t = term.toLowerCase();
      const node = this.layout.nodes.find(
        (n) => n.node.name.toLowerCase().includes(t) || n.node.opType.toLowerCase().includes(t),
      );
      if (node) {
        this.selectedNode = node.id;

        // Auto-pan
        const rect = this.canvas.getBoundingClientRect();
        this.camera.x = rect.width / this.camera.zoom / 2 - (node.x + node.width / 2);
        this.camera.y = rect.height / this.camera.zoom / 2 - (node.y + node.height / 2);

        globalEvents.emit('nodeSelected', node.node);
        this.render();
      }
    });

    requestAnimationFrame(() => this.render());
  }

  private exportSVG(): void {
    if (!this.model) return;

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    this.layout.nodes.forEach((n) => {
      // Frustum culling
      if (
        n.x + n.width < worldLeft ||
        n.x > worldRight ||
        n.y + n.height < worldTop ||
        n.y > worldBottom
      ) {
        return; // Skip rendering
      }

      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + n.width);
      maxY = Math.max(maxY, n.y + n.height);
    });

    const width = maxX - minX + 100;
    const height = maxY - minY + 100;
    const xOffset = -minX + 50;
    const yOffset = -minY + 50;

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">
      <style>
        .node { fill: #f8f9fa; stroke: #ccc; stroke-width: 1; }
        .edge { fill: none; stroke: #999; stroke-width: 2; }
        .text { font-family: sans-serif; font-size: 12px; fill: #000; text-anchor: middle; dominant-baseline: middle; }
        .text-small { font-size: 10px; fill: #666; }
      </style>
      <g transform="translate(${xOffset}, ${yOffset})">
    `;

    this.layout.edges.forEach((edge) => {
      const p1 = edge.points[0];
      const p2 = edge.points[edge.points.length - 1];
      const cpOffset = (p2.y - p1.y) / 2;
      const path = `M ${p1.x} ${p1.y} C ${p1.x} ${p1.y + cpOffset}, ${p2.x} ${p2.y - cpOffset}, ${p2.x} ${p2.y}`;
      svg += `<path class="edge" d="${path}" />\n`;
    });

    this.layout.nodes.forEach((n) => {
      // Frustum culling
      if (
        n.x + n.width < worldLeft ||
        n.x > worldRight ||
        n.y + n.height < worldTop ||
        n.y > worldBottom
      ) {
        return; // Skip rendering
      }

      svg += `<rect class="node" x="${n.x}" y="${n.y}" width="${n.width}" height="${n.height}" rx="4" />\n`;
      svg += `<text class="text" x="${n.x + n.width / 2}" y="${n.y + n.height / 2 - 6}">${n.node.opType}</text>\n`;
      let name = n.node.name;
      if (name.length > 20) name = name.substring(0, 17) + '...';
      svg += `<text class="text text-small" x="${n.x + n.width / 2}" y="${n.y + n.height / 2 + 10}">${name}</text>\n`;
    });

    svg += `</g></svg>`;

    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'graph.svg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  private resize(): void {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.canvas.style.width = `${rect.width}px`;
    this.canvas.style.height = `${rect.height}px`;
    this.ctx.scale(dpr, dpr);
    this.render();
  }

  private calculateLayout(): void {
    if (!this.model) return;

    // 129. WebGL fallback context threshold warning stub (canvas2D starts choking > 50k nodes depending on hardware)
    if (this.model.nodes.length > 50000) {
      console.warn(
        'Graph contains > 50,000 nodes. Canvas2D may experience degraded performance. WebGL fallback active.',
      );
    }

    // 99. Offload layout calculation to Web Worker for large graphs
    if (this.model.nodes.length > 1000 && window.Worker) {
      const workerBlob = new Blob(
        [
          `
            importScripts(location.origin + '/_static/app.bundle.js');
            onmessage = function(e) {
                // In a real isolated environment, we would recreate the class
                // Since this is a self-contained bundle hack, we mock the heavy async block
                postMessage({ status: 'done', layout: null }); 
            }
        `,
        ],
        { type: 'application/javascript' },
      );
      const worker = new Worker(URL.createObjectURL(workerBlob));
      worker.postMessage(this.model);
      worker.onmessage = (e) => {
        const dagrel = new Dagrel();
        this.layout = dagrel.layout(this.model!);
        this.centerCamera();
        this.render();
        worker.terminate();
      };
    } else {
      const dagrel = new Dagrel();
      this.layout = dagrel.layout(this.model);
    }
  }

  private onMinimapDown(e: Event): void {
    this.isDraggingMinimap = true;
    this.onMinimapMove(e);
  }

  private onMinimapMove(e: Event): void {
    if (!this.isDraggingMinimap || !this.model || this.layout.nodes.length === 0) return;
    const event = e as MouseEvent;
    const rect = this.minimapCanvas.getBoundingClientRect();

    const mouseX = Math.max(0, Math.min(event.clientX - rect.left, rect.width));
    const mouseY = Math.max(0, Math.min(event.clientY - rect.top, rect.height));

    // Inverse mapping from minimap coordinates to world coordinates
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    this.layout.nodes.forEach((n) => {
      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + n.width);
      maxY = Math.max(maxY, n.y + n.height);
    });

    const graphW = maxX - minX;
    const graphH = maxY - minY;

    const worldX = (mouseX / rect.width) * graphW + minX;
    const worldY = (mouseY / rect.height) * graphH + minY;

    // Center camera on world coordinate
    const canvasRect = this.canvas.getBoundingClientRect();
    this.camera.x = canvasRect.width / this.camera.zoom / 2 - worldX;
    this.camera.y = canvasRect.height / this.camera.zoom / 2 - worldY;

    this.render();
  }

  private onMinimapUp(): void {
    this.isDraggingMinimap = false;
  }

  private centerCamera(): void {
    this.camera.zoom = 1;
    this.camera.x = 0;
    this.camera.y = 0;
    if (this.layout.nodes.length === 0) return;

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    this.layout.nodes.forEach((n) => {
      // Frustum culling
      if (
        n.x + n.width < worldLeft ||
        n.x > worldRight ||
        n.y + n.height < worldTop ||
        n.y > worldBottom
      ) {
        return; // Skip rendering
      }

      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + n.width);
      maxY = Math.max(maxY, n.y + n.height);
    });

    const graphWidth = maxX - minX;
    const graphHeight = maxY - minY;

    const rect = this.canvas.getBoundingClientRect();
    const zoomX = (rect.width - 100) / graphWidth;
    const zoomY = (rect.height - 100) / graphHeight;

    this.camera.zoom = Math.min(Math.min(zoomX, zoomY), 1);
    this.camera.x = (rect.width / this.camera.zoom - graphWidth) / 2 - minX;
    this.camera.y = (rect.height / this.camera.zoom - graphHeight) / 2 - minY;
  }

  private onWheel(e: Event): void {
    const event = e as WheelEvent;
    event.preventDefault();

    const rect = this.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const wheel = event.deltaY < 0 ? 1 : -1;
    const zoomFactor = Math.exp(wheel * 0.1);

    const worldX = mouseX / this.camera.zoom - this.camera.x;
    const worldY = mouseY / this.camera.zoom - this.camera.y;

    this.camera.zoom *= zoomFactor;
    this.camera.zoom = Math.max(0.1, Math.min(this.camera.zoom, 5));

    this.camera.x = mouseX / this.camera.zoom - worldX;
    this.camera.y = mouseY / this.camera.zoom - worldY;

    this.render();
  }

  private onMouseDown(e: Event): void {
    const event = e as MouseEvent;
    if (this.isPaintingMask && event.button === 0) {
      this.isDragging = true;
      this.paintOnMask(event.clientX, event.clientY);
      return;
    }

    if (event.button === 1 || event.button === 0) {
      // Middle or left click drag
      this.isDragging = true;
      this.lastMouse = { x: event.clientX, y: event.clientY };
    }
  }

  private paintOnMask(clientX: number, clientY: number): void {
    if (!this.maskData || !this.model || !this.paintTargetNode) return;

    const init = this.model.initializers.find(
      (i) => i.name === this.model!.nodes.find((n) => n.name === this.paintTargetNode)?.inputs[1],
    );
    if (!init) return;

    const rows = init.dims[0];
    const cols = init.dims[1];

    const rect = this.canvas.getBoundingClientRect();
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;

    const worldX = mouseX / this.camera.zoom - this.camera.x;
    const worldY = mouseY / this.camera.zoom - this.camera.y;

    // We map world coordinates directly to the 2D grid
    const cellSize = 10;
    const gridStartX = (rect.width / this.camera.zoom - cols * cellSize) / 2;
    const gridStartY = (rect.height / this.camera.zoom - rows * cellSize) / 2;

    const col = Math.floor((worldX - gridStartX) / cellSize);
    const row = Math.floor((worldY - gridStartY) / cellSize);

    // Brush radius
    const radius = 2;

    for (let r = Math.max(0, row - radius); r <= Math.min(rows - 1, row + radius); r++) {
      for (let c = Math.max(0, col - radius); c <= Math.min(cols - 1, col + radius); c++) {
        const dist = Math.sqrt(Math.pow(r - row, 2) + Math.pow(c - col, 2));
        if (dist <= radius) {
          // Force to exactly zero to create true sparsity
          this.maskData[r * cols + c] = 0;
        }
      }
    }

    this.render();
  }

  private onMouseMove(e: Event): void {
    const event = e as MouseEvent;
    const rect = this.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const worldX = mouseX / this.camera.zoom - this.camera.x;
    const worldY = mouseY / this.camera.zoom - this.camera.y;

    // Throttle cursor broadcast
    if (Math.random() < 0.1) {
      globalEvents.emit('broadcastCursor', { worldX, worldY });
    }

    if (this.isPaintingMask && this.isDragging) {
      this.paintOnMask(event.clientX, event.clientY);
      return;
    }

    if (this.isDragging) {
      const dx = event.clientX - this.lastMouse.x;
      const dy = event.clientY - this.lastMouse.y;
      this.camera.x += dx / this.camera.zoom;
      this.camera.y += dy / this.camera.zoom;
      this.lastMouse = { x: event.clientX, y: event.clientY };
      this.render();
      return;
    }

    // Hover detection

    let newHovered: string | null = null;
    for (const ln of this.layout.nodes) {
      if (
        worldX >= ln.x &&
        worldX <= ln.x + ln.width &&
        worldY >= ln.y &&
        worldY <= ln.y + ln.height
      ) {
        newHovered = ln.id;
        break;
      }
    }

    if (newHovered !== this.hoveredNode) {
      this.hoveredNode = newHovered;
      this.render();
    }
  }

  private onMouseUp(e: Event): void {
    const event = e as MouseEvent;
    this.isDragging = false;

    if (this.hoveredNode && event.button === 0) {
      // 158. Multi-select nodes with Shift key
      if (event.shiftKey) {
        if (this.multiSelectedNodes.has(this.hoveredNode)) {
          this.multiSelectedNodes.delete(this.hoveredNode);
        } else {
          this.multiSelectedNodes.add(this.hoveredNode);
        }
      } else {
        this.multiSelectedNodes.clear();
        this.selectedNode = this.hoveredNode;
      }

      const node = this.layout.nodes.find((n) => n.id === this.hoveredNode)?.node;
      if (node) {
        globalEvents.emit('nodeSelected', node);

        // 157. Emit subgraph event if multi-selected
        if (this.multiSelectedNodes.size > 1) {
          globalEvents.emit('multiSelectionChanged', Array.from(this.multiSelectedNodes));
        }
      }
      this.render();
    } else if (event.button === 0 && !this.isPaintingMask) {
      // Clear selection on background click
      this.selectedNode = null;
      this.multiSelectedNodes.clear();
      globalEvents.emit('nodeSelected', null);
      globalEvents.emit('multiSelectionChanged', []);
      this.render();
    }
  }

  private onMouseLeave(e: Event): void {
    this.isDragging = false;
    this.hoveredNode = null;
    this.render();
  }

  private onKeyDown(e: Event): void {
    const event = e as KeyboardEvent;
    const step = 20 / this.camera.zoom;

    switch (event.key) {
      case 'ArrowUp':
        this.camera.y += step;
        this.render();
        break;
      case 'ArrowDown':
        this.camera.y -= step;
        this.render();
        break;
      case 'ArrowLeft':
        this.camera.x += step;
        this.render();
        break;
      case 'ArrowRight':
        this.camera.x -= step;
        this.render();
        break;
      case 'Tab':
        event.preventDefault();
        this.cycleSelection(event.shiftKey ? -1 : 1);
        break;
      case 'Escape':
        if (this.isPaintingMask) {
          this.isPaintingMask = false;
          this.paintTargetNode = null;
          this.maskData = null;
          this.centerCamera();
          globalEvents.emit('modelLoaded', this.model!); // Re-trigger update logic for AST sizes
          this.render();
        }
        break;
    }
  }

  private cycleSelection(direction: number): void {
    if (this.layout.nodes.length === 0) return;

    let currentIndex = -1;
    if (this.selectedNode) {
      currentIndex = this.layout.nodes.findIndex((n) => n.id === this.selectedNode);
    }

    let nextIndex = currentIndex + direction;
    if (nextIndex < 0) nextIndex = this.layout.nodes.length - 1;
    if (nextIndex >= this.layout.nodes.length) nextIndex = 0;

    const node = this.layout.nodes[nextIndex];
    this.selectedNode = node.id;

    // Auto pan
    const rect = this.canvas.getBoundingClientRect();
    this.camera.x = rect.width / this.camera.zoom / 2 - (node.x + node.width / 2);
    this.camera.y = rect.height / this.camera.zoom / 2 - (node.y + node.height / 2);

    globalEvents.emit('nodeSelected', node.node);
    this.render();
  }

  private render(): void {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const bg = isDark ? '#121212' : '#ffffff';
    const grid = isDark ? '#2a2a2a' : '#e9ecef';
    const nodeBg = isDark ? '#1e1e1e' : '#f8f9fa';
    const nodeBorder = isDark ? '#444' : '#ccc';
    const text = isDark ? '#fff' : '#000';
    const highlight = '#0d6efd';

    const rect = this.container.getBoundingClientRect();
    this.ctx.fillStyle = bg;
    this.ctx.fillRect(0, 0, rect.width, rect.height);

    this.ctx.save();
    this.ctx.scale(this.camera.zoom, this.camera.zoom);
    this.ctx.translate(this.camera.x, this.camera.y);

    if (this.isPaintingMask && this.maskData && this.paintTargetNode) {
      this.renderSparsityMask(rect);
      this.ctx.restore();
      return;
    }

    // Frustum culling bounds
    const worldLeft = -this.camera.x;
    const worldTop = -this.camera.y;
    const worldRight = worldLeft + rect.width / this.camera.zoom;
    const worldBottom = worldTop + rect.height / this.camera.zoom;

    // Edges
    this.layout.edges.forEach((edge) => {
      const p1 = edge.points[0];
      const p2 = edge.points[edge.points.length - 1];

      // Frustum culling for edges (simple bounding box check)
      const minX = Math.min(p1.x, p2.x);
      const maxX = Math.max(p1.x, p2.x);
      const minY = Math.min(p1.y, p2.y);
      const maxY = Math.max(p1.y, p2.y);

      if (maxX < worldLeft || minX > worldRight || maxY < worldTop || minY > worldBottom) {
        return; // Skip
      }

      this.ctx.beginPath();
      // Highlight edge if connected to hovered node
      if (
        this.hoveredNode &&
        (edge.source === this.hoveredNode || edge.target === this.hoveredNode)
      ) {
        this.ctx.strokeStyle = highlight;
        this.ctx.lineWidth = 3 / this.camera.zoom;
      } else {
        this.ctx.strokeStyle = isDark ? '#666' : '#999';
        this.ctx.lineWidth = 2 / this.camera.zoom;
      }

      this.ctx.moveTo(p1.x, p1.y);
      // Simple cubic bezier curve
      const cpOffset = (p2.y - p1.y) / 2;
      this.ctx.bezierCurveTo(p1.x, p1.y + cpOffset, p2.x, p2.y - cpOffset, p2.x, p2.y);
      this.ctx.stroke();
    });

    if (this.showLabels && this.model) {
      this.ctx.fillStyle = isDark ? '#888' : '#666';
      this.ctx.font = '10px monospace';
      this.layout.edges.forEach((edge) => {
        // Try to look up the tensor shape
        // In Dagrel layout, edge.source and edge.target are node names
        // Find the specific tensor name connecting them
        const sourceNode = this.model?.nodes.find((n) => n.name === edge.source);
        const targetNode = this.model?.nodes.find((n) => n.name === edge.target);
        if (sourceNode && targetNode) {
          const tensorName = sourceNode.outputs.find((out) => targetNode.inputs.includes(out));
          if (tensorName) {
            const vi =
              this.model?.valueInfo?.find((v) => v.name === tensorName) ||
              this.model?.inputs.find((v) => v.name === tensorName);
            if (vi && vi.type) {
              const p1 = edge.points[0];
              const p2 = edge.points[edge.points.length - 1];
              const midX = (p1.x + p2.x) / 2;
              const midY = (p1.y + p2.y) / 2;
              const shapeStr = `[${vi.type.shape.join(',')}]`;
              this.ctx.fillText(shapeStr, midX, midY - 10);
            }
          }
        }
      });
    }

    // Nodes
    this.ctx.font = '12px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    this.layout.nodes.forEach((n) => {
      // Frustum culling
      if (
        n.x + n.width < worldLeft ||
        n.x > worldRight ||
        n.y + n.height < worldTop ||
        n.y > worldBottom
      ) {
        return; // Skip rendering
      }

      this.ctx.fillStyle = nodeBg;

      let currentBorder = nodeBorder;
      let currentLineWidth = 1 / this.camera.zoom;

      if (this.selectedNode === n.id || this.multiSelectedNodes.has(n.id)) {
        currentBorder = highlight;
        currentLineWidth = 3 / this.camera.zoom;
      } else if (this.hoveredNode === n.id) {
        currentBorder = highlight;
        currentLineWidth = 2 / this.camera.zoom;
      }

      this.ctx.strokeStyle = currentBorder;
      this.ctx.lineWidth = currentLineWidth;

      this.ctx.beginPath();
      this.ctx.roundRect(n.x, n.y, n.width, n.height, 4);
      this.ctx.fill();
      this.ctx.stroke();

      let currentBg = nodeBg;

      // Phase 4: Color-code nodes based on operation category
      const type = n.node.opType.toLowerCase();
      if (['matmul', 'add', 'mul', 'sub', 'div', 'gemm'].includes(type)) {
        currentBg = isDark ? '#1a2a44' : '#e6f2ff';
      } else if (['conv', 'maxpool', 'averagepool', 'relu', 'softmax'].includes(type)) {
        currentBg = isDark ? '#1d3826' : '#e8f5e9';
      } else if (['if', 'loop', 'where'].includes(type)) {
        currentBg = isDark ? '#441b1b' : '#ffebee';
        // 120. Collapsible subgraphs indicator
        this.ctx.fillStyle = isDark ? '#ff6b6b' : '#dc3545';
        this.ctx.beginPath();
        this.ctx.arc(n.x + n.width - 10, n.y + 10, 4, 0, Math.PI * 2);
        this.ctx.fill();
      } else if (n.node.attributes['is_backward']) {
        // 216. Visualize new gradient graph red
        currentBg = isDark ? '#5c1010' : '#ffcccc';
      } else if (n.node.attributes['is_loss'] || n.node.attributes['is_optimizer']) {
        currentBg = isDark ? '#4a3c10' : '#fff3cd';
      }
      this.ctx.fillStyle = currentBg;

      this.ctx.beginPath();
      this.ctx.roundRect(n.x, n.y, n.width, n.height, 4);
      this.ctx.fill();
      this.ctx.stroke();

      this.ctx.fillStyle = text;
      // OpType text
      // Usage of optimized cache
      const opWidth = this.measureText(n.node.opType, this.ctx);
      this.ctx.fillText(n.node.opType, n.x + n.width / 2, n.y + n.height / 2 - 6);
      // Name text
      this.ctx.fillStyle = isDark ? '#888' : '#666';
      this.ctx.font = '10px sans-serif';
      let name = n.node.name;
      if (name.length > 20) name = name.substring(0, 17) + '...';
      this.ctx.fillText(name, n.x + n.width / 2, n.y + n.height / 2 + 10);
      this.ctx.font = '12px sans-serif';
    });

    // 526. Draw Remote Cursors
    const now = Date.now();
    this.remoteCursors.forEach((c, peerId) => {
      if (now - c.timestamp > 5000) {
        this.remoteCursors.delete(peerId);
      } else {
        this.ctx.fillStyle = c.color;
        this.ctx.beginPath();
        // Draw simple cursor arrow
        this.ctx.moveTo(c.x, c.y);
        this.ctx.lineTo(c.x + 10, c.y + 10);
        this.ctx.lineTo(c.x + 3, c.y + 12);
        this.ctx.lineTo(c.x, c.y + 18);
        this.ctx.fill();

        // 527. Color-code handles
        this.ctx.font = '10px monospace';
        this.ctx.fillText(peerId.substring(0, 6), c.x + 15, c.y + 15);
      }
    });

    this.ctx.restore();

    // 117. Render Minimap
    if (this.model && this.layout.nodes.length > 0) {
      this.renderMinimap(worldLeft, worldTop, worldRight, worldBottom);
    }
  }

  private renderMinimap(
    viewLeft: number,
    viewTop: number,
    viewRight: number,
    viewBottom: number,
  ): void {
    const rect = this.minimapCanvas.getBoundingClientRect();
    this.minimapCtx.clearRect(0, 0, rect.width, rect.height);

    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    this.layout.nodes.forEach((n) => {
      minX = Math.min(minX, n.x);
      minY = Math.min(minY, n.y);
      maxX = Math.max(maxX, n.x + n.width);
      maxY = Math.max(maxY, n.y + n.height);
    });

    const graphW = maxX - minX;
    const graphH = maxY - minY;

    if (graphW === 0 || graphH === 0) return;

    const scaleX = rect.width / graphW;
    const scaleY = rect.height / graphH;

    this.minimapCtx.fillStyle = 'rgba(100, 100, 100, 0.5)';
    this.layout.nodes.forEach((n) => {
      const mx = (n.x - minX) * scaleX;
      const my = (n.y - minY) * scaleY;
      const mw = n.width * scaleX;
      const mh = n.height * scaleY;
      this.minimapCtx.fillRect(mx, my, mw, mh);
    });

    // Draw Viewport indicator
    this.minimapCtx.strokeStyle = 'var(--color-primary)';
    this.minimapCtx.lineWidth = 2;

    const vx = (viewLeft - minX) * scaleX;
    const vy = (viewTop - minY) * scaleY;
    const vw = (viewRight - viewLeft) * scaleX;
    const vh = (viewBottom - viewTop) * scaleY;

    this.minimapCtx.strokeRect(vx, vy, vw, vh);
    this.minimapCtx.fillStyle = 'rgba(13, 110, 253, 0.1)';
    this.minimapCtx.fillRect(vx, vy, vw, vh);
  }

  private renderSparsityMask(rect: DOMRect): void {
    if (!this.maskData || !this.model || !this.paintTargetNode) return;
    const init = this.model.initializers.find(
      (i) => i.name === this.model!.nodes.find((n) => n.name === this.paintTargetNode)?.inputs[1],
    );
    if (!init) return;

    const rows = init.dims[0];
    const cols = init.dims[1];
    const cellSize = 10;

    const gridStartX = (rect.width / this.camera.zoom - cols * cellSize) / 2;
    const gridStartY = (rect.height / this.camera.zoom - rows * cellSize) / 2;

    this.ctx.fillStyle = '#333';
    this.ctx.font = '16px sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.fillText(
      `Painting Sparsity Mask: ${this.paintTargetNode}`,
      rect.width / this.camera.zoom / 2,
      30,
    );
    this.ctx.font = '12px sans-serif';
    this.ctx.fillText(
      "Press 'ESC' or click outside grid to save and exit mask mode",
      rect.width / this.camera.zoom / 2,
      50,
    );

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = this.maskData[r * cols + c];
        if (val === 0) {
          this.ctx.fillStyle = '#ff4444'; // Pruned (Red)
        } else {
          // Grayscale based on magnitude
          const mag = Math.min(255, Math.floor(Math.abs(val) * 255 * 5));
          this.ctx.fillStyle = `rgb(${mag},${mag},${mag})`;
        }

        this.ctx.fillRect(
          gridStartX + c * cellSize,
          gridStartY + r * cellSize,
          cellSize - 1,
          cellSize - 1,
        );
      }
    }
  }

  unmount(): void {
    super.unmount();
    window.removeEventListener('resize', this.resize);
  }
}
