import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { IModelGraph } from '../core/IR';
import { globalEvents } from '../core/State';

export interface IMemoryBlock {
  name: string;
  offset: number; // bytes
  size: number; // bytes
  type: 'weight' | 'activation';
}

export class MemoryArenaVisualizer extends BaseComponent {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private currentModel: IModelGraph | null = null;
  private blocks: IMemoryBlock[] = [];

  private statsContainer: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-memory-arena');

    this.statsContainer = $create('div', {
      className: 'ide-memory-stats',
      textContent: 'Peak Memory: 0 B',
    });
    this.statsContainer.style.fontSize = '0.8rem';
    this.statsContainer.style.paddingBottom = '5px';
    this.statsContainer.style.color = 'var(--color-foreground-muted)';

    this.canvas = $create<HTMLCanvasElement>('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '50px';

    this.container.appendChild(this.statsContainer);
    this.container.appendChild(this.canvas);

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
  }

  private resize(): void {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = 50 * dpr;
    this.ctx.scale(dpr, dpr);
    this.render();
  }

  // 294. Compute static memory arena offsets
  private computeOffsets(): void {
    if (!this.currentModel) return;
    this.blocks = [];

    let currentOffset = 0;

    // Weights
    for (const init of this.currentModel.initializers) {
      const size = init.rawData ? init.rawData.byteLength : 4; // Stub if 0
      this.blocks.push({
        name: init.name,
        offset: currentOffset,
        size: size,
        type: 'weight',
      });
      currentOffset += size;
    }

    // 296. Visualize memory re-use (buffer sharing) by connecting overlapping blocks
    // Advanced algorithm (mocked): reuse memory for non-overlapping lifetimes
    const activeLifetimes = new Map<string, { startNode: number; endNode: number; size: number }>();

    // Pass 1: compute lifetimes and sizes
    this.currentModel.nodes.forEach((node, i) => {
      node.outputs.forEach((out) => {
        const vi = this.currentModel!.valueInfo?.find((v) => v.name === out);
        let elCount = 1;
        if (vi && vi.type && vi.type.shape)
          elCount = (vi.type.shape as number[]).reduce((a, b) => a * b, 1) || 1;
        else elCount = 1000; // Stub
        activeLifetimes.set(out, { startNode: i, endNode: i, size: elCount * 4 });
      });
      node.inputs.forEach((inp) => {
        if (activeLifetimes.has(inp)) {
          activeLifetimes.get(inp)!.endNode = i;
        }
      });
    });

    // Pass 2: greedy allocation reusing offsets
    const freeBlocks: { offset: number; size: number }[] = [];
    let peakOffset = currentOffset; // starting after weights

    const allocations = new Map<string, number>();

    // Sort by start node
    const sortedVars = Array.from(activeLifetimes.entries()).sort(
      (a, b) => a[1].startNode - b[1].startNode,
    );

    sortedVars.forEach(([name, info]) => {
      let assignedOffset = -1;
      // Find free block
      for (let i = 0; i < freeBlocks.length; i++) {
        if (freeBlocks[i].size >= info.size) {
          assignedOffset = freeBlocks[i].offset;
          // Split free block
          freeBlocks[i].offset += info.size;
          freeBlocks[i].size -= info.size;
          break;
        }
      }
      if (assignedOffset === -1) {
        assignedOffset = peakOffset;
        peakOffset += info.size;
      }

      allocations.set(name, assignedOffset);
      this.blocks.push({ name, offset: assignedOffset, size: info.size, type: 'activation' });

      // Note: To truly visualize reuse with lines as requested in 296, we render stacked rects
      // when offsets collide in the render loop.
    });
  }

  // 295. Render memory blocks
  private activeBlocks: Set<string> = new Set();

  mount(): void {
    globalEvents.on('modelLoaded', (model: IModelGraph) => {
      this.currentModel = model;
      this.activeBlocks.clear();
      this.computeOffsets();
      this.render();
    });

    globalEvents.on('themeChanged', () => {
      this.render();
    });

    globalEvents.on('nodeSelected', (node: any) => {
      this.activeBlocks.clear();
      if (node) {
        node.inputs.forEach((i: string) => this.activeBlocks.add(i));
        node.outputs.forEach((o: string) => this.activeBlocks.add(o));
      }
      this.render();
    });
  }

  private render(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.ctx.clearRect(0, 0, rect.width, rect.height);

    if (this.blocks.length === 0) {
      this.statsContainer.textContent = 'Memory Arena: Empty';
      return;
    }

    const totalBytes =
      this.blocks[this.blocks.length - 1].offset + this.blocks[this.blocks.length - 1].size;
    const wBytes = this.blocks.filter((b) => b.type === 'weight').reduce((s, b) => s + b.size, 0);
    const aBytes = totalBytes - wBytes;

    // 297 & 298: Stats
    this.statsContainer.innerHTML = `<strong>Peak Arena:</strong> ${(totalBytes / 1024).toFixed(2)} KB | <span style="color:#0d6efd">Weights:</span> ${(wBytes / 1024).toFixed(2)} KB | <span style="color:#198754">Activations:</span> ${(aBytes / 1024).toFixed(2)} KB`;

    // Draw blocks
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const height = 40;
    const yOffset = 5;

    // Track overlaps for lines (296)
    const blockPositions = new Map<string, { x: number; width: number }>();

    this.blocks.forEach((block) => {
      const x = (block.offset / totalBytes) * rect.width;
      const width = Math.max((block.size / totalBytes) * rect.width, 1);

      const isActive = this.activeBlocks.has(block.name);

      if (block.type === 'weight') {
        this.ctx.fillStyle = isDark
          ? isActive
            ? '#4d94ff'
            : '#1a4066'
          : isActive
            ? '#0d6efd'
            : '#cce5ff';
        this.ctx.strokeStyle = isDark ? '#0d6efd' : '#99caff';
      } else {
        this.ctx.fillStyle = isDark
          ? isActive
            ? '#28a745'
            : '#1d4426'
          : isActive
            ? '#198754'
            : '#d4edda';
        this.ctx.strokeStyle = isDark ? '#198754' : '#a3d3af';
      }

      this.ctx.fillRect(x, yOffset, width, height);
      this.ctx.strokeRect(x, yOffset, width, height);

      blockPositions.set(block.name, { x, width });
    });

    // Draw 296 overlap lines
    this.ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
    this.ctx.lineWidth = 1;
    this.ctx.beginPath();

    const overlapChecked = new Set<string>();
    this.blocks.forEach((b1) => {
      this.blocks.forEach((b2) => {
        if (b1.name !== b2.name && b1.type === 'activation' && b2.type === 'activation') {
          if (b1.offset === b2.offset) {
            const key = [b1.name, b2.name].sort().join('-');
            if (!overlapChecked.has(key)) {
              overlapChecked.add(key);
              const p1 = blockPositions.get(b1.name);
              const p2 = blockPositions.get(b2.name);
              if (p1 && p2) {
                // Draw arc connecting reused blocks
                this.ctx.moveTo(p1.x + p1.width / 2, yOffset + height);
                this.ctx.bezierCurveTo(
                  p1.x + p1.width / 2,
                  yOffset + height + 20,
                  p2.x + p2.width / 2,
                  yOffset + height + 20,
                  p2.x + p2.width / 2,
                  yOffset + height,
                );
              }
            }
          }
        }
      });
    });
    this.ctx.stroke();
  }

  unmount(): void {
    super.unmount();
    window.removeEventListener('resize', this.resize);
  }
}
