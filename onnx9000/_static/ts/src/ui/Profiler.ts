import { BaseComponent } from './BaseComponent';
import { $, $create, $on, $off } from '../core/DOM';
import { globalEvents } from '../core/State';

export interface IExecutionTrace {
  opName: string;
  duration: number; // in ms
  startTime: number;
}

export class Profiler extends BaseComponent {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private traces: IExecutionTrace[] = [];

  // Tooltip
  private tooltip: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-profiler-container');
    this.container.style.position = 'relative';

    this.canvas = $create<HTMLCanvasElement>('canvas', { className: 'ide-profiler-canvas' });
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100px';

    this.tooltip = $create('div', { className: 'ide-tooltip hidden' });
    this.tooltip.style.position = 'absolute';
    this.tooltip.style.pointerEvents = 'none';
    this.tooltip.style.background = 'var(--color-background-secondary)';
    this.tooltip.style.border = '1px solid var(--color-background-border)';
    this.tooltip.style.padding = '4px 8px';
    this.tooltip.style.fontSize = '0.8rem';
    this.tooltip.style.zIndex = '100';

    this.container.appendChild(this.canvas);
    this.container.appendChild(this.tooltip);

    // 516. Add a timeline trace viewer compatible with Chrome chrome://tracing
    // 517. Export .json profiling traces for deeper analysis
    const exportBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Export Chrome Trace (.json)',
      attributes: { style: 'position: absolute; right: 10px; top: 10px; z-index: 10;' },
    });
    this.container.appendChild(exportBtn);

    exportBtn.addEventListener('click', () => this.exportTrace());

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Could not get 2D context');
    this.ctx = ctx;

    this.resize = this.resize.bind(this);
    window.addEventListener('resize', this.resize);
    this.resize();
  }

  mount(): void {
    globalEvents.on('profilerData', (traces: IExecutionTrace[]) => {
      this.traces = traces;
      this.render();
    });

    globalEvents.on('themeChanged', () => {
      this.render();
    });

    this.bindEvent(this.canvas, 'mousemove', this.onMouseMove.bind(this));
    this.bindEvent(this.canvas, 'mouseleave', this.onMouseLeave.bind(this));
  }

  private resize(): void {
    const rect = this.container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = 100 * dpr;
    this.ctx.scale(dpr, dpr);
    this.render();
  }

  private getColorForOp(opName: string): string {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    const type = opName.toLowerCase();

    if (['matmul', 'add', 'mul', 'sub', 'div', 'gemm'].includes(type))
      return isDark ? '#1a2a44' : '#e6f2ff';
    if (['conv', 'maxpool', 'averagepool', 'relu', 'softmax'].includes(type))
      return isDark ? '#1d3826' : '#e8f5e9';
    if (['if', 'loop', 'where'].includes(type)) return isDark ? '#441b1b' : '#ffebee';
    return isDark ? '#333' : '#f0f0f0';
  }

  private getBorderColor(opName: string): string {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    return isDark ? '#555' : '#ccc';
  }

  private exportTrace(): void {
    if (this.traces.length === 0) return;

    // Build Chrome Tracing Format (Trace Event Format)
    const traceEvents = this.traces.map((t) => {
      return {
        name: t.opName,
        cat: 'Execution',
        ph: 'X', // Complete event
        ts: t.startTime * 1000, // microseconds
        dur: t.duration * 1000,
        pid: 1, // Main process
        tid: 1, // Main thread
        args: {},
      };
    });

    const payload = JSON.stringify({ traceEvents }, null, 2);
    const blob = new Blob([payload], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'onnx9000_trace.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  private render(): void {
    const rect = this.canvas.getBoundingClientRect();
    this.ctx.clearRect(0, 0, rect.width, rect.height);

    if (this.traces.length === 0) {
      this.ctx.fillStyle = 'var(--color-foreground-muted)';
      this.ctx.font = '12px sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText('No profiling data available', rect.width / 2, rect.height / 2);
      return;
    }

    const minTime = this.traces[0].startTime;
    const maxTime =
      this.traces[this.traces.length - 1].startTime + this.traces[this.traces.length - 1].duration;
    const totalDuration = maxTime - minTime;

    // Draw flame graph
    const height = 30;
    const yOffset = 35; // Center it a bit

    this.traces.forEach((trace) => {
      const x = ((trace.startTime - minTime) / totalDuration) * rect.width;
      const width = Math.max((trace.duration / totalDuration) * rect.width, 1); // Min 1px width

      this.ctx.fillStyle = this.getColorForOp(trace.opName);
      this.ctx.fillRect(x, yOffset, width, height);
      this.ctx.strokeStyle = this.getBorderColor(trace.opName);
      this.ctx.strokeRect(x, yOffset, width, height);
    });
  }

  private onMouseMove(e: Event): void {
    const event = e as MouseEvent;
    if (this.traces.length === 0) return;

    const rect = this.canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;

    const minTime = this.traces[0].startTime;
    const maxTime =
      this.traces[this.traces.length - 1].startTime + this.traces[this.traces.length - 1].duration;
    const totalDuration = maxTime - minTime;

    const yOffset = 35;
    const height = 30;

    let hoveredTrace: IExecutionTrace | null = null;

    if (mouseY >= yOffset && mouseY <= yOffset + height) {
      for (const trace of this.traces) {
        const x = ((trace.startTime - minTime) / totalDuration) * rect.width;
        const width = Math.max((trace.duration / totalDuration) * rect.width, 1);
        if (mouseX >= x && mouseX <= x + width) {
          hoveredTrace = trace;
          break;
        }
      }
    }

    if (hoveredTrace) {
      this.tooltip.classList.remove('hidden');
      this.tooltip.innerHTML = `<strong>${hoveredTrace.opName}</strong><br/>${(hoveredTrace.duration * 1000).toFixed(2)} µs`;

      // Position tooltip
      let ttX = mouseX + 10;
      let ttY = mouseY + 10;
      if (ttX + this.tooltip.offsetWidth > rect.width) ttX = mouseX - this.tooltip.offsetWidth - 10;

      this.tooltip.style.left = `${ttX}px`;
      this.tooltip.style.top = `${ttY}px`;
    } else {
      this.tooltip.classList.add('hidden');
    }
  }

  private onMouseLeave(): void {
    this.tooltip.classList.add('hidden');
  }

  unmount(): void {
    super.unmount();
    window.removeEventListener('resize', this.resize);
  }
}
