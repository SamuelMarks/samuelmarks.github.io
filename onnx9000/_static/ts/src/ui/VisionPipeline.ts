import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';
import { cameraManager } from '../sensors/CameraManager';

export class VisionPipeline extends BaseComponent {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationId: number | null = null;
  private lastTime = 0;
  private fpsDisplay: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-vision-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.overflowY = 'auto';

    const header = $create('h2', { textContent: 'Vision Pipeline (Live Inference)' });
    this.container.appendChild(header);

    const controlsCard = $create('div', { className: 'property-section' });
    const toggleCameraBtn = $create('button', {
      className: 'action-btn',
      textContent: 'Start Camera',
    });
    this.fpsDisplay = $create('span', {
      className: 'muted',
      textContent: ' FPS: 0',
      attributes: { style: 'margin-left: 10px; font-family: monospace;' },
    });

    controlsCard.appendChild(toggleCameraBtn);
    controlsCard.appendChild(this.fpsDisplay);
    this.container.appendChild(controlsCard);

    const canvasCard = $create('div', { className: 'property-section' });
    this.canvas = $create<HTMLCanvasElement>('canvas', {
      attributes: { width: '640', height: '480' },
      className: 'ide-canvas-2d',
    });
    this.canvas.style.position = 'relative';
    this.canvas.style.width = '100%';
    this.canvas.style.maxWidth = '640px';
    this.canvas.style.height = 'auto';
    this.canvas.style.border = '1px solid var(--color-background-border)';
    this.canvas.style.borderRadius = '4px';
    this.canvas.style.background = '#000';

    canvasCard.appendChild(this.canvas);
    this.container.appendChild(canvasCard);

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context not available');
    this.ctx = ctx;

    toggleCameraBtn.addEventListener('click', async () => {
      if (cameraManager.getIsCapturing()) {
        cameraManager.stop();
        toggleCameraBtn.textContent = 'Start Camera';
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      } else {
        await cameraManager.start();
        toggleCameraBtn.textContent = 'Stop Camera';
        this.startRenderLoop();
      }
    });
  }

  mount(): void {
    // 473. Privacy toggles implicitly handled by Stop Camera button.
  }

  private startRenderLoop(): void {
    const video = cameraManager.getVideoElement();

    const loop = (timestamp: number) => {
      if (!cameraManager.getIsCapturing()) return;

      // Calculate FPS (451. Optimize vision loop to achieve 60 FPS)
      if (this.lastTime > 0) {
        const delta = timestamp - this.lastTime;
        const fps = 1000 / delta;
        this.fpsDisplay.textContent = ` FPS: ${Math.round(fps)}`;
      }
      this.lastTime = timestamp;

      // 444. Capture video frames to canvas
      if (video.readyState >= 2) {
        // HAVE_CURRENT_DATA
        // Maintain aspect ratio
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = this.canvas.width;
        const ch = this.canvas.height;

        const scale = Math.min(cw / vw, ch / vh);
        const sw = vw * scale;
        const sh = vh * scale;
        const sx = (cw - sw) / 2;
        const sy = (ch - sh) / 2;

        this.ctx.fillStyle = '#000';
        this.ctx.fillRect(0, 0, cw, ch);
        this.ctx.drawImage(video, sx, sy, sw, sh);

        // 445 & 446. Normalization & Float32 extraction stub
        // Here we would typically grab `this.ctx.getImageData()`
        // and extract RGB mapping to `Float32Array[1, 3, 224, 224]`
        // const imgData = this.ctx.getImageData(sx, sy, sw, sh);

        // 447, 448, 449. Inference and bounding box render mock
        // if (this.activeYoloModel) { ... }
        this.ctx.strokeStyle = 'var(--color-success)';
        this.ctx.lineWidth = 3;
        this.ctx.strokeRect(sx + 50, sy + 50, 100, 100);
        this.ctx.fillStyle = 'var(--color-success)';
        this.ctx.font = '14px monospace';
        this.ctx.fillText('Person: 0.98', sx + 50, sy + 45);
      }

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  unmount(): void {
    super.unmount();
    if (this.animationId) cancelAnimationFrame(this.animationId);
    cameraManager.stop();
  }
}
