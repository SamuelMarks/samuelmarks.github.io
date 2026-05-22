import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';
import { micManager } from '../sensors/MicrophoneManager';

export class AudioPipeline extends BaseComponent {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationId: number | null = null;
  private transcriptionDisplay: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-audio-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.overflowY = 'auto';

    const header = $create('h2', { textContent: 'Audio Pipeline (Live Transcription)' });
    this.container.appendChild(header);

    const controlsCard = $create('div', { className: 'property-section' });
    const toggleMicBtn = $create('button', {
      className: 'action-btn',
      textContent: 'Start Microphone',
    });

    controlsCard.appendChild(toggleMicBtn);
    this.container.appendChild(controlsCard);

    const canvasCard = $create('div', { className: 'property-section' });
    canvasCard.appendChild($create('h3', { textContent: 'Live Waveform' }));
    this.canvas = $create<HTMLCanvasElement>('canvas', {
      attributes: { width: '640', height: '150' },
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

    const transCard = $create('div', { className: 'property-section' });
    transCard.appendChild($create('h3', { textContent: 'Transcription (Mock Whisper)' }));
    this.transcriptionDisplay = $create('div', {
      className: 'ide-chat-messages',
      textContent: 'Awaiting audio...',
      attributes: {
        style:
          'padding: 10px; border: 1px solid var(--color-background-border); border-radius: 4px; font-family: monospace; background: var(--color-background-secondary);',
      },
    });

    transCard.appendChild(this.transcriptionDisplay);
    this.container.appendChild(transCard);

    const ctx = this.canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context not available');
    this.ctx = ctx;

    toggleMicBtn.addEventListener('click', async () => {
      if (micManager.getIsCapturing()) {
        micManager.stop();
        toggleMicBtn.textContent = 'Start Microphone';
        if (this.animationId) cancelAnimationFrame(this.animationId);
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      } else {
        await micManager.start();
        toggleMicBtn.textContent = 'Stop Microphone';
        this.startRenderLoop();
      }
    });
  }

  mount(): void {}

  private startRenderLoop(): void {
    const loop = () => {
      if (!micManager.getIsCapturing()) return;

      const dataArray = micManager.getWaveformData();
      if (dataArray) {
        this.ctx.fillStyle = 'rgb(20, 20, 20)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = 'rgb(0, 255, 0)';

        this.ctx.beginPath();

        const sliceWidth = (this.canvas.width * 1.0) / dataArray.length;
        let x = 0;

        for (let i = 0; i < dataArray.length; i++) {
          const v = dataArray[i] / 128.0;
          const y = (v * this.canvas.height) / 2;

          if (i === 0) {
            this.ctx.moveTo(x, y);
          } else {
            this.ctx.lineTo(x, y);
          }

          x += sliceWidth;
        }

        this.ctx.lineTo(this.canvas.width, this.canvas.height / 2);
        this.ctx.stroke();

        // 460. VAD Stub: Detect amplitude to trigger transcription mock
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += Math.abs(dataArray[i] - 128);
        }
        const avg = sum / dataArray.length;
        if (avg > 25) {
          // Threshold
          if (Math.random() < 0.05) {
            // Mock throttle
            this.transcriptionDisplay.textContent += ' [Speech Detected]';
          }
        }
      }

      this.animationId = requestAnimationFrame(loop);
    };

    this.animationId = requestAnimationFrame(loop);
  }

  unmount(): void {
    super.unmount();
    if (this.animationId) cancelAnimationFrame(this.animationId);
    micManager.stop();
  }
}
