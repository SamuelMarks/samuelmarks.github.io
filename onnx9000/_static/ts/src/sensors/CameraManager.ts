import { globalEvents } from '../core/State';
import { Toast } from '../ui/Toast';

export class CameraManager {
  private videoEl: HTMLVideoElement;
  private stream: MediaStream | null = null;
  private isCapturing = false;

  constructor() {
    this.videoEl = document.createElement('video');
    this.videoEl.setAttribute('playsinline', 'true');
    this.videoEl.style.display = 'none';
    document.body.appendChild(this.videoEl);
  }

  async start(): Promise<void> {
    if (this.isCapturing) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      });

      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();
      this.isCapturing = true;
      globalEvents.emit('cameraStarted', {
        width: this.videoEl.videoWidth,
        height: this.videoEl.videoHeight,
      });
    } catch (err) {
      console.error('Camera error:', err);
      Toast.show('Failed to access camera', 'error');
    }
  }

  stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    this.isCapturing = false;
    globalEvents.emit('cameraStopped');
  }

  getVideoElement(): HTMLVideoElement {
    return this.videoEl;
  }

  getIsCapturing(): boolean {
    return this.isCapturing;
  }
}

export const cameraManager = new CameraManager();
