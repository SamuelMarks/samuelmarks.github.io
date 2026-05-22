import { globalEvents } from '../core/State';
import { Toast } from '../ui/Toast';

export class MicrophoneManager {
  private stream: MediaStream | null = null;
  private audioContext: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private dataArray: Uint8Array | null = null;
  private isCapturing = false;

  async start(): Promise<void> {
    if (this.isCapturing) return;

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      const source = this.audioContext.createMediaStreamSource(this.stream);

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      const bufferLength = this.analyser.frequencyBinCount;
      this.dataArray = new Uint8Array(bufferLength);

      source.connect(this.analyser);

      this.isCapturing = true;
      globalEvents.emit('micStarted');
    } catch (err) {
      console.error('Microphone error:', err);
      Toast.show('Failed to access microphone', 'error');
    }
  }

  stop(): void {
    if (this.stream) {
      this.stream.getTracks().forEach((t) => t.stop());
      this.stream = null;
    }
    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
    this.analyser = null;
    this.dataArray = null;
    this.isCapturing = false;
    globalEvents.emit('micStopped');
  }

  getWaveformData(): Uint8Array | null {
    if (!this.isCapturing || !this.analyser || !this.dataArray) return null;
    this.analyser.getByteTimeDomainData(this.dataArray);
    return this.dataArray;
  }

  getIsCapturing(): boolean {
    return this.isCapturing;
  }
}

export const micManager = new MicrophoneManager();
