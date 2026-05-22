import { $, $create } from '../core/DOM';

export class Spinner {
  private static overlay: HTMLElement | null = null;

  static init(): void {
    if (!this.overlay) {
      this.overlay = $create('div', { className: 'ide-loader-overlay' });
      const spinner = $create('div', { className: 'ide-spinner' });
      this.overlay.appendChild(spinner);
      document.body.appendChild(this.overlay);
    }
  }

  static show(): void {
    this.init();
    if (this.overlay) {
      this.overlay.classList.add('is-active');
    }
  }

  static hide(): void {
    if (this.overlay) {
      this.overlay.classList.remove('is-active');
    }
  }
}
