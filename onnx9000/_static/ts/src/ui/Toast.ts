import { $create } from '../core/DOM';

export class Toast {
  private static container: HTMLElement;

  static init(): void {
    if (!this.container) {
      this.container = $create('div', {
        className: 'ide-toast-container',
      });
      document.body.appendChild(this.container);

      const style = $create('style', {
        textContent: `
          .ide-toast-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
          }
          .ide-toast {
            background: var(--color-background-secondary);
            color: var(--color-foreground-primary);
            border: 1px solid var(--color-background-border);
            border-left: 4px solid var(--color-primary);
            padding: 12px 16px;
            border-radius: 4px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            font-family: sans-serif;
            font-size: 0.9rem;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
          }
          .ide-toast.show {
            opacity: 1;
            transform: translateX(0);
          }
          .ide-toast.error {
            border-left-color: var(--color-danger);
          }
          .ide-toast.success {
            border-left-color: var(--color-success);
          }
          .ide-toast.warn {
            border-left-color: #ffc107;
          }
        `,
      });
      document.head.appendChild(style);
    }
  }

  static show(
    message: string,
    type: 'info' | 'success' | 'warn' | 'error' = 'info',
    duration = 3000,
  ): void {
    this.init();
    const toast = $create('div', {
      className: `ide-toast ${type}`,
      textContent: message,
    });
    this.container.appendChild(toast);

    // Trigger reflow to animate
    void toast.offsetWidth;
    toast.classList.add('show');

    setTimeout(() => {
      toast.classList.remove('show');
      toast.addEventListener('transitionend', () => {
        if (toast.parentNode) {
          toast.parentNode.removeChild(toast);
        }
      });
    }, duration);
  }
}
