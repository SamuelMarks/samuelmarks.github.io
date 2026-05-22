import { $, $on, $off } from '../core/DOM';

export abstract class BaseComponent {
  protected container: HTMLElement;
  protected unmountCallbacks: Array<() => void> = [];

  constructor(containerIdOrElement: string | HTMLElement) {
    if (typeof containerIdOrElement === 'string') {
      const el = $<HTMLElement>(containerIdOrElement);
      if (!el) {
        throw new Error(`Container with selector ${containerIdOrElement} not found.`);
      }
      this.container = el;
    } else {
      this.container = containerIdOrElement;
    }
  }

  // Bind an event and ensure it gets cleaned up on unmount
  protected bindEvent(
    target: EventTarget,
    type: string,
    listener: EventListenerOrEventListenerObject,
    options?: boolean | AddEventListenerOptions,
  ): void {
    // 305. Error boundary wrapping around event listeners
    const safeListener = (e: Event) => {
      try {
        if (typeof listener === 'function') {
          listener(e);
        } else if (listener && typeof listener.handleEvent === 'function') {
          listener.handleEvent(e);
        }
      } catch (err) {
        console.error(`Error boundary caught exception in ${type} event`, err);
      }
    };

    $on(target, type, safeListener, options);
    this.unmountCallbacks.push(() => {
      $off(target, type, safeListener, options);
    });
  }

  // To be implemented by subclasses
  abstract mount(): void;

  unmount(): void {
    this.unmountCallbacks.forEach((cb) => cb());
    this.unmountCallbacks = [];
  }
}
