export function $<T extends HTMLElement = HTMLElement>(
  selector: string,
  parent: ParentNode = document,
): T | null {
  return parent.querySelector<T>(selector);
}

export function $$<T extends HTMLElement = HTMLElement>(
  selector: string,
  parent: ParentNode = document,
): NodeListOf<T> {
  return parent.querySelectorAll<T>(selector);
}

export function $on(
  target: EventTarget,
  type: string,
  listener: EventListenerOrEventListenerObject,
  options?: boolean | AddEventListenerOptions,
): void {
  target.addEventListener(type, listener, options);
}

export function $off(
  target: EventTarget,
  type: string,
  listener: EventListenerOrEventListenerObject,
  options?: boolean | EventListenerOptions,
): void {
  target.removeEventListener(type, listener, options);
}

export function $create<T extends HTMLElement = HTMLElement>(
  tag: string,
  options?: {
    id?: string;
    className?: string;
    innerHTML?: string;
    textContent?: string;
    attributes?: Record<string, string>;
  },
): T {
  const el = document.createElement(tag) as T;
  if (options) {
    if (options.id) el.id = options.id;
    if (options.className) el.className = options.className;
    if (options.innerHTML !== undefined) el.innerHTML = options.innerHTML;
    if (options.textContent !== undefined) el.textContent = options.textContent;
    if (options.attributes) {
      for (const [key, value] of Object.entries(options.attributes)) {
        el.setAttribute(key, value);
      }
    }
  }
  return el;
}
