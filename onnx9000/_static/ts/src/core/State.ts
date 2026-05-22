export type Listener<T> = (val: T) => void;

export class State<T> {
  private value: T;
  private listeners: Listener<T>[] = [];

  constructor(initialValue: T) {
    this.value = initialValue;
  }

  get(): T {
    return this.value;
  }

  set(newValue: T | ((prev: T) => T)): void {
    if (typeof newValue === 'function') {
      this.value = (newValue as (prev: T) => T)(this.value);
    } else {
      this.value = newValue;
    }
    this.notify();
  }

  subscribe(listener: Listener<T>): () => void {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.value);
    }
  }
}

export class PubSub {
  private events: Map<string, Function[]> = new Map();

  on(event: string, callback: Function): void {
    if (!this.events.has(event)) {
      this.events.set(event, []);
    }
    this.events.get(event)!.push(callback);
  }

  off(event: string, callback: Function): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      this.events.set(
        event,
        callbacks.filter((cb) => cb !== callback),
      );
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  emit(event: string, ...args: any[]): void {
    const callbacks = this.events.get(event);
    if (callbacks) {
      for (const callback of callbacks) {
        callback(...args);
      }
    }
  }
}

export const globalEvents = new PubSub();

export const isOfflineMode = new State<boolean>(true);
export const isDistributedMode = new State<boolean>(false);
