export interface IWorkerMessage {
  id: string;
  type: string;
  payload: unknown;
}

export interface IWorkerResponse {
  id: string;
  type: 'success' | 'error' | 'progress';
  payload?: unknown;
  error?: string;
}

export class WebWorkerPool {
  private scriptUrl: string;
  private maxWorkers: number;
  private workers: Worker[] = [];
  private idleWorkers: Worker[] = [];
  private pendingTasks: Array<{
    message: Omit<IWorkerMessage, 'id'>;
    resolve: (val: unknown) => void;
    reject: (err: Error) => void;
    onProgress?: (payload: unknown) => void;
  }> = [];
  private taskMap = new Map<
    string,
    {
      resolve: (val: unknown) => void;
      reject: (err: Error) => void;
      onProgress?: (payload: unknown) => void;
    }
  >();

  constructor(scriptUrl: string, maxWorkers = navigator.hardwareConcurrency || 4) {
    this.scriptUrl = scriptUrl;
    this.maxWorkers = maxWorkers;
  }

  private createWorker(): Worker {
    const worker = new Worker(this.scriptUrl, { type: 'module' });
    worker.onmessage = (e: MessageEvent<IWorkerResponse>) => {
      const { id, type, payload, error } = e.data;
      const handlers = this.taskMap.get(id);

      if (handlers) {
        if (type === 'progress') {
          if (handlers.onProgress) {
            handlers.onProgress(payload);
          }
          return; // Do not complete the task yet
        }

        this.taskMap.delete(id);
        if (type === 'error') {
          handlers.reject(new Error(error || 'Unknown worker error'));
        } else {
          handlers.resolve(payload);
        }
      }

      this.idleWorkers.push(worker);
      this.processNextTask();
    };
    worker.onerror = (e: ErrorEvent) => {
      console.error('Worker generic error:', e);
    };
    return worker;
  }

  private processNextTask(): void {
    if (this.pendingTasks.length === 0) return;

    let worker = this.idleWorkers.pop();
    if (!worker) {
      if (this.workers.length < this.maxWorkers) {
        worker = this.createWorker();
        this.workers.push(worker);
      } else {
        // No workers available
        return;
      }
    }

    const task = this.pendingTasks.shift()!;
    const id = Math.random().toString(36).substring(2, 9);
    this.taskMap.set(id, {
      resolve: task.resolve,
      reject: task.reject,
      onProgress: task.onProgress,
    });

    worker.postMessage({
      id,
      type: task.message.type,
      payload: task.message.payload,
    });
  }

  execute(
    type: string,
    payload: unknown,
    onProgress?: (payload: unknown) => void,
  ): Promise<unknown> {
    return new Promise((resolve, reject) => {
      this.pendingTasks.push({
        message: { type, payload },
        resolve,
        reject,
        onProgress,
      });
      this.processNextTask();
    });
  }

  terminateAll(): void {
    for (const worker of this.workers) {
      worker.terminate();
    }
    this.workers = [];
    this.idleWorkers = [];
    this.taskMap.clear();
    this.pendingTasks = [];
  }
}
