import { IModelGraph } from '../core/IR';

export class IndexedDBVault {
  private dbName = 'onnx9000_ast_cache';
  private storeName = 'models';
  private db: IDBDatabase | null = null;

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onupgradeneeded = (e) => {
        const target = e.target as IDBOpenDBRequest;
        this.db = target.result;
        if (!this.db.objectStoreNames.contains(this.storeName)) {
          this.db.createObjectStore(this.storeName, { keyPath: 'hash' });
        }
      };

      request.onsuccess = (e) => {
        const target = e.target as IDBOpenDBRequest;
        this.db = target.result;
        resolve();
      };

      request.onerror = () => {
        reject(new Error('Failed to initialize IndexedDB'));
      };
    });
  }

  async get(hash: string): Promise<IModelGraph | null> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.get(hash);

      request.onsuccess = () => {
        resolve(request.result ? request.result.model : null);
      };
      request.onerror = () => reject(new Error('Failed to read from cache'));
    });
  }

  async set(hash: string, model: IModelGraph): Promise<void> {
    if (!this.db) await this.init();

    // 583. Ensure no plaintext weights are written to IndexedDB.
    // Strip rawData payload buffers from the model before caching.
    // The user must re-supply the raw file to hydrate the execution buffers,
    // ensuring the indexedDB only caches the structural AST.
    const strippedModel: IModelGraph = JSON.parse(JSON.stringify(model));
    for (let i = 0; i < strippedModel.initializers.length; i++) {
      strippedModel.initializers[i].rawData = undefined;
    }

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.put({ hash, model: strippedModel });

      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to write to cache'));
    });
  }

  async computeHash(buffer: ArrayBuffer): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
  }

  // 407. Show persistent storage quota and usage
  async getStorageEstimate(): Promise<{ usage: number; quota: number } | null> {
    if (navigator.storage && navigator.storage.estimate) {
      const estimate = await navigator.storage.estimate();
      return {
        usage: estimate.usage || 0,
        quota: estimate.quota || 0,
      };
    }
    return null;
  }

  // 406. Provide UI to delete cached models and clear space
  async listKeys(): Promise<string[]> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readonly');
      const store = transaction.objectStore(this.storeName);
      const request = store.getAllKeys();
      request.onsuccess = () => resolve(request.result as string[]);
      request.onerror = () => reject(new Error('Failed to list keys'));
    });
  }

  async delete(hash: string): Promise<void> {
    if (!this.db) await this.init();
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(this.storeName, 'readwrite');
      const store = transaction.objectStore(this.storeName);
      const request = store.delete(hash);
      request.onsuccess = () => resolve();
      request.onerror = () => reject(new Error('Failed to delete from cache'));
    });
  }
}

export const astCache = new IndexedDBVault();
