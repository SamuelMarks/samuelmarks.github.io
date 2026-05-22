import { KerasParser } from './KerasParser';

import { SafetensorsParser } from './Safetensors';
import { ONNXProtoParser } from './ONNXProto';
import { Toast } from '../ui/Toast';
import { IModelGraph } from '../core/IR';
import { WebWorkerPool } from '../core/WebWorkerPool';
import { astCache } from '../storage/IndexedDBVault';
import { logger } from '../core/Logger';
import { Spinner } from '../ui/Spinner';
import { globalEvents } from '../core/State';

export class FileParser {
  private workerPool: WebWorkerPool | null = null;
  private pyodidePool: WebWorkerPool | null = null;

  constructor() {
    try {
      this.workerPool = new WebWorkerPool('_static/assets/parser.worker.js', 2);
      this.pyodidePool = new WebWorkerPool('_static/assets/pyodide.worker.js', 1);
    } catch (e) {
      logger.warn('Failed to initialize WebWorkerPools', e);
    }
  }

  async initPyodide(): Promise<void> {
    if (!this.pyodidePool) return;
    try {
      await this.pyodidePool.execute('INIT', null, (p: any) => {
        Spinner.show();
      });
      Spinner.hide();
    } catch (e) {
      Spinner.hide();
      logger.error('Pyodide init failed', e);
    }
  }

  async processFile(file: File): Promise<IModelGraph | null> {
    const extension = file.name.split('.').pop()?.toLowerCase();
    Toast.show(`Reading file: ${file.name}...`);

    try {
      const buffer = await file.arrayBuffer();

      if (extension === 'safetensors') {
        return this.parseSafetensors(buffer, file.name);
      } else if (extension === 'onnx') {
        return this.parseONNX(buffer, file.name);
      } else if (extension === 'py') {
        return this.parseONNXScript(file);
      } else if (['pb', 'savedmodel', 'pkl', 'pdmodel', 'json', 'gguf'].includes(extension || '')) {
        return this.parseViaWorker(buffer, file.name, extension as string);
      } else {
        throw new Error(`Unsupported file extension: ${extension}`);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      Toast.show(`Failed to parse file: ${msg}`, 'error');
      logger.error(e);
      return null;
    }
  }

  async processDirectory(files: File[]): Promise<IModelGraph | null> {
    Toast.show(`Parsing directory with ${files.length} files...`);
    // Minimal stub for constructing directory hierarchy memory
    logger.info(
      'Directory files:',
      files.map((f) => f.webkitRelativePath),
    );

    // We would package these files into a virtual filesystem map and pass to the worker
    // For now, return a dummy
    return {
      name: 'TF_Directory_Model',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
      docString: JSON.stringify({ files: files.length }),
    };
  }

  private async parseONNXScript(file: File): Promise<IModelGraph | null> {
    if (!this.pyodidePool) throw new Error('Pyodide pool not initialized');

    const text = await file.text();
    Toast.show('Executing ONNXScript in Pyodide...');
    Spinner.show();

    try {
      const result = await this.pyodidePool.execute('PARSE_ONNXSCRIPT', text, (p: any) => {
        logger.info(`[Pyodide] ${p.progress}%: ${p.message}`);
      });
      Spinner.hide();

      // result is a hex string representing ONNX protobuf bytes
      const hex = result as string;
      const bytes = new Uint8Array(Math.ceil(hex.length / 2));
      for (let i = 0; i < hex.length; i += 2) {
        bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
      }

      const parser = new ONNXProtoParser(bytes.buffer);
      const graph = parser.parse();
      graph.name = file.name;
      return graph;
    } catch (e) {
      Spinner.hide();
      const msg = e instanceof Error ? e.message : String(e);
      Toast.show(`ONNXScript Error: ${msg}`, 'error');
      return null;
    }
  }

  private async parseViaWorker(
    buffer: ArrayBuffer,
    name: string,
    ext: string,
  ): Promise<IModelGraph> {
    if (!this.workerPool) {
      throw new Error('Worker pool not initialized');
    }

    const hash = await astCache.computeHash(buffer);
    const cached = await astCache.get(hash);
    if (cached) {
      Toast.show('Loaded AST from IndexedDB Cache', 'success');
      return cached;
    }

    let type = 'PARSE_TF';
    let payload: unknown = buffer;
    if (ext === 'pkl') type = 'PARSE_SKL';
    if (ext === 'pdmodel') type = 'PARSE_PADDLE';
    if (ext === 'json') {
      const text = new TextDecoder().decode(buffer);
      if (text.includes('keras_version') || text.includes('class_name')) {
        const parser = new KerasParser(text);
        const graph = parser.parse();
        await astCache.set(hash, graph);
        return graph;
      }

      type = 'PARSE_XGBOOST';
      payload = new TextDecoder().decode(buffer);
    }
    if (ext === 'gguf') type = 'PARSE_GGUF';

    const result = await this.workerPool.execute(type, payload, (progressPayload) => {
      const p = progressPayload as { progress: number; message: string };
      logger.info(`[Worker] ${p.progress}%: ${p.message}`);
      // Send progress to UI state
      globalEvents.emit('progress', p);
    });

    const graph = result as IModelGraph;
    if (graph && (graph.name === 'Model' || graph.name.includes('_Model'))) {
      graph.name = name;
    }

    await astCache.set(hash, graph);

    return graph;
  }

  private parseSafetensors(buffer: ArrayBuffer, name: string): IModelGraph {
    const parser = new SafetensorsParser(buffer);
    const { metadata, tensors } = parser.parse();

    // 577. Verify watermarks upon model load
    if (metadata && typeof metadata === 'object' && 'watermark' in metadata) {
      const wm = metadata.watermark as string;
      if (wm.startsWith('onnx9000_verified_')) {
        logger.info(`Valid DP Watermark found: ${wm}`);
        Toast.show('Model Watermark Verified', 'success');
      }
    }

    return {
      name,
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: Object.values(tensors).map((t) => ({
        name: t.name,
        dataType: this.mapDtypeToONNX(t.dtype),
        dims: t.shape,
        rawData: new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength),
      })),
      docString: metadata ? JSON.stringify(metadata) : undefined,
    };
  }

  private parseONNX(buffer: ArrayBuffer, name: string): IModelGraph {
    const parser = new ONNXProtoParser(buffer);
    const graph = parser.parse();
    if (graph.name === 'ONNX Model') {
      graph.name = name;
    }
    return graph;
  }

  private mapDtypeToONNX(safetensorDtype: string): number {
    switch (safetensorDtype) {
      case 'F32':
        return 1;
      case 'U8':
        return 2;
      case 'I8':
        return 3;
      case 'U16':
        return 4;
      case 'I16':
        return 5;
      case 'I32':
        return 6;
      case 'I64':
        return 7;
      case 'F16':
        return 10;
      case 'F64':
        return 11;
      case 'U32':
        return 12;
      case 'U64':
        return 13;
      default:
        return 0;
    }
  }

  terminateWorkers(): void {
    if (this.workerPool) {
      this.workerPool.terminateAll();
    }
  }
}
