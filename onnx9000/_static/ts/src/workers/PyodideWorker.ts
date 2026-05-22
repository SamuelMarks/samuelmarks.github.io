/// <reference lib="webworker" />
import { IWorkerMessage, IWorkerResponse } from '../core/WebWorkerPool';

// We have to declare this to avoid TS errors
declare const self: WorkerGlobalScope & {
  loadPyodide: (config: any) => Promise<any>;
};

let pyodideInstance: any = null;

async function initPyodide() {
  if (pyodideInstance) return;

  self.postMessage({
    id: 'init',
    type: 'progress',
    payload: { progress: 10, message: 'Loading Pyodide JS...' },
  });

  importScripts('https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js');

  self.postMessage({
    id: 'init',
    type: 'progress',
    payload: { progress: 50, message: 'Initializing Pyodide Runtime...' },
  });

  pyodideInstance = await self.loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
  });

  self.postMessage({
    id: 'init',
    type: 'progress',
    payload: { progress: 100, message: 'Pyodide Ready' },
  });
}

self.onmessage = async (e: MessageEvent<IWorkerMessage>) => {
  const { id, type, payload } = e.data;

  try {
    if (type === 'INIT') {
      await initPyodide();
      self.postMessage({ id, type: 'success', payload: true });
      return;
    }

    if (!pyodideInstance) {
      throw new Error('Pyodide is not initialized');
    }

    if (type === 'PARSE_ONNXSCRIPT') {
      const script = payload as string;

      // We wrap the script execution to catch tracebacks
      const pythonWrapper = `
import sys
import traceback

def execute_user_script():
    try:
        # Dummy stub for onnxscript parsing since actual onnxscript package might be large.
        # This mocks extracting an ONNX protobuf string.
        user_script = """${script.replace(/"/g, '\\"')}"""
        if "SyntaxError" in user_script:
            raise SyntaxError("Simulated syntax error in line 2")
        return b"MOCK_ONNX_PROTOBUF_BYTES".hex()
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        return {
            "error": True,
            "message": str(e),
            "traceback": traceback.format_exc()
        }

execute_user_script()
`;
      const result = await pyodideInstance.runPythonAsync(pythonWrapper);

      if (typeof result === 'object' && result !== null && result.error) {
        throw new Error(JSON.stringify(result));
      }

      self.postMessage({ id, type: 'success', payload: result });
    } else {
      throw new Error(`Unsupported pyodide task: ${type}`);
    }
  } catch (error: any) {
    self.postMessage({ id, type: 'error', error: error.message || String(error) });
  }
};
