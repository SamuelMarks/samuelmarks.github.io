import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';

declare const require: any;
declare const monaco: any;

export class CodeEditor extends BaseComponent {
  private editorInstance: any = null;
  private debounceTimer: any = null;

  constructor(containerId: string) {
    super(containerId);
  }

  mount(): void {
    // 307. Verify Monaco editor web workers are loaded securely via blob URLs.
    if (typeof (window as any).MonacoEnvironment === 'undefined') {
      (window as any).MonacoEnvironment = {
        getWorkerUrl: function (workerId: string, label: string) {
          const proxy = `self.MonacoEnvironment = { baseUrl: 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/' }; importScripts('https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/base/worker/workerMain.js');`;
          return URL.createObjectURL(new Blob([proxy], { type: 'text/javascript' }));
        },
      };
    }

    if (typeof require !== 'undefined') {
      require(['vs/editor/editor.main'], () => {
        this.initEditor();
      });
    } else {
      console.warn('Monaco editor loader not found.');
    }
  }

  private initEditor(): void {
    const isDark = document.body.getAttribute('data-theme') === 'dark';

    this.editorInstance = monaco.editor.create(this.container, {
      value: [
        'import onnxscript',
        'from onnxscript import opset15 as op',
        '',
        '@onnxscript.script()',
        'def custom_model(X, Y):',
        '    return op.MatMul(X, Y)',
      ].join('\n'),
      language: 'python',
      theme: isDark ? 'vs-dark' : 'vs',
      automaticLayout: true,
      minimap: { enabled: false },
    });

    globalEvents.on('themeChanged', (theme: string) => {
      monaco.editor.setTheme(theme === 'dark' ? 'vs-dark' : 'vs');
    });

    this.editorInstance.onDidChangeModelContent(() => {
      this.handleContentChange();
    });
  }

  private handleContentChange(): void {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }

    this.debounceTimer = setTimeout(() => {
      const code = this.editorInstance.getValue();
      globalEvents.emit('onnxScriptChanged', code);
    }, 1000); // 1 second debounce
  }

  public highlightError(line: number, message: string): void {
    if (!this.editorInstance) return;

    const marker = {
      severity: monaco.MarkerSeverity.Error,
      startLineNumber: line,
      startColumn: 1,
      endLineNumber: line,
      endColumn: 100,
      message: message,
    };
    const model = this.editorInstance.getModel();
    monaco.editor.setModelMarkers(model, 'onnxscript', [marker]);
  }

  public clearErrors(): void {
    if (!this.editorInstance) return;
    const model = this.editorInstance.getModel();
    monaco.editor.setModelMarkers(model, 'onnxscript', []);
  }

  public getValue(): string {
    return this.editorInstance ? this.editorInstance.getValue() : '';
  }

  public setValue(val: string): void {
    if (this.editorInstance) {
      this.editorInstance.setValue(val);
    }
  }

  public setLanguage(lang: string): void {
    if (this.editorInstance) {
      monaco.editor.setModelLanguage(this.editorInstance.getModel(), lang);
    }
  }
}
