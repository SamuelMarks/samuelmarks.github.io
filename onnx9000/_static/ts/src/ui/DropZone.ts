import { BaseComponent } from './BaseComponent';
import { $, $create, $on, $off } from '../core/DOM';
import { globalEvents } from '../core/State';

export class DropZone extends BaseComponent {
  private overlay: HTMLElement;
  private dropMessage: HTMLElement;
  private dragCounter = 0;

  constructor() {
    super(document.body);

    // Create overlay
    this.overlay = $create('div', { className: 'ide-drop-overlay' });
    this.dropMessage = $create('div', {
      className: 'ide-drop-message',
      textContent: 'Drop .onnx, .safetensors, .py, or Directory here',
    });
    this.overlay.appendChild(this.dropMessage);
    document.body.appendChild(this.overlay);
  }

  mount(): void {
    // Bind global drag events
    const body = document.body;

    this.bindEvent(body, 'dragenter', this.onDragEnter.bind(this));
    this.bindEvent(body, 'dragleave', this.onDragLeave.bind(this));
    this.bindEvent(body, 'dragover', this.onDragOver.bind(this));
    this.bindEvent(body, 'drop', this.onDrop.bind(this));
  }

  private onDragEnter(e: Event): void {
    e.preventDefault();
    this.dragCounter++;
    if (this.dragCounter === 1) {
      this.overlay.classList.add('is-active');
    }
  }

  private onDragLeave(e: Event): void {
    e.preventDefault();
    this.dragCounter--;
    if (this.dragCounter === 0) {
      this.overlay.classList.remove('is-active');
    }
  }

  private onDragOver(e: Event): void {
    e.preventDefault();
  }

  private async onDrop(e: Event): void {
    e.preventDefault();
    this.dragCounter = 0;
    this.overlay.classList.remove('is-active');

    const dragEvent = e as DragEvent;
    if (!dragEvent.dataTransfer) return;

    const files: File[] = [];
    const items = dragEvent.dataTransfer.items;

    if (items && items.length > 0) {
      const promises: Promise<void>[] = [];
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file') {
          const entry = item.webkitGetAsEntry();
          if (entry) {
            promises.push(this.traverseFileTree(entry, files));
          }
        }
      }
      await Promise.all(promises);
    } else {
      // Fallback
      for (let i = 0; i < dragEvent.dataTransfer.files.length; i++) {
        files.push(dragEvent.dataTransfer.files[i]);
      }
    }

    if (files.length > 0) {
      if (files.length === 1) {
        globalEvents.emit('filesDropped', files);
      } else {
        globalEvents.emit('directoryDropped', files);
      }
    }
  }

  private traverseFileTree(item: any, files: File[]): Promise<void> {
    return new Promise((resolve) => {
      if (item.isFile) {
        item.file((file: File) => {
          // Keep a reference to its path if possible
          // file.webkitRelativePath is read-only usually, so we just append to files array
          Object.defineProperty(file, 'webkitRelativePath', {
            value: item.fullPath.replace(/^\//, ''),
            writable: false,
          });
          files.push(file);
          resolve();
        });
      } else if (item.isDirectory) {
        const dirReader = item.createReader();
        dirReader.readEntries((entries: any[]) => {
          const promises: Promise<void>[] = [];
          for (let i = 0; i < entries.length; i++) {
            promises.push(this.traverseFileTree(entries[i], files));
          }
          Promise.all(promises).then(() => resolve());
        });
      } else {
        resolve();
      }
    });
  }

  unmount(): void {
    super.unmount();
    if (this.overlay.parentNode) {
      this.overlay.parentNode.removeChild(this.overlay);
    }
  }
}
