import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { IModelGraph, ITensor } from '../core/IR';
import { escapeHTML } from '../core/Sanitize';

export class ModelSummary extends BaseComponent {
  private model: IModelGraph | null = null;
  private tableContainer: HTMLElement;

  constructor(containerId: string) {
    super(containerId);
    this.tableContainer = $create('div', { className: 'model-summary-table-wrapper' });
    this.container.appendChild(this.tableContainer);
  }

  mount(): void {
    // nothing bound yet
  }

  setModel(model: IModelGraph | null): void {
    this.model = model;
    this.render();
  }

  private render(): void {
    this.tableContainer.innerHTML = '';

    if (!this.model) {
      this.tableContainer.innerHTML = '<p>No model loaded.</p>';
      return;
    }

    const header = $create('h3', { textContent: `Model: ${this.model.name}` });

    // 577. Verify watermarks
    if (this.model.docString && this.model.docString.includes('onnx9000_verified_')) {
      const badge = $create('span', {
        className: 'badge success',
        textContent: 'DP Verified',
        attributes: { style: 'margin-left: 10px; font-size: 0.7rem; vertical-align: middle;' },
      });
      header.appendChild(badge);
    }

    this.tableContainer.appendChild(header);

    // 163. Calculate FLOPs / Memory Footprint reductions
    let totalParams = 0;
    let totalBytes = 0;
    let totalSparsity = 0;
    let paramElements = 0;

    this.model.initializers.forEach((init) => {
      if (init.rawData) {
        totalBytes += init.rawData.byteLength;

        // If CSR sparse format
        if (init.dataType === 21) {
          // Roughly parse sparse density out of the buffer header
          const dv = new DataView(init.rawData.buffer, init.rawData.byteOffset, 12);
          const nnz = dv.getUint32(0, true);
          let shapeSize = 1;
          init.dims.forEach((d) => (shapeSize *= d));
          if (shapeSize > 0) {
            totalSparsity += shapeSize - nnz;
            paramElements += shapeSize;
          }
        } else {
          let shapeSize = 1;
          init.dims.forEach((d) => (shapeSize *= d));

          // Check zeros directly
          if (init.dataType === 1) {
            // F32
            const f32 = new Float32Array(
              init.rawData.buffer,
              init.rawData.byteOffset,
              init.rawData.byteLength / 4,
            );
            let zeroCount = 0;
            for (let i = 0; i < f32.length; i++) if (f32[i] === 0) zeroCount++;
            totalSparsity += zeroCount;
          }
          paramElements += shapeSize;
        }
      }
    });

    const sparsityPct = paramElements > 0 ? ((totalSparsity / paramElements) * 100).toFixed(1) : 0;
    const mbSize = (totalBytes / 1024 / 1024).toFixed(2);

    const stats = $create('div', {
      className: 'property-section',
      innerHTML: `
        <div class="property-row"><strong>Nodes:</strong> <span>${this.model.nodes.length}</span></div>
        <div class="property-row"><strong>Memory Footprint:</strong> <span>${mbSize} MB</span></div>
        <div class="property-row"><strong>Global Sparsity:</strong> <span>${sparsityPct}% Zeros</span></div>
      `,
    });
    this.tableContainer.appendChild(stats);

    if (this.model.initializers.length > 0) {
      const table = $create('table', { className: 'ide-table' });
      table.innerHTML = `
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Shape</th>
            <th>Size (Bytes)</th>
          </tr>
        </thead>
        <tbody>
        </tbody>
      `;
      const tbody = table.querySelector('tbody')!;

      // limit to 100 for display
      const displayCount = Math.min(100, this.model.initializers.length);
      for (let i = 0; i < displayCount; i++) {
        const init = this.model.initializers[i];
        const tr = $create('tr');
        const byteSize = init.rawData ? init.rawData.byteLength : 0;
        tr.innerHTML = `
          <td>${escapeHTML(init.name)}</td>
          <td>${init.dataType}</td>
          <td>[${init.dims.join(', ')}]</td>
          <td>${byteSize.toLocaleString()}</td>
        `;
        tbody.appendChild(tr);
      }

      if (this.model.initializers.length > 100) {
        const tr = $create('tr');
        tr.innerHTML = `<td colspan="4" style="text-align: center; color: var(--color-foreground-muted);">... and ${this.model.initializers.length - 100} more</td>`;
        tbody.appendChild(tr);
      }

      this.tableContainer.appendChild(table);
    }
  }
}
