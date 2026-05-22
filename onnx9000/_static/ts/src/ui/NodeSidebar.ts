import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { INode, IModelGraph } from '../core/IR';
import { escapeHTML } from '../core/Sanitize';

export class NodeSidebar extends BaseComponent {
  private contentContainer: HTMLElement;
  private currentModel: IModelGraph | null = null;
  private selectedNode: INode | null = null;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    const header = $create('div', { className: 'sidebar-section' });
    const title = $create('h4', { textContent: 'Node Properties' });
    header.appendChild(title);

    this.contentContainer = $create('div', { className: 'node-properties-content' });
    this.contentContainer.innerHTML = "<p class='muted'>Select a node to view properties.</p>";

    this.container.appendChild(header);
    this.container.appendChild(this.contentContainer);
  }

  mount(): void {
    globalEvents.on('modelLoaded', (model: IModelGraph) => {
      this.currentModel = model;
      this.selectedNode = null;
      this.render();
    });

    globalEvents.on('nodeSelected', (node: INode | null) => {
      this.selectedNode = node;
      this.render();
    });
  }

  private render(): void {
    this.contentContainer.innerHTML = '';

    if (!this.selectedNode) {
      this.contentContainer.innerHTML = "<p class='muted'>Select a node to view properties.</p>";
      return;
    }

    const n = this.selectedNode;

    // Header info
    const infoSection = $create('div', { className: 'property-section' });
    infoSection.innerHTML = `
      <div class="property-row"><strong>Name:</strong> <span>${escapeHTML(n.name)}</span></div>
      <div class="property-row"><strong>OpType:</strong> <span>${escapeHTML(n.opType)}</span></div>
      <div class="property-row"><strong>Domain:</strong> <span>${escapeHTML(n.domain || 'ai.onnx')}</span></div>
    `;

    const btnContainer = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-top: 10px;' },
    });
    const deleteBtn = $create('button', {
      className: 'action-btn danger small',
      textContent: 'Delete Node',
    });
    deleteBtn.addEventListener('click', () => {
      globalEvents.emit('surgeon', `deleteNode:${n.name}`);
    });

    // 510. Allow visually painting sparsity masks
    const maskBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Paint Sparsity Mask',
    });
    maskBtn.style.marginLeft = '10px';
    maskBtn.addEventListener('click', () => {
      globalEvents.emit('paintMask', n.name);
    });

    btnContainer.appendChild(deleteBtn);
    if (n.opType === 'Conv' || n.opType === 'MatMul') {
      btnContainer.appendChild(maskBtn);
    }

    infoSection.appendChild(btnContainer);

    this.contentContainer.appendChild(infoSection);

    // Attributes
    if (Object.keys(n.attributes).length > 0) {
      const attrSection = $create('div', { className: 'property-section' });
      attrSection.innerHTML = `<h5>Attributes</h5>`;
      const table = $create('table', { className: 'ide-table property-table' });
      table.innerHTML = `<tbody></tbody>`;
      const tbody = table.querySelector('tbody')!;

      for (const [key, attr] of Object.entries(n.attributes)) {
        const tr = $create('tr');

        let valueStr = '';
        if (attr.type === 'INT') valueStr = String(attr.i);
        else if (attr.type === 'FLOAT') valueStr = String(attr.f);
        else if (attr.type === 'STRING') valueStr = `"${escapeHTML(attr.s || '')}"`;
        else if (attr.type === 'INTS') valueStr = `[${attr.ints?.join(', ')}]`;
        else if (attr.type === 'FLOATS') valueStr = `[${attr.floats?.join(', ')}]`;
        else valueStr = `<span class="muted">${attr.type}</span>`;

        tr.innerHTML = `
          <td class="prop-key">${escapeHTML(key)}</td>
          <td class="prop-val">
            <input type="text" class="ide-attr-input" data-key="${escapeHTML(key)}" value='${valueStr.replace(/'/g, '&#39;')}' />
          </td>
        `;
        tbody.appendChild(tr);
      }
      attrSection.appendChild(table);
      this.contentContainer.appendChild(attrSection);

      // Bind input changes
      const inputs = attrSection.querySelectorAll<HTMLInputElement>('.ide-attr-input');
      inputs.forEach((input) => {
        input.addEventListener('change', (e) => {
          const target = e.target as HTMLInputElement;
          const key = target.getAttribute('data-key');
          if (key && n.attributes[key]) {
            const attr = n.attributes[key];
            const newVal = target.value;
            try {
              if (attr.type === 'INT') {
                const parsed = parseInt(newVal, 10);
                if (isNaN(parsed)) throw new Error('Must be an integer');
                attr.i = parsed;
              } else if (attr.type === 'FLOAT') {
                const parsed = parseFloat(newVal);
                if (isNaN(parsed)) throw new Error('Must be a float');
                attr.f = parsed;
              } else if (attr.type === 'STRING') attr.s = newVal.replace(/^"|"$/g, '');
              else if (attr.type === 'INTS') {
                const parsed = JSON.parse(newVal);
                if (!Array.isArray(parsed)) throw new Error('Must be an array of integers');
                attr.ints = parsed;
              } else if (attr.type === 'FLOATS') {
                const parsed = JSON.parse(newVal);
                if (!Array.isArray(parsed)) throw new Error('Must be an array of floats');
                attr.floats = parsed;
              }
              // We just manually re-trigger render to update UI
              globalEvents.emit('nodeSelected', n);
            } catch (err) {
              console.error('Invalid attribute format', err);
              // Revert
              target.value = valueStr;
            }
          }
        });
      });
    }

    // Inputs
    if (n.inputs.length > 0) {
      const inputSection = $create('div', { className: 'property-section' });
      inputSection.innerHTML = `<h5>Inputs</h5>`;
      const ul = $create('ul', { className: 'property-list' });
      n.inputs.forEach((inp) => {
        const li = $create('li');

        let shapeStr = '';
        let isDynamic = false;

        if (this.currentModel) {
          // Look for it in value_info or inputs or initializers
          const vi =
            this.currentModel.valueInfo?.find((v) => v.name === inp) ||
            this.currentModel.inputs.find((v) => v.name === inp);

          if (vi && vi.type) {
            shapeStr = ` <span class="muted">(${vi.type.elemType}) [${vi.type.shape.join(', ')}]</span>`;
            isDynamic = vi.type.shape.some(
              (d: any) => typeof d === 'string' || d === '?' || d === null,
            );
          } else {
            const init = this.currentModel.initializers.find((i) => i.name === inp);
            if (init) {
              shapeStr = ` <span class="muted">(INIT) [${init.dims.join(', ')}]</span>`;
              isDynamic = init.dims.some(
                (d: any) => typeof d === 'string' || d === '?' || d === null,
              );
            }
          }
        }

        // 499. Lock dynamic shapes UI stub
        const lockBtn = isDynamic
          ? `<button class="action-btn secondary small" style="margin-left: 5px; font-size: 0.6rem;" onclick="window.dispatchEvent(new CustomEvent('lockShape', {detail: '${inp}'}))">Lock Shape</button>`
          : '';

        li.innerHTML = `<code>${escapeHTML(inp)}</code>${shapeStr}${lockBtn}`;

        // 119. Render tensor initialization data (weights) as sparklines in the sidebar
        if (this.currentModel) {
          const init = this.currentModel.initializers.find((i) => i.name === inp);
          if (init && init.rawData && init.dataType === 1 && init.dims.length >= 2) {
            // 1 = F32
            const floatArray = new Float32Array(
              init.rawData.buffer,
              init.rawData.byteOffset,
              init.rawData.byteLength / 4,
            );
            if (floatArray.length > 0) {
              // Simple histogram sparkline
              const buckets = new Array(10).fill(0);
              let min = Infinity,
                max = -Infinity;
              for (let i = 0; i < floatArray.length; i++) {
                min = Math.min(min, floatArray[i]);
                max = Math.max(max, floatArray[i]);
              }

              const range = max - min || 1;
              for (let i = 0; i < floatArray.length; i++) {
                const bucketIdx = Math.floor(((floatArray[i] - min) / range) * 9.99);
                buckets[bucketIdx]++;
              }

              const maxBucket = Math.max(...buckets);
              let sparklineHTML =
                "<div style='display: flex; align-items: flex-end; height: 30px; gap: 2px; margin-top: 5px;'>";
              buckets.forEach((b) => {
                const h = Math.max((b / maxBucket) * 30, 2);
                sparklineHTML += `<div style='flex: 1; background: var(--color-primary); height: ${h}px;' title='Bucket Size: ${b}'></div>`;
              });
              sparklineHTML += '</div>';

              const sparklineContainer = $create('div', { innerHTML: sparklineHTML });
              li.appendChild(sparklineContainer);

              // 162. Visualize sparsity patterns using a grid canvas in the sidebar
              if (init.dims.length === 2 && Math.max(init.dims[0], init.dims[1]) <= 256) {
                // Don't crash UI on massive matrices
                const rows = init.dims[0];
                const cols = init.dims[1];
                const sc = $create<HTMLCanvasElement>('canvas', {
                  attributes: {
                    width: '100',
                    height: '100',
                    style: 'margin-top: 5px; border: 1px solid var(--color-background-border);',
                  },
                });
                const sctx = sc.getContext('2d');
                if (sctx) {
                  sctx.fillStyle = '#fff';
                  sctx.fillRect(0, 0, 100, 100);
                  const cellW = 100 / cols;
                  const cellH = 100 / rows;
                  for (let r = 0; r < rows; r++) {
                    for (let c = 0; c < cols; c++) {
                      if (floatArray[r * cols + c] !== 0) {
                        sctx.fillStyle = '#000';
                        sctx.fillRect(c * cellW, r * cellH, cellW, cellH);
                      }
                    }
                  }
                }
                li.appendChild(sc);
              }
            }
          }
        }

        ul.appendChild(li);
      });
      inputSection.appendChild(ul);
      this.contentContainer.appendChild(inputSection);
    }

    // Outputs
    if (n.outputs.length > 0) {
      const outSection = $create('div', { className: 'property-section' });
      outSection.innerHTML = `<h5>Outputs</h5>`;
      const ul = $create('ul', { className: 'property-list' });
      n.outputs.forEach((out) => {
        const li = $create('li');
        let shapeStr = '';
        if (this.currentModel) {
          const vi =
            this.currentModel.valueInfo?.find((v) => v.name === out) ||
            this.currentModel.outputs.find((v) => v.name === out);
          if (vi && vi.type) {
            shapeStr = ` <span class="muted">(${vi.type.elemType}) [${vi.type.shape.join(', ')}]</span>`;
          }
        }
        li.innerHTML = `<code>${escapeHTML(out)}</code>${shapeStr}`;
        ul.appendChild(li);
      });
      outSection.appendChild(ul);
      this.contentContainer.appendChild(outSection);
    }
  }
}
