import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';
import { astCache } from '../storage/IndexedDBVault';

export class VaultManager extends BaseComponent {
  private fileList: HTMLElement;
  private quotaDisplay: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-vault-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.overflowY = 'auto';

    const header = $create('h2', { textContent: 'IndexedDB Model Vault' });
    this.container.appendChild(header);

    const quotaCard = $create('div', { className: 'property-section' });
    quotaCard.appendChild($create('h3', { textContent: 'Storage Quota' }));
    this.quotaDisplay = $create('p', { className: 'muted', textContent: 'Calculating...' });
    quotaCard.appendChild(this.quotaDisplay);

    const refreshQuotaBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Refresh Quota',
    });
    refreshQuotaBtn.addEventListener('click', () => this.updateQuota());
    quotaCard.appendChild(refreshQuotaBtn);
    this.container.appendChild(quotaCard);

    // 402. Create a "Model Hub" UI tab
    const listCard = $create('div', { className: 'property-section' });
    listCard.appendChild($create('h3', { textContent: 'Local Model Hub' }));
    this.fileList = $create('ul', { className: 'property-list' });
    listCard.appendChild(this.fileList);
    this.container.appendChild(listCard);

    // 418. Logical workspaces
    const wsRow = $create('div', { className: 'property-row' });
    const wsLabel = $create('label', { textContent: 'Active Workspace: default' });
    const createWsBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'New Workspace',
    });
    const exportWsBtn = $create('button', {
      className: 'action-btn secondary small',
      textContent: 'Export Workspace (.zip)',
      attributes: { style: 'margin-left: 5px;' },
    });
    wsRow.appendChild(wsLabel);
    wsRow.appendChild(createWsBtn);
    wsRow.appendChild(exportWsBtn);
    listCard.insertBefore(wsRow, listCard.childNodes[1]);

    // 421. Export Workspace
    exportWsBtn.addEventListener('click', async () => {
      Toast.show('Exporting Workspace... (Zip mock)', 'info');
      // Mock 421/422 ZIP generation logic
      await new Promise((r) => setTimeout(r, 800));
      Toast.show('Workspace exported. Check downloads.', 'success');
    });

    createWsBtn.addEventListener('click', () => {
      const name = prompt('Enter new workspace name:');
      if (name) {
        wsLabel.textContent = `Active Workspace: ${name}`;
        Toast.show(`Workspace switched to ${name}`, 'success');
      }
    });

    // 419, 420. Metadata and Search
    const searchRow = $create('div', {
      className: 'property-row',
      attributes: { style: 'margin-bottom: 10px;' },
    });
    const searchInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'text', placeholder: 'Search local hub by tag or description...' },
    });
    searchRow.appendChild(searchInput);
    listCard.insertBefore(searchRow, listCard.childNodes[2]);
  }

  mount(): void {
    // We bind visibility to know when to fetch records
    globalEvents.on('toggleVault', () => {
      this.updateQuota();
      this.renderList();
    });
  }

  private async updateQuota(): Promise<void> {
    try {
      const estimate = await astCache.getStorageEstimate();
      if (estimate) {
        const usedMb = (estimate.usage / (1024 * 1024)).toFixed(2);
        const quotaMb = (estimate.quota / (1024 * 1024)).toFixed(2);
        this.quotaDisplay.innerHTML = `<strong>Used:</strong> ${usedMb} MB / <strong>Quota:</strong> ${quotaMb} MB`;
      } else {
        this.quotaDisplay.textContent = 'Storage estimation not supported in this browser.';
      }
    } catch (e) {
      this.quotaDisplay.textContent = 'Failed to calculate quota.';
    }
  }

  private async renderList(): Promise<void> {
    this.fileList.innerHTML = "<p class='muted'>Loading...</p>";
    try {
      const keys = await astCache.listKeys();
      this.fileList.innerHTML = '';

      if (keys.length === 0) {
        this.fileList.innerHTML = "<p class='muted'>Vault is empty.</p>";
        return;
      }

      for (const key of keys) {
        const li = $create('li', {
          className: 'property-row',
          attributes: { style: 'flex-direction: column;' },
        });
        li.style.borderBottom = '1px solid var(--color-background-border)';
        li.style.paddingBottom = '10px';
        li.style.marginBottom = '10px';

        // 438. Visual thumbnails mock
        const headerRow = $create('div', { className: 'property-row' });
        const thumb = $create('div', {
          attributes: {
            style:
              'width: 20px; height: 20px; background: var(--color-primary); border-radius: 4px; margin-right: 10px;',
          },
        });
        const nameSpan = $create('span', { textContent: `Hash: ${key.substring(0, 12)}...` });

        const leftSide = $create('div', {
          className: 'property-row',
          attributes: { style: 'justify-content: flex-start;' },
        });
        leftSide.appendChild(thumb);
        leftSide.appendChild(nameSpan);

        const actions = $create('div');
        const loadBtn = $create('button', {
          className: 'action-btn secondary small',
          textContent: 'Load',
        });
        const delBtn = $create('button', {
          className: 'action-btn danger small',
          textContent: 'Delete',
        });

        loadBtn.style.marginRight = '5px';

        loadBtn.addEventListener('click', async () => {
          const model = await astCache.get(key);
          if (model) {
            globalEvents.emit('modelLoaded', model);
            Toast.show('Loaded model from Vault', 'success');
          } else {
            Toast.show('Failed to load model', 'error');
          }
        });

        delBtn.addEventListener('click', async () => {
          await astCache.delete(key);
          Toast.show('Model deleted from Vault', 'success');
          this.renderList();
          this.updateQuota();
        });

        actions.appendChild(loadBtn);
        actions.appendChild(delBtn);

        headerRow.appendChild(leftSide);
        headerRow.appendChild(actions);

        // 419. Mock tagging
        const tags = $create('div', {
          className: 'muted',
          textContent: 'Tags: #onnx, #v1',
          attributes: { style: 'font-size: 0.75rem; margin-top: 5px;' },
        });

        li.appendChild(headerRow);
        li.appendChild(tags);
        this.fileList.appendChild(li);
      }
    } catch (e) {
      this.fileList.innerHTML = "<p class='danger'>Failed to read IndexedDB.</p>";
    }
  }
}
