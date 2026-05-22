import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';
import { globalAgent } from '../agent/Runner';

export class AgentInterface extends BaseComponent {
  private chatContainer: HTMLElement;
  private inputField: HTMLInputElement;
  private sendBtn: HTMLButtonElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-agent-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.display = 'flex';
    this.container.style.flexDirection = 'column';

    const header = $create('div', { className: 'property-row' });
    header.appendChild($create('h2', { textContent: 'Autonomous Agent (Agent Loop)' }));

    // 628. Cancelable Tasks
    const abortBtn = $create('button', {
      className: 'action-btn danger small',
      textContent: 'Stop Agent',
    });
    abortBtn.style.marginLeft = 'auto';
    header.appendChild(abortBtn);
    this.container.appendChild(header);

    let activeController: AbortController | null = null;
    abortBtn.addEventListener('click', () => {
      if (activeController) {
        activeController.abort();
        Toast.show('Agent execution aborted', 'warn');
        activeController = null;
      }
    });

    // 623. Pre-built Agent Templates
    const templatesRow = $create('div', { className: 'property-row' });
    templatesRow.style.marginBottom = '10px';
    templatesRow.innerHTML = `
       <button class="action-btn secondary small" onclick="document.querySelector('.ide-agent-container input').value = 'Make this model 20% smaller'">Template: Prune Model</button>
       <button class="action-btn secondary small" style="margin-left: 10px;" onclick="document.querySelector('.ide-agent-container input').value = 'List local directory contents'">Template: Read Files</button>
       <button class="action-btn secondary small" style="margin-left: 10px;" onclick="document.querySelector('.ide-agent-container input').value = 'Compile WGSL for MatMul'">Template: WGSL Codegen</button>
    `;
    this.container.appendChild(templatesRow);

    const info = $create('p', {
      textContent:
        "Ask the agent to perform complex workflows. Examples: 'Make this model 20% smaller' or 'Calculate 2 + 2'.",
      className: 'muted',
      attributes: { style: 'margin-bottom: 15px; font-size: 0.85rem;' },
    });
    this.container.appendChild(info);

    this.chatContainer = $create('div', { className: 'ide-chat-messages' });
    this.chatContainer.style.flex = '1';
    this.chatContainer.style.border = '1px solid var(--color-background-border)';
    this.chatContainer.style.borderRadius = '4px';
    this.chatContainer.style.padding = '10px';
    this.chatContainer.style.overflowY = 'auto';
    this.chatContainer.style.marginBottom = '15px';
    this.chatContainer.style.background = 'var(--color-background-secondary)';

    this.container.appendChild(this.chatContainer);

    const inputRow = $create('div', { className: 'property-row' });
    inputRow.style.marginTop = 'auto';

    this.inputField = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'text', placeholder: 'Send instruction to Agent...' },
    });
    this.inputField.style.flex = '1';
    this.inputField.style.marginRight = '10px';

    this.sendBtn = $create('button', { className: 'action-btn', textContent: 'Send' });

    inputRow.appendChild(this.inputField);
    inputRow.appendChild(this.sendBtn);
    this.container.appendChild(inputRow);
  }

  mount(): void {
    this.bindEvent(this.sendBtn, 'click', this.handleSend.bind(this));
    this.bindEvent(this.inputField, 'keydown', (e: Event) => {
      const ev = e as KeyboardEvent;
      if (ev.key === 'Enter') this.handleSend();
    });

    globalEvents.on('agentLog', (msg: string) => {
      const isAction = msg.includes('[Agent Action]');
      const isObs = msg.includes('[Observation]');

      const el = $create('div', { textContent: msg });
      el.style.marginBottom = '8px';
      el.style.fontFamily = 'monospace';
      el.style.fontSize = '0.9rem';

      if (isAction) el.style.color = '#ffc107';
      else if (isObs) el.style.color = '#198754';
      else if (msg.includes('[User]')) {
        el.style.color = 'var(--color-primary)';
        el.style.fontWeight = 'bold';
      }

      this.chatContainer.appendChild(el);
      this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
    });
  }

  private handleSend(): void {
    const text = this.inputField.value.trim();
    if (!text) return;
    this.inputField.value = '';

    // Create new abort controller
    const activeController = new AbortController();
    // 630. Streaming output happens organically via agentLog emitter.
    // 631. Nested agent failures caught internally inside Runner tools.
    globalAgent.runAgentLoop(text, activeController.signal).catch((e) => {
      globalEvents.emit('agentLog', `[Error] Agent crashed: ${e}`);
    });
  }
}
