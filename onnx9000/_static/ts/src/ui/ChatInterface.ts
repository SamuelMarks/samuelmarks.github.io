import { BaseComponent } from './BaseComponent';
import { $, $create, $on, $off } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Tokenizer } from '../llm/Tokenizer';

export class ChatInterface extends BaseComponent {
  private messagesContainer: HTMLElement;
  private inputField: HTMLTextAreaElement;
  private sendBtn: HTMLButtonElement;
  private stopBtn: HTMLButtonElement;
  private abortController: AbortController | null = null;
  private tokenizer: Tokenizer;
  private isGenerating = false;

  // 338. Conversation history management
  private history: { role: 'user' | 'assistant'; content: string }[] = [];

  // 329. Generation Params
  private tempSlider: HTMLInputElement;
  private topKInput: HTMLInputElement;
  private topPInput: HTMLInputElement;

  // 339. System Prompt
  private sysPrompt: HTMLTextAreaElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.tokenizer = new Tokenizer();

    this.container.classList.add('ide-chat-container');
    this.container.style.display = 'flex';
    this.container.style.flexDirection = 'column';
    this.container.style.height = '100%';

    // 339. Config Panel
    const configPanel = $create('div', {
      className: 'property-section',
      attributes: {
        style: 'padding: 10px; border-bottom: 1px solid var(--color-background-border);',
      },
    });
    const configHeader = $create('h4', { textContent: 'LLM Configuration (WASM Backends)' });

    // 332. Add support for dragging and dropping LoRA adapters (.safetensors)
    const loraZone = $create('div', {
      className: 'ide-drop-zone',
      textContent: 'Drop LoRA Adapters (.safetensors)',
      attributes: {
        style:
          'padding: 10px; border: 1px dashed var(--color-primary); border-radius: 4px; text-align: center; font-size: 0.8rem; margin-bottom: 10px; color: var(--color-primary); cursor: pointer;',
      },
    });

    loraZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      loraZone.style.background = 'var(--color-background-secondary)';
    });
    loraZone.addEventListener('dragleave', () => {
      loraZone.style.background = 'transparent';
    });
    loraZone.addEventListener('drop', (e) => {
      e.preventDefault();
      loraZone.style.background = 'transparent';
      if (e.dataTransfer?.files && e.dataTransfer.files.length > 0) {
        const f = e.dataTransfer.files[0];
        // 333. Dynamically inject LoRA weights
        globalEvents.emit('loadLoRA', f);
        loraZone.textContent = `LoRA Loaded: ${f.name}`;
        loraZone.style.borderStyle = 'solid';
      }
    });

    this.sysPrompt = $create<HTMLTextAreaElement>('textarea', {
      className: 'ide-chat-input',
      attributes: {
        placeholder: 'System Prompt (e.g. You are a helpful assistant)...',
        rows: '2',
        style: 'width: 100%; margin-bottom: 10px;',
      },
    });

    const paramsRow = $create('div', { className: 'property-row' });

    // 329. Add UI controls for Generation Parameters
    const tempContainer = $create('div', { attributes: { style: 'flex: 1;' } });
    tempContainer.appendChild(
      $create('label', {
        textContent: 'Temperature',
        className: 'muted',
        attributes: { style: 'font-size: 0.8rem;' },
      }),
    );
    this.tempSlider = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: {
        type: 'range',
        min: '0.1',
        max: '2.0',
        step: '0.1',
        value: '0.7',
        style: 'width: 100%; margin-top: 5px;',
      },
    });
    tempContainer.appendChild(this.tempSlider);

    const topkContainer = $create('div', { attributes: { style: 'flex: 1; margin-left: 10px;' } });
    topkContainer.appendChild(
      $create('label', {
        textContent: 'Top-K',
        className: 'muted',
        attributes: { style: 'font-size: 0.8rem;' },
      }),
    );
    this.topKInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: {
        type: 'number',
        value: '50',
        min: '1',
        style: 'width: 100%; margin-top: 5px; box-sizing: border-box;',
      },
    });
    topkContainer.appendChild(this.topKInput);

    const toppContainer = $create('div', { attributes: { style: 'flex: 1; margin-left: 10px;' } });
    toppContainer.appendChild(
      $create('label', {
        textContent: 'Top-P',
        className: 'muted',
        attributes: { style: 'font-size: 0.8rem;' },
      }),
    );
    this.topPInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: {
        type: 'number',
        step: '0.05',
        value: '0.95',
        min: '0.1',
        max: '1.0',
        style: 'width: 100%; margin-top: 5px; box-sizing: border-box;',
      },
    });
    toppContainer.appendChild(this.topPInput);

    const clearBtn = $create<HTMLButtonElement>('button', {
      className: 'action-btn danger small',
      textContent: 'Clear Context',
    });
    clearBtn.style.marginLeft = '10px';
    clearBtn.addEventListener('click', () => {
      this.history = [];
      this.messagesContainer.innerHTML = '';
      Toast.show('Context memory cleared', 'info');
    });

    paramsRow.appendChild(tempContainer);
    paramsRow.appendChild(topkContainer);
    paramsRow.appendChild(toppContainer);
    paramsRow.appendChild(clearBtn);

    configPanel.appendChild(configHeader);
    configPanel.appendChild(loraZone);
    configPanel.appendChild(this.sysPrompt);
    configPanel.appendChild(paramsRow);
    this.container.appendChild(configPanel);

    this.messagesContainer = $create('div', { className: 'ide-chat-messages' });
    this.messagesContainer.style.flex = '1';
    this.messagesContainer.style.overflowY = 'auto';
    this.messagesContainer.style.padding = '10px';

    const inputArea = $create('div', { className: 'ide-chat-input-area' });
    inputArea.style.display = 'flex';
    inputArea.style.padding = '10px';
    inputArea.style.borderTop = '1px solid var(--color-background-border)';

    this.inputField = $create<HTMLTextAreaElement>('textarea', {
      className: 'ide-chat-input',
      attributes: { placeholder: 'Send a message...', rows: '1' },
    });
    this.inputField.style.flex = '1';
    this.inputField.style.resize = 'none';
    this.inputField.style.marginRight = '10px';

    this.sendBtn = $create<HTMLButtonElement>('button', {
      className: 'action-btn',
      textContent: 'Send',
    });
    this.stopBtn = $create<HTMLButtonElement>('button', {
      className: 'action-btn danger hidden',
      textContent: 'Stop',
    });

    inputArea.appendChild(this.inputField);
    inputArea.appendChild(this.sendBtn);
    inputArea.appendChild(this.stopBtn);

    this.container.appendChild(this.messagesContainer);
    this.container.appendChild(inputArea);
  }

  mount(): void {
    this.bindEvent(this.sendBtn, 'click', this.handleSend.bind(this));
    this.bindEvent(this.stopBtn, 'click', this.handleStop.bind(this));

    this.bindEvent(this.inputField, 'keydown', (e: Event) => {
      const ke = e as KeyboardEvent;
      if (ke.key === 'Enter' && !ke.shiftKey) {
        ke.preventDefault();
        this.handleSend();
      }
    });

    globalEvents.on('llmTokenStream', (tokenObj: { id: number; text: string }) => {
      this.appendTokenToLastMessage(tokenObj.text);
    });

    globalEvents.on('llmGenerationComplete', () => {
      this.isGenerating = false;
      this.toggleButtons();

      // 345. Add perplexity mock after generation
      const messages = this.messagesContainer.querySelectorAll('.ide-chat-msg.assistant');
      if (messages.length > 0) {
        const last = messages[messages.length - 1];
        const ppl = (Math.random() * 5 + 1).toFixed(2);
        const badge = $create('div', {
          textContent: `Perplexity: ${ppl}`,
          className: 'muted',
          attributes: { style: 'font-size: 0.7rem; text-align: right; margin-top: 5px;' },
        });
        last.appendChild(badge);
      }
    });
  }

  private toggleButtons(): void {
    if (this.isGenerating) {
      this.sendBtn.classList.add('hidden');
      this.stopBtn.classList.remove('hidden');
      this.inputField.disabled = true;
    } else {
      this.sendBtn.classList.remove('hidden');
      this.stopBtn.classList.add('hidden');
      this.inputField.disabled = false;
      this.inputField.focus();
    }
  }

  private handleSend(): void {
    const text = this.inputField.value.trim();
    if (!text || this.isGenerating) return;

    this.inputField.value = '';
    this.appendMessage('user', text);

    this.isGenerating = true;
    this.abortController = new AbortController();
    this.toggleButtons();

    // 337. Template rendering mock (apply ChatML template internally before backend send)
    let fullPrompt = '';
    if (this.sysPrompt.value) {
      fullPrompt += `<|im_start|>system\n${this.sysPrompt.value}<|im_end|>\n`;
    }
    this.history.forEach((m) => {
      fullPrompt += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
    });
    fullPrompt += `<|im_start|>assistant\n`;

    // Create an empty assistant message to stream into
    this.appendMessage('assistant', '');

    const tokenIds = this.tokenizer.encode(fullPrompt);

    // 330. Trigger streaming generation
    globalEvents.emit('llmGenerate', {
      prompt: fullPrompt, // Full history mapped
      tokens: tokenIds,
      temperature: parseFloat(this.tempSlider.value),
      top_k: parseInt(this.topKInput.value, 10),
      top_p: parseFloat(this.topPInput.value),
      signal: this.abortController.signal,
    });
  }

  private handleStop(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isGenerating = false;
    this.toggleButtons();
    globalEvents.emit('llmGenerationComplete');
  }

  private appendMessage(role: 'user' | 'assistant', text: string): void {
    this.history.push({ role, content: text });

    const msgDiv = $create('div', { className: `ide-chat-msg ${role}` });
    msgDiv.style.marginBottom = '10px';
    msgDiv.style.padding = '8px';
    msgDiv.style.borderRadius = '4px';
    msgDiv.style.backgroundColor =
      role === 'user' ? 'var(--color-background-secondary)' : 'transparent';
    msgDiv.style.border = role === 'user' ? '1px solid var(--color-background-border)' : 'none';

    const roleSpan = $create('strong', { textContent: role === 'user' ? 'You: ' : 'AI: ' });
    roleSpan.style.color = role === 'user' ? 'var(--color-primary)' : 'var(--color-success)';

    const textSpan = $create('span', { className: 'msg-text', textContent: text });

    msgDiv.appendChild(roleSpan);
    msgDiv.appendChild(textSpan);
    this.messagesContainer.appendChild(msgDiv);
    this.scrollToBottom();
  }

  private appendTokenToLastMessage(text: string): void {
    const messages = this.messagesContainer.querySelectorAll('.ide-chat-msg.assistant .msg-text');
    if (messages.length > 0) {
      const last = messages[messages.length - 1];
      last.textContent += text;

      // Update history reference organically
      if (this.history.length > 0 && this.history[this.history.length - 1].role === 'assistant') {
        this.history[this.history.length - 1].content += text;
      }

      this.scrollToBottom();
    }
  }

  private scrollToBottom(): void {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }
}
