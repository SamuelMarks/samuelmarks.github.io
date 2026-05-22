import { BaseComponent } from './BaseComponent';
import { $, $create } from '../core/DOM';
import { globalEvents } from '../core/State';
import { Toast } from './Toast';

// Minimal BM25 implementation logic for pure text searching (349)
class SimpleBM25 {
  private documents: string[] = [];
  private termFrequencies: Map<string, number>[] = [];
  private documentFrequencies: Map<string, number> = new Map();
  private avgDocLength = 0;

  private k1 = 1.2;
  private b = 0.75;

  addDocument(doc: string): void {
    const terms = this.tokenize(doc);
    const termFreq = new Map<string, number>();

    for (const term of terms) {
      termFreq.set(term, (termFreq.get(term) || 0) + 1);
    }

    for (const term of termFreq.keys()) {
      this.documentFrequencies.set(term, (this.documentFrequencies.get(term) || 0) + 1);
    }

    this.documents.push(doc);
    this.termFrequencies.push(termFreq);

    // Recalculate average doc length
    let totalLength = 0;
    this.termFrequencies.forEach((freq) => {
      for (const count of freq.values()) totalLength += count;
    });
    this.avgDocLength = totalLength / this.documents.length;
  }

  search(query: string): { index: number; score: number; doc: string }[] {
    const queryTerms = this.tokenize(query);
    const scores = this.documents.map((doc, idx) => {
      let score = 0;
      const termFreq = this.termFrequencies[idx];
      let docLength = 0;
      for (const count of termFreq.values()) docLength += count;

      for (const term of queryTerms) {
        if (!termFreq.has(term)) continue;

        const df = this.documentFrequencies.get(term) || 1;
        const idf = Math.log(1 + (this.documents.length - df + 0.5) / (df + 0.5));
        const tf = termFreq.get(term)!;

        const numerator = tf * (this.k1 + 1);
        const denominator = tf + this.k1 * (1 - this.b + this.b * (docLength / this.avgDocLength));

        score += idf * (numerator / denominator);
      }
      return { index: idx, score, doc };
    });

    return scores.filter((s) => s.score > 0).sort((a, b) => b.score - a.score);
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, '')
      .split(/\s+/)
      .filter((t) => t.length > 0);
  }
}

export class RAGInterface extends BaseComponent {
  private bm25 = new SimpleBM25();
  private fileInput: HTMLInputElement;
  private uploadBtn: HTMLButtonElement;
  private queryInput: HTMLInputElement;
  private searchBtn: HTMLButtonElement;
  private resultsContainer: HTMLElement;

  constructor(containerId: string | HTMLElement) {
    super(containerId);

    this.container.classList.add('ide-rag-container');
    this.container.style.padding = '20px';
    this.container.style.height = '100%';
    this.container.style.overflowY = 'auto';

    const header = $create('h2', { textContent: 'Retrieval-Augmented Generation (RAG)' });
    this.container.appendChild(header);

    // 350. Add UI to upload .txt files, parse text, and chunk it
    const uploadSection = $create('div', { className: 'property-section' });
    uploadSection.appendChild($create('h3', { textContent: '1. Upload Knowledge Base' }));

    this.fileInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'file', accept: '.txt,.md,.json' },
    });

    this.uploadBtn = $create<HTMLButtonElement>('button', {
      className: 'action-btn secondary',
      textContent: 'Chunk & Index File',
    });
    this.uploadBtn.style.marginTop = '10px';

    uploadSection.appendChild(this.fileInput);
    uploadSection.appendChild(this.uploadBtn);
    this.container.appendChild(uploadSection);

    // 349. BM25 Search Interface
    const searchSection = $create('div', { className: 'property-section' });
    searchSection.appendChild($create('h3', { textContent: '2. Local Vector / Text Search' }));

    this.queryInput = $create<HTMLInputElement>('input', {
      className: 'ide-file-input',
      attributes: { type: 'text', placeholder: 'Search knowledge base...' },
    });

    this.searchBtn = $create<HTMLButtonElement>('button', {
      className: 'action-btn',
      textContent: 'Search',
    });
    this.searchBtn.style.marginTop = '10px';

    this.resultsContainer = $create('div', { className: 'rag-results-container' });
    this.resultsContainer.style.marginTop = '15px';

    searchSection.appendChild(this.queryInput);
    searchSection.appendChild(this.searchBtn);
    searchSection.appendChild(this.resultsContainer);
    this.container.appendChild(searchSection);
  }

  mount(): void {
    this.bindEvent(this.uploadBtn, 'click', this.handleUpload.bind(this));
    this.bindEvent(this.searchBtn, 'click', this.handleSearch.bind(this));
  }

  private async handleUpload(): void {
    if (!this.fileInput.files || this.fileInput.files.length === 0) {
      Toast.show('Please select a text file first', 'warn');
      return;
    }

    const file = this.fileInput.files[0];
    Toast.show(`Indexing ${file.name}...`, 'info');

    try {
      const text = await file.text();
      // Simple chunking strategy: split by double newlines or paragraphs
      const chunks = text.split(/\n\s*\n/).filter((c) => c.trim().length > 0);

      chunks.forEach((chunk) => {
        this.bm25.addDocument(chunk);
      });

      Toast.show(`Indexed ${chunks.length} chunks successfully using BM25.`, 'success');
    } catch (e) {
      Toast.show('Failed to read file', 'error');
    }
  }

  private handleSearch(): void {
    const query = this.queryInput.value.trim();
    if (!query) return;

    const results = this.bm25.search(query);
    this.resultsContainer.innerHTML = '';

    if (results.length === 0) {
      this.resultsContainer.innerHTML = "<p class='muted'>No matches found.</p>";
      return;
    }

    const topK = Math.min(3, results.length);
    for (let i = 0; i < topK; i++) {
      const res = results[i];
      const resultDiv = $create('div', { className: 'property-row' });
      resultDiv.style.flexDirection = 'column';
      resultDiv.style.border = '1px solid var(--color-background-border)';
      resultDiv.style.padding = '10px';
      resultDiv.style.marginBottom = '10px';
      resultDiv.style.borderRadius = '4px';
      resultDiv.style.background = 'var(--color-background-secondary)';

      const scoreSpan = $create('strong', { textContent: `Score: ${res.score.toFixed(3)}` });
      const docP = $create('p', { textContent: res.doc });
      docP.style.marginTop = '5px';
      docP.style.fontFamily = 'monospace';
      docP.style.fontSize = '0.85rem';

      resultDiv.appendChild(scoreSpan);
      resultDiv.appendChild(docP);
      this.resultsContainer.appendChild(resultDiv);
    }

    Toast.show(`Found ${results.length} matches. Showing top ${topK}.`, 'success');

    // 353. Emit context to main chat
    const contextStr = results
      .slice(0, topK)
      .map((r) => r.doc)
      .join('\n\n');
    globalEvents.emit('ragContextUpdated', contextStr);
  }
}
