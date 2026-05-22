/**
 * Minimal zero-dependency tokenizer stub for BPE/WordPiece tokenization.
 * In a real implementation, this would load `tokenizer.json` and construct a Trie.
 */
export class Tokenizer {
  private vocab = new Map<string, number>();
  private decodes = new Map<number, string>();

  constructor() {
    // Stub vocab
    this.vocab.set('<|endoftext|>', 50256);
    this.decodes.set(50256, '<|endoftext|>');
  }

  /**
   * Load vocabulary from a parsed JSON manifest
   */
  loadVocab(json: any): void {
    if (json.model && json.model.vocab) {
      for (const [token, id] of Object.entries(json.model.vocab)) {
        this.vocab.set(token, id as number);
        this.decodes.set(id as number, token);
      }
    }
  }

  /**
   * Encode a string into an array of integer token IDs.
   */
  encode(text: string): number[] {
    const tokens: number[] = [];
    // Super naive stub for demonstration: split by space, map if exists, else assign arbitrary hash.
    const words = text.split(/(\s+)/);
    for (const w of words) {
      if (this.vocab.has(w)) {
        tokens.push(this.vocab.get(w)!);
      } else {
        // Fallback hash logic for stub
        let hash = 0;
        for (let i = 0; i < w.length; i++) hash = (hash << 5) - hash + w.charCodeAt(i);
        tokens.push(Math.abs(hash) % 50000);
        this.decodes.set(tokens[tokens.length - 1], w); // Temp cache for decode
      }
    }
    return tokens;
  }

  /**
   * Decode an array of integer token IDs back into a string.
   */
  decode(tokens: number[]): string {
    return tokens.map((t) => this.decodes.get(t) || `[UNK:${t}]`).join('');
  }
}
