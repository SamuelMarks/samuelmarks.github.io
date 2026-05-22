export interface IExecutionProvider {
  name: string;
  isAvailable(): Promise<boolean>;
  init(): Promise<void>;
  execute(graphId: string, inputs: Record<string, unknown>): Promise<Record<string, unknown>>;
}

export class Registry {
  private providers = new Map<string, IExecutionProvider>();

  register(provider: IExecutionProvider): void {
    if (this.providers.has(provider.name)) {
      throw new Error(`Execution Provider ${provider.name} is already registered.`);
    }
    this.providers.set(provider.name, provider);
  }

  get(name: string): IExecutionProvider | undefined {
    return this.providers.get(name);
  }

  list(): string[] {
    return Array.from(this.providers.keys());
  }

  async getAvailableProviders(): Promise<string[]> {
    const available: string[] = [];
    for (const [name, provider] of this.providers.entries()) {
      try {
        if (await provider.isAvailable()) {
          available.push(name);
        }
      } catch (e) {
        console.error(`Error checking provider ${name}`, e);
      }
    }
    return available;
  }
}

export const executionRegistry = new Registry();
