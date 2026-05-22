import { IModelGraph } from '../core/IR';
import { globalEvents } from '../core/State';

export interface IBenchmarkResult {
  modelName: string;
  totalSamples: number;
  averageLatencyMs: number;
  throughputIPS: number; // Inferences per second
  accuracy?: number; // Optional if dataset has labels
}

/**
 * 601. Create a `benchmarks/Suite.ts` engine entirely in-browser.
 */
export class BenchmarkSuite {
  /**
   * 602. Download standard micro-datasets (e.g., MNIST, CIFAR-10) directly into the UI.
   * Mocking the dataset fetching logic for now to avoid large binary dependencies.
   */
  public static async loadDataset(
    name: 'MNIST' | 'CIFAR10',
  ): Promise<{ inputs: Float32Array[]; labels: number[] }> {
    // Mock 1000 samples
    const count = 1000;
    const inputs: Float32Array[] = [];
    const labels: number[] = [];

    const size = name === 'MNIST' ? 28 * 28 : 3 * 32 * 32;

    for (let i = 0; i < count; i++) {
      // Random synthetic data
      const t = new Float32Array(size);
      for (let j = 0; j < size; j++) t[j] = Math.random();
      inputs.push(t);
      labels.push(Math.floor(Math.random() * 10)); // 10 classes
    }

    return { inputs, labels };
  }

  /**
   * 603. Run end-to-end inference passes across 1000+ samples automatically.
   * 604. Collect latency, throughput, and accuracy metrics.
   */
  public static async run(
    model: IModelGraph,
    datasetName: 'MNIST' | 'CIFAR10',
  ): Promise<IBenchmarkResult> {
    const { inputs, labels } = await this.loadDataset(datasetName);

    let totalLatency = 0;
    let correctPredictions = 0;

    const tStart = performance.now();

    for (let i = 0; i < inputs.length; i++) {
      const t0 = performance.now();

      // Mock inference execution
      // In reality, this would await provider.execute({ "input": inputs[i] })
      await new Promise((resolve) => setTimeout(resolve, Math.random() * 2 + 1)); // 1-3ms sleep

      const predictedClass = Math.floor(Math.random() * 10); // Mock prediction

      const t1 = performance.now();
      totalLatency += t1 - t0;

      if (predictedClass === labels[i]) {
        correctPredictions++;
      }

      if (i % 100 === 0) {
        globalEvents.emit('benchmarkProgress', { current: i, total: inputs.length });
      }
    }

    const tEnd = performance.now();
    const totalTimeMs = tEnd - tStart;

    return {
      modelName: model.name,
      totalSamples: inputs.length,
      averageLatencyMs: totalLatency / inputs.length,
      throughputIPS: inputs.length / (totalTimeMs / 1000),
      accuracy: (correctPredictions / inputs.length) * 100,
    };
  }
}
