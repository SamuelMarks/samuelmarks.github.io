import { IModelGraph, INode } from '../core/IR';
import { GraphSurgeon } from '../surgeon/GraphSurgeon';

export interface IMutationAction {
  type: 'swap_op' | 'change_attr' | 'prune';
  targetNode: string;
  payload: any;
}

/**
 * 482. Implements Neural Architecture Search (NAS) primitives.
 */
export class NASPrimitives {
  /**
   * 483. Search Space: Randomly mutates the kernel size or stride of Conv/Pool ops
   */
  public static mutateConvKernel(graph: IModelGraph): IModelGraph {
    const cloned = JSON.parse(JSON.stringify(graph)) as IModelGraph;

    const convNodes = cloned.nodes.filter((n) => n.opType === 'Conv' || n.opType === 'MaxPool');
    if (convNodes.length === 0) return cloned;

    // Pick random node
    const target = convNodes[Math.floor(Math.random() * convNodes.length)];

    // 485. Genetic mutation: swap 3x3 to 5x5 or 1x1
    const kernels = [
      [1, 1],
      [3, 3],
      [5, 5],
    ];
    const newKernel = kernels[Math.floor(Math.random() * kernels.length)];

    if (!target.attributes) target.attributes = {};
    target.attributes.kernel_shape = newKernel;

    return cloned;
  }

  /**
   * 485. Creates a population of mutated graphs
   */
  public static generatePopulation(baseGraph: IModelGraph, size: number): IModelGraph[] {
    const population: IModelGraph[] = [];
    for (let i = 0; i < size; i++) {
      let mutated = this.mutateConvKernel(baseGraph);

      // Randomly apply Surgeon pruning
      if (Math.random() > 0.5) {
        const surgeon = new GraphSurgeon(mutated);
        surgeon.sparsify(1e-3);
        mutated = surgeon.getModel();
      }

      population.push(mutated);
    }
    return population;
  }

  /**
   * 484. Micro-benchmark stub to score graphs based on parameter count (as a proxy for latency)
   */
  public static scoreGraph(graph: IModelGraph): number {
    let score = 0;
    // Lower is better (fewer nodes, fewer params)
    score += graph.nodes.length * 10;

    graph.initializers.forEach((t) => {
      let size = 1;
      t.dims.forEach((d) => (size *= d));
      score += size;
    });

    return score;
  }
}
