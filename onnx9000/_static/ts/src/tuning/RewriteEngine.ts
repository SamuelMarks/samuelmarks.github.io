import { IModelGraph, INode } from '../core/IR';

// 488. Expose an API for custom rewrite rules
export type RewriteRule = (graph: IModelGraph) => { mutated: boolean; newGraph: IModelGraph };

export class CustomRewriteEngine {
  private rules: Map<string, RewriteRule> = new Map();

  public registerRule(name: string, rule: RewriteRule): void {
    this.rules.set(name, rule);
  }

  public applyAll(graph: IModelGraph): IModelGraph {
    let current = graph;
    let mutatedOverall = false;

    // Apply until fixpoint or max iterations
    let iterations = 0;
    while (iterations < 10) {
      let mutatedThisPass = false;

      this.rules.forEach((rule, name) => {
        try {
          const { mutated, newGraph } = rule(current);
          if (mutated) {
            current = newGraph;
            mutatedThisPass = true;
            mutatedOverall = true;
            console.log(`[RewriteEngine] Applied rule: ${name}`);
          }
        } catch (e) {
          console.error(`Rule ${name} failed:`, e);
        }
      });

      if (!mutatedThisPass) break;
      iterations++;
    }

    return current;
  }
}

export const globalRewriteEngine = new CustomRewriteEngine();

// Example layer fusion auto-tuning rule (492)
globalRewriteEngine.registerRule('FuseConvBatchNormRelu', (graph: IModelGraph) => {
  // Deep clone
  const newGraph: IModelGraph = JSON.parse(JSON.stringify(graph));
  for (let i = 0; i < graph.initializers.length; i++) {
    if (graph.initializers[i].rawData) {
      newGraph.initializers[i].rawData = graph.initializers[i].rawData;
    }
  }

  let mutated = false;
  const nodesToRemove = new Set<string>();

  for (let i = 0; i < newGraph.nodes.length - 2; i++) {
    const conv = newGraph.nodes[i];
    if (conv.opType !== 'Conv') continue;

    const bn = newGraph.nodes.find(
      (n) => n.inputs[0] === conv.outputs[0] && n.opType === 'BatchNormalization',
    );
    if (!bn) continue;

    const relu = newGraph.nodes.find((n) => n.inputs[0] === bn.outputs[0] && n.opType === 'Relu');
    if (!relu) continue;

    // We found a chain: Conv -> BN -> Relu
    // Fuse them
    conv.opType = 'FusedConvBNRelu';
    conv.outputs = relu.outputs; // Conv bypasses the other two and outputs directly what Relu outputted

    nodesToRemove.add(bn.name);
    nodesToRemove.add(relu.name);
    mutated = true;
  }

  if (mutated) {
    newGraph.nodes = newGraph.nodes.filter((n) => !nodesToRemove.has(n.name));
  }

  return { mutated, newGraph };
});
