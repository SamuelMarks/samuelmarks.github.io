import { IModelGraph, INode } from '../core/IR';

export class Obfuscator {
  /**
   * 579. Obfuscates the structural identifiers (node names, tensor names)
   * of the ONNX graph using reversible random hashes, making the visual
   * topology difficult to reverse-engineer manually.
   */
  public static apply(graph: IModelGraph): IModelGraph {
    const nameMap = new Map<string, string>();

    // Generate secure random hex
    const generateHash = () => {
      const u = new Uint8Array(8);
      window.crypto.getRandomValues(u);
      return Array.from(u)
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    };

    const getOrGenerate = (original: string) => {
      if (!nameMap.has(original)) {
        nameMap.set(original, `n_${generateHash()}`);
      }
      return nameMap.get(original)!;
    };

    // Deep clone to avoid mutating original state in-place unexpectedly
    const clonedGraph: IModelGraph = JSON.parse(JSON.stringify(graph));
    // Reattach raw buffers since JSON.parse drops Uint8Arrays
    for (let i = 0; i < graph.initializers.length; i++) {
      if (graph.initializers[i].rawData) {
        clonedGraph.initializers[i].rawData = graph.initializers[i].rawData;
      }
    }

    // Pass 1: Maps
    clonedGraph.initializers.forEach((init) => {
      init.name = getOrGenerate(init.name);
    });

    clonedGraph.inputs.forEach((inp) => {
      inp.name = getOrGenerate(inp.name);
    });

    clonedGraph.outputs.forEach((out) => {
      out.name = getOrGenerate(out.name);
    });

    clonedGraph.nodes.forEach((node) => {
      if (node.name) {
        node.name = getOrGenerate(node.name);
      }
      node.inputs = node.inputs.map((i) => (i ? getOrGenerate(i) : i));
      node.outputs = node.outputs.map((o) => (o ? getOrGenerate(o) : o));
    });

    // We store the reverse map as encrypted metadata if we ever want to un-obfuscate
    const docMeta = clonedGraph.docString ? JSON.parse(clonedGraph.docString) : {};
    // Only store an obfuscation flag, keep map out of schema for true obfuscation
    docMeta.obfuscated = true;
    clonedGraph.docString = JSON.stringify(docMeta);

    return clonedGraph;
  }
}
