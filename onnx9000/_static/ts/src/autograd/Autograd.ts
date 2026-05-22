import { IModelGraph, INode } from '../core/IR';

export class Autograd {
  private model: IModelGraph;
  private gradients = new Set<string>();

  constructor(model: IModelGraph) {
    this.model = JSON.parse(JSON.stringify(model)); // Deep clone
  }

  getModel(): IModelGraph {
    return this.model;
  }

  // 213. Topologically sort and generate forward tape
  private getForwardTape(): INode[] {
    const nodes = this.model.nodes;
    const sorted: INode[] = [];
    const visited = new Set<string>();
    const tempVisited = new Set<string>();
    const nameToNode = new Map<string, INode>();

    nodes.forEach((n) => nameToNode.set(n.name, n));

    const visit = (nodeName: string) => {
      if (tempVisited.has(nodeName)) throw new Error(`Cycle detected`);
      if (visited.has(nodeName)) return;

      tempVisited.add(nodeName);

      const node = nameToNode.get(nodeName);
      if (node) {
        node.inputs.forEach((inp) => {
          const producer = nodes.find((n) => n.outputs.includes(inp));
          if (producer) visit(producer.name);
        });

        tempVisited.delete(nodeName);
        visited.add(nodeName);
        sorted.push(node);
      }
    };

    nodes.forEach((n) => {
      if (!visited.has(n.name)) visit(n.name);
    });

    return sorted;
  }

  // 214 & 215. Implement backward passes and inject nodes
  generateBackwardPass(): void {
    const tape = this.getForwardTape();
    const backwardNodes: INode[] = [];

    // Assume a scalar loss node exists or target the last output
    const lossName = this.model.outputs[this.model.outputs.length - 1]?.name || 'Loss';
    this.gradients.add(`d_${lossName}`);

    // We walk the tape in reverse to generate the VJPs
    for (let i = tape.length - 1; i >= 0; i--) {
      const node = tape[i];

      // Basic VJP implementations
      if (node.opType === 'Add') {
        // dL/dA = dL/dY, dL/dB = dL/dY (broadcasting not handled in stub)
        const dY = `d_${node.outputs[0]}`;
        const dA = `d_${node.inputs[0]}`;
        const dB = `d_${node.inputs[1]}`;

        backwardNodes.push({
          name: `${node.name}_Backward_Add_A`,
          opType: 'Identity', // Stub representing passing gradient backward
          inputs: [dY],
          outputs: [dA],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
        backwardNodes.push({
          name: `${node.name}_Backward_Add_B`,
          opType: 'Identity',
          inputs: [dY],
          outputs: [dB],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
      } else if (node.opType === 'MatMul') {
        // dL/dA = dY @ B.T, dL/dB = A.T @ dY
        const dY = `d_${node.outputs[0]}`;
        const A = node.inputs[0];
        const B = node.inputs[1];
        const dA = `d_${A}`;
        const dB = `d_${B}`;

        // Transpose B
        backwardNodes.push({
          name: `${node.name}_Backward_TransB`,
          opType: 'Transpose',
          inputs: [B],
          outputs: [`${B}_T`],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
        // dA = dY @ B.T
        backwardNodes.push({
          name: `${node.name}_Backward_MatMulA`,
          opType: 'MatMul',
          inputs: [dY, `${B}_T`],
          outputs: [dA],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });

        // Transpose A
        backwardNodes.push({
          name: `${node.name}_Backward_TransA`,
          opType: 'Transpose',
          inputs: [A],
          outputs: [`${A}_T`],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
        // dB = A.T @ dY
        backwardNodes.push({
          name: `${node.name}_Backward_MatMulB`,
          opType: 'MatMul',
          inputs: [`${A}_T`, dY],
          outputs: [dB],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
      } else if (node.opType === 'Relu') {
        // dL/dX = dY * (X > 0)
        // Left as an exercise or extended later. Minimal stub.
        const dY = `d_${node.outputs[0]}`;
        const dX = `d_${node.inputs[0]}`;
        backwardNodes.push({
          name: `${node.name}_Backward_Relu`,
          opType: 'Identity', // Stub
          inputs: [dY],
          outputs: [dX],
          attributes: { is_backward: { name: 'is_backward', type: 'INT', i: 1 } },
        });
      }
    }

    // Inject the backward nodes into the graph
    this.model.nodes = this.model.nodes.concat(backwardNodes);
  }

  // 218. Append Loss Node
  appendLoss(type: 'CrossEntropy' | 'MSE'): void {
    const finalOut = this.model.outputs[this.model.outputs.length - 1];
    if (!finalOut) throw new Error('No graph outputs found to attach loss');

    const labelsName = 'Target_Labels';
    this.model.inputs.push({
      name: labelsName,
      type: finalOut.type,
    });

    const lossName = 'Loss';
    this.model.nodes.push({
      name: 'Loss_Calculation',
      opType: type === 'MSE' ? 'MSELoss' : 'SoftmaxCrossEntropyLoss',
      inputs: [finalOut.name, labelsName],
      outputs: [lossName],
      attributes: { is_loss: { name: 'is_loss', type: 'INT', i: 1 } },
    });

    this.model.outputs.push({ name: lossName, type: { elemType: 1, shape: [1] } });
  }

  // 220. Inject Optimizer Step
  appendOptimizer(type: 'SGD' | 'Adam', lr: number = 0.01): void {
    // Collect all initializers (weights) and their corresponding gradients
    const initializers = this.model.initializers.map((i) => i.name);

    initializers.forEach((w) => {
      const dw = `d_${w}`;
      // In a real graph, we verify dw was generated
      this.model.nodes.push({
        name: `Opt_Update_${w}`,
        opType: type, // SGD or Adam optimizer operator
        inputs: [w, dw],
        outputs: [w], // In-place update
        attributes: {
          lr: { name: 'lr', type: 'FLOAT', f: lr },
          is_optimizer: { name: 'is_optimizer', type: 'INT', i: 1 },
        },
      });
    });
  }
}
