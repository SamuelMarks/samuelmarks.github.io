import { IModelGraph, INode } from '../core/IR';

// Abstract intermediate representation for lowered nodes
export interface ILoweredNode {
  id: string;
  type: string;
  inputs: string[];
  outputs: string[];
  metadata: unknown;
}

export interface ITIRGraph {
  nodes: ILoweredNode[];
  inputs: string[];
  outputs: string[];
}

export class Lowering {
  static lower(model: IModelGraph): ITIRGraph {
    const tirGraph: ITIRGraph = {
      nodes: [],
      inputs: model.inputs.map((i) => i.name),
      outputs: model.outputs.map((o) => o.name),
    };

    for (const node of model.nodes) {
      // Very basic MLIR/TIR mapping stub
      const lowered: ILoweredNode = {
        id: node.name,
        type: this.mapOpToTIR(node.opType),
        inputs: [...node.inputs],
        outputs: [...node.outputs],
        metadata: { ...node.attributes },
      };

      tirGraph.nodes.push(lowered);
    }

    return tirGraph;
  }

  private static mapOpToTIR(opType: string): string {
    switch (opType) {
      case 'Add':
        return 'tir.add';
      case 'Sub':
        return 'tir.sub';
      case 'Mul':
        return 'tir.mul';
      case 'Div':
        return 'tir.div';
      case 'MatMul':
        return 'tir.matmul';
      case 'Gemm':
        return 'tir.gemm';
      case 'Relu':
        return 'tir.relu';
      case 'Constant':
        return 'tir.constant';
      default:
        return `tir.generic.${opType.toLowerCase()}`;
    }
  }
}
