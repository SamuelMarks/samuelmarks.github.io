import { IModelGraph, INode } from '../core/IR';

/**
 * 496. Support dynamic shape inference algorithms.
 * 497. Handle models with `?` or `None` in their shape definitions correctly.
 * 498. Implement symbolic shape computation via algebraic constraints.
 */
export class ShapeInference {
  public static infer(graph: IModelGraph): IModelGraph {
    const cloned = JSON.parse(JSON.stringify(graph)) as IModelGraph;

    // Create a dictionary of all known shapes
    const shapeDict = new Map<string, (number | string)[]>();

    cloned.inputs.forEach((i) => shapeDict.set(i.name, [...i.dims]));
    cloned.initializers.forEach((i) => shapeDict.set(i.name, [...i.dims]));

    // Pass forward
    cloned.nodes.forEach((node) => {
      if (node.opType === 'MatMul') {
        const shapeA = shapeDict.get(node.inputs[0]);
        const shapeB = shapeDict.get(node.inputs[1]);

        if (shapeA && shapeB) {
          // A: [..., M, K], B: [..., K, N] -> [..., M, N]
          const outShape = [...shapeA];
          outShape[outShape.length - 1] = shapeB[shapeB.length - 1];

          if (node.outputs[0]) {
            shapeDict.set(node.outputs[0], outShape);
          }
        }
      } else if (node.opType === 'Add' || node.opType === 'Mul') {
        // Broadcast logic simplified
        const shapeA = shapeDict.get(node.inputs[0]);
        const shapeB = shapeDict.get(node.inputs[1]);
        if (shapeA && shapeB) {
          const outShape = shapeA.length > shapeB.length ? [...shapeA] : [...shapeB];
          if (node.outputs[0]) {
            shapeDict.set(node.outputs[0], outShape);
          }
        }
      } else if (node.opType === 'Reshape') {
        // Symbolic computation placeholder
        // If second input is a known initializer, compute exact
        if (node.outputs[0]) {
          shapeDict.set(node.outputs[0], ['?', '?']);
        }
      }
    });

    // Update Output ValueInfos
    cloned.outputs.forEach((out) => {
      const inferred = shapeDict.get(out.name);
      if (inferred) {
        out.dims = inferred;
      }
    });

    return cloned;
  }

  /**
   * 499. Lock dynamic shapes to static values
   */
  public static lockShape(
    graph: IModelGraph,
    tensorName: string,
    staticDims: number[],
  ): IModelGraph {
    const cloned = JSON.parse(JSON.stringify(graph)) as IModelGraph;

    const updateDims = (list: any[]) => {
      const item = list.find((x) => x.name === tensorName);
      if (item) item.dims = [...staticDims];
    };

    updateDims(cloned.inputs);
    updateDims(cloned.outputs);
    updateDims(cloned.initializers);

    // Re-run full inference to propagate the static lock
    return this.infer(cloned);
  }
}
