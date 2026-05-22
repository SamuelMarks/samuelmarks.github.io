import { IModelGraph, INode } from '../core/IR';

export class GraphSurgeon {
  private model: IModelGraph;

  constructor(model: IModelGraph) {
    // Clone to avoid mutating original state unless explicitly asked
    this.model = JSON.parse(JSON.stringify(model));
  }

  getModel(): IModelGraph {
    return this.model;
  }

  // 134. Topological sort
  topologicalSort(): void {
    const nodes = this.model.nodes;
    const sorted: INode[] = [];
    const visited = new Set<string>();
    const tempVisited = new Set<string>();
    const nameToNode = new Map<string, INode>();

    nodes.forEach((n) => nameToNode.set(n.name, n));

    const visit = (nodeName: string) => {
      if (tempVisited.has(nodeName)) throw new Error(`Cycle detected in graph at node ${nodeName}`);
      if (visited.has(nodeName)) return;

      tempVisited.add(nodeName);

      const node = nameToNode.get(nodeName);
      if (node) {
        // Find dependencies (nodes that produce inputs for this node)
        node.inputs.forEach((inp) => {
          const producer = nodes.find((n) => n.outputs.includes(inp));
          if (producer) {
            visit(producer.name);
          }
        });

        tempVisited.delete(nodeName);
        visited.add(nodeName);
        sorted.push(node);
      }
    };

    nodes.forEach((n) => {
      if (!visited.has(n.name)) {
        visit(n.name);
      }
    });

    this.model.nodes = sorted;
  }

  // 135. Dead Code Elimination (DCE)
  pruneUnused(): number {
    let removedCount = 0;
    let changed = true;

    while (changed) {
      changed = false;
      const requiredInputs = new Set<string>();

      // Add all model outputs
      this.model.outputs.forEach((out) => requiredInputs.add(out.name));

      // Add all inputs of remaining nodes
      this.model.nodes.forEach((n) => {
        n.inputs.forEach((inp) => requiredInputs.add(inp));
      });

      const newNodes: INode[] = [];
      for (const node of this.model.nodes) {
        // A node is kept if ANY of its outputs are required by another node OR if it's a graph output
        const isRequired = node.outputs.some((out) => requiredInputs.has(out));
        if (isRequired) {
          newNodes.push(node);
        } else {
          removedCount++;
          changed = true;
        }
      }
      this.model.nodes = newNodes;
    }

    return removedCount;
  }

  // 138. Constant Folding
  foldConstants(): number {
    let foldedCount = 0;

    const initializers = new Set<string>();
    this.model.initializers.forEach((i) => initializers.add(i.name));

    const newNodes: INode[] = [];

    for (const node of this.model.nodes) {
      // 139. Identify purely static subgraphs
      const isStatic = node.inputs.every((inp) => initializers.has(inp) || inp === '');

      if (isStatic && node.opType === 'Reshape') {
        // 144. Implement Reshape into Constant folding
        // Stub: Assume we computed the new shape, replace node
        // Note: True WASM execution of the subgraph goes here
        foldedCount++;

        // Promote output to initializer
        node.outputs.forEach((out) => {
          this.model.initializers.push({
            name: out,
            dataType: 1, // Float
            dims: [1], // Stub dimension
            rawData: new Uint8Array(4), // Stub data
          });
          initializers.add(out);
        });
      } else {
        newNodes.push(node);
      }
    }

    this.model.nodes = newNodes;
    return foldedCount;
  }

  // 143. Remove Identity nodes
  removeIdentity(): number {
    let removedCount = 0;
    const newNodes: INode[] = [];
    const replacements = new Map<string, string>(); // old output -> new output

    for (const node of this.model.nodes) {
      if (node.opType === 'Identity' && node.inputs.length === 1 && node.outputs.length === 1) {
        replacements.set(node.outputs[0], node.inputs[0]);
        removedCount++;
      } else {
        newNodes.push(node);
      }
    }

    // Apply replacements to all subsequent nodes
    for (const node of newNodes) {
      for (let i = 0; i < node.inputs.length; i++) {
        let currentInp = node.inputs[i];
        while (replacements.has(currentInp)) {
          currentInp = replacements.get(currentInp)!;
        }
        node.inputs[i] = currentInp;
      }
    }

    this.model.nodes = newNodes;
    return removedCount;
  }

  // 146. Delete Node and re-wire
  deleteNode(nodeName: string): void {
    const nodeIdx = this.model.nodes.findIndex((n) => n.name === nodeName);
    if (nodeIdx === -1) throw new Error(`Node ${nodeName} not found`);

    const node = this.model.nodes[nodeIdx];
    this.model.nodes.splice(nodeIdx, 1);

    // If node has 1 input and 1 output, we can auto re-wire (like an Identity)
    if (node.inputs.length === 1 && node.outputs.length === 1) {
      const inp = node.inputs[0];
      const out = node.outputs[0];

      for (const n of this.model.nodes) {
        for (let i = 0; i < n.inputs.length; i++) {
          if (n.inputs[i] === out) {
            n.inputs[i] = inp;
          }
        }
      }
    } else {
      // Just delete it, downstream nodes might have dangling inputs now
      // This leaves it up to the user to fix via properties panel or pruneUnused
    }
  }

  // 150. Naive Min-Max INT8 Quantization
  quantizeINT8(): number {
    let quantCount = 0;

    // We iterate over initializers (weights)
    for (let i = 0; i < this.model.initializers.length; i++) {
      const init = this.model.initializers[i];
      if (init.dataType === 1 && init.rawData) {
        // F32
        const floatView = new Float32Array(
          init.rawData.buffer,
          init.rawData.byteOffset,
          init.rawData.byteLength / 4,
        );

        let min = Infinity;
        let max = -Infinity;
        for (let j = 0; j < floatView.length; j++) {
          const val = floatView[j];
          if (val < min) min = val;
          if (val > max) max = val;
        }

        // Avoid division by zero
        if (min === max) {
          max = min + 1;
        }

        // asymmetric
        const scale = (max - min) / 255;
        const zp = Math.round(-min / scale);

        const int8View = new Uint8Array(floatView.length);
        for (let j = 0; j < floatView.length; j++) {
          let q = Math.round(floatView[j] / scale) + zp;
          if (q < 0) q = 0;
          if (q > 255) q = 255;
          int8View[j] = q;
        }

        // Update initializer
        init.dataType = 2; // U8
        init.rawData = new Uint8Array(int8View.buffer);

        // Create Quantize nodes in the graph to dequantize on the fly
        // For simplicity in this demo, we just record the count
        // A true implementation inserts DequantizeLinear(init) replacing the init edge
        quantCount++;
      }
    }

    return quantCount;
  }

  // 153. Magnitude-based pruning
  // 508. Compress pruned weights using CSR format
  private encodeCSR(
    floatArray: Float32Array,
    rows: number,
    cols: number,
  ): { values: Float32Array; colIndices: Int32Array; rowPointers: Int32Array } {
    const values: number[] = [];
    const colIndices: number[] = [];
    const rowPointers: number[] = [0];

    let nnz = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = floatArray[r * cols + c];
        if (val !== 0) {
          values.push(val);
          colIndices.push(c);
          nnz++;
        }
      }
      rowPointers.push(nnz);
    }

    return {
      values: new Float32Array(values),
      colIndices: new Int32Array(colIndices),
      rowPointers: new Int32Array(rowPointers),
    };
  }

  // 159. Generate a new IModelGraph containing only the selected nodes and their boundaries
  extractSubgraph(nodeNames: string[]): IModelGraph | null {
    if (nodeNames.length === 0) return null;
    const nodeSet = new Set(nodeNames);

    const cloned = JSON.parse(JSON.stringify(this.model)) as IModelGraph;

    // Filter nodes
    const newNodes = cloned.nodes.filter((n) => nodeSet.has(n.name));
    if (newNodes.length === 0) return null;

    // Determine required inputs and outputs boundary
    const requiredInputs = new Set<string>();
    const generatedOutputs = new Set<string>();

    newNodes.forEach((n) => {
      n.inputs.forEach((i) => requiredInputs.add(i));
      n.outputs.forEach((o) => generatedOutputs.add(o));
    });

    // Filter initializers
    const newInits = cloned.initializers.filter((i) => requiredInputs.has(i.name));

    // True inputs are those required by the subgraph but not generated within it
    // OR those that were already graph inputs
    const trueInputs = cloned.inputs.filter((i) => requiredInputs.has(i.name));
    const originalInputNames = new Set(cloned.inputs.map((i) => i.name));

    requiredInputs.forEach((req) => {
      if (
        !generatedOutputs.has(req) &&
        !newInits.find((i) => i.name === req) &&
        !trueInputs.find((i) => i.name === req)
      ) {
        // We need to elevate an intermediate tensor into a Graph Input
        const vi = cloned.valueInfo?.find((v) => v.name === req);
        trueInputs.push({
          name: req,
          dims: vi?.type?.shape || ['?'], // fallback
          dataType: 1, // F32 fallback
        });
      }
    });

    // Outputs are what was selected, but we could also just output the terminal nodes of the subgraph
    const terminalOutputs = new Set<string>();
    newNodes.forEach((n) => {
      n.outputs.forEach((o) => {
        // If output is not consumed by any OTHER node in the subgraph, it's a terminal
        const isConsumed = newNodes.some(
          (other) => other.name !== n.name && other.inputs.includes(o),
        );
        if (!isConsumed) terminalOutputs.add(o);
      });
    });

    const trueOutputs = Array.from(terminalOutputs).map((o) => {
      const vi =
        cloned.valueInfo?.find((v) => v.name === o) || cloned.outputs.find((v) => v.name === o);
      return {
        name: o,
        dims: vi?.type?.shape || ['?'],
        dataType: 1,
      };
    });

    // Re-attach memory references securely
    for (let i = 0; i < newInits.length; i++) {
      const orig = this.model.initializers.find((x) => x.name === newInits[i].name);
      if (orig && orig.rawData) newInits[i].rawData = orig.rawData;
    }

    return {
      name: `${cloned.name}_subgraph`,
      docString: `Extracted ${newNodes.length} nodes from ${cloned.name}`,
      producerName: 'onnx9000-surgeon',
      producerVersion: '1.0',
      inputs: trueInputs,
      outputs: trueOutputs,
      initializers: newInits,
      nodes: newNodes,
      valueInfo:
        cloned.valueInfo?.filter(
          (v) => requiredInputs.has(v.name) || terminalOutputs.has(v.name),
        ) || [],
    };
  }

  sparsify(threshold: number): number {
    let prunedCount = 0;

    for (let i = 0; i < this.model.initializers.length; i++) {
      const init = this.model.initializers[i];
      if (init.dataType === 1 && init.rawData) {
        // F32
        const floatView = new Float32Array(
          init.rawData.buffer,
          init.rawData.byteOffset,
          init.rawData.byteLength / 4,
        );

        for (let j = 0; j < floatView.length; j++) {
          if (Math.abs(floatView[j]) < threshold && floatView[j] !== 0) {
            floatView[j] = 0;
            prunedCount++;
          }
        }
        // 154. In a real system, we'd replace this with a SparseTensorProto format
        // For this UI demo, writing zeroes simulates the compression potential via ZIP
      }
    }
    return prunedCount;
  }

  // 160 & 161: Promote / Freeze
  promoteInput(nodeName: string): void {
    const node = this.model.nodes.find((n) => n.name === nodeName);
    if (!node) throw new Error('Node not found');
    // This is a stub for promoting a selected node's static input to a model graph input
    // Actual implementation requires identifying the exact tensor edge.
    // We will just create a generic input.
    this.model.inputs.push({
      name: `promoted_${nodeName}_in`,
      type: { elemType: 1, shape: [1] },
    });
    node.inputs[0] = `promoted_${nodeName}_in`;
  }

  freezeInput(nodeName: string): void {
    const node = this.model.nodes.find((n) => n.name === nodeName);
    if (!node) throw new Error('Node not found');

    // Stub for freezing a dynamic input into an initializer
    const targetInput = node.inputs[0];
    const idx = this.model.inputs.findIndex((i) => i.name === targetInput);
    if (idx !== -1) {
      this.model.inputs.splice(idx, 1);
      this.model.initializers.push({
        name: targetInput,
        dataType: 1,
        dims: [1],
        rawData: new Uint8Array([0, 0, 0, 0]),
      });
    }
  }

  // 167. Algebraic rewriting rule stub
  algebraicRewrite(): number {
    let rewriteCount = 0;
    // Example rule: Gemm(A, B, C) -> MatMul(A, B) + Add(Result, C)
    // Left as stub implementation
    return rewriteCount;
  }
}
