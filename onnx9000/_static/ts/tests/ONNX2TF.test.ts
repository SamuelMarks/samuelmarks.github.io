import { describe, it, expect } from 'vitest';
import { ONNX2TF } from '../src/exporters/ONNX2TF';
import { IModelGraph } from '../src/core/IR';

describe('ONNX2TF (onnx2tf / PINTO0309)', () => {
  it('should export an ONNX graph to a TFLite JSON structure by default', () => {
    const graph: IModelGraph = {
      name: 'test',
      inputs: [{ name: 'in1' }],
      outputs: [{ name: 'out1' }],
      nodes: [
        { name: 'conv1', opType: 'Conv', inputs: ['in1', 'w'], outputs: ['out1'], attributes: {} },
      ],
      initializers: [],
    };

    const exporter = new ONNX2TF(graph);
    const result = JSON.parse(exporter.export());

    expect(result.version).toBe(3);
    expect(result.subgraphs[0].nodes.length).toBe(1);
    expect(result.subgraphs[0].nodes[0].op).toBe('Conv2D');
    expect(result.subgraphs[0].nodes[0].attr.data_format).toBe('NHWC');
  });

  it('should export to a TFJS Graph Model format if target is specified', () => {
    const graph: IModelGraph = {
      name: 'test',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [{ name: 'add1', opType: 'Add', inputs: ['a', 'b'], outputs: ['c'], attributes: {} }],
    };

    const exporter = new ONNX2TF(graph, { target: 'tfjs_graph' });
    const result = JSON.parse(exporter.export());

    expect(result.format).toBe('graph-model');
    expect(result.node.length).toBe(1);
    expect(result.node[0].op).toBe('AddV2');
  });

  it('should inject EdgeTPU specific optimizations if enabled', () => {
    const graph: IModelGraph = {
      name: 'test',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [{ name: 'conv1', opType: 'Conv', inputs: ['in'], outputs: ['out'], attributes: {} }],
    };

    const exporter = new ONNX2TF(graph, { edgeTpuOptimization: true });
    const result = JSON.parse(exporter.export());

    expect(result.subgraphs[0].nodes[0].attr.edge_tpu_padding).toBe('SAME');
  });

  it('should map standard ONNX ops to TensorFlow variants', () => {
    const graph: IModelGraph = {
      name: 'test',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [
        { name: 'n1', opType: 'MatMul', inputs: [], outputs: [], attributes: {} },
        { name: 'n2', opType: 'Gemm', inputs: [], outputs: [], attributes: {} },
        { name: 'n3', opType: 'Relu', inputs: [], outputs: [], attributes: {} },
        { name: 'n4', opType: 'MaxPool', inputs: [], outputs: [], attributes: {} },
        { name: 'n5', opType: 'AveragePool', inputs: [], outputs: [], attributes: {} },
        { name: 'n6', opType: 'Add', inputs: [], outputs: [], attributes: {} },
        { name: 'n7', opType: 'Sub', inputs: [], outputs: [], attributes: {} },
        { name: 'n8', opType: 'Mul', inputs: [], outputs: [], attributes: {} },
        { name: 'n9', opType: 'Div', inputs: [], outputs: [], attributes: {} },
        {
          name: 'n10',
          opType: 'Transpose',
          inputs: [],
          outputs: [],
          attributes: { perm: { type: 'INTS', ints: [0, 2, 3, 1] } },
        },
        { name: 'n11', opType: 'Reshape', inputs: [], outputs: [], attributes: {} },
        { name: 'n12', opType: 'Flatten', inputs: [], outputs: [], attributes: {} },
        { name: 'n13', opType: 'Concat', inputs: [], outputs: [], attributes: {} },
        { name: 'n14', opType: 'Softmax', inputs: [], outputs: [], attributes: {} },
        { name: 'n15', opType: 'UnknownOp', inputs: [], outputs: [], attributes: {} },
        { name: 'n16', opType: 'Transpose', inputs: [], outputs: [], attributes: {} }, // missing perm
      ],
    };

    const exporter = new ONNX2TF(graph);
    const result = JSON.parse(exporter.export());
    const nodes = result.subgraphs[0].nodes;

    expect(nodes[0].op).toBe('MatMul');
    expect(nodes[1].op).toBe('FullyConnected');
    expect(nodes[2].op).toBe('Relu');
    expect(nodes[3].op).toBe('MaxPool');
    expect(nodes[3].attr.data_format).toBe('NHWC');
    expect(nodes[4].op).toBe('AvgPool');
    expect(nodes[4].attr.data_format).toBe('NHWC');
    expect(nodes[5].op).toBe('AddV2');
    expect(nodes[6].op).toBe('Sub');
    expect(nodes[7].op).toBe('Mul');
    expect(nodes[8].op).toBe('RealDiv');
    expect(nodes[9].op).toBe('Transpose');
    expect(nodes[9].attr.perm).toEqual([0, 2, 3, 1]);
    expect(nodes[10].op).toBe('Reshape');
    expect(nodes[11].op).toBe('Flatten');
    expect(nodes[12].op).toBe('ConcatV2');
    expect(nodes[13].op).toBe('Softmax');
    expect(nodes[14].op).toBe('Unsupported_UnknownOp');
    expect(nodes[15].attr.perm).toEqual([]); // fallback to empty array
  });
});
