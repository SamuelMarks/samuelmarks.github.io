import { describe, it, expect } from 'vitest';
import { MMdnn } from '../src/mmdnn/MMdnn';
import { IModelGraph } from '../src/core/IR';

describe('MMdnn Core Architecture & Converters', () => {
  it('should parse Caffe prototxt to ONNX', () => {
    const caffeProto = `
      layer {
        name: "conv1"
        type: "Convolution"
      }
      layer {
        name: "relu1"
        type: "ReLU"
      }
      layer {
        name: "pool1"
        type: "Pooling"
      }
      layer {
        name: "fc1"
        type: "InnerProduct"
      }
    `;
    const graph = MMdnn.parseToONNX('caffe', caffeProto);
    expect(graph.nodes.length).toBe(4);
    expect(graph.nodes[0].opType).toBe('Conv');
    expect(graph.nodes[1].opType).toBe('Relu');
    expect(graph.nodes[2].opType).toBe('MaxPool');
    expect(graph.nodes[3].opType).toBe('Gemm');
  });

  it('should generate node name if missing in Caffe', () => {
    const caffeProto = `
      layer {
        type: "Convolution"
      }
    `;
    const graph = MMdnn.parseToONNX('caffe', caffeProto);
    expect(graph.nodes[0].name).toBe('node_0');
    expect(graph.nodes[0].opType).toBe('Conv');
  });

  it('should gracefully handle Caffe blocks without types or empty prototxt', () => {
    const caffeProto = `
      layer {
        name: "unknown_but_no_type"
      }
      // some comment }
    `;
    const graph = MMdnn.parseToONNX('caffe', caffeProto);
    expect(graph.nodes.length).toBe(0);
  });

  it('should handle unmapped Caffe layers as Identity', () => {
    const caffeProto = `
      layer {
        name: "unknown"
        type: "WeirdLayer"
      }
    `;
    const graph = MMdnn.parseToONNX('caffe', caffeProto);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should parse MXNet symbol JSON to ONNX', () => {
    const mxnetJson = JSON.stringify({
      nodes: [
        { op: 'null', name: 'data' },
        { op: 'Convolution', name: 'conv1', inputs: [[0, 0, 0]] },
        { op: 'Activation', name: 'relu1', inputs: [[1, 0, 0]] },
        { op: 'FullyConnected', name: 'fc1', inputs: [[2, 0, 0]] },
      ],
    });
    const graph = MMdnn.parseToONNX('mxnet', mxnetJson);
    expect(graph.nodes.length).toBe(3); // null is skipped
    expect(graph.nodes[0].opType).toBe('Conv');
    expect(graph.nodes[1].opType).toBe('Relu');
    expect(graph.nodes[2].opType).toBe('Gemm');
  });

  it('should generate node name and handle missing inputs in MXNet', () => {
    const mxnetJson = JSON.stringify({
      nodes: [{ op: 'Convolution' }],
    });
    const graph = MMdnn.parseToONNX('mxnet', mxnetJson);
    expect(graph.nodes[0].name).toBe('mx_node_0');
    expect(graph.nodes[0].opType).toBe('Conv');
    expect(graph.nodes[0].inputs.length).toBe(0);
  });

  it('should gracefully handle invalid JSON or missing nodes array in MXNet', () => {
    let graph = MMdnn.parseToONNX('mxnet', 'invalid json {');
    expect(graph.nodes.length).toBe(0);

    graph = MMdnn.parseToONNX('mxnet', '{}');
    expect(graph.nodes.length).toBe(0);
  });

  it('should handle unmapped MXNet layers as Identity', () => {
    const mxnetJson = JSON.stringify({
      nodes: [{ op: 'UnknownLayer', name: 'weird' }],
    });
    const graph = MMdnn.parseToONNX('mxnet', mxnetJson);
    expect(graph.nodes[0].opType).toBe('Identity');
  });

  it('should skip falsy nodes in MXNet', () => {
    const mxnetJson = JSON.stringify({
      nodes: [null, { op: 'Convolution' }],
    });
    const graph = MMdnn.parseToONNX('mxnet', mxnetJson);
    expect(graph.nodes.length).toBe(1);
    expect(graph.nodes[0].opType).toBe('Conv');
  });

  it('should correctly passthrough ONNX model in parseToONNX', () => {
    const onnxStr = JSON.stringify({
      name: 'test',
      nodes: [{ opType: 'Conv', name: 'c1', inputs: [], outputs: [], attributes: {} }],
    });
    const graph = MMdnn.parseToONNX('onnx', onnxStr);
    expect(graph.name).toBe('test');
    expect(graph.nodes[0].opType).toBe('Conv');
  });

  it('should throw error for unsupported source framework in parseToONNX', () => {
    expect(() => MMdnn.parseToONNX('cntk' as any, '')).toThrow(
      'Unsupported source framework: cntk',
    );
  });

  it('should export ONNX to PyTorch Code', () => {
    const graph: IModelGraph = {
      name: 'TestPyTorch',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [
        { name: 'conv1', opType: 'Conv', inputs: ['x'], outputs: ['o1'], attributes: {} },
        { name: 'relu1', opType: 'Relu', inputs: ['o1'], outputs: ['o2'], attributes: {} },
        { name: 'fc1', opType: 'Gemm', inputs: ['o2'], outputs: ['o3'], attributes: {} },
        { name: 'unknown1', opType: 'Weird', inputs: ['o3'], outputs: ['o4'], attributes: {} },
      ],
    };

    const code = MMdnn.exportFromONNX('pytorch_code', graph) as string;
    expect(code).toContain('class TestPyTorch(nn.Module):');
    expect(code).toContain('self.conv1 = nn.Conv2d()');
    expect(code).toContain('self.fc1 = nn.Linear()');
    expect(code).toContain('o1 = self.conv1(x)');
    expect(code).toContain('o2 = F.relu(o1)');
    expect(code).toContain('o4 = o3'); // unknown maps to input
  });

  it('should generate PyTorch init with pass if no stateful ops', () => {
    const graph: IModelGraph = {
      name: '',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [{ name: 'relu1', opType: 'Relu', inputs: ['x'], outputs: ['o1'], attributes: {} }],
    };
    const code = MMdnn.exportFromONNX('pytorch_code', graph) as string;
    expect(code).toContain('class ConvertedModel(nn.Module):');
    expect(code).toContain('pass\n');
  });

  it('should map various ONNX types to PyTorch functional equivalents', () => {
    expect(MMdnn.mapONNXToPyTorchFunc('MaxPool', 'a')).toBe('F.max_pool2d(a, kernel_size=2)');
    expect(MMdnn.mapONNXToPyTorchFunc('AveragePool', 'a')).toBe('F.avg_pool2d(a, kernel_size=2)');
    expect(MMdnn.mapONNXToPyTorchFunc('Softmax', 'a')).toBe('F.softmax(a, dim=-1)');
    expect(MMdnn.mapONNXToPyTorchFunc('Add', 'a')).toBe('a + a');
    expect(MMdnn.mapONNXToPyTorchFunc('Mul', 'a')).toBe('a * a');
    expect(MMdnn.mapONNXToPyTorchFunc('Flatten', 'a')).toBe('torch.flatten(a, 1)');
    expect(MMdnn.mapONNXToPyTorchFunc('Concat', 'a')).toBe('torch.cat((a, a), dim=1)');
    expect(MMdnn.mapONNXToPyTorchFunc('Unknown', 'a')).toBe('a');
  });

  it('should map various ONNX types to PyTorch Modules', () => {
    expect(MMdnn.mapONNXToPyTorch('BatchNormalization')).toBe('BatchNorm2d');
    expect(MMdnn.mapONNXToPyTorch('Unknown')).toBe('Module');
  });

  it('should export ONNX to TFJS Code', () => {
    const graph: IModelGraph = {
      name: 'TestTFJS',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes: [
        { name: 'conv1', opType: 'Conv', inputs: ['x'], outputs: ['o1'], attributes: {} },
        { name: 'relu1', opType: 'Relu', inputs: ['o1'], outputs: ['o2'], attributes: {} },
        { name: 'pool1', opType: 'MaxPool', inputs: ['o2'], outputs: ['o3'], attributes: {} },
        { name: 'apool1', opType: 'AveragePool', inputs: ['o3'], outputs: ['o4'], attributes: {} },
        {
          name: 'bn1',
          opType: 'BatchNormalization',
          inputs: ['o4'],
          outputs: ['o5'],
          attributes: {},
        },
        { name: 'flat1', opType: 'Flatten', inputs: ['o5'], outputs: ['o6'], attributes: {} },
        { name: 'drop1', opType: 'Dropout', inputs: ['o6'], outputs: ['o7'], attributes: {} },
        { name: 'fc1', opType: 'Gemm', inputs: ['o7'], outputs: ['o8'], attributes: {} },
        { name: 'unknown1', opType: 'Weird', inputs: ['o8'], outputs: ['o9'], attributes: {} },
      ],
    };
    const code = MMdnn.exportFromONNX('tfjs_code', graph) as string;
    expect(code).toContain("import * as tf from '@tensorflow/tfjs';");
    expect(code).toContain('model.add(tf.layers.conv2d');
    expect(code).toContain('model.add(tf.layers.dense');
    expect(code).toContain('model.add(tf.layers.activation');
    expect(code).toContain('model.add(tf.layers.maxPooling2d');
    expect(code).toContain('model.add(tf.layers.averagePooling2d');
    expect(code).toContain('model.add(tf.layers.batchNormalization');
    expect(code).toContain('model.add(tf.layers.flatten');
    expect(code).toContain('model.add(tf.layers.dropout');
    expect(code).not.toContain('weird');
  });

  it('should map remaining Caffe and MXNet types correctly', () => {
    expect(MMdnn.mapCaffeTypeToONNX('Softmax')).toBe('Softmax');
    expect(MMdnn.mapCaffeTypeToONNX('BatchNorm')).toBe('BatchNormalization');
    expect(MMdnn.mapCaffeTypeToONNX('Eltwise')).toBe('Add');
    expect(MMdnn.mapCaffeTypeToONNX('Concat')).toBe('Concat');
    expect(MMdnn.mapCaffeTypeToONNX('Dropout')).toBe('Dropout');
    expect(MMdnn.mapCaffeTypeToONNX('Reshape')).toBe('Reshape');
    expect(MMdnn.mapCaffeTypeToONNX('Flatten')).toBe('Flatten');

    expect(MMdnn.mapMXNetTypeToONNX('BatchNorm')).toBe('BatchNormalization');
    expect(MMdnn.mapMXNetTypeToONNX('elemwise_add')).toBe('Add');
    expect(MMdnn.mapMXNetTypeToONNX('elemwise_sub')).toBe('Sub');
    expect(MMdnn.mapMXNetTypeToONNX('elemwise_mul')).toBe('Mul');
    expect(MMdnn.mapMXNetTypeToONNX('Flatten')).toBe('Flatten');
    expect(MMdnn.mapMXNetTypeToONNX('Reshape')).toBe('Reshape');
    expect(MMdnn.mapMXNetTypeToONNX('SoftmaxOutput')).toBe('Softmax');
    expect(MMdnn.mapMXNetTypeToONNX('Concat')).toBe('Concat');
  });

  it('should correctly passthrough ONNX model in exportFromONNX', () => {
    const graph: IModelGraph = {
      name: 'test',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
    };
    const exported = MMdnn.exportFromONNX('onnx', graph);
    expect(exported).toBe(graph);
  });

  it('should throw error for unsupported target framework in exportFromONNX', () => {
    const graph: IModelGraph = {
      name: 'test',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
    };
    expect(() => MMdnn.exportFromONNX('caffe' as any, graph)).toThrow(
      'Unsupported target framework: caffe',
    );
  });

  it('should provide end-to-end convert API functionality', () => {
    const caffeProto = `
      layer {
        name: "conv1"
        type: "Convolution"
      }
    `;
    const resultPyTorch = MMdnn.convert({
      source: 'caffe',
      target: 'pytorch_code',
      modelData: caffeProto,
    }) as string;
    expect(resultPyTorch).toContain('class caffe_imported_model(nn.Module):');
    expect(resultPyTorch).toContain('self.conv1 = nn.Conv2d()');

    const resultTFJS = MMdnn.convert({
      source: 'caffe',
      target: 'tfjs_code',
      modelData: caffeProto,
    }) as string;
    expect(resultTFJS).toContain('tf.layers.conv2d');

    const resultONNX = MMdnn.convert({
      source: 'caffe',
      target: 'onnx',
      modelData: caffeProto,
    }) as IModelGraph;
    expect(resultONNX.nodes[0].opType).toBe('Conv');
  });
});
