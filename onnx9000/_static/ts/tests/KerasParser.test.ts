import { describe, it, expect } from 'vitest';
import { KerasParser } from '../src/parsers/KerasParser';

describe('KerasParser (keras2onnx / tfjs-to-onnx)', () => {
  it('should parse an empty or invalid string safely', () => {
    let parser = new KerasParser('invalid json');
    let graph = parser.parse();
    expect(graph.name).toBe('keras_imported_model');
    expect(graph.nodes.length).toBe(0);

    parser = new KerasParser('{}');
    graph = parser.parse();
    expect(graph.nodes.length).toBe(0);
  });

  it('should parse a standard tf.js topology with Conv2D and Dense', () => {
    const topology = {
      modelTopology: {
        keras_version: '2.4.0',
        model_config: {
          class_name: 'Sequential',
          config: {
            name: 'my_vision_model',
            layers: [
              { class_name: 'InputLayer', config: { name: 'input_1' } },
              {
                class_name: 'Conv2D',
                config: { name: 'conv2d_1', kernel_size: [3, 3], strides: [1, 1] },
              },
              {
                class_name: 'MaxPooling2D',
                config: { name: 'max_pooling2d_1', pool_size: [2, 2] },
              },
              { class_name: 'Flatten', config: { name: 'flatten_1' } },
              { class_name: 'Dense', config: { name: 'dense_1' } },
              { class_name: 'Dropout', config: { name: 'dropout_1', rate: 0.2 } },
              { class_name: 'Activation', config: { name: 'activation_1', activation: 'softmax' } },
            ],
          },
        },
      },
    };

    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();

    expect(graph.name).toBe('my_vision_model');
    // InputLayer is skipped
    expect(graph.nodes.length).toBe(6);

    // Conv2D
    expect(graph.nodes[0].name).toBe('conv2d_1');
    expect(graph.nodes[0].opType).toBe('Conv');
    expect(graph.nodes[0].attributes.kernel_shape.ints).toEqual([3, 3]);

    // MaxPool
    expect(graph.nodes[1].name).toBe('max_pooling2d_1');
    expect(graph.nodes[1].opType).toBe('MaxPool');
    expect(graph.nodes[1].attributes.kernel_shape.ints).toEqual([2, 2]);

    // Flatten
    expect(graph.nodes[2].opType).toBe('Flatten');

    // Dense
    expect(graph.nodes[3].opType).toBe('MatMul');

    // Dropout
    expect(graph.nodes[4].opType).toBe('Dropout');
    expect(graph.nodes[4].attributes.ratio.f).toBe(0.2);

    // Activation (Softmax)
    expect(graph.nodes[5].opType).toBe('Softmax');
  });

  it('should generate node names if not provided in config', () => {
    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers: [{ class_name: 'Dense', config: {} }],
          },
        },
      },
    };
    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();
    expect(graph.nodes[0].name).toBe('layer_0');
  });

  it('should map unknown Keras layers to generic Fallback node types', () => {
    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers: [{ class_name: 'CustomKerasLayer', config: { name: 'custom_1' } }],
          },
        },
      },
    };
    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();
    expect(graph.nodes[0].opType).toBe('Keras_CustomKerasLayer');
  });

  it('should map all supported Keras activations correctly', () => {
    const acts = ['relu', 'softmax', 'sigmoid', 'tanh', 'linear', 'unknown'];
    const expected = ['Relu', 'Softmax', 'Sigmoid', 'Tanh', 'Identity', 'Identity'];

    const layers = acts.map((act, i) => ({
      class_name: 'Activation',
      config: { name: `act_${i}`, activation: act },
    }));

    // Add one without activation property entirely
    layers.push({ class_name: 'Activation', config: { name: 'act_missing' } });
    expected.push('Identity');

    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers,
          },
        },
      },
    };

    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();

    expect(graph.nodes.length).toBe(7);
    graph.nodes.forEach((node, i) => {
      expect(node.opType).toBe(expected[i]);
    });
  });

  it('should handle Conv2D missing properties using defaults', () => {
    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers: [{ class_name: 'Conv2D', config: { name: 'conv1' } }],
          },
        },
      },
    };
    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();
    expect(graph.nodes[0].attributes.kernel_shape.ints).toEqual([]);
    expect(graph.nodes[0].attributes.strides.ints).toEqual([1, 1]);
  });

  it('should handle MaxPooling2D missing properties using defaults', () => {
    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers: [{ class_name: 'MaxPooling2D', config: { name: 'pool1' } }],
          },
        },
      },
    };
    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();
    expect(graph.nodes[0].attributes.kernel_shape.ints).toEqual([2, 2]);
  });

  it('should handle Dropout missing properties using defaults', () => {
    const topology = {
      modelTopology: {
        model_config: {
          config: {
            layers: [{ class_name: 'Dropout', config: { name: 'drop1' } }],
          },
        },
      },
    };
    const parser = new KerasParser(JSON.stringify(topology));
    const graph = parser.parse();
    expect(graph.nodes[0].attributes.ratio.f).toBe(0.5);
  });
});
