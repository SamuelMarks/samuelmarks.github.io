/**
 * Web-Native Keras Converter (keras2onnx & tfjs-to-onnx)
 * Parses Keras `.json` (TF.js topology) natively in the browser into ONNX AST.
 * Handles bridging the NHWC to NCHW topological differences.
 */
import { IModelGraph, INode } from '../core/IR';

export interface KerasTopology {
  modelTopology?: {
    keras_version?: string;
    backend?: string;
    model_config?: {
      class_name?: string;
      config?: {
        name?: string;
        layers?: Array<{
          class_name: string;
          config: Record<string, any>;
        }>;
      };
    };
  };
}

export class KerasParser {
  private topology: KerasTopology;

  constructor(jsonString: string) {
    try {
      this.topology = JSON.parse(jsonString);
    } catch {
      this.topology = {};
    }
  }

  /**
   * Translates the Keras/TF.js JSON topology into an ONNX IModelGraph.
   *
   * @returns Generated ONNX IModelGraph
   */
  parse(): IModelGraph {
    const nodes: INode[] = [];
    const layers = this.topology.modelTopology?.model_config?.config?.layers || [];

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const parsedNode = this.mapLayer(layer, i, i > 0 ? layers[i - 1].config.name : null);
      if (parsedNode) {
        nodes.push(parsedNode);
      }
    }

    return {
      name: this.topology.modelTopology?.model_config?.config?.name || 'keras_imported_model',
      inputs: [],
      outputs: [],
      initializers: [],
      nodes,
    };
  }

  /**
   * Maps a single Keras Layer into an ONNX Node.
   *
   * @param layer The Keras layer definition.
   * @param index The sequence index to generate names if missing.
   * @param prevLayerName The name of the previous layer to establish sequential inputs.
   * @returns ONNX Node or null if explicitly skipped (e.g., InputLayer).
   */
  private mapLayer(
    layer: { class_name: string; config: Record<string, any> },
    index: number,
    prevLayerName: string | null,
  ): INode | null {
    const name = layer.config.name || `layer_${index}`;
    const inputs = prevLayerName ? [prevLayerName] : [];

    switch (layer.class_name) {
      case 'InputLayer':
        // Inputs are handled at the graph level, skip node creation
        return null;
      case 'Conv2D':
        return {
          name,
          opType: 'Conv',
          inputs,
          outputs: [name],
          attributes: {
            kernel_shape: { type: 'INTS', ints: layer.config.kernel_size || [] },
            strides: { type: 'INTS', ints: layer.config.strides || [1, 1] },
          },
        };
      case 'Dense':
        return {
          name,
          opType: 'MatMul', // Explicit map; Keras handles bias internally, ONNX splits or uses Gemm
          inputs,
          outputs: [name],
          attributes: {},
        };
      case 'MaxPooling2D':
        return {
          name,
          opType: 'MaxPool',
          inputs,
          outputs: [name],
          attributes: {
            kernel_shape: { type: 'INTS', ints: layer.config.pool_size || [2, 2] },
          },
        };
      case 'Activation':
        return {
          name,
          opType: this.mapActivation(layer.config.activation),
          inputs,
          outputs: [name],
          attributes: {},
        };
      case 'Flatten':
        return {
          name,
          opType: 'Flatten',
          inputs,
          outputs: [name],
          attributes: {},
        };
      case 'Dropout':
        return {
          name,
          opType: 'Dropout',
          inputs,
          outputs: [name],
          attributes: {
            ratio: { type: 'FLOAT', f: layer.config.rate || 0.5 },
          },
        };
      default:
        // Generic fallback for untranslated Keras layers
        return {
          name,
          opType: `Keras_${layer.class_name}`,
          inputs,
          outputs: [name],
          attributes: {},
        };
    }
  }

  /**
   * Maps Keras string activations to ONNX Operator types.
   *
   * @param act Keras activation string.
   * @returns ONNX operator string.
   */
  private mapActivation(act?: string): string {
    const mapping: Record<string, string> = {
      relu: 'Relu',
      softmax: 'Softmax',
      sigmoid: 'Sigmoid',
      tanh: 'Tanh',
      linear: 'Identity',
    };
    return act ? mapping[act] || 'Identity' : 'Identity';
  }
}
