/**
 * MMdnn core module for N-to-N Neural Network conversion inside the browser.
 * Converts models from legacy formats (Caffe, MXNet, CNTK) into ONNX,
 * and exports ONNX graphs into modern targets (PyTorch Code, TFJS Code).
 */
import { IModelGraph, INode } from '../core/IR';

/**
 * Supported Frameworks for Import and Export.
 */
export type Framework = 'caffe' | 'mxnet' | 'pytorch_code' | 'tfjs_code' | 'onnx';

/**
 * Unified API options for the conversion process.
 */
export interface ConvertOptions {
  /** The source framework to import from. */
  source: Framework;
  /** The target framework to export to. */
  target: Framework;
  /** Primary model architecture (e.g. .prototxt for Caffe, .json for MXNet). */
  modelData: string;
  /** Binary weight data buffer (optional for purely structural translations). */
  weightData?: ArrayBuffer;
}

/**
 * Unified Neural Network Converter.
 */
export class MMdnn {
  /**
   * Main entry point to convert a model from a source framework into a target framework.
   * Uses ONNX as the universal intermediate representation.
   *
   * @param options The conversion parameters specifying source, target, and data string.
   * @returns A string of generated code (for PyTorch/TFJS) or an IModelGraph (for ONNX).
   */
  static convert(options: ConvertOptions): string | IModelGraph {
    const onnxGraph = this.parseToONNX(options.source, options.modelData, options.weightData);
    return this.exportFromONNX(options.target, onnxGraph);
  }

  /**
   * Parses the source architecture string and weight buffer into a canonical ONNX graph.
   *
   * @param source The source legacy framework format.
   * @param modelData Architecture schema (e.g., Caffe prototxt string or MXNet symbol JSON).
   * @param weightData Binary weight buffer.
   * @returns The constructed ONNX IModelGraph.
   */
  static parseToONNX(source: Framework, modelData: string, weightData?: ArrayBuffer): IModelGraph {
    switch (source) {
      case 'caffe':
        return this.parseCaffe(modelData);
      case 'mxnet':
        return this.parseMXNet(modelData);
      case 'onnx':
        return JSON.parse(modelData) as IModelGraph;
      default:
        throw new Error(`Unsupported source framework: ${source}`);
    }
  }

  /**
   * Exports the canonical ONNX graph into the target execution format.
   *
   * @param target The target code generator.
   * @param graph The normalized ONNX Intermediate Representation graph.
   * @returns A generated string of code or the ONNX graph directly.
   */
  static exportFromONNX(target: Framework, graph: IModelGraph): string | IModelGraph {
    switch (target) {
      case 'pytorch_code':
        return this.generatePyTorchCode(graph);
      case 'tfjs_code':
        return this.generateTFJSCode(graph);
      case 'onnx':
        return graph;
      default:
        throw new Error(`Unsupported target framework: ${target}`);
    }
  }

  /**
   * Simplistic Caffe `.prototxt` parser.
   * Scans lines for `type: "Convolution"` etc., and constructs an ONNX AST.
   *
   * @param prototxt String content of a Caffe prototxt file.
   * @returns An ONNX Graph mapping the Caffe layers.
   */
  static parseCaffe(prototxt: string): IModelGraph {
    const nodes: INode[] = [];
    const lines = prototxt.split('\n');
    let currentType = '';
    let currentName = '';

    for (const line of lines) {
      const tMatch = line.match(/type:\s*"([^"]+)"/);
      if (tMatch) currentType = tMatch[1];

      const nMatch = line.match(/name:\s*"([^"]+)"/);
      if (nMatch) currentName = nMatch[1];

      // Block completion heuristic
      if (line.includes('}') && currentType) {
        nodes.push({
          name: currentName || `node_${nodes.length}`,
          opType: this.mapCaffeTypeToONNX(currentType),
          inputs: [`input_${nodes.length}`],
          outputs: [`output_${nodes.length}`],
          attributes: {},
        });
        currentType = '';
        currentName = '';
      }
    }

    return {
      name: 'caffe_imported_model',
      inputs: [{ name: 'input_0' }],
      outputs: [{ name: `output_${nodes.length - 1}` }],
      nodes,
      initializers: [],
    };
  }

  /**
   * Translates a Caffe layer type to its standard ONNX operator equivalent.
   *
   * @param type Caffe layer type string.
   * @returns ONNX standard operator type string.
   */
  static mapCaffeTypeToONNX(type: string): string {
    const mapping: Record<string, string> = {
      Convolution: 'Conv',
      InnerProduct: 'Gemm',
      ReLU: 'Relu',
      Pooling: 'MaxPool',
      Softmax: 'Softmax',
      BatchNorm: 'BatchNormalization',
      Eltwise: 'Add',
      Concat: 'Concat',
      Dropout: 'Dropout',
      Reshape: 'Reshape',
      Flatten: 'Flatten',
    };
    return mapping[type] || 'Identity';
  }

  /**
   * Simplistic MXNet `.json` symbol parser.
   * Maps MXNet operators to ONNX.
   *
   * @param jsonString Stringified MXNet Symbol JSON.
   * @returns An ONNX Graph mapping the MXNet structures.
   */
  static parseMXNet(jsonString: string): IModelGraph {
    let data;
    try {
      data = JSON.parse(jsonString);
    } catch {
      data = {};
    }
    const nodes: INode[] = [];

    if (data && Array.isArray(data.nodes)) {
      data.nodes.forEach((node: any, idx: number) => {
        if (!node || node.op === 'null') return;
        nodes.push({
          name: node.name || `mx_node_${idx}`,
          opType: this.mapMXNetTypeToONNX(node.op),
          inputs: Array.isArray(node.inputs) ? node.inputs.map((i: any[]) => `tensor_${i[0]}`) : [],
          outputs: [`tensor_${idx}`],
          attributes: {},
        });
      });
    }

    return {
      name: 'mxnet_imported_model',
      inputs: [],
      outputs: [],
      nodes,
      initializers: [],
    };
  }

  /**
   * Translates an MXNet layer type to its standard ONNX operator equivalent.
   *
   * @param type MXNet layer type string.
   * @returns ONNX standard operator type string.
   */
  static mapMXNetTypeToONNX(type: string): string {
    const mapping: Record<string, string> = {
      Convolution: 'Conv',
      FullyConnected: 'Gemm',
      Activation: 'Relu',
      Pooling: 'MaxPool',
      BatchNorm: 'BatchNormalization',
      elemwise_add: 'Add',
      elemwise_sub: 'Sub',
      elemwise_mul: 'Mul',
      Flatten: 'Flatten',
      Reshape: 'Reshape',
      SoftmaxOutput: 'Softmax',
      Concat: 'Concat',
    };
    return mapping[type] || 'Identity';
  }

  /**
   * Emits a PyTorch `nn.Module` raw Python string translating the given ONNX topology.
   *
   * @param graph The normalized ONNX Intermediate Representation graph.
   * @returns Generated Python source code string.
   */
  static generatePyTorchCode(graph: IModelGraph): string {
    let code = `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\n`;
    code += `class ${graph.name || 'ConvertedModel'}(nn.Module):\n`;
    code += `    def __init__(self):\n`;
    code += `        super().__init__()\n`;

    const statefulOps = ['Conv', 'Gemm', 'BatchNormalization'];
    let statefulCount = 0;

    graph.nodes.forEach((node) => {
      if (statefulOps.includes(node.opType)) {
        const torchType = this.mapONNXToPyTorch(node.opType);
        code += `        self.${node.name} = nn.${torchType}()  # Requires manual dims mapping\n`;
        statefulCount++;
      }
    });

    if (statefulCount === 0) {
      code += `        pass\n`;
    }

    code += `\n    def forward(self, x):\n`;

    let currentInput = 'x';
    graph.nodes.forEach((node) => {
      if (statefulOps.includes(node.opType)) {
        code += `        ${node.outputs[0]} = self.${node.name}(${currentInput})\n`;
      } else {
        const funcCall = this.mapONNXToPyTorchFunc(node.opType, currentInput);
        code += `        ${node.outputs[0]} = ${funcCall}\n`;
      }
      currentInput = node.outputs[0];
    });

    code += `        return ${currentInput}\n`;
    return code;
  }

  /**
   * Maps an ONNX stateful layer to its PyTorch `nn.Module` class name.
   *
   * @param onnxType The ONNX opType string.
   * @returns The corresponding `nn.` module name.
   */
  static mapONNXToPyTorch(onnxType: string): string {
    const mapping: Record<string, string> = {
      Conv: 'Conv2d',
      Gemm: 'Linear',
      BatchNormalization: 'BatchNorm2d',
    };
    return mapping[onnxType] || 'Module';
  }

  /**
   * Maps an ONNX stateless layer to its PyTorch functional equivalent.
   *
   * @param onnxType The ONNX opType string.
   * @param input Tensor variable name.
   * @returns The PyTorch execution string.
   */
  static mapONNXToPyTorchFunc(onnxType: string, input: string): string {
    const mapping: Record<string, string> = {
      Relu: `F.relu(${input})`,
      MaxPool: `F.max_pool2d(${input}, kernel_size=2)`,
      AveragePool: `F.avg_pool2d(${input}, kernel_size=2)`,
      Softmax: `F.softmax(${input}, dim=-1)`,
      Add: `${input} + ${input}`,
      Mul: `${input} * ${input}`,
      Flatten: `torch.flatten(${input}, 1)`,
      Concat: `torch.cat((${input}, ${input}), dim=1)`,
    };
    return mapping[onnxType] || input;
  }

  /**
   * Emits a TensorFlow.js raw JavaScript code string representing the given ONNX topology.
   *
   * @param graph The normalized ONNX Intermediate Representation graph.
   * @returns Generated JavaScript source code string.
   */
  static generateTFJSCode(graph: IModelGraph): string {
    let code = `import * as tf from '@tensorflow/tfjs';\n\n`;
    code += `export function createModel() {\n`;
    code += `  const model = tf.sequential();\n`;

    graph.nodes.forEach((node) => {
      const layerCall = this.mapONNXToTFJSLayer(node.opType);
      if (layerCall) {
        code += `  model.add(${layerCall});\n`;
      }
    });

    code += `  return model;\n`;
    code += `}\n`;
    return code;
  }

  /**
   * Maps an ONNX layer to its TFJS `tf.layers.` equivalent code snippet.
   *
   * @param onnxType The ONNX opType string.
   * @returns The TFJS construction snippet.
   */
  static mapONNXToTFJSLayer(onnxType: string): string | null {
    const mapping: Record<string, string> = {
      Conv: `tf.layers.conv2d({ filters: 32, kernelSize: 3, activation: 'relu' })`,
      Gemm: `tf.layers.dense({ units: 128 })`,
      Relu: `tf.layers.activation({ activation: 'relu' })`,
      MaxPool: `tf.layers.maxPooling2d({ poolSize: 2 })`,
      AveragePool: `tf.layers.averagePooling2d({ poolSize: 2 })`,
      BatchNormalization: `tf.layers.batchNormalization()`,
      Flatten: `tf.layers.flatten()`,
      Dropout: `tf.layers.dropout({ rate: 0.5 })`,
    };
    return mapping[onnxType] || null;
  }
}
