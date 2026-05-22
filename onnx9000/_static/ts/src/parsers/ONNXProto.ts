import { IModelGraph, INode, ITensor } from '../core/IR';

export class ONNXProtoParser {
  private view: Uint8Array;
  private offset = 0;

  constructor(buffer: ArrayBuffer) {
    this.view = new Uint8Array(buffer);
  }

  // Very rudimentary varint decoder
  private readVarint(): number {
    let result = 0;
    let shift = 0;
    while (true) {
      if (this.offset >= this.view.length) {
        throw new Error('Unexpected end of buffer reading varint');
      }
      const byte = this.view[this.offset++];
      result |= (byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) {
        return result;
      }
      shift += 7;
      if (shift >= 32) {
        // Just reading small varints for tags and lengths, ignore large varints for now
        break;
      }
    }
    return result;
  }

  // Parse a length-delimited string
  private readString(length: number): string {
    const bytes = this.view.subarray(this.offset, this.offset + length);
    this.offset += length;
    return new TextDecoder('utf-8').decode(bytes);
  }

  // Basic structure map: 1 -> ir_version, 2 -> opset_import, 3 -> producer_name, ... 7 -> graph
  public parse(): IModelGraph {
    let graph: IModelGraph = {
      name: 'ONNX Model',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
    };

    while (this.offset < this.view.length) {
      const tag = this.readVarint();
      const fieldNum = tag >> 3;
      const wireType = tag & 0x7;

      if (wireType === 2) {
        const length = this.readVarint();
        if (fieldNum === 7) {
          // GraphProto
          graph = this.parseGraphProto(this.offset, length);
        }
        this.offset += length;
      } else if (wireType === 0) {
        this.readVarint();
      } else if (wireType === 5) {
        this.offset += 4;
      } else if (wireType === 1) {
        this.offset += 8;
      } else {
        throw new Error(`Unsupported wire type: ${wireType}`);
      }
    }

    return graph;
  }

  private parseGraphProto(start: number, length: number): IModelGraph {
    const end = start + length;
    const currentOffset = this.offset;
    this.offset = start;

    const graph: IModelGraph = {
      name: 'graph',
      nodes: [],
      inputs: [],
      outputs: [],
      initializers: [],
    };

    while (this.offset < end) {
      const tag = this.readVarint();
      const fieldNum = tag >> 3;
      const wireType = tag & 0x7;

      if (wireType === 2) {
        const len = this.readVarint();
        if (fieldNum === 1) {
          // node
          graph.nodes.push(this.parseNodeProto(this.offset, len));
        } else if (fieldNum === 2) {
          // name
          graph.name = this.readString(len);
          this.offset -= len; // because readString advances, but we handle it manually
          this.offset += len;
        } else if (fieldNum === 5) {
          // initializer
          graph.initializers.push(this.parseTensorProto(this.offset, len));
        }
        this.offset += len;
      } else {
        // Skip other fields
        if (wireType === 0) this.readVarint();
        else if (wireType === 5) this.offset += 4;
        else if (wireType === 1) this.offset += 8;
      }
    }

    this.offset = currentOffset;
    return graph;
  }

  private parseNodeProto(start: number, length: number): INode {
    // This is a minimal stub for node parsing
    return {
      name: `node_${start}`,
      opType: 'Unknown',
      inputs: [],
      outputs: [],
      attributes: {},
    };
  }

  private parseTensorProto(start: number, length: number): ITensor {
    // This is a minimal stub for tensor parsing
    return {
      name: `tensor_${start}`,
      dataType: 1, // FLOAT default stub
      dims: [],
    };
  }
}
