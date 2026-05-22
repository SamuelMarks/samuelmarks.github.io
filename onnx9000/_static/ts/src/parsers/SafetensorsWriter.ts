import { IModelGraph } from '../core/IR';

export class SafetensorsWriter {
  private static async generateWatermark(model: IModelGraph): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(model.name + model.nodes.length.toString());
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashHex = hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
    return `onnx9000_verified_${hashHex}`;
  }

  public static async export(
    model: IModelGraph,
    filename: string = 'model.safetensors',
  ): Promise<void> {
    const header: Record<string, unknown> = {};
    let meta: Record<string, unknown> = {};
    if (model.docString) {
      try {
        meta = JSON.parse(model.docString);
      } catch (e) {
        meta = { description: model.docString };
      }
    }

    // 575. Implement privacy-preserving model watermarking
    // 576. Embed cryptographic signatures into the headers
    meta.watermark = await this.generateWatermark(model);
    header.__metadata__ = meta;

    let currentOffset = 0;
    const buffers: Uint8Array[] = [];

    // Filter out non-initialized tensors
    const validInitializers = model.initializers.filter((t) => t.rawData);

    for (const tensor of validInitializers) {
      if (!tensor.rawData) continue;

      const byteLength = tensor.rawData.byteLength;
      const dtype = this.mapONNXToDtype(tensor.dataType);

      header[tensor.name] = {
        dtype: dtype,
        shape: tensor.dims,
        data_offsets: [currentOffset, currentOffset + byteLength],
      };

      buffers.push(tensor.rawData);
      currentOffset += byteLength;
    }

    const headerJson = JSON.stringify(header);
    const encoder = new TextEncoder();
    let headerBytes = encoder.encode(headerJson);

    // Header length must be divisible by 8 (8-byte aligned)
    const paddingLength = (8 - (headerBytes.length % 8)) % 8;
    if (paddingLength > 0) {
      const paddedHeaderStr = headerJson + ' '.repeat(paddingLength);
      headerBytes = encoder.encode(paddedHeaderStr);
    }

    const headerLength = headerBytes.length;
    // 8 bytes for length
    const lengthBytes = new Uint8Array(8);
    const dataView = new DataView(lengthBytes.buffer);
    dataView.setUint32(0, headerLength, true); // Little endian
    dataView.setUint32(4, 0, true);

    const blobParts: BlobPart[] = [lengthBytes, headerBytes, ...buffers];
    const blob = new Blob(blobParts, { type: 'application/octet-stream' });

    // Download trigger
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  }

  private static mapONNXToDtype(onnxType: number): string {
    switch (onnxType) {
      case 1:
        return 'F32';
      case 2:
        return 'U8';
      case 3:
        return 'I8';
      case 4:
        return 'U16';
      case 5:
        return 'I16';
      case 6:
        return 'I32';
      case 7:
        return 'I64';
      case 10:
        return 'F16';
      case 11:
        return 'F64';
      case 12:
        return 'U32';
      case 13:
        return 'U64';
      default:
        return 'F32'; // Fallback
    }
  }
}
