import { Tensor } from '../core/Tensor';

export interface SafetensorHeader {
  __metadata__?: Record<string, string>;
  [tensorName: string]: unknown;
}

export interface SafetensorTensorMetadata {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

export class SafetensorsParser {
  private buffer: ArrayBuffer;
  private isLittleEndian: boolean;

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer;

    // Check system endianness
    const uInt32 = new Uint32Array([0x11223344]);
    const uInt8 = new Uint8Array(uInt32.buffer);
    this.isLittleEndian = uInt8[0] === 0x44;
  }

  parse(): { metadata: Record<string, string>; tensors: Record<string, Tensor> } {
    if (this.buffer.byteLength < 8) {
      throw new Error('Buffer too small to be a valid Safetensors file.');
    }

    const dataView = new DataView(this.buffer);

    // Read UInt64 header length (Safetensors is little-endian)
    // We only read 53 bits accurately in JS without BigInt, but header is usually small
    const headerLengthLow = dataView.getUint32(0, true);
    const headerLengthHigh = dataView.getUint32(4, true);

    // For JS limits, header length should fit in low 32 bits
    if (headerLengthHigh !== 0) {
      throw new Error('Safetensors header size is too large for this parser.');
    }
    const headerLength = headerLengthLow;

    if (8 + headerLength > this.buffer.byteLength) {
      throw new Error('Safetensors header length exceeds buffer size.');
    }

    // Extract JSON header bytes
    const headerBytes = new Uint8Array(this.buffer, 8, headerLength);
    const decoder = new TextDecoder('utf-8');
    const jsonString = decoder.decode(headerBytes);

    let header: SafetensorHeader;
    try {
      header = JSON.parse(jsonString);
    } catch (e) {
      throw new Error('Failed to parse Safetensors JSON header.');
    }

    const metadata = header.__metadata__ || {};
    delete header.__metadata__;

    const dataOffsetStart = 8 + headerLength;
    const tensors: Record<string, Tensor> = {};

    for (const [name, meta] of Object.entries(header)) {
      const tensorMeta = meta as SafetensorTensorMetadata;
      if (!tensorMeta.dtype || !tensorMeta.shape || !tensorMeta.data_offsets) {
        throw new Error(`Invalid tensor metadata for tensor: ${name}`);
      }

      const [startOffset, endOffset] = tensorMeta.data_offsets;
      const byteLength = endOffset - startOffset;
      const absoluteStart = dataOffsetStart + startOffset;

      if (absoluteStart + byteLength > this.buffer.byteLength) {
        throw new Error(`Data offset out of bounds for tensor: ${name}`);
      }

      const typedArray = this.mapToTypedArray(absoluteStart, byteLength, tensorMeta.dtype);

      tensors[name] = new Tensor(
        name,
        tensorMeta.dtype,
        tensorMeta.shape,
        typedArray,
        0,
        byteLength / typedArray.BYTES_PER_ELEMENT,
      );
    }

    return { metadata, tensors };
  }

  private mapToTypedArray(start: number, byteLength: number, dtype: string): ArrayBufferView {
    // If not little-endian, we would need to manually swap bytes for zero-copy.
    // However, most modern browsers run on little-endian architectures.
    if (!this.isLittleEndian) {
      console.warn('Big-endian system detected. Zero-copy may result in incorrect values.');
    }

    // Typed arrays must be aligned to their element size
    // For F32, start must be multiple of 4.
    const isAligned = start % 4 === 0;

    if (dtype === 'F32') {
      if (isAligned) {
        return new Float32Array(this.buffer, start, byteLength / 4);
      } else {
        // Fallback: Copy data if unaligned
        const copy = new Uint8Array(this.buffer, start, byteLength).slice();
        return new Float32Array(copy.buffer);
      }
    } else if (dtype === 'I32') {
      if (isAligned) {
        return new Int32Array(this.buffer, start, byteLength / 4);
      } else {
        const copy = new Uint8Array(this.buffer, start, byteLength).slice();
        return new Int32Array(copy.buffer);
      }
    } else if (dtype === 'I64' || dtype === 'F64') {
      const isAligned8 = start % 8 === 0;
      if (dtype === 'I64') {
        if (isAligned8) return new BigInt64Array(this.buffer, start, byteLength / 8);
        const copy = new Uint8Array(this.buffer, start, byteLength).slice();
        return new BigInt64Array(copy.buffer);
      } else {
        if (isAligned8) return new Float64Array(this.buffer, start, byteLength / 8);
        const copy = new Uint8Array(this.buffer, start, byteLength).slice();
        return new Float64Array(copy.buffer);
      }
    } else if (dtype === 'I8') {
      return new Int8Array(this.buffer, start, byteLength);
    } else if (dtype === 'U8') {
      return new Uint8Array(this.buffer, start, byteLength);
    } else if (dtype === 'F16') {
      // Float16Array isn't widely supported yet, fallback to Uint16Array for raw data
      const isAligned2 = start % 2 === 0;
      if (isAligned2) return new Uint16Array(this.buffer, start, byteLength / 2);
      const copy = new Uint8Array(this.buffer, start, byteLength).slice();
      return new Uint16Array(copy.buffer);
    }

    throw new Error(`Unsupported dtype: ${dtype}`);
  }
}
