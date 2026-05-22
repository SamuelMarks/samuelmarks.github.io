export class Tensor {
  constructor(
    public name: string,
    public dtype: string, // e.g., 'F32', 'I64'
    public shape: number[],
    public data: ArrayBufferView,
    public offset: number, // offset from the view
    public length: number, // element count
  ) {}

  get float32Data(): Float32Array {
    if (this.dtype !== 'F32') {
      throw new Error(`Cannot cast ${this.dtype} to Float32Array`);
    }
    return this.data as Float32Array;
  }

  get int32Data(): Int32Array {
    if (this.dtype !== 'I32') {
      throw new Error(`Cannot cast ${this.dtype} to Int32Array`);
    }
    return this.data as Int32Array;
  }

  get int8Data(): Int8Array {
    if (this.dtype !== 'I8') {
      throw new Error(`Cannot cast ${this.dtype} to Int8Array`);
    }
    return this.data as Int8Array;
  }
}
