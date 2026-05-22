export interface ITensorType {
  elemType: number; // e.g., ONNX DataType enum (1 = FLOAT, 7 = INT64, etc.)
  shape: (number | string)[]; // Dimensions, can be string for dynamic
}

export interface IValueInfo {
  name: string;
  type?: ITensorType;
}

export interface IAttribute {
  name: string;
  type: string; // "INT", "FLOAT", "STRING", "TENSOR", "INTS", "FLOATS", "STRINGS"
  i?: number;
  f?: number;
  s?: string;
  t?: ITensor;
  ints?: number[];
  floats?: number[];
  strings?: string[];
}

export interface INode {
  name: string;
  opType: string;
  domain?: string;
  inputs: string[];
  outputs: string[];
  attributes: Record<string, IAttribute>;
}

export interface ITensor {
  name: string;
  dataType: number;
  dims: number[];
  rawData?: Uint8Array;
  floatData?: number[];
  int32Data?: number[];
  int64Data?: number[];
  stringData?: string[];
}

export interface IModelGraph {
  name: string;
  nodes: INode[];
  inputs: IValueInfo[];
  outputs: IValueInfo[];
  initializers: ITensor[];
  valueInfo?: IValueInfo[]; // Intermediate tensor shapes
  docString?: string;
}
