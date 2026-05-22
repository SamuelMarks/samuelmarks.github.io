import { ITIRGraph } from './Lowering';

export class WGSLEmitter {
  private tir: ITIRGraph;

  constructor(tir: ITIRGraph) {
    this.tir = tir;
  }

  // 192. Lower ONNX nodes to WGSL strings
  emit(): string {
    let wgsl = '';

    // We bind a single massive buffer representing our Static Memory Arena
    wgsl += `@group(0) @binding(0) var<storage, read_write> memory: array<f32>;\n\n`;

    // Emit a main entrypoint compute shader
    wgsl += `@compute @workgroup_size(64)\n`;
    wgsl += `fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n`;
    wgsl += `    let id = global_id.x;\n`;
    wgsl += `    // Graph Execution Stub\n`;

    for (const node of this.tir.nodes) {
      if (node.type === 'tir.add') {
        wgsl += `    // Node: ${node.id} (Add)\n`;
        wgsl += `    let a = memory[0 + id];\n`;
        wgsl += `    let b = memory[4 + id];\n`;
        wgsl += `    memory[8 + id] = a + b;\n`;
      } else if (node.type === 'tir.matmul') {
        wgsl += `    // Node: ${node.id} (MatMul)\n`;
        wgsl += `    // Minimal MatMul stub (Vector * Matrix assuming 1D flattening for stub)\n`;
        wgsl += `    let m_a = memory[0];\n`;
        wgsl += `    let m_b = memory[1];\n`;
        wgsl += `    memory[2] = m_a * m_b;\n`;
      } else {
        wgsl += `    // Untranslated node: ${node.type}\n`;
      }
    }

    wgsl += `}\n`;

    return wgsl;
  }
}
