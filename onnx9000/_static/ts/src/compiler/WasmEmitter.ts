import { ITIRGraph, ILoweredNode } from './Lowering';

// Minimal WebAssembly emitter using magic bytes and raw opcode array building.
export class WasmEmitter {
  private tir: ITIRGraph;

  // Standard WASM opcodes
  private static OP = {
    f32_add: 0x92,
    f32_sub: 0x93,
    f32_mul: 0x94,
    f32_div: 0x95,
    local_get: 0x20,
    local_set: 0x21,
    f32_load: 0x2a,
    f32_store: 0x38,
    return: 0x0f,
    end: 0x0b,
  };

  constructor(tir: ITIRGraph) {
    this.tir = tir;
  }

  emit(): Uint8Array {
    // 176. Emit WASM module header
    const magic = new Uint8Array([0x00, 0x61, 0x73, 0x6d]); // '\0asm'
    const version = new Uint8Array([0x01, 0x00, 0x00, 0x00]); // 1

    // Type section (1): Function signatures
    // 1 func type: (i32, i32) -> ()  (e.g., input pointer, output pointer)
    const typeSection = this.createSection(
      1,
      new Uint8Array([
        0x01, // 1 type
        0x60, // func type form
        0x02,
        0x7f,
        0x7f, // 2 params: i32, i32
        0x00, // 0 results
      ]),
    );

    // Import section (2): Import memory
    // env.memory (min 1 page)
    const importSection = this.createSection(
      2,
      new Uint8Array([
        0x01, // 1 import
        0x03,
        0x65,
        0x6e,
        0x76, // "env"
        0x06,
        0x6d,
        0x65,
        0x6d,
        0x6f,
        0x72,
        0x79, // "memory"
        0x02, // memory export
        0x00,
        0x01, // limit flags (min 1)
      ]),
    );

    // Function section (3): Link index to type signature
    const funcSection = this.createSection(
      3,
      new Uint8Array([
        0x01, // 1 function
        0x00, // type index 0
      ]),
    );

    // Export section (7): Export the function as "execute"
    const exportSection = this.createSection(
      7,
      new Uint8Array([
        0x01, // 1 export
        0x07,
        0x65,
        0x78,
        0x65,
        0x63,
        0x75,
        0x74,
        0x65, // "execute"
        0x00, // kind function
        0x00, // func index 0
      ]),
    );

    // Data section (11): Encode static weights (180. Encode static weights directly into the WASM binary data section)
    // Stub: For real impl, we iterate graph.initializers and write bytes into linear memory offsets
    // Memory Index 0, Offset expression: i32.const 0, end
    // data payload: [0x00, 0x00, 0x00, 0x00]
    const dataSection = this.createSection(
      11,
      new Uint8Array([
        0x01, // 1 data segment
        0x00, // memory index 0, active
        0x41,
        0x00,
        0x0b, // i32.const 0, end (offset expr)
        0x04, // payload size
        0x00,
        0x00,
        0x00,
        0x00, // 4 bytes of static data
      ]),
    );

    // Code section (10): Function body
    const bodyBytes = this.emitExecutionCode();
    // 1 func, size, local declarations count (0)
    const funcBody = new Uint8Array([0x01, bodyBytes.length + 1, 0x00, ...bodyBytes]);
    const codeSection = this.createSection(10, funcBody);

    // Concat everything
    const totalLength =
      magic.length +
      version.length +
      typeSection.length +
      importSection.length +
      funcSection.length +
      exportSection.length +
      codeSection.length +
      dataSection.length;
    const finalWasm = new Uint8Array(totalLength);
    let offset = 0;
    const append = (buf: Uint8Array) => {
      finalWasm.set(buf, offset);
      offset += buf.length;
    };

    append(magic);
    append(version);
    append(typeSection);
    append(importSection);
    append(funcSection);
    append(exportSection);
    append(codeSection);
    append(dataSection);

    return finalWasm;
  }

  private emitExecutionCode(): Uint8Array {
    const code: number[] = [];

    for (const node of this.tir.nodes) {
      if (node.type === 'tir.add' || node.type === 'tir.sub' || node.type === 'tir.mul') {
        // Stub: load two F32, add, store
        // In a real JIT, this tracks memory layout offsets
        code.push(WasmEmitter.OP.local_get, 0x00); // base pointer
        code.push(WasmEmitter.OP.f32_load, 0x02, 0x00); // load A
        code.push(WasmEmitter.OP.local_get, 0x00);
        code.push(WasmEmitter.OP.f32_load, 0x02, 0x04); // load B (offset 4)

        if (node.type === 'tir.add') code.push(WasmEmitter.OP.f32_add);
        if (node.type === 'tir.sub') code.push(WasmEmitter.OP.f32_sub);
        if (node.type === 'tir.mul') code.push(WasmEmitter.OP.f32_mul);

        code.push(WasmEmitter.OP.local_get, 0x01); // dest pointer
        code.push(WasmEmitter.OP.f32_store, 0x02, 0x00); // store result
      } else if (node.type === 'tir.matmul') {
        // 178. Implement nested loop generators
        // This is a minimal stub for a MatMul emission byte trace
        // A true implementation dynamically emits looping constructs (Block, Loop, Br_if)
        // We will just do a simple fallback sequence
        code.push(WasmEmitter.OP.local_get, 0x00);
        code.push(WasmEmitter.OP.f32_load, 0x02, 0x00);
        code.push(WasmEmitter.OP.local_get, 0x00);
        code.push(WasmEmitter.OP.f32_load, 0x02, 0x04);
        code.push(WasmEmitter.OP.f32_mul);
        code.push(WasmEmitter.OP.local_get, 0x01);
        code.push(WasmEmitter.OP.f32_store, 0x02, 0x00);
      }
    }

    // Always return cleanly in stub
    code.push(WasmEmitter.OP.return);

    code.push(WasmEmitter.OP.end);
    return new Uint8Array(code);
  }

  private createSection(id: number, data: Uint8Array): Uint8Array {
    // Basic uleb128 for size. Assuming size < 128 for these simple stubs
    return new Uint8Array([id, data.length, ...data]);
  }
}
