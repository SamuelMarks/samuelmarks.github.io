# Generic AOT Compiler (`compiler`)

ONNX9000 provides a generic Ahead-of-Time (AOT) compiler infrastructure. It parses the ONNX IR and lowers it into a generic representation that can be further targeted by specific backend plugins.

## Web Demo

A standalone web demo is available to visualize the compilation steps.
```bash
cd apps/demo-compile
npm run dev
```

## CLI Usage

Use the CLI to perform AOT compilation on a given model:
```bash
onnx9000 compile my_model.onnx
```

## SDK Integration

The JS/TS SDK natively exports the `compiler` tools for programmatically lowering the IR in the browser or Node.js.
