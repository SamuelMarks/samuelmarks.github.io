# Keras2ONNX

Integration guide for converting Keras models to ONNX using `onnx9000-converters`.

## CLI Usage

```bash
onnx9000 convert --from keras --src my_model.h5 --to onnx -o model.onnx
```

## Demo

See the standalone web demo:
```bash
cd apps/demo-keras2onnx
npm run dev
```
