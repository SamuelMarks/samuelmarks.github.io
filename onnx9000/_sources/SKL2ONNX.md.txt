# SKL2ONNX

Integration guide for converting Scikit-Learn models to ONNX using `onnx9000-converters`.

## CLI Usage

```bash
onnx9000 convert --from sklearn --src my_model.pkl --to onnx -o model.onnx
```

## Demo

See the standalone web demo:
```bash
cd apps/demo-skl2onnx
npm run dev
```
