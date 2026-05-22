import { A as n, N as l, T as N } from "./main-Mf2qmUsB.js";
class v {
  /**
   * Translates a parsed TensorFlow node into an equivalent array of ONNX nodes.
   *
   * @param node The TensorFlow node definition parsed from a .pbtxt file.
   * @param graph The ONNX graph to append initializers and tensors to.
   * @returns An array of translated ONNX nodes.
   */
  map(i, r) {
    const e = {}, d = [i.name];
    let a = i.op;
    for (const [t, s] of Object.entries(i.attr))
      s.list && s.list.i ? e[t] = new n(t, "INTS", s.list.i) : s.i !== void 0 ? e[t] = new n(t, "INT", s.i) : s.f !== void 0 ? e[t] = new n(t, "FLOAT", s.f) : s.s !== void 0 ? e[t] = new n(t, "STRING", s.s) : s.shape && (e[t] = new n(t, "INTS", s.shape));
    if (a === "Placeholder")
      return a = "Identity", [new l(
        a,
        i.input.length ? i.input : [i.name + "_input_dummy"],
        d,
        e,
        i.name
      )];
    if (a === "Const") {
      a = "Constant", r.initializers.push(i.name);
      const t = i.attr.value;
      let s = [1];
      t && t.tensor && t.tensor.shape && t.tensor.shape.length > 0 && (s = t.tensor.shape);
      const o = new N(i.name, s, "float32"), p = s.reduce((u, f) => u * Math.abs(f), 1) || 1;
      return o.data = new Uint8Array(p * 4), r.tensors[i.name] = o, [];
    } else if (a === "Relu6")
      a = "Relu";
    else if (a === "Conv2D" || a === "DepthwiseConv2dNative") {
      if (a = "Conv", e.strides && e.strides.type === "INTS" && Array.isArray(e.strides.value) && e.strides.value.length === 4) {
        const t = e.strides.value;
        e.strides = new n("strides", "INTS", [t[1], t[2]]);
      }
      if (e.padding && e.padding.type === "STRING") {
        const t = e.padding.value;
        t === "SAME" ? e.auto_pad = new n("auto_pad", "STRING", "SAME_UPPER") : t === "VALID" && (e.auto_pad = new n("auto_pad", "STRING", "VALID")), delete e.padding;
      }
    } else if (a === "MaxPool" || a === "AvgPool") {
      if (e.ksize && e.ksize.type === "INTS" && Array.isArray(e.ksize.value) && e.ksize.value.length === 4) {
        const t = e.ksize.value;
        e.kernel_shape = new n("kernel_shape", "INTS", [t[1], t[2]]), delete e.ksize;
      }
      if (e.strides && e.strides.type === "INTS" && Array.isArray(e.strides.value) && e.strides.value.length === 4) {
        const t = e.strides.value;
        e.strides = new n("strides", "INTS", [t[1], t[2]]);
      }
      if (e.padding && e.padding.type === "STRING") {
        const t = e.padding.value;
        t === "SAME" ? e.auto_pad = new n("auto_pad", "STRING", "SAME_UPPER") : t === "VALID" && (e.auto_pad = new n("auto_pad", "STRING", "VALID")), delete e.padding;
      }
    }
    return [new l(a, i.input, d, e, i.name)];
  }
}
export {
  v as TFMapper
};
