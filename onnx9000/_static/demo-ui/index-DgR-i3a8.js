var m = Object.defineProperty;
var g = (t, e, s) => e in t ? m(t, e, { enumerable: !0, configurable: !0, writable: !0, value: s }) : t[e] = s;
var n = (t, e, s) => g(t, typeof e != "symbol" ? e + "" : e, s);
import { G as u, V as l, N as c, A as a } from "./main-BIlw4j2H.js";
function h(t) {
  if (t.trim().startsWith("{"))
    try {
      return JSON.parse(t);
    } catch {
      return {};
    }
  return {};
}
class N {
  constructor(e) {
    n(this, "modelData");
    this.modelData = e;
  }
  map() {
    const e = new u("H2O_Model");
    e.opsetImports = { "": 14, "ai.onnx.ml": 3 }, e.inputs.push(new l("X", [-1, 10], "float32")), e.outputs.push(new l("Y", [-1, 1], "float32"));
    const s = typeof this.modelData.algo == "string" ? this.modelData.algo : "";
    let r = "TreeEnsembleRegressor";
    const o = {};
    s === "xgboost" ? (r = "TreeEnsembleRegressor", o.n_targets = new a("n_targets", "INT", 1)) : s === "deeplearning" ? r = "MatMul" : (r = "TreeEnsembleRegressor", o.n_targets = new a("n_targets", "INT", 1), o.post_transform = new a("post_transform", "STRING", "NONE"));
    const p = r.startsWith("Tree") ? "ai.onnx.ml" : "", i = new c(r, ["X"], ["Y"], o, r, p);
    return e.addNode(i), e;
  }
}
export {
  N as H2OMapper,
  h as parseH2O
};
