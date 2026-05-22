var m = Object.defineProperty;
var d = (o, e, t) => e in o ? m(o, e, { enumerable: !0, configurable: !0, writable: !0, value: t }) : o[e] = t;
var c = (o, e, t) => d(o, typeof e != "symbol" ? e + "" : e, t);
import { G as u, V as h, A as p, N as y } from "./main-Mf2qmUsB.js";
function M(o) {
  const e = o.trim().split(`
`);
  let t = "c_svc", r = "rbf", n = 0;
  const l = [];
  let a = !1;
  for (let s of e)
    if (s = s.trim(), !!s)
      if (a) {
        const i = s.split(/\s+/);
        if (i.length > 0 && i[0] !== void 0) {
          const f = parseFloat(i[0]);
          isNaN(f) || l.push(f);
        }
      } else if (s.startsWith("svm_type"))
        t = s.split(/\s+/)[1] || "c_svc";
      else if (s.startsWith("kernel_type"))
        r = s.split(/\s+/)[1] || "rbf";
      else if (s.startsWith("rho")) {
        const i = s.split(/\s+/)[1];
        i && (n = parseFloat(i));
      } else s === "SV" && (a = !0);
  return { svmType: t, kernelType: r, rho: n, coefs: l };
}
class T {
  constructor(e) {
    c(this, "model");
    this.model = e;
  }
  map() {
    const e = new u("LibSVM_Model");
    e.opsetImports = { "": 14, "ai.onnx.ml": 3 }, e.inputs.push(new h("X", [-1, 10], "float32")), e.outputs.push(new h("Y", [-1, 1], "float32"));
    const t = this.model.kernelType.toUpperCase(), r = this.model.svmType.includes("svr") ? "SVMRegressor" : "SVMClassifier", n = {
      kernel_type: new p("kernel_type", "STRING", t),
      rho: new p("rho", "FLOATS", [this.model.rho])
    };
    this.model.coefs.length > 0 && (n.coefficients = new p("coefficients", "FLOATS", this.model.coefs));
    const l = new y(r, ["X"], ["Y"], n, r, "ai.onnx.ml");
    return e.addNode(l), e;
  }
}
export {
  T as LibSVMMapper,
  M as parseLibSVM
};
