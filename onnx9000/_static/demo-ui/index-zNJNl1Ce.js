var d = Object.defineProperty;
var g = (s, t, a) => t in s ? d(s, t, { enumerable: !0, configurable: !0, writable: !0, value: a }) : s[t] = a;
var f = (s, t, a) => g(s, typeof t != "symbol" ? t + "" : t, a);
import { G as m, T as c, V as h, N as v, A as T } from "./main-Mf2qmUsB.js";
function A(s) {
  return JSON.parse(s);
}
class S {
  constructor(t, a) {
    f(this, "jaxpr");
    f(this, "flaxState");
    f(this, "graph");
    this.jaxpr = t, a && (this.flaxState = a), this.graph = new m("JaxModel");
  }
  getTensor(t) {
    if (!this.graph.tensors[t]) {
      const a = new c(t, [], "float32");
      return this.graph.addTensor(a), a;
    }
    return this.graph.tensors[t];
  }
  map() {
    for (const t of this.jaxpr.invars)
      this.getTensor(t), this.graph.inputs.push(new h(t, [], "float32"));
    for (const t of this.jaxpr.constvars)
      if (this.flaxState && this.flaxState[t]) {
        const a = this.flaxState[t];
        let o = [];
        if ("flat" in a && typeof a.flat == "function") {
          const n = a.flat();
          o = Array.isArray(n) ? n : [];
        } else Array.isArray(a) && (o = a);
        const i = new Float32Array(o), u = new c(t, [], "float32", !0, !1, new Uint8Array(i.buffer));
        this.graph.addTensor(u);
      } else
        this.getTensor(t), this.graph.inputs.push(new h(t, [], "float32"));
    for (const t of this.jaxpr.eqns) {
      const a = t.invars.map((r) => this.getTensor(r).name), o = t.outvars.map((r) => this.getTensor(r).name);
      let i = t.primitive;
      const n = {
        add: "Add",
        sub: "Sub",
        mul: "Mul",
        div: "Div",
        dot_general: "MatMul",
        broadcast_in_dim: "Expand",
        reshape: "Reshape",
        conv_general_dilated: "Conv",
        max_pool: "MaxPool",
        reduce_sum: "ReduceSum"
      }[t.primitive];
      n && (i = n);
      const l = new v(i, a, o, {}, `${t.primitive}_node`);
      for (const r in t.params) {
        const e = t.params[r];
        let p = "STRING";
        typeof e == "number" ? p = Number.isInteger(e) ? "INT" : "FLOAT" : typeof e == "string" ? p = "STRING" : Array.isArray(e) && (p = typeof e[0] == "number" ? Number.isInteger(e[0]) ? "INTS" : "FLOATS" : "STRING"), l.attributes[r] = new T(r, p, e);
      }
      this.graph.addNode(l);
    }
    for (const t of this.jaxpr.outvars)
      this.getTensor(t), this.graph.outputs.push(new h(t, [], "float32"));
    return this.graph;
  }
}
export {
  S as JaxMapper,
  A as parseJaxpr
};
