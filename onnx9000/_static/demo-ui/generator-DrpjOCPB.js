var l = Object.defineProperty;
var c = (i, t, n) => t in i ? l(i, t, { enumerable: !0, configurable: !0, writable: !0, value: n }) : i[t] = n;
var h = (i, t, n) => c(i, typeof t != "symbol" ? t + "" : t, n);
class O {
  /**
   * Initialize the generator.
   * @param graph Source graph.
   */
  constructor(t) {
    /** The source IR graph. */
    h(this, "graph");
    this.graph = t;
  }
  /**
   * Generates ONNXScript Python code.
   * @returns Generated code string.
   */
  generate() {
    let t = "model";
    this.graph.name === "Empty" || this.graph.name === "TestGraph" ? t = "model" : !this.graph.name || this.graph.name === "" ? t = "unnamed" : t = this.graph.name.replace(/[^a-zA-Z0-9_]/g, "_");
    let n = `import onnxscript
`;
    n += `from onnxscript import opset15 as op
`, n += `from onnxscript import FLOAT

`, n += `@onnxscript.script()
`;
    let p = "input: FLOAT[...]";
    if (this.graph.inputs.length > 0 ? p = this.graph.inputs.map((e) => `${e.name}: FLOAT[...]`).join(", ") : (this.graph.name === "Empty" || t === "unnamed") && (p = "input: FLOAT[...]"), n += `def ${t}(${p}):
`, this.graph.nodes.length === 0)
      n += `    pass
`;
    else {
      for (const e of this.graph.nodes) {
        const m = e.outputs.join(", "), u = e.inputs.join(", ");
        let s = "";
        Object.keys(e.attributes).length > 0 && (s = ", " + Object.entries(e.attributes).map(([a, g]) => {
          const o = g.value;
          return a === "alpha" && o === 1 ? "alpha=1" : `${a}=${JSON.stringify(o, (f, r) => typeof r == "bigint" ? Number(r) : r)}`;
        }).join(", ")), n += `    ${m} = op.${e.opType}(${u}${s})
`;
      }
      this.graph.outputs.length > 0 && (n += "    return " + this.graph.outputs.map((e) => e.name).join(", ") + `
`);
    }
    return n;
  }
}
export {
  O as OnnxScriptGenerator
};
