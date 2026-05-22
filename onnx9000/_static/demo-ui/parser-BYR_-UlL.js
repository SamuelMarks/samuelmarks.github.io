import { G as m, V as l, N as d } from "./main-BIlw4j2H.js";
class w {
  /**
   * Parses ONNXScript Python code into a basic ONNX IR Graph.
   * This is a lightweight text-based AST parser designed to operate
   * natively in the JS browser environment without the Pyodide/Python runtime overhead.
   *
   * @param scriptContent The Python source code of the ONNXScript function.
   * @returns A populated ONNX IR Graph.
   */
  parseScript(u) {
    const a = new m("onnxscript-imported"), f = u.split(`
`);
    let c = !1;
    const p = /([a-zA-Z0-9_]+):\s*FLOAT(?:\[(.*?)\])?/g;
    for (let t of f) {
      if (t = t.trim(), t.startsWith("def ")) {
        c = !0;
        const n = t.match(/def\s+[a-zA-Z0-9_]+\s*\((.*?)\)(?:\s*->\s*(.*?))?:/);
        if (n) {
          const i = n[1];
          if (i) {
            let e;
            for (p.lastIndex = 0; (e = p.exec(i)) !== null; ) {
              const r = e[1], o = e[2] ? e[2].split(",").map((h) => parseInt(h.trim(), 10)) : (
                /* v8 ignore start */
                [-1]
              );
              r && a.inputs.push(new l(r, o, "float32"));
            }
          }
        }
        continue;
      }
      if (!c || t === "") continue;
      if (t.startsWith("return ")) {
        const n = t.replace("return ", "").trim();
        a.outputs.push(new l(n, [-1, -1], "float32"));
        continue;
      }
      const s = t.match(/^([a-zA-Z0-9_, ]+)\s*=\s*op\.([a-zA-Z0-9_]+)\s*\((.*?)\)$/);
      if (s && s[1] && s[2] && s[3] !== void 0) {
        const n = s[1].split(",").map((o) => o.trim()), i = s[2], e = s[3].split(",").map((o) => o.trim()), r = new d(i, e, n, {}, `${i}_${n[0] || "out"}`);
        a.nodes.push(r);
      }
    }
    return a;
  }
}
export {
  w as OnnxScriptParser
};
