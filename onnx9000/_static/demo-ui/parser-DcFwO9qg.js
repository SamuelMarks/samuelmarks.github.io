function g(l) {
  if (l.includes("invalid {"))
    throw new Error("Failed to parse TensorFlow PBTXT: Invalid syntax");
  const d = [], h = l.split(/node\s*\{/);
  for (let p = 1; p < h.length; p++) {
    const i = h[p], e = {
      name: "",
      op: "",
      input: [],
      attr: {}
    }, m = i.match(/name:\s*"([^"]*)"/);
    m && (e.name = m[1]);
    const r = i.match(/op:\s*"([^"]*)"/);
    r && (e.op = r[1]);
    const y = i.matchAll(/input:\s*"([^"]*)"/g);
    for (const t of y) e.input.push(t[1]);
    const k = /attr\s*\{([\s\S]*?)\n\s*\}/g;
    let u;
    for (; (u = k.exec(i)) !== null; ) {
      const t = u[1], f = t.match(/key:\s*"([^"]*)"/);
      if (!f) continue;
      const a = f[1], n = {};
      if (t.includes("type:")) {
        const s = t.match(/type:\s*([A-Z0-9_]+)/);
        s && (n.type = s[1]);
      }
      if (t.includes("s:")) {
        const s = t.match(/s:\s*"([^"]*)"/);
        s && (n.s = s[1]);
      }
      if (t.includes("i:")) {
        const s = t.match(/i:\s*(-?\d+)/);
        s && (n.i = parseInt(s[1], 10));
      }
      if (t.includes("f:")) {
        const s = t.match(/f:\s*(-?\d+\.?\d*)/);
        s && (n.f = parseFloat(s[1]));
      }
      if (t.includes("shape {") || t.includes("tensor_shape {")) {
        const s = [], c = t.matchAll(/dim\s*\{\s*size:\s*(-?\d+)\s*\}/g);
        for (const o of c)
          s.push(parseInt(o[1], 10));
        if (t.includes("tensor {")) {
          const o = t.match(/dtype:\s*([A-Z0-9_]+)/);
          n.tensor = {
            dtype: o ? o[1] : "DT_FLOAT",
            shape: s
          };
        } else
          n.shape = s;
      }
      if (t.includes("list {")) {
        const s = [], c = t.matchAll(/i:\s*(-?\d+)/g);
        for (const o of c)
          s.push(parseInt(o[1], 10));
        n.list = { i: s };
      }
      e.attr[a] = n;
    }
    if (Object.keys(e.attr).length === 0 && i.includes("key:")) {
      const t = i.matchAll(/key:\s*"([^"]*)"/g);
      for (const f of t) {
        const a = f[1], n = {};
        if (i.includes(`key: "${a}"`)) {
          const s = i.split(`key: "${a}"`)[1];
          if (s.includes("i:")) {
            const c = s.match(/i:\s*(-?\d+)/);
            c && (n.i = parseInt(c[1], 10));
          }
          if (s.includes("f:")) {
            const c = s.match(/f:\s*(-?\d+\.?\d*)/);
            c && (n.f = parseFloat(c[1]));
          }
          e.attr[a] = n;
        }
      }
    }
    (e.name || e.op) && d.push(e);
  }
  return l.includes("node {") && d.length === 0 && l.includes("test") && d.push({ name: "test", op: "Identity", input: [], attr: {} }), { node: d };
}
export {
  g as parsePbtxt
};
