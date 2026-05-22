import { B as y, r as g, W as p, a as w, b as m, s as r, c as b, d as N, e as P } from "./main-Mf2qmUsB.js";
function E(t) {
  const s = { layer: [], input: [], input_dim: [], input_shape: [] }, a = t.split(`
`), i = [s];
  let e = s;
  for (let n = 0; n < a.length; n++) {
    const f = a[n];
    if (f === void 0) continue;
    let c = f.split("#")[0].trim();
    if (c)
      if (c.endsWith("{")) {
        const o = c.slice(0, -1).trim(), l = {};
        o === "layer" || o === "layers" ? (s.layer.push(l), i.push(l), e = l) : (e[o] ? (Array.isArray(e[o]) || (e[o] = [e[o]]), e[o].push(l)) : e[o] = l, i.push(l), e = l);
      } else if (c === "}")
        i.pop(), e = i[i.length - 1];
      else {
        const o = c.indexOf(":");
        if (o !== -1) {
          const l = c.substring(0, o).trim();
          let u = c.substring(o + 1).trim(), h = u;
          u.startsWith('"') && u.endsWith('"') || u.startsWith("'") && u.endsWith("'") ? h = u.substring(1, u.length - 1) : !isNaN(Number(u)) && u !== "true" && u !== "false" ? h = Number(u) : u === "true" ? h = !0 : u === "false" && (h = !1), l === "input" || l === "input_dim" || l === "bottom" || l === "top" || l === "dim" ? (e[l] ? Array.isArray(e[l]) || (e[l] = [e[l]]) : e[l] = [], e[l].push(h)) : e[l] = h;
        }
      }
  }
  return s;
}
async function L(t) {
  const s = new y(t), a = { layer: [] };
  for (; s.getPosition() < s.getLength(); ) {
    const { fieldNumber: i, wireType: e } = await g(s);
    if (i === 1 && e === p) {
      const n = await w(s);
      a.name = await m(s, n);
    } else if ((i === 100 || i === 2) && e === p) {
      const n = await w(s), f = new y(await s.readBytes(n)), c = await T(f);
      a.layer.push(c);
    } else
      await r(s, e);
  }
  return a;
}
async function T(t) {
  const s = { blobs: [] };
  for (; t.getPosition() < t.getLength(); ) {
    const { fieldNumber: a, wireType: i } = await g(t);
    if (a === 1 && i === p) {
      const e = await w(t);
      s.name = await m(t, e);
    } else if (a === 2 && i === p) {
      const e = await w(t);
      s.type = await m(t, e);
    } else if ((a === 50 || a === 6) && i === p) {
      const e = await w(t), n = new y(await t.readBytes(e)), f = await _(n);
      s.blobs.push(f);
    } else
      await r(t, i);
  }
  return s;
}
async function _(t) {
  const s = { data: [] }, a = [];
  for (; t.getPosition() < t.getLength(); ) {
    const { fieldNumber: i, wireType: e } = await g(t);
    if (i === 1 && e === b)
      a[0] = await w(t);
    else if (i === 2 && e === b)
      a[1] = await w(t);
    else if (i === 3 && e === b)
      a[2] = await w(t);
    else if (i === 4 && e === b)
      a[3] = await w(t);
    else if (i === 5)
      if (e === N) {
        const n = await t.readBytes(4), f = new DataView(n.buffer, n.byteOffset, n.byteLength).getFloat32(
          0,
          !0
        );
        s.data.push(f);
      } else if (e === p) {
        const n = await w(t), f = await t.readBytes(n), c = new DataView(f.buffer, f.byteOffset, f.byteLength);
        for (let o = 0; o < n; o += 4)
          s.data.push(c.getFloat32(o, !0));
      } else
        await r(t, e);
    else if (i === 7 && e === p) {
      const n = await w(t), f = new y(await t.readBytes(n));
      s.shape = await B(f);
    } else
      await r(t, e);
  }
  return !s.shape && a.length > 0 && (s.shape = a.filter((i) => i !== void 0)), s;
}
async function B(t) {
  const s = [];
  for (; t.getPosition() < t.getLength(); ) {
    const { fieldNumber: a, wireType: i } = await g(t);
    if (a === 1 && i === b)
      s.push(Number(await P(t)));
    else if (a === 1 && i === p) {
      const e = await w(t), n = t.getPosition() + e;
      for (; t.getPosition() < n; )
        s.push(Number(await P(t)));
    } else
      await r(t, i);
  }
  return s;
}
export {
  L as parseCaffemodel,
  E as parsePrototxt
};
