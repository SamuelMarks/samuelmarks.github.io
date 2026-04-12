const fs = require('fs');
const wasmBuffer = fs.readFileSync('/Users/samuel/repos/dash/web/public/dash.wasm');
WebAssembly.instantiate(wasmBuffer, {
  a: new Proxy({}, { get: (target, prop) => function() { console.log("called", prop); return 0; } }),
  env: new Proxy({}, { get: (target, prop) => function() { console.log("called env", prop); return 0; } })
}).then(result => {
  console.log("Instantiated!", Object.keys(result.instance.exports));
}).catch(err => {
  console.error("Error:", err);
});
