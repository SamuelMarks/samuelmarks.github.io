// Trick to set Module properly in Node
globalThis.Module = {
  arguments: ['-c', 'echo Hello from dash!'],
  print: console.log,
  printErr: console.error,
  onExit: (code) => { console.log("EXIT:", code); process.exit(code); },
};
// evaluate dash.js in global scope instead of require to avoid Node's Module shadowing
(() => {
  const fs = require('fs');
  const code = fs.readFileSync('/Users/samuel/repos/dash/web/public/dash.js', 'utf8');
  eval(code);
})();
