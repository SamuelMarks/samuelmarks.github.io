global.Module = {
  arguments: ['-c', 'ls -l'],
  print: console.log,
  printErr: console.error,
  onExit: (code) => { console.log("EXIT:", code); process.exit(code); },
};
Module["Asyncify"] = require('./public/dash.js').Asyncify;
const fs = require('fs');
const code = fs.readFileSync('/Users/samuel/repos/dash/web/public/dash.js', 'utf8');
eval(code);
