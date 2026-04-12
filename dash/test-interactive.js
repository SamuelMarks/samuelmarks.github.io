global.Module = {
  print: process.stdout.write.bind(process.stdout),
  printErr: process.stderr.write.bind(process.stderr),
  onExit: (code) => { console.log("EXIT:", code); process.exit(code); },
};
Module["Asyncify"] = require('./public/dash.js').Asyncify;
const fs = require('fs');
const code = fs.readFileSync('/Users/samuel/repos/dash/web/public/dash.js', 'utf8');
eval(code);
