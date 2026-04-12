global.Module = {
  onRuntimeInitialized: () => {
    console.log("stdinStream:", window.Module.FS.getStream(0) !== null);
  }
};
Module["Asyncify"] = require('./public/dash.js').Asyncify;
const fs = require('fs');
const code = fs.readFileSync('/Users/samuel/repos/dash/web/public/dash.js', 'utf8');
eval(code);
