#!/usr/bin/env node

const fs = require('fs');

global.Module = {
  print: (text) => process.stdout.write(text + '\n'),
  printErr: (text) => process.stderr.write(text + '\n'),
  onExit: (code) => { process.exit(code); },
  preRun: [],
  postRun: [],
};

const wasmPath = __dirname + '/public/dash.wasm';
const jsPath = __dirname + '/public/dash.js';

Module.Asyncify = require(jsPath).Asyncify;
// Execute the script
const code = fs.readFileSync(jsPath, 'utf8');
eval(code);

