global.Module = {
  arguments: ['-c', 'echo Hello from dash!'],
  print: console.log,
  printErr: console.error,
  instantiateWasm: function(info, receiveInstance) {
    console.log("instantiateWasm called!");
    return false; // let emscripten handle it
  },
  preRun: [() => console.log("PreRun called")],
  postRun: [() => console.log("PostRun called")],
  onRuntimeInitialized: () => console.log("Runtime initialized")
};
console.log("Starting test.js");
require('./public/dash.js');
console.log("After require");
