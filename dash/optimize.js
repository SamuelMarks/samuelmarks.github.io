const { minify } = require('terser');
const fs = require('fs');

async function optimize() {
  const code = fs.readFileSync('public/dash.js', 'utf8');
  const result = await minify(code, {
    compress: {
      dead_code: true,
      drop_console: false,
      drop_debugger: true,
      keep_classnames: false,
      keep_fargs: true,
      keep_fnames: false,
      keep_infinity: false
    },
    mangle: {
      eval: false,
      keep_classnames: false,
      keep_fnames: false,
      toplevel: false,
      safari10: false
    },
    module: false,
    sourceMap: false
  });
  fs.writeFileSync('public/dash.min.js', result.code);
  console.log('Optimized dash.js to dash.min.js');
}

optimize().catch(err => {
  console.error(err);
  process.exit(1);
});
