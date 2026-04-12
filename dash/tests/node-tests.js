#!/usr/bin/env node

const { execSync } = require('child_process');
const path = require('path');

const dashPath = path.resolve(__dirname, '../public/dash.js');

const tests = [
  { name: 'echo builtin', script: 'echo hello', expected: 'hello\n' },
  { name: 'variable assignment', script: 'a=123\necho $a', expected: '123\n' },
  { name: 'for loop', script: 'for i in 1 2 3; do echo $i; done', expected: '1\n2\n3\n' },
  { name: 'arithmetic expansion', script: 'echo $((2+3))', expected: '5\n' },
  { name: 'if statement', script: 'if true; then echo yes; fi', expected: 'yes\n' },
];

let failed = 0;

for (const test of tests) {
  try {
    const output = execSync(`node ${dashPath} -c '${test.script.replace(/'/g, "'\\''")}'`, { encoding: 'utf8' });
    if (output === test.expected) {
      console.log(`PASS: ${test.name}`);
    } else {
      console.error(`FAIL: ${test.name}`);
      console.error(`  Expected: ${JSON.stringify(test.expected)}`);
      console.error(`  Got:      ${JSON.stringify(output)}`);
      failed++;
    }
  } catch (e) {
    console.error(`FAIL: ${test.name} (Error)`);
    console.error(e.message);
    failed++;
  }
}

if (failed > 0) {
  process.exit(1);
} else {
  console.log("All headless Node tests passed!");
  process.exit(0);
}
