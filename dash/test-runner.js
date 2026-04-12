#!/usr/bin/env node

const { execSync } = require('child_process');
const fs = require('fs');

const dashPath = __dirname + '/public/dash.js';

const tests = [
  { name: 'echo', script: 'echo hello', expected: 'hello\n' },
  { name: 'variables', script: 'a=123\necho $a', expected: '123\n' },
  { name: 'loops', script: 'for i in 1 2 3; do echo $i; done', expected: '1\n2\n3\n' },
  { name: 'math', script: 'echo $((2+3))', expected: '5\n' },
  { name: 'pipe', script: 'echo "a\nb\nc" | grep b', expected: 'b\n' }
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
  console.log("All tests passed!");
  process.exit(0);
}
