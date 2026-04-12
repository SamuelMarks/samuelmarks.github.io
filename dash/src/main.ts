import { Terminal } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import { WebLinksAddon } from '@xterm/addon-web-links';
import '@xterm/xterm/css/xterm.css';

const term = new Terminal({
  cursorBlink: true,
  fontFamily: 'monospace',
  fontSize: 14,
  theme: { background: '#1e1e1e' }
});

const fitAddon = new FitAddon();
term.loadAddon(fitAddon);
term.loadAddon(new WebLinksAddon());

const container = document.getElementById('terminal-container')!;
term.open(container);
fitAddon.fit();

window.addEventListener('resize', () => fitAddon.fit());

term.writeln('Welcome to \x1b[1;32mdash\x1b[0m compiled to WebAssembly!');
term.writeln('Loading shell in Web Worker...');

const worker = new Worker('/worker.js');

worker.onmessage = (e) => {
  const msg = e.data;
  switch (msg.type) {
    case 'LOADED':
      term.writeln('Shell loaded.\r\n');
      if (window.location.search.includes('test=1')) {
        worker.postMessage({ type: 'TEST_CMD' });
      }
      break;
    case 'STDOUT':
      term.write(msg.data.replace(/\n/g, '\r\n') + '\r\n');
      break;
    case 'STDERR':
      term.write('\x1b[31m' + msg.data.replace(/\n/g, '\r\n') + '\x1b[0m\r\n');
      break;
    case 'STDOUT_CHAR':
      const char = String.fromCharCode(msg.data);
      if (char === '\n') term.write('\r\n');
      else term.write(char);
      break;
  }
};

let inputBuffer = '';

term.onData(e => {
  // Ignore escape sequences (e.g., arrow keys) since we are in basic canonical mode
  if (e.startsWith('\x1b')) return;

  const charCodes = [];
  for (let i = 0; i < e.length; i++) {
    const char = e.charAt(i);
    const charCode = e.charCodeAt(i);

    if (charCode === 13) { // Enter
      term.write('\r\n');
      inputBuffer += '\n';
      for (let j = 0; j < inputBuffer.length; j++) {
        charCodes.push(inputBuffer.charCodeAt(j));
      }
      worker.postMessage({ type: 'INPUT', data: charCodes });
      inputBuffer = '';
      charCodes.length = 0;
    } else if (charCode === 127 || charCode === 8) { // Backspace or Ctrl+H
      if (inputBuffer.length > 0) {
        inputBuffer = inputBuffer.slice(0, -1);
        term.write('\b \b');
      }
    } else if (charCode === 21) { // Ctrl+U (Clear line)
      while (inputBuffer.length > 0) {
        inputBuffer = inputBuffer.slice(0, -1);
        term.write('\b \b');
      }
    } else if (charCode === 23) { // Ctrl+W (Erase word)
      // Erase trailing spaces
      while (inputBuffer.length > 0 && inputBuffer.endsWith(' ')) {
        inputBuffer = inputBuffer.slice(0, -1);
        term.write('\b \b');
      }
      // Erase word characters
      while (inputBuffer.length > 0 && !inputBuffer.endsWith(' ')) {
        inputBuffer = inputBuffer.slice(0, -1);
        term.write('\b \b');
      }
    } else if (charCode === 3) { // Ctrl+C
      term.write('^C\r\n');
      inputBuffer = '';
      worker.postMessage({ type: 'INPUT', data: [3] });
    } else if (charCode === 4) { // Ctrl+D
      if (inputBuffer.length === 0) {
        worker.postMessage({ type: 'INPUT', data: [4] });
      }
    } else if (charCode >= 32 && charCode <= 126) { // Normal printable char
      inputBuffer += char;
      term.write(char);
    }
  }
});
