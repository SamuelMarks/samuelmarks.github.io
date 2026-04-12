let stdinBuffer = [];
let resolveStdin = null;
globalThis.stdinBuffer = stdinBuffer;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'INPUT') {
    stdinBuffer.push(...msg.data);
    if (self.__DASH_RESOLVE) {
      let rs = self.__DASH_RESOLVE;
      self.__DASH_RESOLVE = null;
      rs();
    }
  } else if (msg.type === 'TEST_CMD') {
    const testCmd = 'echo test\n';
    for (let i = 0; i < testCmd.length; i++) {
      stdinBuffer.push(testCmd.charCodeAt(i));
    }
    if (self.__DASH_RESOLVE) {
      let rs = self.__DASH_RESOLVE;
      self.__DASH_RESOLVE = null;
      rs();
    }
  }
};

self.Module = {
  preRun: [
    () => {
      const FS = self.Module.FS;
      const IDBFS = self.IDBFS || self.Module.IDBFS;
      try { FS.mkdir('/etc'); } catch (e) {}
      try { FS.mkdir('/bin'); } catch (e) {}
      FS.writeFile('/bin/uname', '');
      FS.chmod('/bin/uname', 0o777);
      FS.writeFile('/bin/ls', '');
      FS.chmod('/bin/ls', 0o777);
      FS.writeFile('/bin/cat', '');
      FS.chmod('/bin/cat', 0o777);
      FS.writeFile('/bin/grep', '');
      FS.chmod('/bin/grep', 0o777);
      try { FS.mkdir('/tmp'); } catch (e) {}
      FS.writeFile('/etc/profile', 'export PATH=/bin\nexport PS1="wasm-shell$ "\n');
      try { FS.mkdir('/home'); } catch (e) {}
      try { FS.mkdir('/home/web_user'); } catch (e) {}
      
      FS.mount(IDBFS, {}, '/home/web_user');
      self.Module.addRunDependency('syncfs');
      FS.syncfs(true, (err) => {
        if (err) console.error('IDBFS sync error:', err);
        try { FS.stat('/home/web_user/.profile'); } 
        catch (e) { FS.writeFile('/home/web_user/.profile', 'alias ll="ls -l"\n'); }
        self.Module.removeRunDependency('syncfs');
      });
    }
  ],
  print: (text) => {
    self.postMessage({ type: 'STDOUT', data: text });
  },
  printErr: (text) => {
    // console.log("STDERR:", text);
    self.postMessage({ type: 'STDERR', data: text });
  },
  onExit: (code) => {
    self.postMessage({ type: 'STDOUT', data: '\r\n[dash process exited with code ' + code + ']\r\n' });
  },
  onRuntimeInitialized: () => {
    self.postMessage({ type: 'LOADED' });
    const FS = self.Module.FS;
    setInterval(() => {
      FS.syncfs(false, () => {});
    }, 5000);

    if (self.Module.TTY) {
      const put_char = function(tty, val) {
        if (val === null || val === undefined) return;
        self.postMessage({ type: 'STDOUT_CHAR', data: val });
      };
      if (self.Module.TTY.default_tty_ops) self.Module.TTY.default_tty_ops.put_char = put_char;
      if (self.Module.TTY.default_tty1_ops) self.Module.TTY.default_tty1_ops.put_char = put_char;
    }

    const stdinStream = FS.getStream(0);
    if (stdinStream) {
      // We now handle stdin asynchronously directly in dash via EM_ASM_INT in input.c
    }
  }
};

importScripts('./dash.js');
