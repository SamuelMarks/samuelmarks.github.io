mergeInto(LibraryManager.library, {
  dash_async_read: function(fd, bufPtr, length) {
    return Asyncify.handleSleep(function(wakeUp) {
        globalThis.__DASH_RESOLVE = function() {
            let bytesRead = 0;
            while (globalThis.stdinBuffer.length > 0 && bytesRead < length) {
                HEAP8[bufPtr + bytesRead] = globalThis.stdinBuffer.shift();
                bytesRead++;
            }
            wakeUp(bytesRead);
        };
        if (globalThis.stdinBuffer && globalThis.stdinBuffer.length > 0) {
            let rs = globalThis.__DASH_RESOLVE;
            globalThis.__DASH_RESOLVE = null;
            setTimeout(rs, 0);
        }
    });
  }
});
