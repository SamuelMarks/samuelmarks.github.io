let stdinBuffer = [];
let resolveStdin = null;
globalThis.stdinBuffer = stdinBuffer;

globalThis.processExternalCommand = function (args) {
  const cmd = args[0].split("/").pop();

  if (args.includes("--version")) {
    self.Module.print(`${cmd} (dash-wasm) 0.0.1`);
    return 0;
  }

  const customHelpCommands = [
    "grep",
    "tar",
    "env",
    "sed",
    "sudo",
    "date",
    "df",
    "stat",
    "file",
    "uname",
    "cowsay",
    "whoami",
    "nano",
    "vi",
    "vim",
    "less",
    "more",
    "top",
    "htop",
  ];

  if (args.includes("--help") && !customHelpCommands.includes(cmd)) {
    self.Module.print(`Usage: ${cmd} [OPTION]...`);
    self.Module.print(`This is a WebAssembly port of ${cmd}.`);
    return 0;
  }

  if (cmd === "cowsay") {
    let eyes = "oo";
    let tongue = "  ";
    let wrapColumn = 40;
    let files = [];
    let messages = [];
    let showHelp = false;

    let i = 1;
    while (i < args.length) {
      let arg = args[i];
      if (arg === "-e") {
        eyes = (args[++i] || "oo").substring(0, 2).padEnd(2, " ");
      } else if (arg === "-T") {
        tongue = (args[++i] || "  ").substring(0, 2).padEnd(2, " ");
      } else if (arg === "-W") {
        wrapColumn = parseInt(args[++i], 10) || 40;
      } else if (arg === "-b") {
        eyes = "==";
        tongue = "  ";
      } else if (arg === "-d") {
        eyes = "xx";
        tongue = "U ";
      } else if (arg === "-g") {
        eyes = "$$";
        tongue = "  ";
      } else if (arg === "-p") {
        eyes = "@@";
        tongue = "  ";
      } else if (arg === "-s") {
        eyes = "**";
        tongue = "U ";
      } else if (arg === "-t") {
        eyes = "--";
        tongue = "  ";
      } else if (arg === "-w") {
        eyes = "OO";
        tongue = "  ";
      } else if (arg === "-y") {
        eyes = "..";
        tongue = "  ";
      } else if (arg === "-h" || arg === "--help") {
        showHelp = true;
      } else if (arg.startsWith("-")) {
        // ignore other unknown flags or treat as message
      } else {
        messages.push(arg);
      }
      i++;
    }

    if (showHelp) {
      self.Module.print(
        "Usage: cowsay [-bdgpstwy] [-h] [-e eyes] [-T tongue] [-W wrapcolumn] [message]",
      );
      return 0;
    }

    function wordWrap(text, width) {
      if (width <= 0) return text.split("\n");
      let lines = [];
      let rawLines = text.split("\n");
      for (let rline of rawLines) {
        if (rline === "") {
          lines.push("");
          continue;
        }
        let currentLine = "";
        let words = rline.split(/[ \t]+/);
        for (let w of words) {
          if (currentLine.length + w.length + 1 > width) {
            if (currentLine.length > 0) {
              lines.push(currentLine);
              currentLine = "";
            }
            while (w.length > width) {
              lines.push(w.substring(0, width));
              w = w.substring(width);
            }
            currentLine = w;
          } else {
            if (currentLine.length > 0) currentLine += " ";
            currentLine += w;
          }
        }
        if (currentLine.length > 0) lines.push(currentLine);
      }
      return lines;
    }

    function generateBubble(lines) {
      let maxLen = 0;
      for (let l of lines) {
        if (l.length > maxLen) maxLen = l.length;
      }

      let out = "";
      out += " " + "_".repeat(maxLen + 2) + "\n";
      if (lines.length === 1) {
        out += `< ${lines[0].padEnd(maxLen, " ")} >\n`;
      } else {
        for (let idx = 0; idx < lines.length; idx++) {
          let l = lines[idx];
          if (idx === 0) out += `/ ${l.padEnd(maxLen, " ")} \\\n`;
          else if (idx === lines.length - 1)
            out += `\\ ${l.padEnd(maxLen, " ")} /\n`;
          else out += `| ${l.padEnd(maxLen, " ")} |\n`;
        }
      }
      out += " " + "-".repeat(maxLen + 2) + "\n";
      return out;
    }

    function printCowsay(text) {
      let lines = wordWrap(text, wrapColumn);
      if (lines.length === 0) {
        lines = [""];
      }
      let bubble = generateBubble(lines);
      let cow = `        \\   ^__^\n         \\  (${eyes})\\_______\n            (__)\\       )\\/\\\n             ${tongue} ||----w |\n                ||     ||\n`;
      self.Module.print(bubble + cow);
    }

    if (messages.length > 0) {
      printCowsay(messages.join(" "));
      return 0;
    } else {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          printCowsay(inputStr.trim());
          wakeUp(0);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let j = 0; j < bytesRead; j++) {
                    inputStr += String.fromCharCode(buf[j]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    }
  }

  if (cmd === "jq") {
    let rawOutput = false;
    let filter = null;
    let files = [];

    let i = 1;
    while (i < args.length) {
      if (args[i] === "-r" || args[i] === "--raw-output") {
        rawOutput = true;
        i++;
      } else if (!filter) {
        filter = args[i];
        i++;
      } else {
        files.push(args[i]);
        i++;
      }
    }

    if (!filter) {
      self.Module.printErr("jq: usage: jq [-r] filter [file...]");
      return 1;
    }

    function evalJq(obj, pathStr) {
      if (pathStr === ".") return obj;
      let path = pathStr.replace(/^\./, "");
      if (!path) return obj;
      let parts = [];
      let currentPart = "";
      for (let j = 0; j < path.length; j++) {
        if (path[j] === ".") {
          if (currentPart) parts.push(currentPart);
          currentPart = "";
        } else if (path[j] === "[") {
          if (currentPart) parts.push(currentPart);
          currentPart = "[";
        } else if (path[j] === "]") {
          currentPart += "]";
          parts.push(currentPart);
          currentPart = "";
        } else {
          currentPart += path[j];
        }
      }
      if (currentPart) parts.push(currentPart);

      let current = obj;
      for (let part of parts) {
        if (current == null) return null;
        if (part.startsWith("[") && part.endsWith("]")) {
          let idx = parseInt(part.substring(1, part.length - 1), 10);
          current = current[idx];
        } else {
          current = current[part];
        }
      }
      return current;
    }

    function processJSON(str) {
      if (!str.trim()) return;
      try {
        let obj = JSON.parse(str);
        let res = evalJq(obj, filter);
        if (res === undefined || res === null) {
          self.Module.print("null");
        } else if (rawOutput && typeof res === "string") {
          self.Module.print(res);
        } else {
          self.Module.print(JSON.stringify(res, null, 2));
        }
      } catch (e) {
        self.Module.printErr("jq: parse error: " + e.message);
      }
    }

    if (files.length > 0) {
      let ret = 0;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          processJSON(text);
        } catch (e) {
          self.Module.printErr(`jq: cannot open ${f} (${e.message})`);
          ret = 1;
        }
      }
      return ret;
    } else {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          processJSON(inputStr);
          wakeUp(0);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let j = 0; j < bytesRead; j++) {
                    inputStr += String.fromCharCode(buf[j]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    }
  }
  if (cmd === "grep") {
    let opts = { i: false, v: false, n: false, c: false, e: null };
    let files = [];
    let patterns = [];

    let i = 1;
    while (i < args.length) {
      if (args[i] === "--help") {
        self.Module.print("Usage: grep [OPTION]... PATTERNS [FILE]...");
        return 0;
      } else if (args[i] === "-i") {
        opts.i = true;
        i++;
      } else if (args[i] === "-v") {
        opts.v = true;
        i++;
      } else if (args[i] === "-n") {
        opts.n = true;
        i++;
      } else if (args[i] === "-c") {
        opts.c = true;
        i++;
      } else if (args[i] === "-e") {
        if (i + 1 < args.length) {
          patterns.push(args[i + 1]);
          i += 2;
        } else {
          self.Module.printErr("grep: option requires an argument -- 'e'");
          return 1;
        }
      } else if (args[i].startsWith("-e")) {
        patterns.push(args[i].substring(2));
        i++;
      } else if (args[i].startsWith("-") && args[i] !== "-") {
        let invalid = false;
        for (let j = 1; j < args[i].length; j++) {
          let ch = args[i][j];
          if (ch === "i") opts.i = true;
          else if (ch === "v") opts.v = true;
          else if (ch === "n") opts.n = true;
          else if (ch === "c") opts.c = true;
          else {
            self.Module.printErr("grep: invalid option -- '" + ch + "'");
            invalid = true;
            break;
          }
        }
        if (invalid) return 1;
        i++;
      } else if (patterns.length === 0) {
        patterns.push(args[i]);
        i++;
      } else {
        files.push(args[i]);
        i++;
      }
    }

    if (patterns.length === 0) {
      self.Module.printErr("Usage: grep [OPTION]... PATTERNS [FILE]...");
      return 1;
    }

    let regexList = [];
    for (let p of patterns) {
      try {
        regexList.push(new RegExp(p, opts.i ? "i" : ""));
      } catch (e) {
        regexList.push({
          test: function (str) {
            if (opts.i) return str.toLowerCase().includes(p.toLowerCase());
            return str.includes(p);
          },
        });
      }
    }

    let matchCount = 0;
    let lineNum = 0;

    function processLine(line, file, printFilename) {
      lineNum++;
      let matched = false;
      for (let r of regexList) {
        if (r.test(line)) {
          matched = true;
          break;
        }
      }
      if (opts.v) matched = !matched;

      if (matched) {
        matchCount++;
        if (!opts.c) {
          let prefix = "";
          if (printFilename) prefix += file + ":";
          if (opts.n) prefix += lineNum + ":";
          self.Module.print(prefix + line);
        }
      }
    }

    let printFilename = files.length > 1;

    if (files.length > 0) {
      let ret = 0;
      let totalMatches = 0;
      for (let f of files) {
        matchCount = 0;
        lineNum = 0;
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          let lines = text.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) {
            processLine(line, f, printFilename);
          }
          if (opts.c) {
            if (printFilename) self.Module.print(f + ":" + matchCount);
            else self.Module.print(matchCount.toString());
          }
          totalMatches += matchCount;
        } catch (e) {
          self.Module.printErr(`grep: ${f}: No such file or directory`);
          ret = 1;
        }
      }
      return totalMatches > 0 ? 0 : 1;
    } else {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          let lines = inputStr.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          matchCount = 0;
          lineNum = 0;
          for (let line of lines) {
            processLine(line, "(standard input)", false);
          }
          if (opts.c) {
            self.Module.print(matchCount.toString());
          }
          wakeUp(matchCount > 0 ? 0 : 1);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let j = 0; j < bytesRead; j++) {
                    inputStr += String.fromCharCode(buf[j]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            execute();
          } else {
            execute();
          }
        } catch (e) {
          execute();
        }
      });
    }
  }

  if (cmd === "awk") {
    let script = null;
    let files = [];
    let separator = " ";

    let i = 1;
    while (i < args.length) {
      if (args[i] === "-F") {
        separator = args[i + 1];
        i += 2;
      } else if (args[i].startsWith("-F")) {
        separator = args[i].substring(2);
        i++;
      } else if (!script) {
        script = args[i];
        i++;
      } else {
        files.push(args[i]);
        i++;
      }
    }

    if (!script) {
      self.Module.printErr("awk: usage: awk [-F fs] 'pattern {action}' [file]");
      return 1;
    }

    let pattern = null;
    let action = null;

    let match = script.match(/^\s*(?:\/(.+?)\/\s*)?\{(.+)\}\s*$/);
    if (match) {
      if (match[1]) pattern = new RegExp(match[1]);
      action = match[2].trim();
    } else {
      match = script.match(/^\s*\/(.+?)\/\s*$/);
      if (match) {
        pattern = new RegExp(match[1]);
        action = "print $0";
      } else {
        action = script.trim();
        if (action.startsWith("{") && action.endsWith("}")) {
          action = action.substring(1, action.length - 1).trim();
        }
      }
    }

    function processLine(line) {
      if (!line) return;
      if (pattern && !pattern.test(line)) return;

      let fields;
      if (separator === " ") {
        fields = line.trim().split(/\s+/);
      } else {
        fields = line.split(separator);
      }

      if (action.startsWith("print")) {
        let printArgs = action.substring(5).trim();
        if (!printArgs || printArgs === "$0") {
          self.Module.print(line);
        } else {
          let vars = printArgs.split(/,\s*/);
          let output = [];
          for (let v of vars) {
            if (v.startsWith("$")) {
              let idx = parseInt(v.substring(1), 10);
              if (idx === 0) output.push(line);
              else output.push(fields[idx - 1] || "");
            } else {
              if (v.startsWith('"') && v.endsWith('"')) {
                output.push(v.substring(1, v.length - 1));
              } else {
                output.push(v);
              }
            }
          }
          self.Module.print(output.join(" "));
        }
      } else if (action === "") {
        self.Module.print(line);
      }
    }

    if (files.length > 0) {
      let ret = 0;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          let lines = text.split("\n");
          if (lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) {
            processLine(line);
          }
        } catch (e) {
          self.Module.printErr(`awk: cannot open ${f} (${e.message})`);
          ret = 1;
        }
      }
      return ret;
    } else {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          let lines = inputStr.split("\n");
          if (lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) {
            processLine(line);
          }
          wakeUp(0);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let i = 0; i < bytesRead; i++) {
                    inputStr += String.fromCharCode(buf[i]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    }
  }

  if (cmd === "sed") {
    let scripts = [];
    let files = [];

    let i = 1;
    while (i < args.length) {
      if (args[i] === "-e") {
        if (i + 1 < args.length) {
          scripts.push(args[i + 1]);
          i += 2;
        } else {
          self.Module.printErr("sed: option requires an argument -- 'e'");
          return 1;
        }
      } else if (args[i] === "-n") {
        i++;
      } else if (!scripts.length && !args[i].startsWith("-")) {
        scripts.push(args[i]);
        i++;
      } else {
        files.push(args[i]);
        i++;
      }
    }

    if (scripts.length === 0) {
      self.Module.printErr("sed: usage: sed script [file]");
      return 1;
    }

    let commands = [];
    for (let script of scripts) {
      let parts = script.split(";");
      for (let part of parts) {
        part = part.trim();
        if (!part) continue;
        if (part.startsWith("s")) {
          let delim = part[1];
          let secondDelim = part.indexOf(delim, 2);
          if (secondDelim !== -1) {
            let patternStr = part.substring(2, secondDelim);
            let thirdDelim = part.indexOf(delim, secondDelim + 1);
            if (thirdDelim !== -1) {
              let replStr = part.substring(secondDelim + 1, thirdDelim);
              let flags = part.substring(thirdDelim + 1);
              let isGlobal = flags.includes("g");
              commands.push({
                type: "s",
                pattern: new RegExp(patternStr, isGlobal ? "g" : ""),
                replacement: replStr,
              });
            }
          }
        }
      }
    }

    function processLine(line) {
      if (line === undefined || line === null) return;
      let out = line;
      for (let cmd of commands) {
        if (cmd.type === "s") {
          out = out.replace(cmd.pattern, cmd.replacement);
        }
      }
      self.Module.print(out);
    }

    if (files.length > 0) {
      let ret = 0;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          let lines = text.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) {
            processLine(line);
          }
        } catch (e) {
          self.Module.printErr(`sed: cannot open ${f} (${e.message})`);
          ret = 1;
        }
      }
      return ret;
    } else {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          let lines = inputStr.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) {
            processLine(line);
          }
          wakeUp(0);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let i = 0; i < bytesRead; i++) {
                    inputStr += String.fromCharCode(buf[i]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 10);
      });
    }
  }

  if (cmd === "tar") {
    let mode = null; // 'c', 'x', 't'
    let verbose = false;
    let file = null;
    let gzip = false;
    let paths = [];

    // Basic arg parsing
    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "--help") {
        self.Module.print("Usage: tar [OPTION...] [FILE]...");
        self.Module.print(
          "GNU `tar' saves many files together into a single tape or disk archive, and can",
        );
        self.Module.print("restore individual files from the archive.");
        self.Module.print("  -c, --create               create a new archive");
        self.Module.print(
          "  -x, --extract, --get       extract files from an archive",
        );
        self.Module.print(
          "  -t, --list                 list the contents of an archive",
        );
        self.Module.print(
          "  -v, --verbose              verbosely list files processed",
        );
        self.Module.print(
          "  -f, --file=ARCHIVE         use archive file or device ARCHIVE",
        );
        self.Module.print(
          "  -z, --gzip, --gunzip       filter the archive through gzip (not supported)",
        );
        return 0;
      }

      if (arg.startsWith("-") && arg.length > 1) {
        if (arg.startsWith("--file=")) file = arg.substring(7);
        else if (arg === "--create") mode = "c";
        else if (arg === "--extract" || arg === "--get") mode = "x";
        else if (arg === "--list") mode = "t";
        else if (arg === "--verbose") verbose = true;
        else if (arg === "--gzip" || arg === "--gunzip") gzip = true;
        else {
          for (let k = 1; k < arg.length; k++) {
            let c = arg[k];
            if (c === "c") mode = "c";
            else if (c === "x") mode = "x";
            else if (c === "t") mode = "t";
            else if (c === "v") verbose = true;
            else if (c === "z") gzip = true;
            else if (c === "f") {
              if (k === arg.length - 1 && j + 1 < args.length) {
                file = args[++j];
              } else if (k < arg.length - 1) {
                file = arg.substring(k + 1);
                break;
              } else {
                self.Module.printErr("tar: option requires an argument -- 'f'");
                return 2;
              }
            } else {
              self.Module.printErr("tar: invalid option -- '" + c + "'");
              return 2;
            }
          }
        }
      } else {
        // Can be old-style tar arguments if j=1 and doesn't start with -
        if (j === 1 && !arg.startsWith("-")) {
          for (let k = 0; k < arg.length; k++) {
            let c = arg[k];
            if (c === "c") mode = "c";
            else if (c === "x") mode = "x";
            else if (c === "t") mode = "t";
            else if (c === "v") verbose = true;
            else if (c === "z") gzip = true;
            else if (c === "f") {
              if (j + 1 < args.length) {
                file = args[++j];
              } else {
                self.Module.printErr("tar: option requires an argument -- 'f'");
                return 2;
              }
            } else {
              self.Module.printErr("tar: invalid option -- '" + c + "'");
              return 2;
            }
          }
        } else {
          paths.push(arg);
        }
      }
    }

    if (!mode) {
      self.Module.printErr("tar: Must specify one of -c, -r, -t, -u, -x");
      return 2;
    }
    if (gzip) {
      self.Module.printErr(
        "tar: -z/--gzip is currently stubbed/not fully implemented in this Wasm build. Falling back to uncompressed.",
      );
    }
    if (!file) {
      self.Module.printErr(
        "tar: Refusing to read/write archive content from/to terminal (missing -f).",
      );
      return 2;
    }

    // Helper: read all files recursively
    function getFiles(path) {
      let result = [];
      try {
        let stat = self.Module.FS.lstat(path);
        result.push({ path: path, stat: stat });
        if (self.Module.FS.isDir(stat.mode)) {
          let items = self.Module.FS.readdir(path).filter(
            (n) => n !== "." && n !== "..",
          );
          for (let item of items) {
            let childPath = path === "/" ? "/" + item : path + "/" + item;
            result = result.concat(getFiles(childPath));
          }
        }
      } catch (e) {
        self.Module.printErr(
          `tar: ${path}: Cannot stat: No such file or directory`,
        );
      }
      return result;
    }

    // Helper: encode string to Uint8Array
    const encoder = new TextEncoder();
    const decoder = new TextDecoder();
    function strToBytes(str, len) {
      let b = new Uint8Array(len);
      let e = encoder.encode(str);
      b.set(e.subarray(0, len));
      return b;
    }
    function padOctal(val, len) {
      let s = val.toString(8);
      while (s.length < len - 1) s = "0" + s;
      return strToBytes(s + "\0", len);
    }

    if (mode === "c") {
      if (paths.length === 0) {
        self.Module.printErr(
          "tar: Cowardly refusing to create an empty archive",
        );
        return 2;
      }

      let allEntries = [];
      for (let p of paths) {
        allEntries = allEntries.concat(getFiles(p));
      }

      let buffers = [];

      for (let entry of allEntries) {
        if (verbose) self.Module.print(entry.path);

        let isDir = self.Module.FS.isDir(entry.stat.mode);
        let isSymlink = self.Module.FS.isLink(entry.stat.mode);
        let fileSize = isDir || isSymlink ? 0 : entry.stat.size;
        let name = entry.path.replace(/^\//, ""); // Strip leading slash
        if (isDir && !name.endsWith("/")) name += "/";
        if (name === "") name = "./"; // Root dir representation

        let header = new Uint8Array(512);
        // 0: name (100)
        header.set(strToBytes(name, 100), 0);
        // 100: mode (8)
        header.set(padOctal(entry.stat.mode & 0o7777, 8), 100);
        // 108: uid (8)
        header.set(padOctal(entry.stat.uid, 8), 108);
        // 116: gid (8)
        header.set(padOctal(entry.stat.gid, 8), 116);
        // 124: size (12)
        header.set(padOctal(fileSize, 12), 124);
        // 136: mtime (12)
        header.set(
          padOctal(Math.floor(entry.stat.mtime.getTime() / 1000), 12),
          136,
        );
        // 156: typeflag (1)
        let typeflag = isDir ? "5" : isSymlink ? "2" : "0";
        header.set(strToBytes(typeflag, 1), 156);
        // 157: linkname (100)
        if (isSymlink) {
          let linkname = self.Module.FS.readlink(entry.path);
          header.set(strToBytes(linkname, 100), 157);
        }
        // 257: magic (6) - ustar
        header.set(strToBytes("ustar\0", 6), 257);
        // 263: version (2)
        header.set(strToBytes("00", 2), 263);

        // Compute checksum (148: 8 bytes)
        // First fill checksum field with spaces
        header.set(strToBytes("        ", 8), 148);
        let chksum = 0;
        for (let i = 0; i < 512; i++) chksum += header[i];
        header.set(padOctal(chksum, 8), 148);
        // space/null terminate checksum
        header[154] = 0;
        header[155] = 0x20;

        buffers.push(header);

        if (fileSize > 0) {
          let fd = self.Module.FS.open(entry.path, "r");
          let content = new Uint8Array(fileSize);
          self.Module.FS.read(fd, content, 0, fileSize, 0);
          self.Module.FS.close(fd);

          let numBlocks = Math.ceil(fileSize / 512);
          let contentPadded = new Uint8Array(numBlocks * 512);
          contentPadded.set(content);
          buffers.push(contentPadded);
        }
      }
      // Two 512-byte blocks of zeros to signify end of archive
      buffers.push(new Uint8Array(1024));

      let totalLength = buffers.reduce((acc, val) => acc + val.length, 0);
      let outBuf = new Uint8Array(totalLength);
      let offset = 0;
      for (let b of buffers) {
        outBuf.set(b, offset);
        offset += b.length;
      }

      try {
        self.Module.FS.writeFile(file, outBuf);
      } catch (err) {
        self.Module.printErr(`tar: Cannot write to ${file}: ${err.message}`);
        return 2;
      }
      return 0;
    }

    if (mode === "x" || mode === "t") {
      let inBuf;
      try {
        inBuf = self.Module.FS.readFile(file);
      } catch (err) {
        self.Module.printErr(
          `tar: ${file}: Cannot open: No such file or directory`,
        );
        return 2;
      }

      let offset = 0;
      while (offset + 512 <= inBuf.length) {
        let header = inBuf.subarray(offset, offset + 512);
        // Check for end of archive (all zeros in name field)
        if (header[0] === 0) break;

        let nameEnd = header.indexOf(0);
        if (nameEnd === -1 || nameEnd > 100) nameEnd = 100;
        let name = decoder.decode(header.subarray(0, nameEnd));
        if (name.length === 0) break;

        // Parse size (octal)
        let sizeStr = decoder.decode(header.subarray(124, 135)).trim();
        let size = parseInt(sizeStr, 8);
        if (isNaN(size)) size = 0;

        // Parse typeflag
        let typeflag = String.fromCharCode(header[156]);

        // Parse mode
        let modeStr = decoder.decode(header.subarray(100, 107)).trim();
        let fileMode = parseInt(modeStr, 8);

        if (mode === "t") {
          if (verbose) {
            // Approximate verbose list (drwxr-xr-x ...)
            self.Module.print(
              `${typeflag === "5" ? "d" : "-"}${fileMode.toString(8)} ${size} ${name}`,
            );
          } else {
            self.Module.print(name);
          }
        } else if (mode === "x") {
          if (verbose) self.Module.print(name);

          let extractPath = name.startsWith("/") ? name : FS.cwd() + "/" + name;

          // Ensure parent directories exist
          let parts = extractPath.split("/");
          let cur = "";
          for (let i = 0; i < parts.length - 1; i++) {
            if (parts[i] === "") {
              cur += "/";
              continue;
            }
            cur += (cur === "/" ? "" : "/") + parts[i];
            try {
              self.Module.FS.mkdir(cur);
            } catch (e) {} // ignore exists
          }

          if (typeflag === "5" || name.endsWith("/")) {
            try {
              self.Module.FS.mkdir(extractPath, fileMode);
            } catch (e) {}
          } else if (typeflag === "2") {
            // Symlink
            let linknameEnd = header.subarray(157, 257).indexOf(0);
            if (linknameEnd === -1) linknameEnd = 100;
            let linkname = decoder.decode(
              header.subarray(157, 157 + linknameEnd),
            );
            try {
              self.Module.FS.symlink(linkname, extractPath);
            } catch (e) {}
          } else {
            // Regular file
            let fileData = inBuf.subarray(offset + 512, offset + 512 + size);
            try {
              self.Module.FS.writeFile(extractPath, fileData);
              self.Module.FS.chmod(extractPath, fileMode);
            } catch (err) {
              self.Module.printErr(
                `tar: Cannot extract ${name}: ${err.message}`,
              );
            }
          }
        }

        let numBlocks = Math.ceil(size / 512);
        offset += 512 + numBlocks * 512;
      }
      return 0;
    }

    return 0;
  }

  if (cmd === "env") {
    let ignoreEnv = false;
    let setVars = {};
    let i = 1;
    while (i < args.length) {
      if (args[i] === "-i" || args[i] === "--ignore-environment") {
        ignoreEnv = true;
        i++;
      } else if (args[i] === "--help") {
        self.Module.print(
          "Usage: env [OPTION]... [-] [NAME=VALUE]... [COMMAND [ARG]...]",
        );
        return 0;
      } else if (args[i].includes("=")) {
        let idx = args[i].indexOf("=");
        let key = args[i].substring(0, idx);
        let val = args[i].substring(idx + 1);
        setVars[key] = val;
        i++;
      } else {
        break;
      }
    }

    if (i === args.length) {
      let currentEnv = ignoreEnv ? {} : Object.assign({}, ENV);
      for (let k in setVars) {
        currentEnv[k] = setVars[k];
      }
      for (let k in currentEnv) {
        if (currentEnv[k] !== undefined) {
          self.Module.print(`${k}=${currentEnv[k]}`);
        }
      }
      return 0;
    } else {
      // Execute command
      let savedEnv = Object.assign({}, ENV);
      if (ignoreEnv) {
        for (let k in ENV) delete ENV[k];
      }
      for (let k in setVars) {
        ENV[k] = setVars[k];
      }
      // Build quoted command line
      let cmdStr = args
        .slice(i)
        .map((a) => `'${a.replace(/'/g, "'\\''")}'`)
        .join(" ");

      let ret = self.Module.ccall(
        "dash_eval_script",
        "number",
        ["string"],
        [cmdStr],
      );

      for (let k in ENV) delete ENV[k];
      for (let k in savedEnv) ENV[k] = savedEnv[k];
      return ret;
    }
  }

  if (cmd === "su") {
    let j = 1;
    let showHelp = false;
    let evalStr = null;
    let user = null;

    while (j < args.length) {
      let arg = args[j];
      if (arg === "-") {
        j++;
      } else if (arg === "-h" || arg === "--help") {
        showHelp = true;
        break;
      } else if (arg === "-c") {
        if (j + 1 < args.length) {
          evalStr = args[++j];
        } else {
          self.Module.printErr(`su: option requires an argument -- 'c'`);
          return 2;
        }
        j++;
      } else if (!arg.startsWith("-")) {
        if (!user) {
          user = arg;
        }
        j++;
      } else {
        j++;
      }
    }

    if (showHelp) {
      self.Module.print("usage: su [-] [username] [-c command]");
      return 0;
    }

    if (evalStr !== null) {
      return self.Module.ccall(
        "dash_eval_script",
        "number",
        ["string"],
        [evalStr],
      );
    } else {
      self.Module.printErr(
        `su: interactive subshells are not supported in this Wasm build`,
      );
      return 1;
    }
  }

  if (cmd === "sh" || cmd === "bash") {
    let script = null;
    let evalStr = null;
    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "-c") {
        if (j + 1 < args.length) {
          evalStr = args[++j];
        } else {
          self.Module.printErr(`${cmd}: option requires an argument -- 'c'`);
          return 2;
        }
      } else if (!arg.startsWith("-")) {
        script = arg;
        break;
      }
    }

    if (evalStr !== null) {
      return self.Module.ccall(
        "dash_eval_script",
        "number",
        ["string"],
        [evalStr],
      );
    } else if (script !== null) {
      try {
        let content = self.Module.FS.readFile(script, { encoding: "utf8" });
        return self.Module.ccall(
          "dash_eval_script",
          "number",
          ["string"],
          [content],
        );
      } catch (e) {
        self.Module.printErr(`${cmd}: ${script}: No such file or directory`);
        return 127;
      }
    } else {
      self.Module.printErr(
        `${cmd}: interactive subshells are not supported in this Wasm build`,
      );
      return 1;
    }
  }

  if (cmd === "sudo") {
    let j = 1;
    let showHelp = false;

    while (j < args.length && args[j].startsWith("-")) {
      let arg = args[j];
      if (arg === "-h" || arg === "--help") {
        showHelp = true;
        break;
      } else if (arg === "-u") {
        j += 2;
      } else {
        j++;
      }
    }

    if (showHelp) {
      self.Module.print("usage: sudo [-h] [-u user] command");
      return 0;
    }

    if (j < args.length) {
      let newArgs = args.slice(j);
      return globalThis.processExternalCommand(newArgs);
    } else {
      self.Module.print("usage: sudo [-h] [-u user] command");
      return 1;
    }
  }
  if (cmd === "find") {
    let argsIdx = 1;
    if (argsIdx < args.length && !args[argsIdx].startsWith("-")) {
      startPath = args[argsIdx++];
    }

    while (argsIdx < args.length) {
      let arg = args[argsIdx];
      if (arg === "-name" && argsIdx + 1 < args.length) {
        nameOpt = args[++argsIdx];
      } else if (arg === "-type" && argsIdx + 1 < args.length) {
        typeOpt = args[++argsIdx];
      }
      argsIdx++;
    }

    // Simplified glob-to-regex for -name
    let nameRegex = null;
    if (nameOpt) {
      let escaped = nameOpt
        .replace(/[.+^${}()|[\]\\]/g, "\\$&")
        .replace(/\*/g, ".*")
        .replace(/\?/g, ".");
      nameRegex = new RegExp("^" + escaped + "$");
    }

    function walk(dir) {
      let items;
      try {
        items = self.Module.FS.readdir(dir);
      } catch (e) {
        self.Module.printErr(`find: \`${dir}': No such file or directory`);
        return;
      }

      for (let item of items) {
        if (item === "." || item === "..") continue;
        let fullPath = dir === "/" ? "/" + item : dir + "/" + item;
        let stat;
        try {
          stat = self.Module.FS.lstat(fullPath);
        } catch (e) {
          continue;
        }

        let isDir = self.Module.FS.isDir(stat.mode);
        let isFile = self.Module.FS.isFile(stat.mode);
        let matchType =
          !typeOpt || (typeOpt === "d" && isDir) || (typeOpt === "f" && isFile);
        let matchName = !nameRegex || nameRegex.test(item);

        if (matchType && matchName) {
          self.Module.print(fullPath);
        }

        if (isDir) {
          walk(fullPath);
        }
      }
    }

    // Check if start path itself matches
    try {
      let stat = self.Module.FS.lstat(startPath);
      let isDir = self.Module.FS.isDir(stat.mode);
      let isFile = self.Module.FS.isFile(stat.mode);
      let matchType =
        !typeOpt || (typeOpt === "d" && isDir) || (typeOpt === "f" && isFile);
      let startName = startPath.split("/").pop() || startPath;
      let matchName = !nameRegex || nameRegex.test(startName);
      if (matchType && matchName) {
        self.Module.print(startPath);
      }
      if (isDir) {
        walk(startPath);
      }
    } catch (e) {
      self.Module.printErr(`find: \`${startPath}': No such file or directory`);
      return 1;
    }

    return 0;
  }
  if (cmd === "sleep") {
    if (args.length < 2) {
      self.Module.printErr("sleep: missing operand");
      return 1;
    }
    let duration = parseFloat(args[1]);
    if (isNaN(duration) || duration < 0) {
      self.Module.printErr("sleep: invalid time interval '" + args[1] + "'");
      return 1;
    }
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      setTimeout(function () {
        wakeUp(0);
      }, duration * 1000);
    });
  }

  if (cmd === "find") {
    let startPath = ".";
    let nameOpt = null;
    let typeOpt = null;

    let argsIdx = 1;
    if (argsIdx < args.length && !args[argsIdx].startsWith("-")) {
      startPath = args[argsIdx++];
    }

    while (argsIdx < args.length) {
      let arg = args[argsIdx];
      if (arg === "-name" && argsIdx + 1 < args.length) {
        nameOpt = args[++argsIdx];
      } else if (arg === "-type" && argsIdx + 1 < args.length) {
        typeOpt = args[++argsIdx];
      }
      argsIdx++;
    }

    let nameRegex = null;
    if (nameOpt) {
      let escaped = nameOpt
        .replace(/[.+^\\${}()|[\]\\\\]/g, "\\\\$&")
        .replace(/\\*/g, ".*")
        .replace(/\\?/g, ".");
      nameRegex = new RegExp("^" + escaped + "$");
    }

    function walk(dir) {
      let items;
      try {
        items = self.Module.FS.readdir(dir);
      } catch (e) {
        return;
      }
      for (let item of items) {
        if (item === "." || item === "..") continue;
        let fullPath = dir === "/" ? "/" + item : dir + "/" + item;
        let stat;
        try {
          stat = self.Module.FS.lstat(fullPath);
        } catch (e) {
          continue;
        }
        let isDir = self.Module.FS.isDir(stat.mode);
        let isFile = self.Module.FS.isFile(stat.mode);
        if (
          (!typeOpt ||
            (typeOpt === "d" && isDir) ||
            (typeOpt === "f" && isFile)) &&
          (!nameRegex || nameRegex.test(item))
        ) {
          self.Module.print(fullPath);
        }
        if (isDir) walk(fullPath);
      }
    }
    try {
      let stat = self.Module.FS.lstat(startPath);
      if (
        (!typeOpt ||
          (typeOpt === "d" && self.Module.FS.isDir(stat.mode)) ||
          (typeOpt === "f" && self.Module.FS.isFile(stat.mode))) &&
        (!nameRegex || nameRegex.test(startPath.split("/").pop() || startPath))
      ) {
        self.Module.print(startPath);
      }
      if (self.Module.FS.isDir(stat.mode)) walk(startPath);
    } catch (e) {
      self.Module.printErr(`find: ${startPath}: No such file or directory`);
      return 1;
    }
    return 0;
  }

  if (cmd === "xargs") {
    let argsCmd = ["echo"];

    if (args.length > 1) {
      argsCmd = args.slice(1);
    }

    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      let inputStr = "";

      function execute() {
        let tokens = inputStr
          .trim()
          .split(/\s+/)
          .filter((x) => x.length > 0);

        if (tokens.length > 0) {
          argsCmd.push(...tokens);
        }

        if (argsCmd[0] === "echo") {
          self.Module.print(argsCmd.slice(1).join(" "));
          wakeUp(0);
        } else {
          let res = globalThis.processExternalCommand(argsCmd);
          if (res !== undefined && res !== null) {
            wakeUp(res);
          } else {
            wakeUp(
              self.Module.ccall(
                "dash_eval_script",
                "number",
                ["string"],
                [argsCmd.join(" ")],
              ),
            );
          }
        }
      }

      // We can't rely just on stdinBuffer when running via dash_eval_script piping.
      // Try reading from file descriptor 0 directly
      let fd = 0;
      try {
        let buf = new Uint8Array(4096);
        let bytesRead = self.Module.FS.read(
          self.Module.FS.getStream(0),
          buf,
          0,
          4096,
          undefined,
        );
        if (bytesRead > 0) {
          for (let i = 0; i < bytesRead; i++) {
            inputStr += String.fromCharCode(buf[i]);
          }
        }
        execute();
      } catch (e) {
        // fd 0 might block or error, fallback to pulling from stdinBuffer for web interactive
        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }

            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);

          setTimeout(() => {
            clearInterval(readLoop);
            execute();
          }, 300);
        }, 100);
      }
    });
  }
  if (cmd === "sleep") {
    if (args.length < 2) {
      self.Module.printErr("sleep: missing operand");
      return 1;
    }
    let duration = parseFloat(args[1]);
    if (isNaN(duration) || duration < 0) {
      self.Module.printErr("sleep: invalid time interval '" + args[1] + "'");
      return 1;
    }
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      setTimeout(function () {
        wakeUp(0);
      }, duration * 1000);
    });
  }

  if (cmd === "date") {
    let opts = { u: false, d: null, r: null, format: null };
    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "--help") {
        self.Module.print("Usage: date [OPTION]... [+FORMAT]");
        self.Module.print("Display the current time in the given FORMAT.");
        self.Module.print(
          "  -d, --date=STRING      display time described by STRING, not `now'",
        );
        self.Module.print(
          "  -r, --reference=FILE   display the last modification time of FILE",
        );
        self.Module.print(
          "  -u, --utc, --universal print or set Coordinated Universal Time (UTC)",
        );
        return 0;
      }
      if (arg.startsWith("+")) {
        opts.format = arg.substring(1);
      } else if (arg === "-u" || arg === "--utc" || arg === "--universal") {
        opts.u = true;
      } else if (arg === "-d") {
        if (j + 1 < args.length) opts.d = args[++j];
        else {
          self.Module.printErr("date: option requires an argument -- 'd'");
          return 1;
        }
      } else if (arg.startsWith("--date=")) {
        opts.d = arg.substring(7);
      } else if (arg === "-r") {
        if (j + 1 < args.length) opts.r = args[++j];
        else {
          self.Module.printErr("date: option requires an argument -- 'r'");
          return 1;
        }
      } else if (arg.startsWith("--reference=")) {
        opts.r = arg.substring(12);
      } else {
        self.Module.printErr("date: invalid date '" + arg + "'");
        return 1;
      }
    }

    let d;
    if (opts.r) {
      try {
        let stat = self.Module.FS.stat(opts.r);
        d = new Date(stat.mtime);
      } catch (e) {
        self.Module.printErr("date: " + opts.r + ": No such file or directory");
        return 1;
      }
    } else if (opts.d) {
      if (opts.d.toLowerCase() === "yesterday") {
        d = new Date();
        d.setDate(d.getDate() - 1);
      } else if (opts.d.toLowerCase() === "tomorrow") {
        d = new Date();
        d.setDate(d.getDate() + 1);
      } else {
        d = new Date(opts.d);
        if (isNaN(d.getTime())) {
          // Try treating as timestamp if numeric
          let num = Number(opts.d);
          if (!isNaN(num)) d = new Date(num * 1000);
        }
      }
      if (isNaN(d.getTime())) {
        self.Module.printErr("date: invalid date '" + opts.d + "'");
        return 1;
      }
    } else {
      d = new Date();
    }

    if (opts.format !== null) {
      let out = "";
      let format = opts.format;
      let pad = (n, width = 2) => n.toString().padStart(width, "0");

      let year = opts.u ? d.getUTCFullYear() : d.getFullYear();
      let month = opts.u ? d.getUTCMonth() : d.getMonth();
      let date = opts.u ? d.getUTCDate() : d.getDate();
      let hours = opts.u ? d.getUTCHours() : d.getHours();
      let mins = opts.u ? d.getUTCMinutes() : d.getMinutes();
      let secs = opts.u ? d.getUTCSeconds() : d.getSeconds();
      let day = opts.u ? d.getUTCDay() : d.getDay();
      let ms = opts.u ? d.getUTCMilliseconds() : d.getMilliseconds();

      for (let i = 0; i < format.length; i++) {
        if (format[i] === "%" && i + 1 < format.length) {
          let c = format[++i];
          switch (c) {
            case "Y":
              out += year;
              break;
            case "y":
              out += pad(year % 100);
              break;
            case "m":
              out += pad(month + 1);
              break;
            case "d":
              out += pad(date);
              break;
            case "H":
              out += pad(hours);
              break;
            case "M":
              out += pad(mins);
              break;
            case "S":
              out += pad(secs);
              break;
            case "T":
              out += `${pad(hours)}:${pad(mins)}:${pad(secs)}`;
              break;
            case "F":
              out += `${year}-${pad(month + 1)}-${pad(date)}`;
              break;
            case "s":
              out += Math.floor(d.getTime() / 1000);
              break;
            case "N":
              out += pad(ms, 3) + "000000";
              break; // nanoseconds mock
            case "w":
              out += day;
              break; // 0 (Sunday) to 6
            case "u":
              out += day === 0 ? 7 : day;
              break; // 1 (Monday) to 7
            case "%":
              out += "%";
              break;
            case "n":
              out += "\n";
              break;
            case "t":
              out += "\t";
              break;
            default:
              out += "%" + c;
              break;
          }
        } else if (format[i] === "\\" && i + 1 < format.length) {
          let n = format[++i];
          if (n === "n") out += "\n";
          else if (n === "t") out += "\t";
          else out += "\\" + n;
        } else {
          out += format[i];
        }
      }
      self.Module.print(out);
    } else {
      if (opts.u) {
        self.Module.print(d.toUTCString().replace("GMT", "UTC"));
      } else {
        // Strip out GMT-xxxx bit to look cleaner
        self.Module.print(d.toString().split(" GMT")[0]);
      }
    }
    return 0;
  }

  if (cmd === "df") {
    let opts = { h: false, k: false };
    for (let j = 1; j < args.length; j++) {
      if (args[j] === "--help") {
        self.Module.print("Usage: df [OPTION]... [FILE]...");
        self.Module.print(
          "Show information about the file system on which each FILE resides,",
        );
        self.Module.print("or all file systems by default.");
        self.Module.print(
          "  -h, --human-readable  print sizes in powers of 1024 (e.g., 1023M)",
        );
        self.Module.print("  -k                    like --block-size=1K");
        return 0;
      }
      if (args[j] === "-h" || args[j] === "--human-readable") opts.h = true;
      else if (args[j] === "-k") opts.k = true;
      else if (args[j].startsWith("-") && args[j] !== "-") {
        self.Module.printErr("df: invalid option -- '" + args[j] + "'");
        return 1;
      }
    }

    function formatSize(bytes) {
      if (!opts.h) return Math.ceil(bytes / 1024).toString();
      const units = ["B", "K", "M", "G", "T", "P"];
      let u = 0;
      let val = bytes;
      while (val >= 1024 && u < units.length - 1) {
        val /= 1024;
        u++;
      }
      return (u === 0 ? val : val.toFixed(1)) + units[u];
    }

    function padRight(str, len) {
      str = str.toString();
      while (str.length < len) str += " ";
      return str;
    }

    function padLeft(str, len) {
      str = str.toString();
      while (str.length < len) str = " " + str;
      return str;
    }

    if (opts.h) {
      self.Module.print("Filesystem      Size  Used Avail Use% Mounted on");
    } else {
      self.Module.print(
        "Filesystem     1K-blocks  Used Available Use% Mounted on",
      );
    }

    // Emscripten doesn't expose easy FS capacity. We'll mock typical MEMFS/IDBFS usage.
    // MEMFS is RAM bounded, let's say 2GB for display purposes.
    const TOTAL_MEMFS = 2 * 1024 * 1024 * 1024;

    // We can try to traverse FS to get actual used bytes for MEMFS but it's slow.
    // For a mock, we will just say a nominal amount is used.
    let usedMemfs = 10 * 1024 * 1024; // 10MB
    let availMemfs = TOTAL_MEMFS - usedMemfs;
    let usePct = Math.ceil((usedMemfs / TOTAL_MEMFS) * 100) + "%";

    let fsName = padRight("MEMFS", 15);
    let sizeStr = padLeft(formatSize(TOTAL_MEMFS), opts.h ? 4 : 9);
    let usedStr = padLeft(formatSize(usedMemfs), opts.h ? 5 : 5);
    let availStr = padLeft(formatSize(availMemfs), opts.h ? 5 : 9);
    let pctStr = padLeft(usePct, 4);

    self.Module.print(`${fsName}${sizeStr} ${usedStr} ${availStr} ${pctStr} /`);

    return 0;
  }

  if (cmd === "du") {
    let opts = { h: false, s: false };
    let files = [];

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-") {
        for (let k = 1; k < arg.length; k++) {
          if (arg[k] === "h") opts.h = true;
          else if (arg[k] === "s") opts.s = true;
          else {
            self.Module.printErr(`du: invalid option -- '${arg[k]}'`);
            return 1;
          }
        }
      } else {
        files.push(arg);
      }
      j++;
    }

    if (files.length === 0) files.push(".");

    let ret = 0;

    function formatSize(bytes) {
      if (!opts.h) return Math.ceil(bytes / 1024).toString();
      const units = ["B", "K", "M", "G", "T", "P"];
      let u = 0;
      let val = bytes;
      while (val >= 1024 && u < units.length - 1) {
        val /= 1024;
        u++;
      }
      return (u === 0 ? val : val.toFixed(1)) + units[u];
    }

    function computeSize(dirPath) {
      let totalSize = 0;
      try {
        let stat = self.Module.FS.lstat(dirPath);
        totalSize += stat.size || 0;
        if (self.Module.FS.isDir(stat.mode)) {
          let items = self.Module.FS.readdir(dirPath);
          for (let item of items) {
            if (item === "." || item === "..") continue;
            let fullPath = dirPath === "/" ? "/" + item : dirPath + "/" + item;
            totalSize += computeSize(fullPath);
          }
        }
        if (!opts.s) {
          self.Module.print(`${formatSize(totalSize)}\t${dirPath}`);
        }
      } catch (e) {
        self.Module.printErr(
          `du: cannot access '${dirPath}': No such file or directory`,
        );
        ret = 1;
      }
      return totalSize;
    }

    for (let f of files) {
      let sz = computeSize(f);
      if (opts.s && ret === 0) {
        self.Module.print(`${formatSize(sz)}\t${f}`);
      }
    }
    return ret;
  }

  if (cmd === "stat") {
    let opts = { L: false, c: null };
    let files = [];
    for (let j = 1; j < args.length; j++) {
      if (args[j] === "--help") {
        self.Module.print("Usage: stat [OPTION]... FILE...");
        self.Module.print("Display file or file system status.");
        self.Module.print("  -L, --dereference     follow links");
        self.Module.print(
          "  -c, --format=FORMAT   use the specified FORMAT instead of the default;",
        );
        self.Module.print(
          "                        output a newline after each use of FORMAT",
        );
        return 0;
      }
      if (args[j] === "-L" || args[j] === "--dereference") {
        opts.L = true;
      } else if (args[j] === "-c") {
        if (j + 1 < args.length) {
          opts.c = args[++j];
        } else {
          self.Module.printErr("stat: option requires an argument -- 'c'");
          return 1;
        }
      } else if (args[j].startsWith("--format=")) {
        opts.c = args[j].substring(9);
      } else if (args[j].startsWith("-") && args[j] !== "-") {
        self.Module.printErr("stat: invalid option -- '" + args[j] + "'");
        return 1;
      } else {
        files.push(args[j]);
      }
    }

    if (files.length === 0) {
      self.Module.printErr("stat: missing operand");
      self.Module.printErr("Try 'stat --help' for more information.");
      return 1;
    }

    function pad(str, len, char = " ") {
      str = str.toString();
      while (str.length < len) str = char + str;
      return str;
    }

    function modeToPerms(mode) {
      const types = {
        0o010000: "p",
        0o020000: "c",
        0o040000: "d",
        0o060000: "b",
        0o100000: "-",
        0o120000: "l",
        0o140000: "s",
      };
      let type = types[mode & 0o170000] || "?";
      const rwx = ["---", "--x", "-w-", "-wx", "r--", "r-x", "rw-", "rwx"];
      let u = rwx[(mode >> 6) & 7];
      let g = rwx[(mode >> 3) & 7];
      let o = rwx[mode & 7];
      if (mode & 0o4000) u = u.substring(0, 2) + (u[2] === "x" ? "s" : "S");
      if (mode & 0o2000) g = g.substring(0, 2) + (g[2] === "x" ? "s" : "S");
      if (mode & 0o1000) o = o.substring(0, 2) + (o[2] === "x" ? "t" : "T");
      return type + u + g + o;
    }

    function getTypeName(mode) {
      if (self.Module.FS.isDir(mode)) return "directory";
      if (self.Module.FS.isLink(mode)) return "symbolic link";
      if (self.Module.FS.isChrdev(mode)) return "character special file";
      if (self.Module.FS.isBlkdev(mode)) return "block special file";
      if (self.Module.FS.isFIFO(mode)) return "fifo";
      if (self.Module.FS.isSocket(mode)) return "socket";
      return "regular file";
    }

    for (let f of files) {
      try {
        let stat = opts.L ? self.Module.FS.stat(f) : self.Module.FS.lstat(f);

        if (opts.c !== null) {
          let out = "";
          for (let i = 0; i < opts.c.length; i++) {
            if (opts.c[i] === "%" && i + 1 < opts.c.length) {
              let c = opts.c[++i];
              switch (c) {
                case "a":
                  out += (stat.mode & 0o7777).toString(8);
                  break;
                case "A":
                  out += modeToPerms(stat.mode);
                  break;
                case "b":
                  out += stat.blocks;
                  break;
                case "B":
                  out += "512";
                  break; // Default block size
                case "d":
                  out += stat.dev;
                  break;
                case "F":
                  out += getTypeName(stat.mode);
                  break;
                case "g":
                  out += stat.gid;
                  break;
                case "i":
                  out += stat.ino;
                  break;
                case "n":
                  out += f;
                  break;
                case "N":
                  out +=
                    `'${f}'` +
                    (self.Module.FS.isLink(stat.mode)
                      ? ` -> '${self.Module.FS.readlink(f)}'`
                      : "");
                  break;
                case "s":
                  out += stat.size;
                  break;
                case "u":
                  out += stat.uid;
                  break;
                case "x":
                  out += new Date(stat.atime).toISOString();
                  break;
                case "y":
                  out += new Date(stat.mtime).toISOString();
                  break;
                case "z":
                  out += new Date(stat.ctime).toISOString();
                  break;
                case "%":
                  out += "%";
                  break;
                default:
                  out += "%" + c;
                  break;
              }
            } else if (opts.c[i] === "\\" && i + 1 < opts.c.length) {
              let n = opts.c[++i];
              if (n === "n") out += "\n";
              else if (n === "t") out += "\t";
              else out += "\\" + n;
            } else {
              out += opts.c[i];
            }
          }
          self.Module.print(out);
        } else {
          let typeName = getTypeName(stat.mode);
          let nameStr = `  File: ${f}`;
          if (self.Module.FS.isLink(stat.mode)) {
            nameStr += ` -> ${self.Module.FS.readlink(f)}`;
          }

          self.Module.print(nameStr);
          self.Module.print(
            `  Size: ${pad(stat.size, 10, " ")}\tBlocks: ${pad(stat.blocks || Math.ceil(stat.size / 512), 10, " ")} IO Block: ${pad(stat.blksize || 4096, 6, " ")}   ${typeName}`,
          );
          self.Module.print(
            `Device: ${stat.dev}d\tInode: ${stat.ino}\tLinks: ${stat.nlink}`,
          );
          self.Module.print(
            `Access: (${pad((stat.mode & 0o7777).toString(8), 4, "0")}/${modeToPerms(stat.mode)})\tUid: (${stat.uid})\tGid: (${stat.gid})`,
          );
          self.Module.print(`Access: ${new Date(stat.atime).toISOString()}`);
          self.Module.print(`Modify: ${new Date(stat.mtime).toISOString()}`);
          self.Module.print(`Change: ${new Date(stat.ctime).toISOString()}`);
        }
      } catch (err) {
        self.Module.printErr(
          `stat: cannot stat '${f}': No such file or directory`,
        );
      }
    }
    return 0;
  }

  if (cmd === "file") {
    let opts = { b: false, L: false, h: true };
    let files = [];
    for (let j = 1; j < args.length; j++) {
      if (args[j] === "--help") {
        self.Module.print("Usage: file [OPTION...] [FILE...]");
        self.Module.print("Determine type of FILEs.");
        self.Module.print(
          "  -b, --brief              do not prepend filenames to output lines",
        );
        self.Module.print(
          "  -h, --no-dereference     don't follow symlinks (default)",
        );
        self.Module.print("  -L, --dereference        follow symlinks");
        return 0;
      }
      if (args[j].startsWith("-") && args[j] !== "-") {
        for (let k = 1; k < args[j].length; k++) {
          let c = args[j][k];
          if (c === "b") opts.b = true;
          else if (c === "L") {
            opts.L = true;
            opts.h = false;
          } else if (c === "h") {
            opts.h = true;
            opts.L = false;
          } else {
            self.Module.printErr("file: invalid option -- '" + c + "'");
            return 1;
          }
        }
      } else {
        files.push(args[j]);
      }
    }

    if (files.length === 0) {
      self.Module.printErr("Usage: file [OPTION...] [FILE...]");
      return 1;
    }

    for (let f of files) {
      try {
        let stat = opts.L ? self.Module.FS.stat(f) : self.Module.FS.lstat(f);
        let typeStr = "data";

        if (self.Module.FS.isDir(stat.mode)) typeStr = "directory";
        else if (self.Module.FS.isLink(stat.mode)) {
          let linkTarget = self.Module.FS.readlink(f);
          typeStr = "symbolic link to " + linkTarget;
        } else if (self.Module.FS.isChrdev(stat.mode))
          typeStr = "character special";
        else if (self.Module.FS.isBlkdev(stat.mode)) typeStr = "block special";
        else if (self.Module.FS.isFIFO(stat.mode))
          typeStr = "fifo (named pipe)";
        else if (self.Module.FS.isSocket(stat.mode)) typeStr = "socket";
        else if (stat.size === 0) typeStr = "empty";
        else {
          // Read first 256 bytes for magic
          let fd = self.Module.FS.open(f, "r");
          let buf = new Uint8Array(256);
          let bytesRead = self.Module.FS.read(fd, buf, 0, 256, 0);
          self.Module.FS.close(fd);

          let isAscii = true;
          let hasNewline = false;
          for (let i = 0; i < bytesRead; i++) {
            if (
              buf[i] > 127 ||
              (buf[i] < 32 && buf[i] !== 9 && buf[i] !== 10 && buf[i] !== 13)
            ) {
              isAscii = false;
              break;
            }
            if (buf[i] === 10) hasNewline = true;
          }

          if (
            bytesRead >= 4 &&
            buf[0] === 0x00 &&
            buf[1] === 0x61 &&
            buf[2] === 0x73 &&
            buf[3] === 0x6d
          ) {
            typeStr = "WebAssembly (wasm) binary module";
          } else if (bytesRead >= 2 && buf[0] === 0x23 && buf[1] === 0x21) {
            typeStr = "POSIX shell script, ASCII text executable";
          } else if (isAscii) {
            typeStr = "ASCII text";
          }
        }

        if (opts.b) {
          self.Module.print(typeStr);
        } else {
          self.Module.print(f + ": " + typeStr);
        }
      } catch (err) {
        self.Module.printErr(
          f + ": cannot open `" + f + "' (No such file or directory)",
        );
      }
    }
    return 0;
  }

  if (cmd === "find") {
    let startPath = ".";
    let nameOpt = null;
    let typeOpt = null;

    let argsIdx = 1;
    if (argsIdx < args.length && !args[argsIdx].startsWith("-")) {
      startPath = args[argsIdx++];
    }

    while (argsIdx < args.length) {
      let arg = args[argsIdx];
      if (arg === "-name" && argsIdx + 1 < args.length) {
        nameOpt = args[++argsIdx];
      } else if (arg === "-type" && argsIdx + 1 < args.length) {
        typeOpt = args[++argsIdx];
      }
      argsIdx++;
    }

    let nameRegex = null;
    if (nameOpt) {
      let escaped = nameOpt
        .replace(/[.+^\\${}()|[\]\\\\]/g, "\\\\$&")
        .replace(/\\*/g, ".*")
        .replace(/\\?/g, ".");
      nameRegex = new RegExp("^" + escaped + "$");
    }

    function walk(dir) {
      let items;
      try {
        items = self.Module.FS.readdir(dir);
      } catch (e) {
        return;
      }
      for (let item of items) {
        if (item === "." || item === "..") continue;
        let fullPath = dir === "/" ? "/" + item : dir + "/" + item;
        let stat;
        try {
          stat = self.Module.FS.lstat(fullPath);
        } catch (e) {
          continue;
        }
        let isDir = self.Module.FS.isDir(stat.mode);
        let isFile = self.Module.FS.isFile(stat.mode);
        if (
          (!typeOpt ||
            (typeOpt === "d" && isDir) ||
            (typeOpt === "f" && isFile)) &&
          (!nameRegex || nameRegex.test(item))
        ) {
          self.Module.print(fullPath);
        }
        if (isDir) walk(fullPath);
      }
    }
    try {
      let stat = self.Module.FS.lstat(startPath);
      if (
        (!typeOpt ||
          (typeOpt === "d" && self.Module.FS.isDir(stat.mode)) ||
          (typeOpt === "f" && self.Module.FS.isFile(stat.mode))) &&
        (!nameRegex || nameRegex.test(startPath.split("/").pop() || startPath))
      ) {
        self.Module.print(startPath);
      }
      if (self.Module.FS.isDir(stat.mode)) walk(startPath);
    } catch (e) {
      self.Module.printErr(`find: ${startPath}: No such file or directory`);
      return 1;
    }
    return 0;
  }
  if (cmd === "sleep") {
    if (args.length < 2) {
      self.Module.printErr("sleep: missing operand");
      return 1;
    }
    let duration = parseFloat(args[1]);
    if (isNaN(duration) || duration < 0) {
      self.Module.printErr("sleep: invalid time interval '" + args[1] + "'");
      return 1;
    }
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      setTimeout(function () {
        wakeUp(0);
      }, duration * 1000);
    });
  }

  if (cmd === "whoami") {
    if (args.includes("--help")) {
      self.Module.print("Usage: whoami [OPTION]...");
      self.Module.print(
        "Print the user name associated with the current effective user ID.",
      );
      self.Module.print("Same as id -un.");
      self.Module.print("");
      self.Module.print("      --help     display this help and exit");
      self.Module.print("      --version  output version information and exit");
      return 0;
    }
    if (args.length > 1) {
      self.Module.printErr(`whoami: extra operand '${args[1]}'`);
      self.Module.printErr(`Try 'whoami --help' for more information.`);
      return 1;
    }
    self.Module.print("web_user");
    return 0;
  }

  if (cmd === "uname") {
    let opts = {
      a: false,
      s: false,
      n: false,
      r: false,
      v: false,
      m: false,
      p: false,
      i: false,
      o: false,
    };
    let hasOpt = false;
    for (let j = 1; j < args.length; j++) {
      if (args[j] === "--help") {
        self.Module.print("Usage: uname [OPTION]...");
        self.Module.print(
          "Print certain system information.  With no OPTION, same as -s.",
        );
        self.Module.print("  -a, --all                print all information");
        self.Module.print("  -s, --kernel-name        print the kernel name");
        self.Module.print(
          "  -n, --nodename           print the network node hostname",
        );
        self.Module.print(
          "  -r, --kernel-release     print the kernel release",
        );
        self.Module.print(
          "  -v, --kernel-version     print the kernel version",
        );
        self.Module.print(
          "  -m, --machine            print the machine hardware name",
        );
        self.Module.print(
          "  -p, --processor          print the processor type (non-portable)",
        );
        self.Module.print(
          "  -i, --hardware-platform  print the hardware platform (non-portable)",
        );
        self.Module.print(
          "  -o, --operating-system   print the operating system",
        );
        return 0;
      }
      if (args[j] === "--all") {
        opts.a = hasOpt = true;
        continue;
      }
      if (args[j] === "--kernel-name") {
        opts.s = hasOpt = true;
        continue;
      }
      if (args[j] === "--nodename") {
        opts.n = hasOpt = true;
        continue;
      }
      if (args[j] === "--kernel-release") {
        opts.r = hasOpt = true;
        continue;
      }
      if (args[j] === "--kernel-version") {
        opts.v = hasOpt = true;
        continue;
      }
      if (args[j] === "--machine") {
        opts.m = hasOpt = true;
        continue;
      }
      if (args[j] === "--processor") {
        opts.p = hasOpt = true;
        continue;
      }
      if (args[j] === "--hardware-platform") {
        opts.i = hasOpt = true;
        continue;
      }
      if (args[j] === "--operating-system") {
        opts.o = hasOpt = true;
        continue;
      }

      if (args[j].startsWith("-")) {
        for (let k = 1; k < args[j].length; k++) {
          let c = args[j][k];
          if (opts[c] !== undefined) {
            opts[c] = true;
            hasOpt = true;
          } else {
            self.Module.printErr("uname: invalid option -- '" + c + "'");
            return 1;
          }
        }
      } else {
        self.Module.printErr("uname: extra operand '" + args[j] + "'");
        return 1;
      }
    }

    if (!hasOpt) opts.s = true;
    if (opts.a) {
      opts.s =
        opts.n =
        opts.r =
        opts.v =
        opts.m =
        opts.p =
        opts.i =
        opts.o =
          true;
    }

    let out = [];
    if (opts.s) out.push("Emscripten");
    if (opts.n) out.push("emscripten");
    if (opts.r) out.push("1.0");
    if (opts.v) out.push("#1");
    if (opts.m) out.push("wasm32");
    if (opts.p) out.push("wasm32");
    if (opts.i) out.push("wasm32");
    if (opts.o) out.push("Emscripten");

    self.Module.print(out.join(" "));
    return 0;
  }

  if (cmd === "which") {
    let showAll = false;
    let targets = [];
    for (let i = 1; i < args.length; i++) {
      if (args[i] === "-a") {
        showAll = true;
      } else if (!args[i].startsWith("-")) {
        targets.push(args[i]);
      }
    }

    if (targets.length === 0) return 1;

    let pathEnv =
      typeof ENV !== "undefined" && ENV.PATH ? ENV.PATH : "/bin:/usr/bin";
    let paths = pathEnv.split(":");
    let FS = self.Module.FS;
    let ret = 0;

    for (let target of targets) {
      let found = false;
      if (target.includes("/")) {
        try {
          let stat = FS.stat(target);
          if (!FS.isDir(stat.mode)) {
            self.Module.print(target);
            found = true;
          }
        } catch (e) {}
      } else {
        for (let p of paths) {
          if (!p) p = ".";
          let fullPath = p === "/" ? "/" + target : p + "/" + target;
          try {
            let stat = FS.stat(fullPath);
            // Check if not directory (simplified executable check for wasm)
            if (!FS.isDir(stat.mode)) {
              self.Module.print(fullPath);
              found = true;
              if (!showAll) break;
            }
          } catch (e) {}
        }
      }
      if (!found) {
        ret = 1;
      }
    }
    return ret;
  }

  if (cmd === "mktemp") {
    let isDir = false;
    let dryRun = false;
    let template = "";
    let quiet = false;
    let hasTemplate = false;

    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "-d" || arg === "--directory") {
        isDir = true;
      } else if (arg === "-u" || arg === "--dry-run") {
        dryRun = true;
      } else if (arg === "-q" || arg === "--quiet") {
        quiet = true;
      } else if (!arg.startsWith("-")) {
        template = arg;
        hasTemplate = true;
      } else if (arg === "-p" || arg === "--tmpdir") {
        // simplified: ignore tmpdir specifics for now or handle next arg
      }
    }

    if (!hasTemplate) {
      template = "/tmp/tmp.XXXXXX";
    }

    let numXs = 0;
    for (let i = template.length - 1; i >= 0; i--) {
      if (template[i] === "X") {
        numXs++;
      } else {
        break;
      }
    }

    if (numXs < 3) {
      if (!quiet)
        self.Module.printErr(`mktemp: too few X's in template '${template}'`);
      return 1;
    }

    let baseTemplate = template.substring(0, template.length - numXs);
    let chars =
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    let maxTries = 100;
    let path = "";
    let success = false;

    for (let t = 0; t < maxTries; t++) {
      let randomStr = "";
      for (let i = 0; i < numXs; i++) {
        randomStr += chars.charAt(Math.floor(Math.random() * chars.length));
      }
      path = baseTemplate + randomStr;

      if (dryRun) {
        success = true;
        break;
      }

      try {
        if (isDir) {
          self.Module.FS.mkdir(path, 0o700);
        } else {
          let stream = self.Module.FS.open(path, "wx", 0o600);
          self.Module.FS.close(stream);
        }
        success = true;
        break;
      } catch (e) {
        // try again
      }
    }

    if (!success) {
      if (!quiet)
        self.Module.printErr(
          `mktemp: failed to create ${isDir ? "directory" : "file"} via template '${template}'`,
        );
      return 1;
    }

    self.Module.print(path);
    return 0;
  }

  if (cmd === "basename") {
    let argsList = args.slice(1);

    // Basic -- handling
    if (argsList.length > 0 && argsList[0] === "--") {
      argsList.shift();
    }

    if (argsList.length === 0) {
      self.Module.printErr("basename: missing operand");
      return 1;
    }

    let name = argsList[0];
    let suffix = argsList.length > 1 ? argsList[1] : "";

    // GNU basename handles -s option, but we are keeping it simple for POSIX compliance.
    // We can also add very basic -a support if needed, but not strictly required.
    if (argsList[0] === "-a") {
      let hasA = true;
    }

    if (name === "") {
      self.Module.print("");
      return 0;
    }

    // Strip trailing slashes, unless the string is entirely slashes
    let end = name.length - 1;
    while (end > 0 && name[end] === "/") {
      end--;
    }
    if (end < name.length - 1) {
      name = name.slice(0, end + 1);
    }
    // If it became empty after stripping trailing slashes, it means it was all slashes.
    // It should be '/'
    if (name.length === 0) {
      name = "/";
    }

    let base = name === "/" ? "/" : name.substring(name.lastIndexOf("/") + 1);

    if (suffix && base.endsWith(suffix) && base !== suffix) {
      base = base.slice(0, -suffix.length);
    }

    self.Module.print(base);
    return 0;
  }

  if (cmd === "dirname") {
    let argsList = args.slice(1);

    if (argsList.length > 0 && argsList[0] === "--") {
      argsList.shift();
    }

    if (argsList.length === 0) {
      self.Module.printErr("dirname: missing operand");
      return 1;
    }

    let name = argsList[0];

    if (name === "") {
      self.Module.print(".");
      return 0;
    }

    let end = name.length - 1;
    while (end > 0 && name[end] === "/") {
      end--;
    }
    if (end < name.length - 1) {
      name = name.slice(0, end + 1);
    }

    if (name.length === 0 || name === "/") {
      self.Module.print("/");
      return 0;
    }

    let lastSlash = name.lastIndexOf("/");

    if (lastSlash === -1) {
      self.Module.print(".");
      return 0;
    }

    if (lastSlash === 0) {
      self.Module.print("/");
      return 0;
    }

    let dirEnd = lastSlash;
    while (dirEnd > 0 && name[dirEnd - 1] === "/") {
      dirEnd--;
    }

    if (dirEnd === 0) {
      self.Module.print("/");
    } else {
      self.Module.print(name.slice(0, dirEnd));
    }
    return 0;
  }

  if (cmd === "tee") {
    let append = false;
    let files = [];
    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "-a" || arg === "--append") {
        append = true;
      } else if (!arg.startsWith("-") || arg === "-") {
        files.push(arg);
      }
    }

    let streams = [];
    for (let f of files) {
      if (f === "-") continue;
      try {
        let flags = append ? "a" : "w";
        let stream = self.Module.FS.open(f, flags);
        streams.push(stream);
      } catch (e) {
        self.Module.printErr(`tee: ${f}: No such file or directory`);
      }
    }

    let ret = 0;
    try {
      let stream0 = self.Module.FS.getStream(0);
      if (!stream0) {
        self.Module.printErr("tee: standard input is not available");
        ret = 1;
      } else {
        let buf = new Uint8Array(4096);
        while (true) {
          let bytesRead = 0;
          try {
            bytesRead = self.Module.FS.read(stream0, buf, 0, 4096, undefined);
          } catch (e) {
            break;
          }
          if (bytesRead <= 0) break;

          for (let i = 0; i < bytesRead; i++) {
            self.postMessage({ type: "STDOUT_CHAR", data: buf[i] });
          }

          for (let stream of streams) {
            try {
              self.Module.FS.write(stream, buf, 0, bytesRead, undefined);
            } catch (e) {
              ret = 1;
            }
          }
        }
      }
    } catch (e) {
      self.Module.printErr(`tee: error reading from stdin: ${e}`);
      ret = 1;
    }

    for (let stream of streams) {
      try {
        self.Module.FS.close(stream);
      } catch (e) {}
    }
    return ret;
  }

  if (cmd === "cat") {
    let ret = 0;
    let files = args.slice(1);
    if (files.length === 0) {
      files = ["-"];
    }
    for (let f of files) {
      if (f === "-") {
        try {
          let stream0 = self.Module.FS.getStream(0);
          if (!stream0) {
            self.Module.printErr("cat: standard input is not available");
            ret = 1;
          } else {
            let buf = new Uint8Array(4096);
            while (true) {
              let bytesRead = 0;
              try {
                bytesRead = self.Module.FS.read(
                  stream0,
                  buf,
                  0,
                  4096,
                  undefined,
                );
              } catch (e) {
                break;
              }
              if (bytesRead <= 0) break;
              for (let i = 0; i < bytesRead; i++) {
                self.postMessage({ type: "STDOUT_CHAR", data: buf[i] });
              }
            }
          }
        } catch (e) {
          self.Module.printErr(`cat: error reading from stdin: ${e}`);
          ret = 1;
        }
        continue;
      }
      try {
        let stat = self.Module.FS.stat(f);
        if (self.Module.FS.isDir(stat.mode)) {
          self.Module.printErr(`cat: ${f}: Is a directory`);
          ret = 1;
          continue;
        }
        let buf = self.Module.FS.readFile(f);
        for (let i = 0; i < buf.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: buf[i] });
        }
      } catch (e) {
        self.Module.printErr(`cat: ${f}: No such file or directory`);
        ret = 1;
      }
    }
    return ret;
  }

  if (cmd === "cp") {
    let ret = 0;
    let recursive = false;
    let args_cp = [];

    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg === "-r" || arg === "-R" || arg === "--recursive") {
        recursive = true;
      } else {
        args_cp.push(arg);
      }
    }

    if (args_cp.length < 2) {
      self.Module.printErr("cp: missing file operand");
      return 1;
    }

    let dest = args_cp.pop();
    let isDestDir = false;
    try {
      let stat = self.Module.FS.stat(dest);
      isDestDir = self.Module.FS.isDir(stat.mode);
    } catch (e) {}

    if (args_cp.length > 1 && !isDestDir) {
      self.Module.printErr(`cp: target '${dest}' is not a directory`);
      return 1;
    }

    function doCopy(src, dst) {
      try {
        let stat = self.Module.FS.stat(src);
        if (self.Module.FS.isDir(stat.mode)) {
          if (!recursive) {
            self.Module.printErr(
              `cp: -r not specified; omitting directory '${src}'`,
            );
            ret = 1;
            return;
          }
          try {
            self.Module.FS.mkdir(dst);
          } catch (e) {} // ignore if exists

          let items = self.Module.FS.readdir(src);
          for (let item of items) {
            if (item === "." || item === "..") continue;
            doCopy(src + "/" + item, dst + "/" + item);
          }
        } else {
          let content = self.Module.FS.readFile(src);
          self.Module.FS.writeFile(dst, content);
        }
      } catch (e) {
        self.Module.printErr(
          `cp: cannot stat '${src}': No such file or directory`,
        );
        ret = 1;
      }
    }

    for (let src of args_cp) {
      let srcName = src.split("/").pop();
      let targetDst = isDestDir
        ? dest + (dest.endsWith("/") ? "" : "/") + srcName
        : dest;
      doCopy(src, targetDst);
    }

    return ret;
  }

  if (cmd === "mv") {
    let ret = 0;
    let args_mv = args.slice(1);

    if (args_mv.length < 2) {
      self.Module.printErr("mv: missing file operand");
      return 1;
    }

    let dest = args_mv.pop();
    let isDestDir = false;
    try {
      let stat = self.Module.FS.stat(dest);
      isDestDir = self.Module.FS.isDir(stat.mode);
    } catch (e) {}

    if (args_mv.length > 1 && !isDestDir) {
      self.Module.printErr(`mv: target '${dest}' is not a directory`);
      return 1;
    }

    for (let src of args_mv) {
      let srcName = src.split("/").pop();
      let targetDst = isDestDir
        ? dest + (dest.endsWith("/") ? "" : "/") + srcName
        : dest;
      try {
        self.Module.FS.rename(src, targetDst);
      } catch (e) {
        self.Module.printErr(
          `mv: cannot stat '${src}': No such file or directory`,
        );
        ret = 1;
      }
    }

    return ret;
  }

  if (cmd === "rm") {
    let ret = 0;
    let recursive = false;
    let force = false;
    let args_rm = [];

    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-") {
        if (arg === "--recursive") recursive = true;
        else if (arg === "--force") force = true;
        else {
          for (let k = 1; k < arg.length; k++) {
            if (arg[k] === "r" || arg[k] === "R") recursive = true;
            else if (arg[k] === "f") force = true;
          }
        }
      } else {
        args_rm.push(arg);
      }
    }

    if (args_rm.length === 0 && !force) {
      self.Module.printErr("rm: missing operand");
      return 1;
    }

    function doRemove(p) {
      try {
        let stat = self.Module.FS.lstat(p);
        if (self.Module.FS.isDir(stat.mode)) {
          if (!recursive) {
            self.Module.printErr(`rm: cannot remove '${p}': Is a directory`);
            ret = 1;
            return;
          }
          let items = self.Module.FS.readdir(p);
          for (let item of items) {
            if (item === "." || item === "..") continue;
            doRemove(p + "/" + item);
          }
          self.Module.FS.rmdir(p);
        } else {
          self.Module.FS.unlink(p);
        }
      } catch (e) {
        if (!force) {
          self.Module.printErr(
            `rm: cannot remove '${p}': No such file or directory`,
          );
          ret = 1;
        }
      }
    }

    for (let p of args_rm) {
      doRemove(p);
    }

    return ret;
  }

  if (cmd === "ls") {
    let opts = { l: false, a: false, h: false };
    let files = [];
    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-") {
        for (let k = 1; k < arg.length; k++) {
          if (arg[k] === "l") opts.l = true;
          else if (arg[k] === "a") opts.a = true;
          else if (arg[k] === "h") opts.h = true;
          else {
            self.Module.printErr("ls: invalid option -- '" + arg[k] + "'");
            return 2;
          }
        }
      } else {
        files.push(arg);
      }
    }

    if (files.length === 0) files.push(self.Module.FS.cwd());

    let ret = 0;

    function modeToPerms(mode) {
      const types = {
        0o010000: "p",
        0o020000: "c",
        0o040000: "d",
        0o060000: "b",
        0o100000: "-",
        0o120000: "l",
        0o140000: "s",
      };
      let type = types[mode & 0o170000] || "?";
      const rwx = ["---", "--x", "-w-", "-wx", "r--", "r-x", "rw-", "rwx"];
      let u = rwx[(mode >> 6) & 7];
      let g = rwx[(mode >> 3) & 7];
      let o = rwx[mode & 7];
      return type + u + g + o;
    }

    function formatSize(bytes) {
      if (!opts.h) return bytes.toString();
      const units = ["B", "K", "M", "G", "T", "P"];
      let u = 0;
      let val = bytes;
      while (val >= 1024 && u < units.length - 1) {
        val /= 1024;
        u++;
      }
      return (u === 0 ? val : val.toFixed(1)) + units[u];
    }

    for (let i = 0; i < files.length; i++) {
      let f = files[i];
      if (files.length > 1) {
        if (i > 0) self.Module.print("");
        self.Module.print(f + ":");
      }
      try {
        let stat = self.Module.FS.lstat(f);
        if (self.Module.FS.isDir(stat.mode)) {
          let items = self.Module.FS.readdir(f);
          if (!opts.a) {
            items = items.filter((n) => !n.startsWith("."));
          }
          items.sort();

          if (opts.l) {
            for (let item of items) {
              let fullPath = f === "/" ? "/" + item : f + "/" + item;
              let itemStat = self.Module.FS.lstat(fullPath);
              let perms = modeToPerms(itemStat.mode);
              let links = itemStat.nlink || 1;
              let uid = itemStat.uid || 0;
              let gid = itemStat.gid || 0;
              let sizeStr = formatSize(itemStat.size || 0);
              let mtime = new Date(itemStat.mtime).toLocaleString("en-US", {
                month: "short",
                day: "2-digit",
                hour: "2-digit",
                minute: "2-digit",
              });
              self.Module.print(
                `${perms} ${links} ${uid} ${gid} ${sizeStr.padStart(5)} ${mtime} ${item}`,
              );
            }
          } else {
            self.Module.print(items.join("  "));
          }
        } else {
          if (opts.l) {
            let perms = modeToPerms(stat.mode);
            let links = stat.nlink || 1;
            let uid = stat.uid || 0;
            let gid = stat.gid || 0;
            let sizeStr = formatSize(stat.size || 0);
            let mtime = new Date(stat.mtime).toLocaleString("en-US", {
              month: "short",
              day: "2-digit",
              hour: "2-digit",
              minute: "2-digit",
            });
            self.Module.print(
              `${perms} ${links} ${uid} ${gid} ${sizeStr.padStart(5)} ${mtime} ${f}`,
            );
          } else {
            self.Module.print(f);
          }
        }
      } catch (e) {
        self.Module.printErr(
          `ls: cannot access '${f}': No such file or directory`,
        );
        ret = 1;
      }
    }
    return ret;
  }

  if (cmd === "mkdir") {
    let ret = 0;
    let parents = false;
    let args_mkdir = [];

    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-") {
        if (arg === "--parents") parents = true;
        else {
          for (let k = 1; k < arg.length; k++) {
            if (arg[k] === "p") parents = true;
            else {
              self.Module.printErr(`mkdir: invalid option -- '${arg[k]}'`);
              return 1;
            }
          }
        }
      } else {
        args_mkdir.push(arg);
      }
    }

    if (args_mkdir.length === 0) {
      self.Module.printErr("mkdir: missing operand");
      return 1;
    }

    for (let p of args_mkdir) {
      if (parents) {
        let parts = p.split("/").filter((x) => x);
        let currentPath = p.startsWith("/") ? "" : self.Module.FS.cwd();
        if (!currentPath && p.startsWith("/")) {
          currentPath = "";
        }
        for (let part of parts) {
          currentPath = currentPath + "/" + part;
          try {
            self.Module.FS.mkdir(currentPath);
          } catch (e) {
            let isDir = false;
            try {
              let stat = self.Module.FS.stat(currentPath);
              isDir = self.Module.FS.isDir(stat.mode);
            } catch (e2) {}

            if (!isDir) {
              self.Module.printErr(
                `mkdir: cannot create directory '${p}': File exists`,
              );
              ret = 1;
              break;
            }
          }
        }
      } else {
        try {
          self.Module.FS.mkdir(p);
        } catch (e) {
          self.Module.printErr(
            `mkdir: cannot create directory '${p}': File exists or no such file or directory`,
          );
          ret = 1;
        }
      }
    }

    return ret;
  }

  if (cmd === "chmod") {
    let ret = 0;
    let recursive = false;
    let args_chmod = [];

    for (let j = 1; j < args.length; j++) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-" && arg.length > 1) {
        if (arg === "--recursive") recursive = true;
        else {
          for (let k = 1; k < arg.length; k++) {
            if (arg[k] === "R") recursive = true;
            // Ignore other flags for now
          }
        }
      } else {
        args_chmod.push(arg);
      }
    }

    if (args_chmod.length < 2) {
      self.Module.printErr("chmod: missing operand");
      return 1;
    }

    let modeStr = args_chmod[0];
    let files = args_chmod.slice(1);

    function parseMode(currentMode, modeStr, isDir) {
      if (/^[0-7]+$/.test(modeStr)) {
        return parseInt(modeStr, 8);
      }

      let newMode = currentMode & 0o7777; // Keep only permissions

      // Simple symbolic mode parsing: [ugoa]*[+-=][rwxXst]*
      let parts = modeStr.split(",");
      for (let part of parts) {
        let match = part.match(/^([ugoa]*)([+-=])([rwxXst]*)$/);
        if (!match) continue; // Skip invalid, or we could throw

        let who = match[1];
        let op = match[2];
        let perms = match[3];

        if (who === "") who = "a"; // Default to 'a' if not specified (should apply umask, but 'a' is close enough for minimal)

        let mask = 0;
        if (perms.includes("r")) mask |= 0o444;
        if (perms.includes("w")) mask |= 0o222;
        if (perms.includes("x")) mask |= 0o111;
        if (perms.includes("X") && (isDir || (currentMode & 0o111) !== 0))
          mask |= 0o111;

        let whoMask = 0;
        if (who.includes("u") || who === "a") whoMask |= 0o700;
        if (who.includes("g") || who === "a") whoMask |= 0o070;
        if (who.includes("o") || who === "a") whoMask |= 0o007;

        let appliedMask = mask & whoMask;

        if (op === "+") {
          newMode |= appliedMask;
        } else if (op === "-") {
          newMode &= ~appliedMask;
        } else if (op === "=") {
          newMode &= ~whoMask;
          newMode |= appliedMask;
        }
      }
      return newMode;
    }

    function doChmod(p) {
      try {
        let stat = self.Module.FS.lstat(p);
        let isDir = self.Module.FS.isDir(stat.mode);
        let newMode = parseMode(stat.mode, modeStr, isDir);

        self.Module.FS.chmod(p, newMode);

        if (isDir && recursive) {
          let items = self.Module.FS.readdir(p);
          for (let item of items) {
            if (item === "." || item === "..") continue;
            doChmod(p + "/" + item);
          }
        }
      } catch (e) {
        self.Module.printErr(
          `chmod: cannot access '${p}': No such file or directory`,
        );
        ret = 1;
      }
    }

    for (let f of files) {
      doChmod(f);
    }

    return ret;
  }

  if (cmd === "tr") {
    let opts = { d: false, s: false, c: false };
    let set1 = "";
    let set2 = "";

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg === "-d") opts.d = true;
      else if (arg === "-s") opts.s = true;
      else if (arg === "-c" || arg === "-C") opts.c = true;
      else if (arg.startsWith("-") && arg !== "-") {
        for (let k = 1; k < arg.length; k++) {
          if (arg[k] === "d") opts.d = true;
          else if (arg[k] === "s") opts.s = true;
          else if (arg[k] === "c" || arg[k] === "C") opts.c = true;
        }
      } else if (!set1) set1 = arg;
      else if (!set2) set2 = arg;
      j++;
    }

    if (!set1) {
      self.Module.printErr("tr: missing operand");
      return 1;
    }

    function expandSet(s) {
      let res = "";
      let i = 0;
      while (i < s.length) {
        if (s[i] === "\\" && i + 1 < s.length) {
          let next = s[i + 1];
          if (next === "n") res += "\n";
          else if (next === "t") res += "\t";
          else if (next === "r") res += "\r";
          else res += next;
          i += 2;
        } else if (i + 2 < s.length && s[i + 1] === "-") {
          let start = s.charCodeAt(i);
          let end = s.charCodeAt(i + 2);
          if (start <= end) {
            for (let c = start; c <= end; c++) res += String.fromCharCode(c);
          } else {
            res += s[i] + s[i + 1] + s[i + 2];
          }
          i += 3;
        } else {
          res += s[i];
          i++;
        }
      }
      return res;
    }

    let expSet1 = expandSet(set1);
    let expSet2 = set2 ? expandSet(set2) : "";

    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      let inputStr = "";

      function execute() {
        let output = "";
        let lastChar = null;

        for (let i = 0; i < inputStr.length; i++) {
          let ch = inputStr[i];
          let idx = expSet1.indexOf(ch);
          let match = idx !== -1;

          if (opts.c) match = !match;

          if (opts.d) {
            if (match) continue;
            if (opts.s && expSet2.indexOf(ch) !== -1 && lastChar === ch)
              continue;
            output += ch;
            lastChar = ch;
          } else {
            let replaceChar = ch;
            if (match && expSet2.length > 0) {
              let repIdx = opts.c
                ? expSet2.length - 1
                : Math.min(idx, expSet2.length - 1);
              replaceChar = expSet2[repIdx];
            }
            if (opts.s) {
              let squeezeSet = expSet2.length > 0 ? expSet2 : expSet1;
              if (
                squeezeSet.indexOf(replaceChar) !== -1 &&
                lastChar === replaceChar
              ) {
                continue;
              }
            }
            output += replaceChar;
            lastChar = replaceChar;
          }
        }
        for (let i = 0; i < output.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: output.charCodeAt(i) });
        }
        wakeUp(0);
      }

      try {
        let stream = self.Module.FS.getStream(0);
        if (stream) {
          let buf = new Uint8Array(4096);
          let bytesRead = 1;
          while (bytesRead > 0) {
            try {
              bytesRead = self.Module.FS.read(stream, buf, 0, 4096, undefined);
              if (bytesRead > 0) {
                for (let k = 0; k < bytesRead; k++) {
                  inputStr += String.fromCharCode(buf[k]);
                }
              }
            } catch (e) {
              bytesRead = 0;
            }
          }
          if (inputStr.length > 0) {
            execute();
            return;
          }
        }
      } catch (e) {}

      setTimeout(() => {
        let readLoop = setInterval(() => {
          let didRead = false;
          while (globalThis.stdinBuffer && globalThis.stdinBuffer.length > 0) {
            inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
            didRead = true;
          }
          if (didRead && globalThis.stdinBuffer.length === 0) {
            clearInterval(readLoop);
            execute();
          } else if (inputStr.length > 0 && !didRead) {
            clearInterval(readLoop);
            execute();
          }
        }, 50);
      }, 0);
    });
  }

  if (cmd === "cut") {
    let delim = "\t";
    let fieldsStr = "";
    let charsStr = "";
    let files = [];
    let suppress = false;

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg === "-d") {
        delim = args[++j];
      } else if (arg.startsWith("-d")) {
        delim = arg.substring(2);
      } else if (arg === "-f") {
        fieldsStr = args[++j];
      } else if (arg.startsWith("-f")) {
        fieldsStr = arg.substring(2);
      } else if (arg === "-c") {
        charsStr = args[++j];
      } else if (arg.startsWith("-c")) {
        charsStr = arg.substring(2);
      } else if (arg === "-s" || arg === "--only-delimited") {
        suppress = true;
      } else if (!arg.startsWith("-")) {
        files.push(arg);
      }
      j++;
    }

    if (!fieldsStr && !charsStr) {
      self.Module.printErr(
        "cut: you must specify a list of bytes, characters, or fields",
      );
      return 1;
    }

    function getIndices(listStr, maxLen) {
      let indices = new Set();
      let ranges = listStr.split(",");
      for (let r of ranges) {
        if (r.includes("-")) {
          let parts = r.split("-");
          let start = parts[0] === "" ? 1 : parseInt(parts[0], 10);
          let end = parts[1] === "" ? maxLen : parseInt(parts[1], 10);
          for (let i = start; i <= end && i <= maxLen; i++) {
            indices.add(i);
          }
        } else {
          let n = parseInt(r, 10);
          if (n <= maxLen) indices.add(n);
        }
      }
      let arr = Array.from(indices);
      arr.sort((a, b) => a - b);
      return arr;
    }

    function processLine(line) {
      if (charsStr) {
        let indices = getIndices(charsStr, line.length);
        let out = "";
        for (let i of indices) {
          out += line[i - 1]; // 1-based
        }
        self.Module.print(out);
      } else if (fieldsStr) {
        if (!line.includes(delim)) {
          if (!suppress) self.Module.print(line);
        } else {
          let parts = line.split(delim);
          let indices = getIndices(fieldsStr, parts.length);
          let outParts = [];
          for (let i of indices) {
            outParts.push(parts[i - 1]);
          }
          self.Module.print(outParts.join(delim));
        }
      }
    }

    if (files.length === 0) {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          let lines = inputStr.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) processLine(line);
          wakeUp(0);
        }
        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let k = 0; k < bytesRead; k++) {
                    inputStr += String.fromCharCode(buf[k]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    } else {
      let ret = 0;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          let lines = text.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          for (let line of lines) processLine(line);
        } catch (e) {
          self.Module.printErr(`cut: ${f}: No such file or directory`);
          ret = 1;
        }
      }
      return ret;
    }
  }

  if (cmd === "top" || cmd === "htop") {
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      function writeRaw(str) {
        for (let i = 0; i < str.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: str.charCodeAt(i) });
        }
      }

      function render() {
        writeRaw("\x1b[2J\x1b[H"); // Clear
        writeRaw("\x1b[7mtotal tasks: 3, running: 1, sleeping: 2\x1b[0m\r\n");
        writeRaw(
          "\x1b[7m  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND\x1b[0m\r\n",
        );
        writeRaw(
          `    1 root      20   0    2.1m   1.2m   0.8m S   0.0   0.1   0:00.01 init\r\n`,
        );
        writeRaw(
          `   10 web_user  20   0    4.5m   2.3m   1.5m S   0.0   1.2   0:00.10 dash\r\n`,
        );

        // Randomize CPU a bit to make it look alive
        let cpu = (Math.random() * 5).toFixed(1);
        writeRaw(
          `   25 web_user  20   0    1.0m   0.5m   0.2m R   ${cpu}   0.0   0:00.01 ${cmd}\r\n`,
        );
        writeRaw("\r\n(press q to quit)");
      }

      self.postMessage({ type: "SET_RAW_MODE", data: true });
      writeRaw("\x1b[?1049h"); // Enter alternate screen buffer
      render();

      let ticks = 0;
      let inputLoop = setInterval(() => {
        ticks++;
        if (ticks % 60 === 0) {
          render();
        }
        while (globalThis.stdinBuffer && globalThis.stdinBuffer.length > 0) {
          let code = globalThis.stdinBuffer.shift();
          let ch = String.fromCharCode(code);
          if (ch === "q" || ch === "Q" || code === 3) {
            clearInterval(inputLoop);
            self.postMessage({ type: "SET_RAW_MODE", data: false });
            writeRaw("\x1b[?1049l");
            wakeUp(0);
            return;
          }
        }
      }, 50);
    });
  }

  if (cmd === "vi" || cmd === "vim" || cmd === "nano" || cmd === "pico") {
    let files = args.slice(1).filter((a) => !a.startsWith("-"));
    let filename = files.length > 0 ? files[0] : "Untitled";
    let text = "";
    if (files.length > 0) {
      try {
        text = self.Module.FS.readFile(files[0], { encoding: "utf8" });
      } catch (e) {} // It's a new file
    }

    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      const termRows = 24;
      let lines = text.split("\n");
      if (lines.length === 0) lines = [""];
      let cursorY = 0;
      let cursorX = 0;
      let scrollY = 0;

      function writeRaw(str) {
        for (let i = 0; i < str.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: str.charCodeAt(i) });
        }
      }

      function render() {
        writeRaw("\x1b[?25l"); // Hide cursor
        writeRaw("\x1b[2J\x1b[H"); // Clear
        // header
        writeRaw(
          `\x1b[7m  GNU nano-ish                        ${filename} \x1b[0m\r\n`,
        );

        let displayLines = termRows - 3;
        if (cursorY < scrollY) scrollY = cursorY;
        if (cursorY >= scrollY + displayLines)
          scrollY = cursorY - displayLines + 1;

        for (let i = 0; i < displayLines; i++) {
          let lineIdx = scrollY + i;
          if (lineIdx < lines.length) {
            writeRaw(lines[lineIdx] + "\r\n");
          } else {
            writeRaw("\r\n");
          }
        }
        // footer
        writeRaw(`\x1b[7m^S Save    ^X Exit\x1b[0m`);

        // Position cursor
        writeRaw(`\x1b[${cursorY - scrollY + 2};${cursorX + 1}H`);
        writeRaw("\x1b[?25h"); // Show cursor
      }

      self.postMessage({ type: "SET_RAW_MODE", data: true });
      writeRaw("\x1b[?1049h"); // Enter alternate screen buffer
      render();

      let escapeBuffer = "";
      let inputLoop = setInterval(() => {
        while (globalThis.stdinBuffer && globalThis.stdinBuffer.length > 0) {
          let code = globalThis.stdinBuffer.shift();
          let ch = String.fromCharCode(code);

          if (escapeBuffer.length > 0) {
            escapeBuffer += ch;
            if (escapeBuffer === "\x1b[A") {
              // Up
              if (cursorY > 0) {
                cursorY--;
                if (cursorX > lines[cursorY].length)
                  cursorX = lines[cursorY].length;
              }
              escapeBuffer = "";
              render();
            } else if (escapeBuffer === "\x1b[B") {
              // Down
              if (cursorY < lines.length - 1) {
                cursorY++;
                if (cursorX > lines[cursorY].length)
                  cursorX = lines[cursorY].length;
              }
              escapeBuffer = "";
              render();
            } else if (escapeBuffer === "\x1b[C") {
              // Right
              if (cursorX < lines[cursorY].length) cursorX++;
              escapeBuffer = "";
              render();
            } else if (escapeBuffer === "\x1b[D") {
              // Left
              if (cursorX > 0) cursorX--;
              escapeBuffer = "";
              render();
            } else if (escapeBuffer.length >= 3) {
              escapeBuffer = ""; // Unknown or longer sequence
            }
            continue;
          }

          if (ch === "\x1b") {
            escapeBuffer = ch;
            continue;
          }

          if (code === 24) {
            // Ctrl+X
            clearInterval(inputLoop);
            self.postMessage({ type: "SET_RAW_MODE", data: false });
            writeRaw("\x1b[?1049l");
            wakeUp(0);
            return;
          } else if (code === 19) {
            // Ctrl+S
            try {
              self.Module.FS.writeFile(filename, lines.join("\n"));
            } catch (e) {}
          } else if (code === 127 || code === 8) {
            // Backspace
            if (cursorX > 0) {
              lines[cursorY] =
                lines[cursorY].slice(0, cursorX - 1) +
                lines[cursorY].slice(cursorX);
              cursorX--;
            } else if (cursorY > 0) {
              let oldLen = lines[cursorY - 1].length;
              lines[cursorY - 1] += lines[cursorY];
              lines.splice(cursorY, 1);
              cursorY--;
              cursorX = oldLen;
            }
            render();
          } else if (code === 13 || code === 10) {
            // Enter
            let remainder = lines[cursorY].slice(cursorX);
            lines[cursorY] = lines[cursorY].slice(0, cursorX);
            lines.splice(cursorY + 1, 0, remainder);
            cursorY++;
            cursorX = 0;
            render();
          } else if (code >= 32 && code <= 126) {
            lines[cursorY] =
              lines[cursorY].slice(0, cursorX) +
              ch +
              lines[cursorY].slice(cursorX);
            cursorX++;
            render();
          }
        }
      }, 50);
    });
  }
  if (cmd === "less" || cmd === "more") {
    let files = args.slice(1).filter((a) => !a.startsWith("-"));
    let text = "";
    if (files.length > 0) {
      try {
        text = self.Module.FS.readFile(files[0], { encoding: "utf8" });
      } catch (e) {
        self.Module.printErr(`${cmd}: ${files[0]}: No such file or directory`);
        return 1;
      }
    } else {
      self.Module.printErr(
        `${cmd}: reading from stdin not implemented yet in this Wasm build`,
      );
      return 1;
    }

    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      const termRows = 24;
      const lines = text.split("\n");
      let offset = 0;

      function writeRaw(str) {
        for (let i = 0; i < str.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: str.charCodeAt(i) });
        }
      }

      function render() {
        writeRaw("\x1b[2J\x1b[H"); // Clear screen and move to top
        let out = lines.slice(offset, offset + termRows - 1);
        for (let i = 0; i < out.length; i++) {
          writeRaw(out[i] + "\r\n");
        }
        writeRaw(
          "\x1b[7m:(press q to quit, space/j for next, k for prev)\x1b[0m",
        );
      }

      self.postMessage({ type: "SET_RAW_MODE", data: true });
      writeRaw("\x1b[?1049h"); // Enter alternate screen buffer
      render();

      let inputLoop = setInterval(() => {
        while (globalThis.stdinBuffer && globalThis.stdinBuffer.length > 0) {
          let ch = String.fromCharCode(globalThis.stdinBuffer.shift());
          if (ch === "q" || ch === "Q") {
            clearInterval(inputLoop);
            self.postMessage({ type: "SET_RAW_MODE", data: false });
            writeRaw("\x1b[?1049l"); // Exit alternate screen buffer
            wakeUp(0);
            return;
          } else if (
            ch === " " ||
            ch === "\x1b[B" ||
            ch === "j" ||
            ch === "\r" ||
            ch === "\n"
          ) {
            if (offset + termRows - 1 < lines.length) {
              offset++;
              render();
            }
          } else if (ch === "k" || ch === "\x1b[A") {
            if (offset > 0) {
              offset--;
              render();
            }
          }
        }
      }, 50);
    });
  }

  if (cmd === "tail") {
    let linesToPrint = 10;
    let linesPlus = false;
    let bytesToPrint = null;
    let bytesPlus = false;
    let quiet = false;
    let verbose = false;
    let files = [];

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg === "-n") {
        let val = args[++j];
        if (val && val.startsWith("+")) {
          linesPlus = true;
          linesToPrint = parseInt(val.substring(1), 10);
        } else {
          linesToPrint = parseInt(val, 10);
        }
      } else if (arg.startsWith("-n")) {
        let val = arg.substring(2);
        if (val.startsWith("+")) {
          linesPlus = true;
          linesToPrint = parseInt(val.substring(1), 10);
        } else {
          linesToPrint = parseInt(val, 10);
        }
      } else if (arg === "-c") {
        let val = args[++j];
        if (val && val.startsWith("+")) {
          bytesPlus = true;
          bytesToPrint = parseInt(val.substring(1), 10);
        } else {
          bytesToPrint = parseInt(val, 10);
        }
      } else if (arg.startsWith("-c")) {
        let val = arg.substring(2);
        if (val.startsWith("+")) {
          bytesPlus = true;
          bytesToPrint = parseInt(val.substring(1), 10);
        } else {
          bytesToPrint = parseInt(val, 10);
        }
      } else if (arg === "-q" || arg === "--quiet" || arg === "--silent") {
        quiet = true;
      } else if (arg === "-v" || arg === "--verbose") {
        verbose = true;
      } else if (!arg.startsWith("-")) {
        files.push(arg);
      } else if (
        arg.startsWith("-") &&
        (arg[1] === "+" || !isNaN(parseInt(arg.substring(1), 10)))
      ) {
        let val = arg.substring(1);
        if (val.startsWith("+")) {
          linesPlus = true;
          linesToPrint = parseInt(val.substring(1), 10);
        } else {
          linesToPrint = parseInt(val, 10);
        }
      }
      j++;
    }

    if (files.length === 0) files.push("-");

    if (files.length > 1 && !quiet) verbose = true;

    function processTail(text, isFile, filename) {
      if (verbose) {
        self.Module.print(`==> ${filename} <==`);
      }
      if (bytesToPrint !== null) {
        let out = "";
        if (bytesPlus) {
          out = text.substring(Math.max(0, bytesToPrint - 1));
        } else {
          let start = Math.max(0, text.length - bytesToPrint);
          out = text.substring(start);
        }
        for (let i = 0; i < out.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: out.charCodeAt(i) });
        }
      } else {
        let lines = text.split("\n");
        if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
        let start = 0;
        if (linesPlus) {
          start = Math.max(0, linesToPrint - 1);
        } else {
          start = Math.max(0, lines.length - Math.abs(linesToPrint));
        }
        for (let i = start; i < lines.length; i++) {
          self.Module.print(lines[i]);
        }
      }
    }

    if (files.length === 1 && files[0] === "-") {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          processTail(inputStr, false, "standard input");
          wakeUp(0);
        }

        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let k = 0; k < bytesRead; k++) {
                    inputStr += String.fromCharCode(buf[k]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    } else {
      let ret = 0;
      let isFirst = true;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          if (!isFirst && verbose) self.Module.print("");
          processTail(text, true, f);
          isFirst = false;
        } catch (e) {
          self.Module.printErr(
            `tail: cannot open '${f}' for reading: No such file or directory`,
          );
          ret = 1;
        }
      }
      return ret;
    }
  }

  if (cmd === "head") {
    let linesToPrint = 10;
    let bytesToPrint = null;
    let quiet = false;
    let verbose = false;
    let files = [];

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg === "-n") {
        linesToPrint = parseInt(args[++j], 10);
      } else if (arg.startsWith("-n")) {
        linesToPrint = parseInt(arg.substring(2), 10);
      } else if (arg === "-c") {
        bytesToPrint = parseInt(args[++j], 10);
      } else if (arg.startsWith("-c")) {
        bytesToPrint = parseInt(arg.substring(2), 10);
      } else if (arg === "-q" || arg === "--quiet" || arg === "--silent") {
        quiet = true;
      } else if (arg === "-v" || arg === "--verbose") {
        verbose = true;
      } else if (!arg.startsWith("-")) {
        files.push(arg);
      } else if (
        arg.startsWith("-") &&
        !isNaN(parseInt(arg.substring(1), 10))
      ) {
        linesToPrint = parseInt(arg.substring(1), 10);
      }
      j++;
    }

    if (files.length === 0) files.push("-");

    if (files.length > 1 && !quiet) verbose = true;

    function processHead(text, isFile, filename) {
      if (verbose) {
        self.Module.print(`==> ${filename} <==`);
      }
      if (bytesToPrint !== null) {
        let out = text.substring(0, bytesToPrint);
        for (let i = 0; i < out.length; i++) {
          self.postMessage({ type: "STDOUT_CHAR", data: out.charCodeAt(i) });
        }
      } else {
        let lines = text.split("\n");
        if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
        let limit = linesToPrint;
        if (limit < 0) {
          limit = lines.length + limit;
          if (limit < 0) limit = 0;
        }
        for (let i = 0; i < Math.min(limit, lines.length); i++) {
          self.Module.print(lines[i]);
        }
      }
    }

    if (files.length === 1 && files[0] === "-") {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          processHead(inputStr, false, "standard input");
          wakeUp(0);
        }

        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let k = 0; k < bytesRead; k++) {
                    inputStr += String.fromCharCode(buf[k]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    } else {
      let ret = 0;
      let isFirst = true;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          if (!isFirst && verbose) self.Module.print("");
          processHead(text, true, f);
          isFirst = false;
        } catch (e) {
          self.Module.printErr(
            `head: cannot open '${f}' for reading: No such file or directory`,
          );
          ret = 1;
        }
      }
      return ret;
    }
  }

  if (cmd === "pbcopy") {
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      let inputStr = "";
      function execute() {
        globalThis.resolveClipboardWrite = function () {
          wakeUp(0);
        };
        self.postMessage({ type: "CLIPBOARD_WRITE", data: inputStr });
      }

      try {
        let stream = self.Module.FS.getStream(0);
        if (stream) {
          let buf = new Uint8Array(4096);
          let bytesRead = 1;
          while (bytesRead > 0) {
            try {
              bytesRead = self.Module.FS.read(stream, buf, 0, 4096, undefined);
              if (bytesRead > 0) {
                for (let k = 0; k < bytesRead; k++) {
                  inputStr += String.fromCharCode(buf[k]);
                }
              }
            } catch (e) {
              bytesRead = 0;
            }
          }
          if (inputStr.length > 0) {
            execute();
            return;
          }
        }
      } catch (e) {}
      execute();
    });
  }

  if (cmd === "pbpaste") {
    return self.Module.Asyncify.handleSleep(function (wakeUp) {
      globalThis.resolveClipboardRead = function (text) {
        if (text) {
          for (let i = 0; i < text.length; i++) {
            self.postMessage({ type: "STDOUT_CHAR", data: text.charCodeAt(i) });
          }
        }
        wakeUp(0);
      };
      self.postMessage({ type: "CLIPBOARD_READ" });
    });
  }

  if (cmd === "sort") {
    let opts = { r: false, n: false, u: false, h: false };
    let files = [];

    let j = 1;
    while (j < args.length) {
      let arg = args[j];
      if (arg.startsWith("-") && arg !== "-") {
        for (let k = 1; k < arg.length; k++) {
          if (arg[k] === "r") opts.r = true;
          else if (arg[k] === "n") opts.n = true;
          else if (arg[k] === "u") opts.u = true;
          else if (arg[k] === "h") opts.h = true;
        }
      } else {
        files.push(arg);
      }
      j++;
    }

    function parseHuman(str) {
      let match = str.trim().match(/^([+-]?\d*\.?\d+)\s*([KMGTP]?)/i);
      if (!match) return NaN;
      let num = parseFloat(match[1]);
      let suffix = match[2].toUpperCase();
      const multipliers = { K: 1e3, M: 1e6, G: 1e9, T: 1e12, P: 1e15 };
      if (multipliers[suffix]) {
        num *= multipliers[suffix];
      }
      return num;
    }

    function doSort(lines) {
      if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();

      lines.sort((a, b) => {
        if (opts.h) {
          let numA = parseHuman(a);
          let numB = parseHuman(b);
          let isNumA = !isNaN(numA);
          let isNumB = !isNaN(numB);

          if (isNumA && isNumB) {
            if (numA !== numB) return numA - numB;
          } else if (isNumA && !isNumB) {
            return 1; // Numbers come after non-numbers in GNU sort -h
          } else if (!isNumA && isNumB) {
            return -1;
          }
        } else if (opts.n) {
          let numA = parseFloat(a);
          let numB = parseFloat(b);
          if (isNaN(numA)) numA = 0;
          if (isNaN(numB)) numB = 0;
          if (numA !== numB) return numA - numB;
        }
        return a.localeCompare(b);
      });

      if (opts.r) lines.reverse();

      if (opts.u) {
        let uniqueLines = [];
        let seen = new Set();
        for (let line of lines) {
          if (!seen.has(line)) {
            seen.add(line);
            uniqueLines.push(line);
          }
        }
        lines = uniqueLines;
      }

      for (let line of lines) {
        self.Module.print(line);
      }
    }

    if (files.length === 0 || files.includes("-")) {
      return self.Module.Asyncify.handleSleep(function (wakeUp) {
        let inputStr = "";
        function execute() {
          let lines = inputStr.split("\n");
          doSort(lines);
          wakeUp(0);
        }

        try {
          let stream = self.Module.FS.getStream(0);
          if (stream) {
            let buf = new Uint8Array(4096);
            let bytesRead = 1;
            while (bytesRead > 0) {
              try {
                bytesRead = self.Module.FS.read(
                  stream,
                  buf,
                  0,
                  4096,
                  undefined,
                );
                if (bytesRead > 0) {
                  for (let k = 0; k < bytesRead; k++) {
                    inputStr += String.fromCharCode(buf[k]);
                  }
                }
              } catch (e) {
                bytesRead = 0;
              }
            }
            if (inputStr.length > 0) {
              execute();
              return;
            }
          }
        } catch (e) {}

        setTimeout(() => {
          let readLoop = setInterval(() => {
            let didRead = false;
            while (
              globalThis.stdinBuffer &&
              globalThis.stdinBuffer.length > 0
            ) {
              inputStr += String.fromCharCode(globalThis.stdinBuffer.shift());
              didRead = true;
            }
            if (didRead && globalThis.stdinBuffer.length === 0) {
              clearInterval(readLoop);
              execute();
            } else if (inputStr.length > 0 && !didRead) {
              clearInterval(readLoop);
              execute();
            }
          }, 50);
        }, 0);
      });
    } else {
      let allLines = [];
      let ret = 0;
      for (let f of files) {
        try {
          let text = self.Module.FS.readFile(f, { encoding: "utf8" });
          let lines = text.split("\n");
          if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
          allLines.push(...lines);
        } catch (e) {
          self.Module.printErr(`sort: ${f}: No such file or directory`);
          ret = 1;
        }
      }
      doSort(allLines);
      return ret;
    }
  }

  return null; // Fallback to built-ins in jobs.c
};

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "INIT_LOCALSTORAGE") {
    try {
      self.Module.FS.mkdir("/sys");
    } catch (e) {}
    try {
      self.Module.FS.mkdir("/sys/fs");
    } catch (e) {}
    self.Module.FS.mkdir("/sys/fs/localstorage");
    // Create a readme file to provide help text
    self.Module.FS.writeFile(
      "/sys/fs/localstorage/README.txt",
      "This directory maps to the browser localStorage.\nCreate/update files here to set localStorage keys.\nDelete files to remove keys.\n",
    );
    for (const [key, value] of Object.entries(msg.data)) {
      self.Module.FS.writeFile("/sys/fs/localstorage/" + key, value);
    }

    const originalClose = self.Module.FS.close;
    self.Module.FS.close = function (stream) {
      originalClose.apply(this, arguments);
      if (stream.path && stream.path.startsWith("/sys/fs/localstorage/")) {
        const key = stream.path.split("/").pop();
        if (key !== "README.txt") {
          try {
            const content = self.Module.FS.readFile(stream.path, {
              encoding: "utf8",
            });
            postMessage({
              type: "UPDATE_LOCALSTORAGE",
              key: key,
              value: content,
            });
          } catch (e) {}
        }
      }
    };

    const originalUnlink = self.Module.FS.unlink;
    self.Module.FS.unlink = function (path) {
      originalUnlink.apply(this, arguments);
      if (path.startsWith("/sys/fs/localstorage/")) {
        const key = path.split("/").pop();
        if (key !== "README.txt") {
          postMessage({ type: "DELETE_LOCALSTORAGE", key: key });
        }
      }
    };
  } else if (msg.type === "INPUT") {
    stdinBuffer.push(...msg.data);
    if (self.__DASH_RESOLVE) {
      let rs = self.__DASH_RESOLVE;
      self.__DASH_RESOLVE = null;
      rs();
    }
  } else if (msg.type === "CLIPBOARD_WRITE_ACK") {
    if (globalThis.resolveClipboardWrite) {
      globalThis.resolveClipboardWrite();
      globalThis.resolveClipboardWrite = null;
    }
  } else if (msg.type === "CLIPBOARD_READ_ACK") {
    if (globalThis.resolveClipboardRead) {
      globalThis.resolveClipboardRead(msg.data);
      globalThis.resolveClipboardRead = null;
    }
  } else if (msg.type === "TEST_CMD") {
    const testCmd = "echo test\n";
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
  arguments: ["-l"],
  preRun: [
    () => {
      const FS = self.Module.FS;
      const IDBFS = self.IDBFS || self.Module.IDBFS;
      try {
        FS.mkdir("/etc");
      } catch (e) {}
      try {
        FS.mkdir("/bin");
      } catch (e) {}
      FS.writeFile("/bin/htop", "");
      FS.chmod("/bin/htop", 0o777);
      FS.writeFile("/bin/top", "");
      FS.chmod("/bin/top", 0o777);
      FS.writeFile("/bin/more", "");
      FS.chmod("/bin/more", 0o777);
      FS.writeFile("/bin/less", "");
      FS.chmod("/bin/less", 0o777);
      FS.writeFile("/bin/vim", "");
      FS.chmod("/bin/vim", 0o777);
      FS.writeFile("/bin/vi", "");
      FS.chmod("/bin/vi", 0o777);
      FS.writeFile("/bin/nano", "");
      FS.chmod("/bin/nano", 0o777);
      FS.writeFile("/bin/whoami", "");
      FS.chmod("/bin/whoami", 0o777);
      FS.writeFile("/bin/uname", "");
      FS.chmod("/bin/uname", 0o777);
      FS.writeFile("/bin/file", "");
      FS.chmod("/bin/file", 0o777);
      FS.writeFile("/bin/stat", "");
      FS.chmod("/bin/stat", 0o777);
      FS.writeFile("/bin/df", "");
      FS.chmod("/bin/df", 0o777);
      FS.writeFile("/bin/du", "");
      FS.chmod("/bin/du", 0o777);
      FS.writeFile("/bin/date", "");
      FS.chmod("/bin/date", 0o777);
      FS.writeFile("/bin/su", "");
      FS.chmod("/bin/su", 0o777);
      FS.writeFile("/bin/sudo", "");
      FS.chmod("/bin/sudo", 0o777);
      FS.writeFile("/bin/sleep", "");
      FS.chmod("/bin/sleep", 0o777);
      FS.writeFile("/bin/find", "");
      FS.chmod("/bin/find", 0o777);

      FS.writeFile("/bin/xargs", "");
      FS.chmod("/bin/xargs", 0o777);
      FS.writeFile("/bin/which", "");
      FS.chmod("/bin/which", 0o777);
      FS.writeFile("/bin/jq", "");
      FS.chmod("/bin/jq", 0o777);
      FS.writeFile("/bin/tar", "");
      FS.chmod("/bin/tar", 0o777);
      FS.writeFile("/bin/sh", "");
      FS.writeFile("/bin/env", "");
      FS.chmod("/bin/env", 0o777);

      FS.chmod("/bin/sh", 0o777);
      FS.writeFile("/bin/bash", "");
      FS.chmod("/bin/bash", 0o777);
      FS.writeFile("/bin/tee", "");
      FS.writeFile("/bin/mktemp", "");
      FS.writeFile("/bin/basename", "");
      FS.chmod("/bin/basename", 0o777);
      FS.writeFile("/bin/dirname", "");
      FS.chmod("/bin/dirname", 0o777);
      FS.chmod("/bin/mktemp", 0o777);
      FS.chmod("/bin/tee", 0o777);
      FS.writeFile("/bin/ls", "");
      FS.chmod("/bin/ls", 0o777);
      FS.writeFile("/bin/cat", "");
      FS.chmod("/bin/cat", 0o777);
      FS.writeFile("/bin/cp", "");
      FS.chmod("/bin/cp", 0o777);
      FS.writeFile("/bin/mv", "");
      FS.chmod("/bin/mv", 0o777);
      FS.writeFile("/bin/rm", "");
      FS.chmod("/bin/rm", 0o777);
      FS.writeFile("/bin/mkdir", "");
      FS.chmod("/bin/mkdir", 0o777);
      FS.writeFile("/bin/chmod", "");
      FS.chmod("/bin/chmod", 0o777);
      FS.writeFile("/bin/sed", "");
      FS.chmod("/bin/sed", 0o777);
      FS.writeFile("/bin/cut", "");
      FS.chmod("/bin/cut", 0o777);
      FS.writeFile("/bin/tr", "");
      FS.chmod("/bin/tr", 0o777);
      FS.writeFile("/bin/sort", "");
      FS.chmod("/bin/sort", 0o777);
      FS.writeFile("/bin/head", "");
      FS.chmod("/bin/head", 0o777);
      FS.writeFile("/bin/tail", "");
      FS.chmod("/bin/tail", 0o777);
      FS.writeFile("/bin/cowsay", "");
      FS.chmod("/bin/cowsay", 0o777);

      FS.writeFile("/bin/grep", "");
      FS.writeFile("/bin/awk", "");
      FS.chmod("/bin/grep", 0o777);
      FS.chmod("/bin/awk", 0o777);
      FS.writeFile("/bin/curl", "");
      FS.chmod("/bin/curl", 0o777);
      try {
        FS.mkdir("/tmp");
      } catch (e) {}
      FS.writeFile(
        "/etc/profile",
        'export PATH=/bin\nexport PS1="wasm-shell$ "\n',
      );
      try {
        FS.mkdir("/home");
      } catch (e) {}
      try {
        FS.mkdir("/home/web_user");
      } catch (e) {}

      FS.mount(IDBFS, {}, "/home/web_user");
      self.Module.addRunDependency("syncfs");
      FS.syncfs(true, (err) => {
        if (err) console.error("IDBFS sync error:", err);
        try {
          FS.stat("/home/web_user/.profile");
        } catch (e) {
          FS.writeFile("/home/web_user/.profile", 'alias ll="ls -l"\n');
        }
        self.Module.removeRunDependency("syncfs");
      });
    },
  ],
  print: (text) => {
    self.postMessage({ type: "STDOUT", data: text });
  },
  printErr: (text) => {
    // console.log("STDERR:", text);
    self.postMessage({ type: "STDERR", data: text });
  },
  onExit: (code) => {
    self.postMessage({
      type: "STDOUT",
      data: "\r\n[dash process exited with code " + code + "]\r\n",
    });
  },
  onRuntimeInitialized: () => {
    self.postMessage({ type: "LOADED" });
    const FS = self.Module.FS;
    setInterval(() => {
      FS.syncfs(false, () => {});
    }, 5000);

    if (self.Module.TTY) {
      const put_char = function (tty, val) {
        if (val === null || val === undefined) return;
        self.postMessage({ type: "STDOUT_CHAR", data: val });
      };
      if (self.Module.TTY.default_tty_ops)
        self.Module.TTY.default_tty_ops.put_char = put_char;
      if (self.Module.TTY.default_tty1_ops)
        self.Module.TTY.default_tty1_ops.put_char = put_char;
    }

    const stdinStream = FS.getStream(0);
    if (stdinStream) {
      // We now handle stdin asynchronously directly in dash via EM_ASM_INT in input.c
    }
  },
};

importScripts("./dash.js");
