import { globalEvents } from './State';

export enum LogLevel {
  DEBUG = 0,
  INFO = 1,
  WARN = 2,
  ERROR = 3,
}

export interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: number;
}

export class Logger {
  private level: LogLevel = LogLevel.INFO;
  private originalConsole: typeof console;

  constructor() {
    this.originalConsole = { ...console };
  }

  setLevel(level: LogLevel): void {
    this.level = level;
  }

  intercept(): void {
    console.log = (...args: unknown[]) => {
      this.originalConsole.log(...args);
      this.log(LogLevel.INFO, args.join(' '));
    };
    console.warn = (...args: unknown[]) => {
      this.originalConsole.warn(...args);
      this.log(LogLevel.WARN, args.join(' '));
    };
    console.error = (...args: unknown[]) => {
      this.originalConsole.error(...args);
      this.log(LogLevel.ERROR, args.join(' '));
    };
    console.info = (...args: unknown[]) => {
      this.originalConsole.info(...args);
      this.log(LogLevel.INFO, args.join(' '));
    };
    console.debug = (...args: unknown[]) => {
      this.originalConsole.debug(...args);
      this.log(LogLevel.DEBUG, args.join(' '));
    };
  }

  private log(level: LogLevel, message: string): void {
    if (level < this.level) return;
    const entry: LogEntry = {
      level,
      message,
      timestamp: Date.now(),
    };
    globalEvents.emit('log', entry);
  }
}

export const logger = new Logger();
