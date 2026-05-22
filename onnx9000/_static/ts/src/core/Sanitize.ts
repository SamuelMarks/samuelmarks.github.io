export function escapeHTML(str: string): string {
  if (!str) return '';
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

export function assertNotNull<T>(val: T | null | undefined, message?: string): T {
  if (val === null || val === undefined) {
    throw new Error(message || 'Assertion failed: Expected value not to be null or undefined');
  }
  return val;
}
