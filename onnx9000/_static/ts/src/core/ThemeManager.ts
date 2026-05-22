import { globalEvents } from './State';

export class ThemeManager {
  private mediaQuery: MediaQueryList;

  constructor() {
    this.mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
  }

  init(): void {
    // Furo theme changes 'data-theme' on the body/html element.
    // We observe that or the media query.
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      this.setTheme(savedTheme as 'light' | 'dark' | 'auto');
    } else {
      this.setTheme('auto');
    }

    this.mediaQuery.addEventListener('change', (e) => {
      if (localStorage.getItem('theme') === 'auto' || !localStorage.getItem('theme')) {
        this.applyTheme(e.matches ? 'dark' : 'light');
      }
    });

    // Observer for body data-theme attribute
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
          const newTheme = document.body.getAttribute('data-theme');
          if (newTheme === 'light' || newTheme === 'dark') {
            this.applyTheme(newTheme);
          }
        }
      }
    });

    observer.observe(document.body, { attributes: true });
  }

  setTheme(theme: 'light' | 'dark' | 'auto'): void {
    localStorage.setItem('theme', theme);
    if (theme === 'auto') {
      this.applyTheme(this.mediaQuery.matches ? 'dark' : 'light');
    } else {
      this.applyTheme(theme);
    }
  }

  private applyTheme(theme: 'light' | 'dark'): void {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.setAttribute('data-theme', theme);
    globalEvents.emit('themeChanged', theme);
  }
}

export const themeManager = new ThemeManager();
