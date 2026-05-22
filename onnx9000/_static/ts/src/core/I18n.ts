import { globalEvents } from './State';

/**
 * 636. Add internationalization (i18n) support, loading locale JSONs
 */
export class I18nManager {
  private currentLocale = 'en';
  private locales: Record<string, Record<string, string>> = {
    en: {
      'nav.upload': 'Upload Model',
      'nav.surgeon': 'Graph Surgeon',
      'nav.benchmark': 'Micro-Benchmarks',
      'toast.loaded': 'Model loaded successfully',
      'action.execute': 'Run Inference',
      'action.download': 'Download',
    },
    es: {
      'nav.upload': 'Subir Modelo',
      'nav.surgeon': 'Cirujano de Grafos',
      'nav.benchmark': 'Micro-Puntos de Referencia',
      'toast.loaded': 'Modelo cargado con éxito',
      'action.execute': 'Ejecutar Inferencia',
      'action.download': 'Descargar',
    },
  };

  public setLocale(locale: string): void {
    if (this.locales[locale]) {
      this.currentLocale = locale;
      globalEvents.emit('localeChanged', locale);
    }
  }

  public t(key: string): string {
    const dict = this.locales[this.currentLocale];
    return dict[key] || key;
  }
}

export const i18n = new I18nManager();
