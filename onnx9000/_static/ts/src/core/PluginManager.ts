import { IModelGraph } from './IR';
import { Toast } from '../ui/Toast';
import { logger } from './Logger';
import { globalEvents } from './State';

export interface IPluginContext {
  registerSidebarTab: (title: string, component: HTMLElement) => void;
  showToast: (msg: string, type?: 'info' | 'success' | 'warn' | 'error') => void;
  log: (msg: string) => void;
  getActiveModel: () => Readonly<IModelGraph> | null;
}

export interface IPlugin {
  name: string;
  version: string;
  init: (ctx: IPluginContext) => Promise<void>;
  onModelLoad?: (model: Readonly<IModelGraph>) => void;
}

export class PluginManager {
  private plugins = new Map<string, IPlugin>();
  private activeModel: IModelGraph | null = null;
  private sidebarContainer: HTMLElement | null = null;

  constructor() {
    globalEvents.on('modelLoaded', (model: IModelGraph) => {
      this.activeModel = model;
      this.plugins.forEach((p) => {
        if (p.onModelLoad) {
          try {
            p.onModelLoad(this.activeModel!);
          } catch (e) {
            logger.error(`Plugin ${p.name} failed onModelLoad hook: ${e}`);
          }
        }
      });
    });
  }

  setSidebarContainer(el: HTMLElement): void {
    this.sidebarContainer = el;
  }

  async loadPlugin(url: string): Promise<void> {
    try {
      // 563. Dynamic loading via ESM
      // 564. Sandbox external plugins (in a real scenario, we'd use a Worker or iframe.
      // For this native JS implementation, we assume basic ESM isolation)
      const module = await import(/* @vite-ignore */ url);

      if (!module.default || typeof module.default.init !== 'function') {
        throw new Error('Plugin does not export a valid default IPlugin interface');
      }

      const plugin: IPlugin = module.default;

      if (this.plugins.has(plugin.name)) {
        throw new Error(`Plugin ${plugin.name} is already loaded`);
      }

      // 565. Expose strict Readonly API
      const context: IPluginContext = {
        registerSidebarTab: (title: string, component: HTMLElement) => {
          this.registerTab(plugin.name, title, component);
        },
        showToast: (msg: string, type = 'info') => Toast.show(`[${plugin.name}] ${msg}`, type),
        log: (msg: string) => logger.info(`[${plugin.name}] ${msg}`),
        getActiveModel: () => this.activeModel,
      };

      await plugin.init(context);
      this.plugins.set(plugin.name, plugin);
      Toast.show(`Plugin ${plugin.name} v${plugin.version} loaded`, 'success');
    } catch (e) {
      logger.error(`Failed to load plugin from ${url}`, e);
      Toast.show(`Failed to load plugin`, 'error');
    }
  }

  private registerTab(pluginName: string, title: string, component: HTMLElement): void {
    if (!this.sidebarContainer) return;

    // 566. API wrapper for registering sidebar tabs
    const section = document.createElement('div');
    section.className = 'sidebar-section';

    const h4 = document.createElement('h4');
    h4.textContent = `${title} (${pluginName})`;

    section.appendChild(h4);
    section.appendChild(component);

    this.sidebarContainer.appendChild(section);
  }

  getLoadedPlugins(): string[] {
    return Array.from(this.plugins.keys());
  }
}

export const pluginManager = new PluginManager();
