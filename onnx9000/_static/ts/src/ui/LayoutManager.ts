import { BaseComponent } from './BaseComponent';
import { $, $on, $off } from '../core/DOM';

export class LayoutManager extends BaseComponent {
  private sidebar: HTMLElement;
  private bottomPanel: HTMLElement;
  private resizerV: HTMLElement;
  private resizerH: HTMLElement;

  private isResizingV = false;
  private isResizingH = false;

  constructor(containerId: string) {
    super(containerId);
    this.sidebar = $<HTMLElement>('#ide-sidebar', this.container)!;
    this.bottomPanel = $<HTMLElement>('#ide-bottom', this.container)!;
    this.resizerV = $<HTMLElement>('#resizer-v', this.container)!;
    this.resizerH = $<HTMLElement>('#resizer-h', this.container)!;
  }

  mount(): void {
    const savedSidebarWidth = localStorage.getItem('ide-sidebar-width');
    if (savedSidebarWidth) {
      this.sidebar.style.width = `${savedSidebarWidth}px`;
    }

    const savedBottomHeight = localStorage.getItem('ide-bottom-height');
    if (savedBottomHeight) {
      this.bottomPanel.style.height = `${savedBottomHeight}px`;
    }

    this.bindEvent(this.resizerV, 'mousedown', this.onMouseDownV.bind(this));
    this.bindEvent(this.resizerH, 'mousedown', this.onMouseDownH.bind(this));
  }

  private onMouseDownV(e: Event): void {
    const mouseEvent = e as MouseEvent;
    mouseEvent.preventDefault();
    this.isResizingV = true;
    this.resizerV.classList.add('is-resizing');

    const onMouseMove = this.onMouseMoveV.bind(this);
    const onMouseUp = () => {
      this.isResizingV = false;
      this.resizerV.classList.remove('is-resizing');
      localStorage.setItem('ide-sidebar-width', this.sidebar.style.width.replace('px', ''));
      $off(document, 'mousemove', onMouseMove);
      $off(document, 'mouseup', onMouseUp);
      window.dispatchEvent(new Event('resize'));
    };

    $on(document, 'mousemove', onMouseMove);
    $on(document, 'mouseup', onMouseUp);
  }

  private onMouseMoveV(e: Event): void {
    if (!this.isResizingV) return;
    const mouseEvent = e as MouseEvent;
    const containerRect = this.container.getBoundingClientRect();
    const newWidth = mouseEvent.clientX - containerRect.left;
    if (newWidth > 150 && newWidth < 500) {
      this.sidebar.style.width = `${newWidth}px`;
    }
  }

  private onMouseDownH(e: Event): void {
    const mouseEvent = e as MouseEvent;
    mouseEvent.preventDefault();
    this.isResizingH = true;
    this.resizerH.classList.add('is-resizing');

    const onMouseMove = this.onMouseMoveH.bind(this);
    const onMouseUp = () => {
      this.isResizingH = false;
      this.resizerH.classList.remove('is-resizing');
      localStorage.setItem('ide-bottom-height', this.bottomPanel.style.height.replace('px', ''));
      $off(document, 'mousemove', onMouseMove);
      $off(document, 'mouseup', onMouseUp);
      window.dispatchEvent(new Event('resize'));
    };

    $on(document, 'mousemove', onMouseMove);
    $on(document, 'mouseup', onMouseUp);
  }

  private onMouseMoveH(e: Event): void {
    if (!this.isResizingH) return;
    const mouseEvent = e as MouseEvent;
    const containerRect = this.container.getBoundingClientRect();
    const newHeight = containerRect.bottom - mouseEvent.clientY;
    if (newHeight > 100 && newHeight < 600) {
      this.bottomPanel.style.height = `${newHeight}px`;
    }
  }
}
