/**
 * Navegación: vistas por hash (#analizar, #historial…), sidebar colapsable
 * en escritorio/tablet y drawer accesible en móvil.
 */
import { $, $$ } from './utils.js';

const DEFAULT_VIEW = 'analizar';
const SIDEBAR_KEY = 'news-intelligence.sidebar';
const mobileQuery = window.matchMedia('(max-width: 767.98px)');
const desktopQuery = window.matchMedia('(min-width: 1200px)');

function readPreference() {
    try { return localStorage.getItem(SIDEBAR_KEY); } catch { return null; }
}
function savePreference(value) {
    try { localStorage.setItem(SIDEBAR_KEY, value); } catch { /* almacenamiento no disponible */ }
}

export function initNavigation({ onViewChange } = {}) {
    const app = $('#app');
    const shell = $('.shell');
    const sidebar = $('#sidebar');
    const backdrop = $('.drawer-backdrop');
    const openButton = $('#drawer-open');
    const collapseButton = $('#sidebar-collapse');
    const views = new Map($$('[data-view]').map((view) => [view.dataset.view, view]));
    let current = null;

    /* ---- Sidebar colapsable ---- */
    function syncCollapseButton() {
        const expanded = app.dataset.sidebar !== 'collapsed';
        collapseButton.setAttribute('aria-expanded', String(expanded));
        $('.side-link__text', collapseButton).textContent = expanded ? 'Contraer menú' : 'Expandir menú';
        collapseButton.dataset.tooltip = expanded ? 'Contraer menú' : 'Expandir menú';
    }

    function applyDefaultSidebar() {
        app.dataset.sidebar = readPreference() || (desktopQuery.matches ? 'expanded' : 'collapsed');
        syncCollapseButton();
    }

    collapseButton.addEventListener('click', () => {
        const next = app.dataset.sidebar === 'collapsed' ? 'expanded' : 'collapsed';
        app.dataset.sidebar = next;
        savePreference(next);
        syncCollapseButton();
    });

    desktopQuery.addEventListener('change', () => { if (!readPreference()) applyDefaultSidebar(); });
    applyDefaultSidebar();

    /* ---- Drawer móvil ---- */
    const isDrawerOpen = () => app.classList.contains('drawer-open');

    function openDrawer() {
        app.classList.add('drawer-open');
        backdrop.hidden = false;
        openButton.setAttribute('aria-expanded', 'true');
        shell.inert = true;
        document.body.style.overflow = 'hidden';
        const target = $('.side-link[aria-current="page"]', sidebar) || $('.side-link', sidebar);
        window.requestAnimationFrame(() => target?.focus());
    }

    function closeDrawer(returnFocus = true) {
        if (!isDrawerOpen()) return;
        app.classList.remove('drawer-open');
        backdrop.hidden = true;
        openButton.setAttribute('aria-expanded', 'false');
        shell.inert = false;
        document.body.style.overflow = '';
        if (returnFocus) openButton.focus();
    }

    openButton.addEventListener('click', openDrawer);
    for (const closer of $$('[data-drawer-close]')) closer.addEventListener('click', () => closeDrawer());
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && isDrawerOpen()) closeDrawer();
    });
    mobileQuery.addEventListener('change', (event) => { if (!event.matches) closeDrawer(false); });

    /* ---- Vistas ---- */
    function show(name, { focus = false } = {}) {
        if (!views.has(name)) name = DEFAULT_VIEW;
        closeDrawer(false);
        if (name === current) return;

        current = name;
        for (const [key, view] of views) view.hidden = key !== name;
        for (const link of $$('[data-view-link]')) {
            if (link.dataset.viewLink === name) link.setAttribute('aria-current', 'page');
            else link.removeAttribute('aria-current');
        }

        const view = views.get(name);
        document.title = `${view.dataset.title} · News Intelligence`;

        if (focus) {
            window.scrollTo({ top: 0, behavior: 'instant' });
            $('.page-title', view)?.focus({ preventScroll: true });
        }
        onViewChange?.(name);
    }

    window.addEventListener('hashchange', () => {
        const name = location.hash.slice(1);
        // Ignora anclas que no son vistas (p. ej. #main del enlace de salto).
        if (views.has(name)) show(name, { focus: true });
    });

    // Enlace "Saltar al contenido": mueve el foco sin alterar la vista activa.
    $('.skip-link')?.addEventListener('click', (event) => {
        event.preventDefault();
        $('.page-title', views.get(current))?.focus();
    });

    show(location.hash.slice(1) || DEFAULT_VIEW);

    return { show, current: () => current };
}
