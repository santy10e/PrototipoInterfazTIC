/** Utilidades de formato y DOM compartidas. */

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

export const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const numberFormat = new Intl.NumberFormat('es');
export const formatNumber = (value) => numberFormat.format(value);

/** Cuenta palabras con la misma noción de "palabra" que un lector humano. */
export function countWords(text) {
    const trimmed = text.trim();
    return trimmed ? trimmed.split(/\s+/).length : 0;
}

export function plural(count, singular, pluralForm = `${singular}s`) {
    return `${formatNumber(count)} ${count === 1 ? singular : pluralForm}`;
}

/** Réplica de truncate_text() de modelo.py, para previsualizar el PDF con fidelidad. */
export function truncateWords(text, maxWords = 6) {
    const words = text.split(' ');
    return words.length > maxWords ? `${words.slice(0, maxWords).join(' ')}...` : text;
}

/** Convierte saltos de línea y espacios repetidos en un único espacio (para fragmentos de una línea). */
export const singleLine = (text) => text.replace(/\s+/g, ' ').trim();

const dateTimeFormat = new Intl.DateTimeFormat('es', { dateStyle: 'medium', timeStyle: 'short' });
const relativeFormat = new Intl.RelativeTimeFormat('es', { numeric: 'auto' });

/** @returns {Date|null} */
export function parseTimestamp(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
}

export const formatDateTime = (date) => (date ? dateTimeFormat.format(date) : 'Sin fecha');

export function formatRelative(date) {
    if (!date) return 'Sin fecha';
    const seconds = Math.round((date.getTime() - Date.now()) / 1000);
    const abs = Math.abs(seconds);
    if (abs < 45) return 'Hace un momento';
    if (abs < 3600) return capitalize(relativeFormat.format(Math.round(seconds / 60), 'minute'));
    if (abs < 86400) return capitalize(relativeFormat.format(Math.round(seconds / 3600), 'hour'));
    return formatDateTime(date);
}

const capitalize = (text) => text.charAt(0).toUpperCase() + text.slice(1);

/**
 * Crea un elemento de forma segura (el contenido de texto nunca se interpreta como HTML).
 * @param {string} tag
 * @param {Object} [props] className, text, attrs, dataset
 * @param {Array<Node|string>} [children]
 */
export function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    if (props.className) node.className = props.className;
    if (props.text != null) node.textContent = props.text;
    if (props.attrs) {
        for (const [key, value] of Object.entries(props.attrs)) {
            if (value != null && value !== false) node.setAttribute(key, value === true ? '' : value);
        }
    }
    if (props.dataset) Object.assign(node.dataset, props.dataset);
    for (const child of children) {
        if (child != null) node.append(child);
    }
    return node;
}

export const icon = (name) => el('i', { className: `bi bi-${name}`, attrs: { 'aria-hidden': 'true' } });
