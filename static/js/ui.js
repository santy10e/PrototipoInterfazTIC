/** Componentes de feedback compartidos: etiquetas de resultado, avisos, estado del servicio. */
import { PREDICTION } from './api.js';
import { $, $$, el, icon } from './utils.js';

const RESULT_META = {
    [PREDICTION.FAKE]: { label: 'Falsa', icon: 'exclamation-triangle-fill', className: 'tag--fake' },
    [PREDICTION.TRUE]: { label: 'Verdadera', icon: 'check-circle-fill', className: 'tag--true' },
};

/** Etiqueta de resultado con icono + texto (no depende solo del color). */
export function resultTag(prediction) {
    const meta = RESULT_META[prediction] || { label: prediction, icon: 'dash-circle', className: 'tag--neutral' };
    return el('span', { className: `tag ${meta.className}` }, [icon(meta.icon), el('span', { text: meta.label })]);
}

/** Anuncia un mensaje a lectores de pantalla sin cambiar el foco. */
export function announce(message) {
    const region = $('#sr-announcer');
    if (!region) return;
    region.textContent = '';
    // Pequeño retraso para que el lector detecte el cambio aunque el texto se repita.
    window.setTimeout(() => { region.textContent = message; }, 60);
}

const TOAST_ICONS = { success: 'check-circle-fill', error: 'exclamation-octagon-fill', info: 'info-circle-fill' };

/** Aviso temporal no bloqueante. */
export function toast(message, type = 'info', duration = 4200) {
    const region = $('#toast-region');
    if (!region) return;
    const item = el('div', { className: `toast-item toast-item--${type}`, attrs: { role: type === 'error' ? 'alert' : 'status' } }, [
        icon(TOAST_ICONS[type] || TOAST_ICONS.info),
        el('span', { text: message }),
    ]);
    region.append(item);
    window.setTimeout(() => {
        item.classList.add('is-leaving');
        item.addEventListener('animationend', () => item.remove(), { once: true });
        window.setTimeout(() => item.remove(), 400); // por si las animaciones están desactivadas
    }, duration);
}

const STATUS_TEXT = {
    checking: 'Comprobando servicio…',
    online: 'Servicio disponible',
    offline: 'Servidor no disponible',
};

/**
 * Refleja la disponibilidad real del servidor, determinada por el resultado
 * de las peticiones al backend (no se presupone).
 * @param {'checking'|'online'|'offline'} status
 */
export function setServiceStatus(status) {
    for (const pill of $$('[data-service-status]')) {
        pill.dataset.serviceStatus = status;
        pill.title = STATUS_TEXT[status];
        const text = $('[data-service-status-text]', pill);
        if (text) text.textContent = STATUS_TEXT[status];
    }
}

/** Activa/desactiva el estado de carga de un botón conservando su etiqueta original. */
export function setButtonLoading(button, loading, loadingLabel) {
    const label = $('[data-label]', button);
    if (label && !label.dataset.original) label.dataset.original = label.textContent;
    button.classList.toggle('is-loading', loading);
    button.disabled = loading;
    button.setAttribute('aria-busy', String(loading));
    if (label) label.textContent = loading && loadingLabel ? loadingLabel : label.dataset.original;
}
