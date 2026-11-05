/**
 * Historial: contadores, resumen de actividad (analizador), tabla/tarjetas
 * y detalle de cada registro en un modal.
 */
import { HISTORY_LIMIT, PREDICTION } from './api.js';
import { clearHighlight, refreshHistory, subscribe } from './store.js';
import { announce, resultTag, setButtonLoading, toast } from './ui.js';
import { $, $$, el, formatDateTime, formatRelative, icon, singleLine } from './utils.js';

const RECENT_ITEMS = 3;
const pad = (n) => String(n).padStart(2, '0');

export function initHistory({ isViewActive }) {
    const tableWrap = $('#history-table-wrap');
    const tableBody = $('#history-body');
    const loading = $('#history-loading');
    const empty = $('#history-empty');
    const errorBox = $('#history-error');
    const refreshButton = $('#history-refresh');
    const recentList = $('#recent-list');
    const recentEmpty = $('#recent-empty');
    const capacityFill = $('#capacity-fill');

    const modalElement = $('#entry-modal');
    const modal = window.bootstrap.Modal.getOrCreateInstance(modalElement);
    let modalEntry = null;
    let modalTrigger = null;

    /* ---- Detalle ---- */
    function openEntry(entry, trigger) {
        modalEntry = entry;
        modalTrigger = trigger;
        $('#entry-modal-meta').textContent = `Registro N.º ${pad(entry.number)} · ${formatDateTime(entry.date)}`;
        $('#entry-modal-title').textContent = 'Noticia analizada';
        $('#entry-modal-result').replaceChildren(resultTag(entry.prediction));
        $('#entry-modal-text').textContent = entry.text;
        modal.show();
    }

    modalElement.addEventListener('hidden.bs.modal', () => {
        // Si la lista se re-renderizó con el modal abierto, se busca el control equivalente del mismo registro.
        const target = modalTrigger?.isConnected
            ? modalTrigger
            : $(`[data-entry="${modalEntry?.number}"][data-origin="${modalTrigger?.dataset.origin}"]`);
        (target || $('#history-table-title'))?.focus?.();
        modalTrigger = null;
    });

    $('#entry-copy').addEventListener('click', async () => {
        if (!modalEntry) return;
        try {
            await navigator.clipboard.writeText(modalEntry.text);
            toast('Texto copiado al portapapeles.', 'success');
        } catch {
            toast('No se pudo copiar el texto. Selecciónalo y cópialo manualmente.', 'error');
        }
    });

    /* ---- Render ---- */
    function renderCounts(entries) {
        for (const node of $$('[data-history-count]')) node.textContent = entries.length;
        capacityFill.style.width = `${(entries.length / HISTORY_LIMIT) * 100}%`;

        $('[data-stat="total"]').textContent = entries.length;
        $('[data-stat="falsa"]').textContent = entries.filter((e) => e.prediction === PREDICTION.FAKE).length;
        $('[data-stat="verdadera"]').textContent = entries.filter((e) => e.prediction === PREDICTION.TRUE).length;
    }

    function renderRecent(entries) {
        const latest = entries.slice(-RECENT_ITEMS).reverse();
        recentEmpty.hidden = latest.length > 0;
        recentList.hidden = latest.length === 0;
        recentList.replaceChildren(...latest.map((entry) => {
            const button = el('button', {
                className: 'activity__item',
                dataset: { entry: entry.number, origin: 'recent' },
                attrs: { type: 'button', 'aria-label': `Ver registro ${entry.number}: ${singleLine(entry.text).slice(0, 80)}` },
            }, [
                el('span', { className: 'activity__text', text: singleLine(entry.text) }),
                el('time', { className: 'activity__time', text: formatRelative(entry.date), attrs: { datetime: entry.timestamp } }),
                resultTag(entry.prediction),
            ]);
            button.addEventListener('click', () => openEntry(entry, button));
            return el('li', {}, [button]);
        }));
    }

    function renderRow(entry, highlight) {
        const viewButton = el('button', {
            className: 'icon-button icon-button--sm',
            dataset: { entry: entry.number, origin: 'table' },
            attrs: { type: 'button', 'aria-label': `Ver noticia completa del registro ${entry.number}`, title: 'Ver noticia completa' },
        }, [icon('arrows-angle-expand')]);
        viewButton.addEventListener('click', () => openEntry(entry, viewButton));

        const text = singleLine(entry.text);
        return el('tr', { className: highlight ? 'is-new' : '' }, [
            el('td', { className: 'cell-num', text: pad(entry.number) }),
            el('td', { className: 'cell-date' }, [
                entry.date
                    ? el('time', { text: formatDateTime(entry.date), attrs: { datetime: entry.timestamp } })
                    : el('span', { text: 'Sin fecha' }),
            ]),
            el('td', { className: 'cell-text' }, [
                el('span', { className: 'cell-text__inner', text, attrs: { title: text.length > 300 ? `${text.slice(0, 300)}…` : text } }),
            ]),
            el('td', { className: 'cell-result' }, [resultTag(entry.prediction)]),
            el('td', { className: 'cell-actions' }, [viewButton]),
        ]);
    }

    function renderTable(state) {
        const { entries, status } = state;
        const hasEntries = entries.length > 0;

        loading.hidden = !(status === 'loading' && !hasEntries);
        errorBox.hidden = !(status === 'error' && !hasEntries);
        empty.hidden = !(status === 'ready' && !hasEntries);
        tableWrap.hidden = !hasEntries;

        if (!errorBox.hidden) $('[data-error-text]', errorBox).textContent = state.error;

        // Durante un refresco los datos no han cambiado todavía: no se recrean las filas.
        if (status === 'refreshing') return;

        // Más reciente primero; el número conserva el orden cronológico (igual que el PDF).
        tableBody.replaceChildren(...entries.slice().reverse().map((entry) => renderRow(entry, entry.number === state.highlight)));

        if (state.highlight != null && isViewActive('historial')) clearHighlight();
    }

    subscribe((state) => {
        renderCounts(state.entries);
        renderRecent(state.entries);
        renderTable(state);
    });

    /* ---- Acciones ---- */
    refreshButton.addEventListener('click', async () => {
        setButtonLoading(refreshButton, true);
        const ok = await refreshHistory();
        setButtonLoading(refreshButton, false);
        if (ok) announce('Historial actualizado.');
        else toast('No se pudo actualizar el historial.', 'error');
    });

    $('[data-action="history-retry"]').addEventListener('click', () => refreshHistory());
}
