/**
 * Estado compartido del historial. Fuente única: GET /get_predictions.
 * Las vistas (actividad, historial, informes) se suscriben a los cambios.
 */
import { getPredictions, HISTORY_LIMIT } from './api.js';
import { setServiceStatus } from './ui.js';
import { parseTimestamp } from './utils.js';

/**
 * @typedef {Object} HistoryEntry
 * @property {number} number      Posición cronológica (1 = más antigua), igual que en el PDF.
 * @property {string} text
 * @property {string} prediction  'falsa' | 'verdadera'
 * @property {Date|null} date
 * @property {string|null} timestamp
 */

let state = {
    /** @type {HistoryEntry[]} */
    entries: [],
    /** 'idle' | 'loading' | 'refreshing' | 'ready' | 'error' */
    status: 'idle',
    error: null,
    /** Número de registro que debe resaltarse (recién añadido desde esta pestaña). */
    highlight: null,
};

const listeners = new Set();
let pending = null;

export const getState = () => state;

export function subscribe(listener) {
    listeners.add(listener);
    listener(state);
    return () => listeners.delete(listener);
}

function setState(patch) {
    state = { ...state, ...patch };
    for (const listener of listeners) listener(state);
}

export function clearHighlight() {
    if (state.highlight != null) state = { ...state, highlight: null };
}

/**
 * Vuelve a cargar el historial desde el servidor. Las llamadas simultáneas se agrupan.
 * @param {{highlightNewest?: boolean}} [options]
 */
export function refreshHistory({ highlightNewest = false } = {}) {
    if (pending) return pending;

    setState({ status: state.status === 'ready' ? 'refreshing' : 'loading', error: null });

    pending = getPredictions(HISTORY_LIMIT)
        .then((predictions) => {
            const entries = predictions.map((item, index) => ({
                number: index + 1,
                text: String(item.text ?? ''),
                prediction: item.prediction,
                timestamp: item.timestamp ?? null,
                date: parseTimestamp(item.timestamp),
            }));
            setServiceStatus('online');
            setState({
                entries,
                status: 'ready',
                highlight: highlightNewest && entries.length ? entries[entries.length - 1].number : state.highlight,
            });
            return true;
        })
        .catch((error) => {
            if (error.isNetwork) setServiceStatus('offline');
            setState({ status: 'error', error: error.message });
            return false;
        })
        .finally(() => { pending = null; });

    return pending;
}
