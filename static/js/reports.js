/**
 * Informes PDF: selección del número de predicciones, vista previa fiel
 * del contenido y descarga del PDF generado por ReportLab (/download_report).
 */
import { fetchReport } from './api.js';
import { getState, subscribe } from './store.js';
import { announce, resultTag, setButtonLoading, setServiceStatus, toast } from './ui.js';
import { $, el, icon, plural, singleLine, truncateWords } from './utils.js';

const pad = (n) => String(n).padStart(2, '0');

export function initReports() {
    const form = $('#report-form');
    const button = $('#report-btn');
    const summary = $('#report-summary-text');
    const status = $('#report-status');
    const previewList = $('#report-preview-list');
    const emptyBox = $('#report-empty');
    let selected = Number(new FormData(form).get('num_predictions')) || 5;
    let downloading = false;

    const includedCount = () => Math.min(selected, getState().entries.length);

    function setStatus(message, type = 'info') {
        status.className = `form-status is-${type}`;
        status.replaceChildren();
        if (!message) return;
        const iconName = { success: 'check-circle-fill', error: 'exclamation-circle-fill', info: 'hourglass-split' }[type];
        status.append(icon(iconName), el('span', { text: message }));
    }

    function render(state) {
        const available = state.entries.length;
        const included = Math.min(selected, available);

        if (available === 0) {
            summary.replaceChildren(state.status === 'loading' || state.status === 'idle'
                ? 'Cargando el historial…'
                : 'No hay predicciones en el historial.');
        } else {
            summary.replaceChildren(
                'Se incluirán ',
                el('strong', { text: String(included) }),
                ` de ${plural(available, 'predicción disponible', 'predicciones disponibles')}.`,
            );
        }

        // Mismo criterio que el backend: predictions_history[:num_predictions]
        const items = state.entries.slice(0, included);
        previewList.hidden = items.length === 0;
        emptyBox.hidden = items.length > 0 || state.status === 'loading' || state.status === 'idle';
        previewList.replaceChildren(...items.map((entry) => el('li', {}, [
            el('span', { className: 'preview-list__n', text: pad(entry.number) }),
            el('span', { className: 'preview-list__text', text: singleLine(truncateWords(entry.text)) }),
            resultTag(entry.prediction),
        ])));

        if (!downloading) button.disabled = available === 0;
    }

    function saveBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = el('a', { attrs: { href: url, download: filename, hidden: true } });
        document.body.append(link);
        link.click();
        link.remove();
        window.setTimeout(() => URL.revokeObjectURL(url), 2000);
    }

    async function download() {
        const included = includedCount();
        if (downloading || included === 0) return;

        downloading = true;
        setButtonLoading(button, true, 'Generando PDF…');
        setStatus('Generando el informe en el servidor…', 'info');

        try {
            const { blob, filename } = await fetchReport(selected);
            saveBlob(blob, filename);
            setServiceStatus('online');
            const message = `Informe descargado: ${filename} (${plural(included, 'predicción', 'predicciones')}).`;
            setStatus(message, 'success');
            toast('Informe PDF descargado.', 'success');
            announce(message);
        } catch (error) {
            if (error.isNetwork) setServiceStatus('offline');
            setStatus(error.message, 'error');
            toast('No se pudo generar el informe.', 'error');
        } finally {
            downloading = false;
            setButtonLoading(button, false);
            render(getState());
        }
    }

    form.addEventListener('change', (event) => {
        if (event.target.name !== 'num_predictions') return;
        selected = Number(event.target.value);
        setStatus('');
        render(getState());
    });

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        download();
    });

    subscribe(render);
}
