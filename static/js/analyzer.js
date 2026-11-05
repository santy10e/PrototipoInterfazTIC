/**
 * Módulo del analizador: entrada, validación, envío a /predict y
 * presentación del resultado (vacío, cargando, clasificación, no procesable, error).
 */
import { predict, PREDICTION } from './api.js';
import { announce, setButtonLoading, setServiceStatus } from './ui.js';
import { $, $$, countWords, plural, prefersReducedMotion } from './utils.js';

/** Textos del veredicto. No se añaden motivos ni porcentajes: el backend no los proporciona. */
const VERDICTS = {
    [PREDICTION.FAKE]: {
        label: 'Posible noticia falsa',
        icon: 'bi-exclamation-triangle-fill',
        title: 'El modelo clasificó esta noticia como falsa',
        description: 'La clasificación la realiza el modelo de regresión logística a partir del texto introducido. Contrasta la información con fuentes fiables antes de darla por cierta o compartirla.',
        announcement: 'Análisis completado. Resultado: posible noticia falsa.',
    },
    [PREDICTION.TRUE]: {
        label: 'Clasificada como verdadera',
        icon: 'bi-patch-check-fill',
        title: 'El modelo clasificó esta noticia como verdadera',
        description: 'La clasificación la realiza el modelo de regresión logística a partir del texto introducido. Que el modelo la clasifique como verdadera no garantiza que su contenido sea exacto.',
        announcement: 'Análisis completado. Resultado: clasificada como verdadera.',
    },
};

/** Tiempo mínimo del estado de carga para evitar un parpadeo cuando la respuesta es instantánea. */
const MIN_LOADING_MS = 250;
const wait = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));

export function initAnalyzer({ onPredicted } = {}) {
    const form = $('#analyze-form');
    const textarea = $('#news-text');
    const counter = $('#char-count');
    const clearButton = $('#clear-btn');
    const analyzeButton = $('#analyze-btn');
    const errorBox = $('#news-error');
    const panel = $('#result-panel');
    let busy = false;

    /* ---- Entrada ---- */
    function updateCounter() {
        const text = textarea.value;
        counter.textContent = `${plural(text.length, 'carácter', 'caracteres')} · ${plural(countWords(text), 'palabra')}`;
        clearButton.disabled = busy || text.length === 0;
    }

    function showError(message) {
        $('[data-error-text]', errorBox).textContent = message;
        errorBox.hidden = false;
        form.classList.add('has-error');
        textarea.setAttribute('aria-invalid', 'true');
        announce(message);
    }

    function clearError() {
        errorBox.hidden = true;
        form.classList.remove('has-error');
        textarea.removeAttribute('aria-invalid');
    }

    /* ---- Panel de resultado ---- */
    function setResultState(name) {
        panel.dataset.state = name;
        panel.setAttribute('aria-busy', String(name === 'loading'));
        if (name !== 'outcome') delete panel.dataset.verdict;
        for (const view of $$('[data-state-view]', panel)) view.hidden = view.dataset.stateView !== name;
    }

    function renderVerdict(prediction, text) {
        const verdict = VERDICTS[prediction];
        panel.dataset.verdict = prediction;
        $('#verdict').dataset.verdict = prediction;
        $('#verdict-icon').className = `bi ${verdict.icon}`;
        $('#verdict-label').textContent = verdict.label;
        $('#verdict-title').textContent = verdict.title;
        $('#verdict-desc').textContent = verdict.description;
        $('#analyzed-text').textContent = text.trim();
        $('#analyzed-meta').textContent = plural(countWords(text), 'palabra');
    }

    /** En pantallas donde el resultado queda debajo del formulario, lo lleva a la vista. */
    function revealResult() {
        const rect = panel.getBoundingClientRect();
        if (rect.top > window.innerHeight * 0.55 || rect.bottom < 0) {
            panel.scrollIntoView({ behavior: prefersReducedMotion() ? 'auto' : 'smooth', block: 'start' });
        }
    }

    /* ---- Análisis ---- */
    async function analyze() {
        if (busy) return;
        const text = textarea.value;

        if (!text.trim()) {
            showError('Introduce el contenido de una noticia para analizarla.');
            textarea.focus();
            return;
        }

        busy = true;
        clearError();
        setButtonLoading(analyzeButton, true, 'Analizando…');
        textarea.readOnly = true;
        updateCounter();
        setResultState('loading');
        announce('Analizando la noticia…');
        revealResult();

        try {
            const [{ prediction }] = await Promise.all([predict(text), wait(prefersReducedMotion() ? 0 : MIN_LOADING_MS)]);
            setServiceStatus('online');

            if (VERDICTS[prediction]) {
                renderVerdict(prediction, text);
                setResultState('outcome');
                announce(VERDICTS[prediction].announcement);
                onPredicted?.(prediction);
            } else {
                // El backend devuelve un mensaje cuando el texto queda vacío tras el preprocesado.
                setResultState('unprocessable');
                announce('No se pudo procesar la noticia. Prueba con un texto más extenso.');
            }
        } catch (error) {
            if (error.isNetwork) setServiceStatus('offline');
            $('#result-error-text').textContent = error.message;
            setResultState('error');
            announce(`No se pudo completar el análisis. ${error.message}`);
        } finally {
            busy = false;
            textarea.readOnly = false;
            setButtonLoading(analyzeButton, false);
            updateCounter();
        }
    }

    function resetComposer() {
        textarea.value = '';
        clearError();
        updateCounter();
    }

    /* ---- Eventos ---- */
    form.addEventListener('submit', (event) => {
        event.preventDefault();
        analyze();
    });

    textarea.addEventListener('input', () => {
        if (!errorBox.hidden && textarea.value.trim()) clearError();
        updateCounter();
    });

    textarea.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            analyze();
        }
    });

    clearButton.addEventListener('click', () => {
        resetComposer();
        textarea.focus();
        announce('Contenido eliminado.');
    });

    panel.addEventListener('click', (event) => {
        const action = event.target.closest('[data-action]')?.dataset.action;
        if (action === 'new-analysis') {
            resetComposer();
            setResultState('empty');
            textarea.focus();
        } else if (action === 'edit-text') {
            textarea.focus();
        } else if (action === 'retry') {
            analyze();
        }
    });

    updateCounter();
    setResultState('empty');
}
