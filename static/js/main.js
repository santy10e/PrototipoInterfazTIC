/**
 * Punto de entrada de la interfaz de News Intelligence.
 */
import { initAnalyzer } from './analyzer.js';
import { initHistory } from './history.js';
import { initNavigation } from './navigation.js';
import { initReports } from './reports.js';
import { refreshHistory } from './store.js';

let activeView = null;

initHistory({ isViewActive: (name) => activeView === name });
initReports();
initAnalyzer({ onPredicted: () => refreshHistory({ highlightNewest: true }) });

initNavigation({
    onViewChange(view) {
        activeView = view;
        // Las vistas que muestran el historial se sincronizan con el servidor al abrirse.
        if (view === 'historial' || view === 'informes') refreshHistory();
    },
});

// Carga inicial: alimenta el resumen y determina la disponibilidad real del servicio.
refreshHistory();

// Al volver a la pestaña, sincroniza el historial (puede haber cambiado desde otro cliente).
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') refreshHistory();
});
