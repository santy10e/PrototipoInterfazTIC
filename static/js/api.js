/**
 * Cliente de los endpoints Flask existentes.
 * No altera contratos: /predict (POST), /get_predictions (GET), /download_report (GET).
 */

/** Capacidad del historial definida en el backend (modelo.py). */
export const HISTORY_LIMIT = 20;

/** Valores de predicción que devuelve /predict. */
export const PREDICTION = Object.freeze({ FAKE: 'falsa', TRUE: 'verdadera' });

export class ApiError extends Error {
    /**
     * @param {string} message Mensaje listo para mostrar al usuario.
     * @param {number} status  Código HTTP (0 = sin conexión).
     */
    constructor(message, status = 0) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
    }
    get isNetwork() { return this.status === 0; }
}

const NETWORK_MESSAGE = 'No se pudo conectar con el servidor. Comprueba que la aplicación Flask está en ejecución e inténtalo de nuevo.';

async function request(url, options) {
    try {
        return await fetch(url, options);
    } catch {
        throw new ApiError(NETWORK_MESSAGE, 0);
    }
}

async function readJson(response) {
    try {
        return await response.json();
    } catch {
        return null;
    }
}

/**
 * Envía una noticia al modelo.
 * @returns {Promise<{prediction: string}>} 'falsa' | 'verdadera' | mensaje del servidor si no pudo procesarse.
 */
export async function predict(text) {
    const response = await request('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
    });
    const data = await readJson(response);

    if (!response.ok) {
        throw new ApiError(data?.error || `El servidor respondió con un error (${response.status}).`, response.status);
    }
    if (!data || typeof data.prediction !== 'string') {
        throw new ApiError('El servidor devolvió una respuesta inesperada.', response.status);
    }
    return data;
}

/**
 * Obtiene las predicciones del historial (orden cronológico, de la más antigua a la más reciente).
 * @returns {Promise<Array<{text: string, prediction: string, timestamp?: string}>>}
 */
export async function getPredictions(limit = HISTORY_LIMIT) {
    const response = await request(`/get_predictions?num_predictions=${encodeURIComponent(limit)}`);
    const data = await readJson(response);

    if (!response.ok || !data || !Array.isArray(data.predictions)) {
        throw new ApiError(`No se pudo obtener el historial (código ${response.status}).`, response.status);
    }
    return data.predictions;
}

/**
 * Solicita el informe PDF generado por ReportLab y lo devuelve como Blob.
 * @returns {Promise<{blob: Blob, filename: string}>}
 */
export async function fetchReport(count) {
    const response = await request(`/download_report?num_predictions=${encodeURIComponent(count)}`);

    if (!response.ok) {
        throw new ApiError(`No se pudo generar el informe (código ${response.status}).`, response.status);
    }
    const blob = await response.blob();
    return { blob, filename: filenameFrom(response) || 'informe_predicciones.pdf' };
}

function filenameFrom(response) {
    const header = response.headers.get('Content-Disposition') || '';
    const encoded = header.match(/filename\*=UTF-8''([^;]+)/i);
    if (encoded) return decodeURIComponent(encoded[1]);
    const plain = header.match(/filename="?([^";]+)"?/i);
    return plain ? plain[1] : null;
}
