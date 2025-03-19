from flask import Flask, render_template, request, jsonify, send_file
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Descargar recursos de NLTK (si no los tienes)
nltk.download('punkt')
nltk.download('stopwords')

# Definir la clase del modelo
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # No aplicar sigmoid aquí

# Cargar el modelo y los preprocesadores
def load_model_and_preprocessors():
    # Cargar el modelo
    model = torch.load('modelo_optimizado_sgd.pt', map_location=torch.device('cpu'))
    model.eval()

    # Cargar el vectorizador TF-IDF
    tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")

    # Cargar el selector de características
    selector = joblib.load("selector.pkl")

    # Cargar el normalizador
    scaler = joblib.load("scaler.pkl")

    return model, tfidf_vectorizer, selector, scaler

# Preprocesamiento de texto
def preprocess_text(text, tfidf_vectorizer, selector, scaler):
    # Limpieza del texto
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Eliminar caracteres especiales
    text = re.sub(r'\d', ' ', text)  # Eliminar números

    # Tokenización y eliminación de stopwords
    words = word_tokenize(text)
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]

    # Verificar si el texto quedó vacío
    if not words:
        print("Advertencia: El texto se ha vaciado después de preprocesarlo.")
        return None  # Devuelve None para evitar problemas más adelante

    text = ' '.join(words)

    # Vectorización TF-IDF
    text_vec = tfidf_vectorizer.transform([text])

    # Selección de características
    text_reduced = selector.transform(text_vec)

    # Normalización
    text_normalized = scaler.transform(text_reduced)

    # Convertir a tensor
    text_tensor = torch.tensor(text_normalized.toarray(), dtype=torch.float32)

    return text_tensor

# Función para truncar el texto a un máximo de 6 palabras
def truncate_text(text, max_words=6):
    words = text.split(' ')
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return text

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo y los preprocesadores
model, tfidf_vectorizer, selector, scaler = load_model_and_preprocessors()

# Lista para almacenar las últimas predicciones (hasta 20)
predictions_history = []

# Ruta principal (interfaz web)
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para predecir noticias
@app.route('/predict', methods=['POST'])
def predict():
    # Obtener el texto de la solicitud
    data = request.json
    text = data.get('text', '')

    if not text.strip():
        return jsonify({'error': 'Por favor, ingresa una noticia.'}), 400

    # Preprocesar el texto
    text_tensor = preprocess_text(text, tfidf_vectorizer, selector, scaler)

    # Si el preprocesamiento falló, retornar un mensaje
    if text_tensor is None:
        return jsonify({
            'prediction': "No se pudo procesar la noticia."
        })

    print("Tensor de entrada:", text_tensor)  # Debugging

    # Realizar la predicción
    with torch.no_grad():
        output = model(text_tensor)
        print("Salida del modelo antes de clamp:", output)

        # Asegurar que la salida es escalar
        if output.numel() > 1:
            output = output.mean()

        output = torch.clamp(output, -10, 10)
        print("Salida del modelo después de clamp:", output)

        prediction_prob = torch.sigmoid(output).item()

    # Verificar NaN
    if prediction_prob != prediction_prob:
        print("Advertencia: La probabilidad es NaN. Asignando valor predeterminado.")
        prediction_prob = 0.5

    # Determinar la predicción
    prediction = "falsa" if prediction_prob > 0.7 else "verdadera"

    # Guardar en historial
    predictions_history.append({
        'text': text,
        'prediction': prediction
    })

    # Mantener solo las últimas 20 predicciones
    if len(predictions_history) > 20:
        predictions_history.pop(0)

    # Devolver el resultado
    return jsonify({
        'prediction': prediction
    })

# Ruta para obtener las predicciones (para el modal)
@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    # Obtener el número de predicciones a incluir en el informe
    num_predictions = int(request.args.get('num_predictions', 5))  # Valor predeterminado: 5

    # Limitar el número de predicciones al máximo disponible
    num_predictions = min(num_predictions, len(predictions_history))

    # Devolver las predicciones seleccionadas
    return jsonify({
        'predictions': predictions_history[:num_predictions]
    })

# Ruta para generar y descargar el informe en PDF
@app.route('/download_report', methods=['GET'])
def download_report():
    # Obtener el número de predicciones a incluir en el informe
    num_predictions = int(request.args.get('num_predictions', 5))  # Valor predeterminado: 5

    # Limitar el número de predicciones al máximo disponible
    num_predictions = min(num_predictions, len(predictions_history))

    # Crear un buffer para el PDF
    buffer = BytesIO()

    # Crear el PDF
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica", 12)

    # Título del informe
    pdf.drawString(50, 750, "Informe de Predicciones de Noticias Falsas")
    pdf.drawString(50, 730, "-------------------------------------------")

    # Agregar detalles de las predicciones
    y_position = 700
    for i, pred in enumerate(predictions_history[:num_predictions]):
        pdf.drawString(50, y_position, f"Predicción {i + 1}:")
        pdf.drawString(70, y_position - 20, f"Noticia: {truncate_text(pred['text'])}")
        pdf.drawString(70, y_position - 40, f"Resultado: {pred['prediction']}")
        y_position -= 60

    # Finalizar el PDF
    pdf.save()

    # Preparar el buffer para la descarga
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="informe_predicciones.pdf", mimetype='application/pdf')

# Iniciar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True)