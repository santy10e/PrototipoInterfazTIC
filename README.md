# Predicción de Noticias Falsas

Este proyecto es una aplicación web que utiliza un modelo de aprendizaje automático para predecir si una noticia es falsa o verdadera. La aplicación está construida con Flask (backend) y Bootstrap (frontend), y utiliza un modelo de regresión logística entrenado con PyTorch generado en el TIC "Optimización de la Precisión en la Detección de Noticias Falsas de política en español mediante la Aplicación de Algoritmos de Optimización en la Regresión Logística".

## Características principales

- **Predicción en tiempo real:** Ingresa una noticia y obtén una predicción inmediata sobre si es falsa o verdadera.
- **Historial de predicciones:** Mantiene un registro de las últimas 20 predicciones realizadas.
- **Informe en PDF:** Genera y descarga un informe en PDF con las predicciones recientes.
- **Interfaz amigable:** Diseño moderno y responsive, compatible con dispositivos móviles y de escritorio.

## Tecnologías utilizadas

- **Backend:**
  - Flask: Framework web para Python.
  - PyTorch: Biblioteca de aprendizaje automático para entrenar y cargar el modelo.
  - Scikit-learn: Para el preprocesamiento de texto (TF-IDF, selección de características, normalización).
  - Joblib: Para cargar los preprocesadores guardados.

- **Frontend:**
  - Bootstrap: Framework CSS para el diseño de la interfaz.
  - HTML5 y JavaScript: Para la estructura y la lógica de la interfaz.
  - ReportLab: Para generar informes en PDF.

- **Herramientas adicionales:**
  - NLTK: Para el preprocesamiento de texto (tokenización, eliminación de stopwords).

## Requisitos previos

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

- Python 3.8 o superior.
- Pip (gestor de paquetes de Python).

## Instalación

1. **Clona el repositorio:**
   ```bash
   git clone https://github.com/tu-usuario/prediccion-noticias-falsas.git
   cd prediccion-noticias-falsas
   
   
