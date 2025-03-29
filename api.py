from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
import json
from fastapi.middleware.cors import CORSMiddleware

# Importar funciones del pipeline
from pipeline import (
    load_model_and_predict,
    train_and_save_model,
    expand_contractions_es,
    tokenize_with_spacy_batch,
    preprocessing,
    process_batch,
)

app = FastAPI(title="API para detección y reentrenamiento de Fake News en Español")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir cualquier origen
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos
    allow_headers=["*"],  # Permitir todos los headers
)

# Modelo de entrada para predicción con múltiples textos
class NewsBatchInput(BaseModel):
    noticias: List[dict]  # Lista de diccionarios con "Titulo" y "Descripcion"

# Función para leer `metrics.json`
def load_metrics():
    try:
        with open("metrics.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Métricas no encontradas. ¿Entrenaste el modelo?")

# Endpoint de predicción para múltiples noticias
@app.post("/predict")
def predict(news_batch: NewsBatchInput):
    resultados = []
    for noticia in news_batch.noticias:
        titulo = noticia.get("Titulo", "")
        descripcion = noticia.get("Descripcion", "")
        texto_completo = f"{titulo}. {descripcion}"
        resultado = load_model_and_predict(texto_completo)
        resultados.append({"Titulo": titulo, "Descripcion": descripcion, "Prediccion": resultado})
    return {"resultados": resultados}

# Endpoint para obtener métricas del modelo
@app.get("/model-metrics")
async def get_model_metrics():
    data = load_metrics()
    metrics = data.get("metrics", {})  # Ahora accede correctamente a "metrics"
    return {
        "accuracy": metrics.get("accuracy", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "f1_score": metrics.get("f1_score", 0.0),
    }


# Endpoint para obtener las palabras más influyentes en Fake News
@app.get("/top-words")
async def get_top_words():
    metrics = load_metrics()
    return {"words": metrics.get("top_words", [])}

# Endpoint para reentrenamiento desde CSV
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=";")

        # Procesar DataFrame
        processed_df = process_dataframe(df)

        # Reentrenar y guardar el modelo
        train_and_save_model(processed_df)

        return {"message": "Modelo reentrenado exitosamente."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")

# Endpoint para verificar palabras importantes en una noticia
class CheckWordsInput(BaseModel):
    title: str
    description: str

@app.post("/check_important_words/")
def check_important_words(input_data: CheckWordsInput):
    metrics = load_metrics()
    top_words = set(metrics.get("top_words", []))  # Convertir a conjunto para búsqueda rápida
    words_in_text = set(input_data.title.lower().split() + input_data.description.lower().split())

    matched_words = list(words_in_text & top_words)

    return {"important_words_found": matched_words}
