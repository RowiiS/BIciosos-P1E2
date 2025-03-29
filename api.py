from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
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

# Función para procesar un DataFrame
def process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    # Validar que existan las columnas necesarias
    required_columns = ["Label", "Titulo", "Descripcion"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Error: La columna '{col}' no está presente en el dataset.")

    # Rellenar valores nulos
    data["Titulo"] = data["Titulo"].fillna("")
    data["Descripcion"] = data["Descripcion"].fillna("")

    # Expansión de contracciones
    data["Titulo"] = data["Titulo"].apply(expand_contractions_es)
    data["Descripcion"] = data["Descripcion"].apply(expand_contractions_es)

    # Tokenización
    data["Titles"] = tokenize_with_spacy_batch(data["Titulo"].tolist())
    data["Descriptions"] = tokenize_with_spacy_batch(data["Descripcion"].tolist())

    # Preprocesamiento de texto
    data["Titles1"] = data["Titles"].apply(preprocessing)
    data["Descriptions1"] = data["Descriptions"].apply(preprocessing)

    # Stemming y lematización (para Descripcion preprocesada)
    stems, lemmas = process_batch(data["Descriptions1"].tolist())
    data["Stems"] = stems
    data["Lemmas"] = lemmas

    return data

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

# Endpoint para reentrenamiento desde CSV
@app.post("/retrain")
async def retrain(file: UploadFile = File(...)):
    try:
        # Leer el archivo CSV
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")), sep=";")

        # Procesar el DataFrame
        processed_df = process_dataframe(df)

        # Reentrenar y guardar el modelo
        train_and_save_model(processed_df)

        return {"message": "Modelo reentrenado exitosamente."}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar el archivo: {str(e)}")
    
import json

@app.get("/metrics")
def get_metrics():
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Métricas no encontradas. ¿Entrenaste el modelo?")


class CheckWordsInput(BaseModel):
    title: str
    description: str

@app.post("/check_important_words/")
def check_important_words(input_data: CheckWordsInput):
    try:
        with open("metrics.json", "r") as f:
            data = json.load(f)
        
        top_words = set(data.get("top_words", []))  # Convertir a conjunto para búsqueda rápida
        words_in_text = set(input_data.title.lower().split() + input_data.description.lower().split())

        matched_words = list(words_in_text & top_words)

        return {"important_words_found": matched_words}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No se encontraron palabras importantes. ¿Entrenaste el modelo?")
