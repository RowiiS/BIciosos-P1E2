from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd

# Importar las funciones de tu pipeline
from pipeline import load_model_and_predict, train_and_save_model, expand_contractions_es, tokenize_with_spacy_batch, preprocessing, process_batch

# Si no cuentas con una función que procese un DataFrame (sin cargar desde CSV),
# la definimos aquí para el endpoint de reentrenamiento:
def process_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    # Validar que existan las columnas requeridas
    required_columns = ["Titulo", "Descripcion", "Label"]
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

# Definición de modelos para los endpoints usando Pydantic
class NewsInput(BaseModel):
    text: str

class RetrainData(BaseModel):
    data: List[Dict]  # Cada diccionario debe incluir: Titulo, Descripcion y Label (además de otros campos si los hay)

# Crear la instancia de FastAPI
app = FastAPI(title="API para detección y reentrenamiento de Fake News en Español")

# Endpoint de predicción
@app.post("/predict")
def predict(news: NewsInput):
    result = load_model_and_predict(news.text)
    return result

# Endpoint de reentrenamiento
@app.post("/retrain")
def retrain(retrain_data: RetrainData):
    # Convertir la lista de diccionarios a DataFrame
    df_new = pd.DataFrame(retrain_data.data)
    # Procesar el DataFrame para obtener las columnas necesarias y aplicarle el preprocesamiento
    processed_df = process_dataframe(df_new)
    # Reentrenar y guardar el modelo usando los nuevos datos
    train_and_save_model(processed_df)
    return {"message": "Modelo reentrenado exitosamente."}
