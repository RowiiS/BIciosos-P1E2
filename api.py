from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
from scipy.sparse import hstack
import spacy
import nltk
import re
from nltk.corpus import stopwords
from num2words import num2words
import unicodedata

# Cargar modelo entrenado y vectorizadores
model = joblib.load("modelo.pkl")
vectorizers = joblib.load("vectorizers.pkl")

# Cargar modelo de lenguaje en español de spaCy
nlp = spacy.load("es_core_news_sm")

app = FastAPI()

# Definir estructura de entrada para predicción
class PredictionInput(BaseModel):
    titulos: List[str]
    descripciones: List[str]

# Funciones de preprocesamiento
def remove_non_ascii(words):
    return [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]

def to_lowercase(words):
    return [word.lower() for word in words if word]

def remove_punctuation(words):
    return [re.sub(r'[^\w\sáéíóúñÁÉÍÓÚÑ]', '', word) for word in words if word]

def replace_numbers(words):
    return [num2words(word, lang='es') if word.isdigit() else word for word in words]

def remove_stopwords(words):
    stop_words = set(stopwords.words('spanish'))
    return [word for word in words if word not in stop_words]

def tokenize_with_spacy(text):
    return [token.text for token in nlp(text)]

def preprocessing(text):
    words = tokenize_with_spacy(text)
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Validar que las listas tienen el mismo tamaño
    if len(input_data.titulos) != len(input_data.descripciones):
        return {"error": "Las listas de títulos y descripciones deben tener el mismo tamaño"}

    # Tokenización y preprocesamiento en batch
    titulos_tokens = [preprocessing(titulo) for titulo in input_data.titulos]
    descripciones_tokens = [preprocessing(desc) for desc in input_data.descripciones]

    # Vectorización en batch
    X_titulos = vectorizers["Titles1"].transform([" ".join(tokens) for tokens in titulos_tokens])
    X_descripciones = vectorizers["Descriptions1"].transform([" ".join(tokens) for tokens in descripciones_tokens])

    # Combinar características
    X_combined = hstack([X_titulos, X_descripciones])

    # Predicciones en batch
    predictions = model.predict(X_combined)
    probabilities = model.predict_proba(X_combined).max(axis=1)

    # Construcción de la respuesta
    results = [
        {"prediction": int(pred), "probability": float(prob)}
        for pred, prob in zip(predictions, probabilities)
    ]

    return {"results": results}

@app.get("/")
def read_root():
    return {"message": "API de clasificación de noticias activa"}
