import pandas as pd
import numpy as np
import joblib
import spacy
import nltk
import unicodedata
import re
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

# Descargar recursos de NLTK (solo si no están descargados)
nltk.download('stopwords')

# Cargar modelo de lenguaje en español de spaCy
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Descargando modelo de spaCy para español...")
    import os
    os.system("python -m spacy download es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")


"""
🔹 Preprocesamiento de Texto
"""
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

def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words

def expand_contractions_es(text):
    text = re.sub(r'\bal\b', 'a el', text)
    text = re.sub(r'\bdel\b', 'de el', text)
    return text

def tokenize_with_spacy_batch(texts, batch_size=1000):
    return [[token.text for token in doc] for doc in nlp.pipe(texts, batch_size=batch_size)]

def stem_words(words):
    stemmer = SnowballStemmer("spanish")
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    doc = nlp(" ".join(words))
    return [token.lemma_ for token in doc if token.pos_ == "VERB"]

def process_batch(texts):
    stems = [stem_words(words) for words in texts]
    lemmas = [lemmatize_verbs(words) for words in texts]
    return stems, lemmas


"""
🔹 Carga y Preprocesamiento de Datos
"""
def load_and_process_data(file_path):
    # Cargar los datos en un DataFrame. Se usa el separador ';'
    data = pd.read_csv(file_path, sep=";")

    # Validar que las columnas necesarias existen
    required_columns = ["Titulo", "Descripcion", "Label"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Error: La columna '{col}' no está presente en el dataset.")

    # Llenar valores nulos en las columnas de texto
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

    # Stemming y lematización
    stems, lemmas = process_batch(data["Descriptions1"].tolist())
    data["Stems"] = stems
    data["Lemmas"] = lemmas

    # Separar en entrenamiento (80%) y prueba (20%)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Label"])

    return train_data, test_data


"""
🔹 Vectorización y Entrenamiento del Modelo
"""
def train_and_save_model(train_data):
    y = train_data["Label"]
    vectorizers = {}

    # Vectorizar columnas
    def vectorize_column(column):
        vectorizer = TfidfVectorizer(max_features=5000)
        matrix = vectorizer.fit_transform(train_data[column].apply(lambda x: " ".join(x)))
        vectorizers[column] = vectorizer
        return matrix

    X_titles = vectorize_column("Titles1")
    X_desc = vectorize_column("Descriptions1")
    X_stems = vectorize_column("Stems")
    X_lemmas = vectorize_column("Lemmas")

    X_combined = hstack([X_titles, X_desc, X_stems, X_lemmas])

    # Dividir datos usando un split adicional para evaluar el modelo
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42, stratify=y)

    # Entrenar modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Evaluar
    y_pred = model.predict(X_test)
    print("Precisión:", np.mean(y_pred == y_test))

    # Guardar modelo y vectorizadores
    joblib.dump(model, "modelo.pkl")
    joblib.dump(vectorizers, "vectorizers.pkl")

    print("Modelo guardado exitosamente.")


"""
🔹 Carga y Predicción del Modelo
"""
def load_model_and_predict(text_data):
    model = joblib.load("modelo.pkl")
    vectorizers = joblib.load("vectorizers.pkl")

    # Tokenizar y preprocesar el texto
    tokens = tokenize_with_spacy_batch([text_data])[0]
    processed_text = preprocessing(tokens)

    # Vectorizar con cada modelo de Tfidf
    X_input = hstack([vectorizers[col].transform([" ".join(processed_text)]) for col in ["Titles1", "Descriptions1", "Stems", "Lemmas"]])

    # Predecir
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input).max()

    return {"prediction": int(prediction), "probability": float(probability)}


"""
🔹 Ejecución del Pipeline
"""
if __name__ == "__main__":
    file_path = "fake_news_spanish.csv"  # Ruta del dataset

    print("📥 Cargando y procesando datos...")
    train_data, test_data = load_and_process_data(file_path)

    print("🎯 Entrenando modelo...")
    train_and_save_model(train_data)

    # Prueba con un texto
    resultado = load_model_and_predict("El presidente anunció una nueva reforma educativa.")
    print("📢 Resultado de predicción:", resultado)
