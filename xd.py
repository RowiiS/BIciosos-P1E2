import pandas as pd

# Cargar el dataset completo
df = pd.read_csv("fake_news_spanish.csv", sep=";")

# Seleccionar solo las primeras 20 filas
df_sample = df.head(20)

# Guardar el nuevo archivo reducido
df_sample.to_csv("fake_news_sample.csv", sep=";")

print("Archivo reducido guardado como fake_news_sample.csv")
