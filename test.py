import requests
import pandas as pd

# Define el endpoint de la API
API_URL = "http://127.0.0.1:5000/predict_batch"

# Define la ruta del dataset
DATASET_PATH = "data/raw_testing_dataset.csv"

# Carga el dataset en un DataFrame
try:
    print("Cargando dataset...")
    data = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en {DATASET_PATH}")
    exit(1)

# Convierte el DataFrame a JSON
data_json = data.to_dict(orient="records")

# Envía la solicitud POST al endpoint
print("Enviando datos al endpoint...")
response = requests.post(API_URL, json=data_json)

# Verifica si la respuesta es exitosa
if response.status_code == 200:
    # Procesa la respuesta
    response_data = response.json()
    # Convierte la respuesta a un DataFrame
    response_df = pd.DataFrame(response_data)
    # Ordena el DataFrame por la columna 'proba' en orden descendente
    response_df = response_df.sort_values(by="proba", ascending=False)
    # Muestra las 10 primeras filas con las columnas requeridas
    print("\nTop 10 resultados ordenados por 'proba':\n")
    print(response_df[['age', 'job', 'marital', 'education', 'proba']].head(10))
else:
    print(f"Error: {response.status_code}")
    print("Detalles:", response.text)
