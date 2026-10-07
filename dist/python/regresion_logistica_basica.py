"""Regresión logística básica: horas de estudio -> aprobar o no aprobar."""

# Importamos NumPy para construir la matriz de entradas y el vector objetivo.
import numpy as np

# Importamos el clasificador de regresión logística.
from sklearn.linear_model import LogisticRegression

# X representa las horas de estudio observadas en cada estudiante.
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])

# y vale 0 si el estudiante no aprobó y 1 si aprobó.
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Creamos el modelo; random_state fija el resultado para hacerlo reproducible.
modelo = LogisticRegression(random_state=42)

# El modelo aprende qué probabilidades corresponden a cada cantidad de horas.
modelo.fit(X, y)

# Definimos un nuevo caso: un estudiante que estudió 4.5 horas.
nuevo_estudiante = np.array([[4.5]])

# predict_proba devuelve [P(clase 0), P(clase 1)]; tomamos la clase positiva.
probabilidad_aprobar = modelo.predict_proba(nuevo_estudiante)[0, 1]

# Convertimos la probabilidad en clase usando el umbral estándar de 0.50.
clase_predicha = int(probabilidad_aprobar >= 0.50)

# Mostramos una probabilidad legible como porcentaje.
print(f"Probabilidad de aprobar: {probabilidad_aprobar:.1%}")

# Traducimos la clase numérica a una etiqueta fácil de interpretar.
print("Predicción:", "Aprueba" if clase_predicha == 1 else "No aprueba")
