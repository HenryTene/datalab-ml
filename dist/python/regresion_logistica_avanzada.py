"""Regresión logística avanzada con Pipeline, umbral y métricas de clasificación."""

# Importamos NumPy para generar datos y convertir probabilidades en clases.
import numpy as np

# Importamos la función que separa entrenamiento y prueba conservando la proporción de clases.
from sklearn.model_selection import train_test_split

# Importamos Pipeline para aplicar el escalado sin filtrar información de prueba.
from sklearn.pipeline import Pipeline

# Importamos StandardScaler para estandarizar las variables numéricas.
from sklearn.preprocessing import StandardScaler

# Importamos el modelo de regresión logística.
from sklearn.linear_model import LogisticRegression

# Importamos métricas que responden preguntas distintas sobre el desempeño.
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score

# Creamos un generador aleatorio reproducible.
rng = np.random.default_rng(42)

# Generamos 240 frecuencias cardiacas entre 65 y 180 latidos por minuto.
frecuencia = rng.uniform(65, 180, 240)

# Generamos una segunda variable: edad entre 18 y 80 años.
edad = rng.uniform(18, 80, 240)

# Calculamos un puntaje lineal; los coeficientes expresan el peso de cada variable.
logit_real = -9.5 + 0.075 * frecuencia + 0.025 * edad

# Convertimos el puntaje lineal en una probabilidad entre cero y uno.
probabilidad_real = 1 / (1 + np.exp(-logit_real))

# Sorteamos la clase real usando la probabilidad de cada observación.
y = rng.binomial(1, probabilidad_real)

# Unimos frecuencia y edad como columnas de la matriz de entrada X.
X = np.column_stack([frecuencia, edad])

# Separamos entrenamiento y prueba; stratify conserva el balance de clases.
X_train, X_test, y_train, y_test = train_test_split(
    # Entregamos todas las variables predictoras.
    X,
    # Entregamos la clase real.
    y,
    # Reservamos 25 % para una evaluación no usada durante el ajuste.
    test_size=0.25,
    # Fijamos la semilla para reproducir la misma separación.
    random_state=42,
    # Conservamos la proporción de positivos y negativos en ambos grupos.
    stratify=y,
)

# Construimos un Pipeline que estandariza y clasifica en un único flujo.
modelo = Pipeline([
    # El escalador aprende sus parámetros únicamente con X_train.
    ("escalado", StandardScaler()),
    # class_weight="balanced" compensa automáticamente el desbalance de clases.
    ("logistica", LogisticRegression(class_weight="balanced", random_state=42)),
])

# Ajustamos todos los pasos del Pipeline usando solo entrenamiento.
modelo.fit(X_train, y_train)

# Obtenemos la probabilidad de la clase positiva para cada caso de prueba.
probabilidades = modelo.predict_proba(X_test)[:, 1]

# Elegimos un umbral menor a 0.50 para priorizar la detección de positivos.
umbral = 0.35

# Convertimos las probabilidades en clases de acuerdo con el umbral elegido.
predicciones = (probabilidades >= umbral).astype(int)

# Calculamos VN, FP, FN y VP a partir de la matriz de confusión.
vn, fp, fn, vp = confusion_matrix(y_test, predicciones).ravel()

# Calculamos precision: de las alertas emitidas, cuántas fueron correctas.
precision = precision_score(y_test, predicciones, zero_division=0)

# Calculamos recall: de los positivos reales, cuántos fueron detectados.
recall = recall_score(y_test, predicciones, zero_division=0)

# Calculamos ROC-AUC usando probabilidades, sin fijar un único umbral.
auc = roc_auc_score(y_test, probabilidades)

# Mostramos el umbral que convirtió probabilidades en decisiones.
print(f"Umbral utilizado: {umbral:.2f}")

# Mostramos cada celda de la matriz de confusión.
print(f"VN={vn}, FP={fp}, FN={fn}, VP={vp}")

# Mostramos la calidad de las alertas positivas.
print(f"Precision: {precision:.3f}")

# Mostramos la cobertura de los casos positivos reales.
print(f"Recall: {recall:.3f}")

# Mostramos la capacidad global de ordenar positivos sobre negativos.
print(f"ROC-AUC: {auc:.3f}")
