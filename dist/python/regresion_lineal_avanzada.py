"""Regresión lineal avanzada con Pipeline, Ridge y validación cruzada."""

# Importamos NumPy para crear un conjunto de datos reproducible.
import numpy as np

# Importamos cross_validate para evaluar el modelo en varias particiones.
from sklearn.model_selection import cross_validate

# Importamos Pipeline para encadenar transformación y modelo sin fuga de datos.
from sklearn.pipeline import Pipeline

# Importamos StandardScaler para colocar todas las variables en escalas comparables.
from sklearn.preprocessing import StandardScaler

# Importamos RidgeCV para regularizar y elegir automáticamente el mejor valor de alpha.
from sklearn.linear_model import RidgeCV

# Creamos un generador aleatorio con semilla fija para obtener siempre los mismos datos.
rng = np.random.default_rng(42)

# Generamos 120 observaciones de horas de trabajo entre 25 y 55.
horas_trabajo = rng.uniform(25, 55, 120)

# Generamos 120 observaciones de horas de descanso entre 1 y 5.
horas_descanso = rng.uniform(1, 5, 120)

# Combinamos ambas variables como columnas de la matriz X.
X = np.column_stack([horas_trabajo, horas_descanso])

# Generamos ruido para representar factores reales que el modelo no observa.
ruido = rng.normal(0, 1.8, 120)

# Construimos y con una relación conocida más el ruido aleatorio.
y = 5.5 + 0.51 * horas_trabajo - 0.41 * horas_descanso + ruido

# Creamos un Pipeline: primero escala X y después ajusta una regresión Ridge.
modelo = Pipeline([
    # Este paso aprende media y desviación solamente dentro de cada entrenamiento.
    ("escalado", StandardScaler()),
    # Este paso prueba distintos alpha y conserva el que generaliza mejor.
    ("ridge", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])),
])

# Evaluamos R² y MAE mediante cinco particiones de validación cruzada.
resultados = cross_validate(
    # Indicamos qué Pipeline queremos evaluar.
    modelo,
    # Entregamos las variables independientes.
    X,
    # Entregamos la variable objetivo.
    y,
    # Usamos cinco particiones para reducir la dependencia de un único split.
    cv=5,
    # Solicitamos dos métricas complementarias.
    scoring={"r2": "r2", "mae": "neg_mean_absolute_error"},
)

# Ajustamos finalmente el Pipeline con todos los datos disponibles.
modelo.fit(X, y)

# Recuperamos el alpha elegido dentro del paso llamado "ridge".
mejor_alpha = modelo.named_steps["ridge"].alpha_

# Convertimos MAE a positivo porque scikit-learn lo devuelve negado internamente.
mae_promedio = -resultados["test_mae"].mean()

# Calculamos el promedio de R² obtenido en las cinco particiones.
r2_promedio = resultados["test_r2"].mean()

# Predecimos la producción para 45 horas de trabajo y 3 de descanso.
prediccion = modelo.predict([[45, 3]])[0]

# Mostramos el nivel de regularización seleccionado.
print(f"Mejor alpha: {mejor_alpha}")

# Mostramos cuánto error absoluto comete el modelo en promedio.
print(f"MAE de validación: {mae_promedio:.2f}")

# Mostramos qué fracción de variabilidad explica el modelo fuera de entrenamiento.
print(f"R² de validación: {r2_promedio:.3f}")

# Mostramos la predicción del escenario nuevo.
print(f"Producción estimada: {prediccion:.2f}")
