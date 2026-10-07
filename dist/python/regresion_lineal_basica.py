"""Regresión lineal básica: horas de estudio -> calificación."""

# Importamos NumPy para representar los datos como arreglos numéricos.
import numpy as np

# Importamos el modelo de regresión lineal de scikit-learn.
from sklearn.linear_model import LinearRegression

# X contiene la variable independiente: horas de estudio de cada estudiante.
# Cada observación va dentro de una lista porque scikit-learn espera una matriz 2D.
X = np.array([[1], [2], [3], [4], [5], [6]])

# y contiene la variable dependiente: calificación obtenida por cada estudiante.
y = np.array([52, 57, 64, 70, 76, 83])

# Creamos el objeto que aprenderá la mejor recta para relacionar X con y.
modelo = LinearRegression()

# Ajustamos la recta usando todos los ejemplos disponibles.
modelo.fit(X, y)

# Guardamos las horas de estudio del nuevo estudiante que queremos evaluar.
nuevo_estudiante = np.array([[7]])

# Calculamos la calificación estimada para siete horas de estudio.
prediccion = modelo.predict(nuevo_estudiante)

# Mostramos el intercepto: valor de y cuando las horas de estudio son cero.
print(f"Intercepto β0: {modelo.intercept_:.2f}")

# Mostramos la pendiente: cambio esperado en y por cada hora adicional.
print(f"Pendiente β1: {modelo.coef_[0]:.2f}")

# Mostramos la predicción; [0] extrae el único valor devuelto por el modelo.
print(f"Calificación estimada: {prediccion[0]:.2f}")
