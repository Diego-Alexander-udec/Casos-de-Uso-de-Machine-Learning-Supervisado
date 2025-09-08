# Importación de bibliotecas necesarias
import pandas as pd  # Para manipulación y análisis de datos
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.linear_model import LinearRegression  # Para crear el modelo de regresión

# Creación del conjunto de datos con más registros y mejor distribución
datos = {
    'Temperatura_C': [25, 30, 22, 28, 32, 27, 29, 24, 31, 26, 33, 23, 29, 31, 28],
    'Dia_Semana': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 1],  # 1=Lunes, 7=Domingo
    'Ventas_Helados': [150, 220, 100, 180, 250, 190, 210, 130, 200, 160, 270, 120, 200, 230, 175]
}

# Creación del DataFrame a partir del diccionario de datos
df = pd.DataFrame(datos)

# Separación de características (X) y variable objetivo (y)
X = df[['Temperatura_C', 'Dia_Semana']]  # Variables independientes
y = df['Ventas_Helados']  # Variable dependiente

# Creación y entrenamiento del modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)  # Ajuste del modelo con los datos de entrenamiento

# Función para estimar ventas de helados
def EstimarVentasHelados(temperatura, dia_semana):
    """
    Estima las ventas de helados según la temperatura y día de la semana
    
    Args:
        temperatura: Temperatura en grados Celsius (float)
        dia_semana: Día de la semana (int: 1=Lunes, 7=Domingo)
    
    Returns:
        int: Cantidad estimada de helados a vender
    """
    # Realiza la predicción usando el modelo entrenado
    result = modelo.predict([[temperatura, dia_semana]])[0]
    return round(result)  # Redondea el resultado al número entero más cercano

# Visualización de los datos
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperatura_C'], df['Ventas_Helados'], c=df['Dia_Semana'], cmap='viridis')
plt.colorbar(label='Día de la semana')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Ventas de Helados')
plt.title('Relación entre Temperatura y Ventas de Helados')
plt.show()

# Ejemplo de uso del modelo
temperatura_ejemplo = 12
dia_ejemplo = 2  # Martes

# Realizar predicción con el modelo
ventas_estimadas = EstimarVentasHelados(temperatura_ejemplo, dia_ejemplo)

# Mostrar resultados
print(f"\nCoeficientes del modelo:")
print(f"Impacto de la temperatura: {modelo.coef_[0]:.2f}")
print(f"Impacto del día de la semana: {modelo.coef_[1]:.2f}")
print(f"\nLas ventas estimadas para una temperatura de {temperatura_ejemplo}°C")
print(f"en día {dia_ejemplo} (Sábado) son de {ventas_estimadas} helados.")

# Mostrar estadísticas descriptivas del DataFrame
print("\nEstadísticas descriptivas de los datos:")
print(df.describe())