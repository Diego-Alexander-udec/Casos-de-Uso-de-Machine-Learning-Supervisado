import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


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
    import pandas as pd
    if temperatura > 50:
        return 0  # No hay ventas si la temperatura es superior a 50°C
    if temperatura < 0:
        return -1  # Ventas negativas si la temperatura es menor a 0°C
    X_new = pd.DataFrame({'Temperatura_C': [temperatura], 'Dia_Semana': [dia_semana]})
    result = modelo.predict(X_new)[0]
    return round(result)  # Redondea el resultado al número entero más cercano

def generar_grafica(temperatura=None, dia_semana=None, resultado=None, filename="grafica.png"):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Temperatura_C'], df['Ventas_Helados'], c=df['Dia_Semana'], cmap='viridis', label="Datos entrenamiento")
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Ventas de Helados')
    plt.title('Relación entre Temperatura y Ventas de Helados')
    temp_range = sorted(df['Temperatura_C'].unique())
    # Solo una línea de tendencia para el día digitado por el usuario
    if dia_semana is not None:
        X_pred = pd.DataFrame({'Temperatura_C': temp_range, 'Dia_Semana': [dia_semana]*len(temp_range)})
        y_pred = modelo.predict(X_pred)
        plt.plot(temp_range, y_pred, color='gray', label=f"Tendencia Día {dia_semana}")
    if temperatura is not None and dia_semana is not None and resultado is not None:
        plt.scatter([temperatura], [resultado], color='red', s=120, label="Tu predicción")
    plt.legend()
    plt.tight_layout()
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    filepath = os.path.join(static_folder, filename)
    plt.savefig(filepath)
    plt.close()

# Visualización de los datos
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperatura_C'], df['Ventas_Helados'], c=df['Dia_Semana'], cmap='viridis')
plt.colorbar(label='Día de la semana')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Ventas de Helados')
plt.title('Relación entre Temperatura y Ventas de Helados')

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