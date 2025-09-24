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
    X_new = pd.DataFrame({'Temperatura_C': [temperatura], 'Dia_Semana': [dia_semana]})
    result = modelo.predict(X_new)[0]
    return round(result)  # Redondea el resultado al número entero más cercano

def generar_grafica(temperatura=None, dia_semana=None, resultado=None, filename="grafica.png"):
    """Generar gráfica mejorada para web con mejor calidad y cache busting"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Temperatura_C'], df['Ventas_Helados'], c=df['Dia_Semana'], cmap='viridis', 
               label="Datos entrenamiento", alpha=0.7)
    plt.xlabel('Temperatura (°C)', fontsize=12)
    plt.ylabel('Ventas de Helados', fontsize=12)
    plt.title('Relación entre Temperatura y Ventas de Helados', fontsize=14, fontweight='bold')
    
    temp_range = sorted(df['Temperatura_C'].unique())
    # Solo una línea de tendencia para el día digitado por el usuario
    if dia_semana is not None:
        X_pred = pd.DataFrame({'Temperatura_C': temp_range, 'Dia_Semana': [dia_semana]*len(temp_range)})
        y_pred = modelo.predict(X_pred)
        plt.plot(temp_range, y_pred, color='#E74C3C', linewidth=2, label=f"Tendencia Día {dia_semana}")
    
    if temperatura is not None and dia_semana is not None and resultado is not None:
        plt.scatter([temperatura], [resultado], color='red', s=150, 
                   edgecolor='darkred', linewidth=2, label="Tu predicción", zorder=5)
    
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Asegurar que existe la carpeta static
    static_folder = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_folder, exist_ok=True)
    
    filepath = os.path.join(static_folder, filename)
    plt.savefig(filepath, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()  # Importante cerrar para liberar memoria

# Generar gráfica inicial para demostración (opcional)
if __name__ == "__main__":
    # Ejemplo de uso del modelo
    temperatura_ejemplo = 12
    dia_ejemplo = 2  # Martes
    
    # Realizar predicción con el modelo
    ventas_predichas = EstimarVentasHelados(temperatura_ejemplo, dia_ejemplo)
    
    # Mostrar resultados
    print(f"\nCoeficientes del modelo:")
    print(f"Impacto de la temperatura: {modelo.coef_[0]:.2f}")
    print(f"Impacto del día de la semana: {modelo.coef_[1]:.2f}")
    print(f"\nLas ventas estimadas para una temperatura de {temperatura_ejemplo}°C")
    print(f"en día {dia_ejemplo} (Sábado) son de {ventas_predichas} helados.")
    
    # Mostrar estadísticas descriptivas del DataFrame
    print("\nEstadísticas descriptivas de los datos:")
    print(df.describe())
    
    # Generar gráfica con la predicción
    generar_grafica(temperatura_ejemplo, dia_ejemplo, ventas_predichas, "demo_regression.png")