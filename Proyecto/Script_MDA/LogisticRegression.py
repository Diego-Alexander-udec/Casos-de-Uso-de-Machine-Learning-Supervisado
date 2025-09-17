import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import joblib # Para guardar el modelo

# --- 1. Creación de un Conjunto de Datos Simulado ---
# En un caso real, cargarías tus datos desde un archivo (ej: pd.read_csv('citas.csv'))
np.random.seed(42) # Para reproducibilidad
n_muestras = 500

datos = {
    'Edad': np.random.randint(5, 85, size=n_muestras),
    'TiempoEspera': np.random.randint(0, 120, size=n_muestras), # Días de espera desde que se pide la cita
    'CitasPrevias': np.random.randint(0, 15, size=n_muestras),
    'DiaSemana': np.random.choice(['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes'], size=n_muestras),
    'Asistio': np.random.choice([0, 1], size=n_muestras, p=[0.3, 0.7]) # 0=No, 1=Sí
}
df = pd.DataFrame(datos)

# Ajustar la variable 'Asistio' para que tenga más sentido con las otras variables
# Por ejemplo, mayor tiempo de espera o ser más joven podría aumentar la probabilidad de no asistir.
df['Asistio'] = df.apply(
    lambda row: 0 if (row['TiempoEspera'] > 90 or row['Edad'] < 18 and row['CitasPrevias'] < 2) else row['Asistio'],
    axis=1
)

print("--- Vista Previa de los Datos ---")
print(df.head())
print("\n--- Información del DataFrame ---")
df.info()


# --- 2. Separación de Datos y Preprocesamiento ---
# Variable objetivo (y) y características (X)
X = df.drop('Asistio', axis=1)
y = df['Asistio']

# Identificar columnas numéricas y categóricas
columnas_numericas = ['Edad', 'TiempoEspera', 'CitasPrevias']
columnas_categoricas = ['DiaSemana']

# Crear un transformador para preprocesar las columnas
# - Las numéricas se escalarán (estandarizarán).
# - Las categóricas se convertirán a formato numérico (One-Hot Encoding).
preprocesador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), columnas_numericas),
        ('cat', OneHotEncoder(), columnas_categoricas)
    ])

# Dividir los datos en conjuntos de entrenamiento y prueba (80% / 20%)
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --- 3. Creación y Entrenamiento del Modelo ---
# Se usará un Pipeline para encadenar el preprocesamiento y el modelo
# Esto evita la fuga de datos y simplifica el código.
modelo_logistico = Pipeline(steps=[
    ('preprocesador', preprocesador),
    ('clasificador', LogisticRegression(random_state=42))
])

# Entrenar el modelo
print("\n--- Entrenando el modelo de Regresión Logística... ---")
modelo_logistico.fit(X_entrena, y_entrena)
print("¡Modelo entrenado!")


# --- 4. Evaluación del Modelo ---
# Realizar predicciones en el conjunto de prueba
y_pred = modelo_logistico.predict(X_prueba)
y_pred_proba = modelo_logistico.predict_proba(X_prueba)[:, 1] # Probabilidades para la clase '1' (Asistió)

# a) Exactitud (Accuracy)
accuracy = accuracy_score(y_prueba, y_pred)
print(f'\nExactitud del modelo: {accuracy * 100:.2f}%')

# b) Matriz de Confusión
conf_matrix = confusion_matrix(y_prueba, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Asistió', 'Sí Asistió'],
            yticklabels=['No Asistió', 'Sí Asistió'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

# c) Reporte de Clasificación (Precisión, Recall, F1-Score)
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_prueba, y_pred, target_names=['No Asistió', 'Sí Asistió']))

# d) Curva ROC y Área Bajo la Curva (AUC)
auc = roc_auc_score(y_prueba, y_pred_proba)
print(f'Área Bajo la Curva (AUC-ROC): {auc:.4f}')

fpr, tpr, _ = roc_curve(y_prueba, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()


# --- 5. Ejemplo de Predicción con Nuevos Datos ---
# Creación de un nuevo paciente (DataFrame)
nuevo_paciente = pd.DataFrame({
    'Edad': [35],
    'TiempoEspera': [15],
    'CitasPrevias': [4],
    'DiaSemana': ['Martes']
})

# Predecir si asistirá
prediccion = modelo_logistico.predict(nuevo_paciente)[0]
# Predecir la probabilidad de que asista
probabilidad = modelo_logistico.predict_proba(nuevo_paciente)[0][1]

print("\n--- Ejemplo de Predicción Individual ---")
print(f"Datos del paciente:\n{nuevo_paciente.to_string(index=False)}\n")
print(f"Predicción: {'Sí asistirá' if prediccion == 1 else 'No asistirá'}")
print(f"Probabilidad de que asista: {probabilidad * 100:.2f}%")


# --- 6. Guardar el Modelo y los Datos de Prueba para la App Web ---
print("\n--- Guardando artefactos para la aplicación web... ---")
joblib.dump(modelo_logistico, 'modelo_citas.joblib')
X_prueba.to_csv('X_prueba.csv', index=False)
y_prueba.to_csv('y_prueba.csv', index=False)
print("¡Modelo y datos de prueba guardados!")
