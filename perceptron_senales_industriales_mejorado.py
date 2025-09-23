"""
Perceptrón para Clasificación de Señales Eléctricas en Control Industrial
========================================================================

Caso: Clasificación de señales eléctricas para determinar el estado del sistema
Variables independientes:
- Voltaje promedio (V): Nivel de tensión de la señal eléctrica
- Frecuencia (Hz): Número de oscilaciones por segundo
- Duración de la señal (ms): Tiempo de duración de la medición
- Tasa de cambio (V/s): Velocidad de variación del voltaje

Variable objetivo (binaria):
- 0: Sistema Apagado (voltaje bajo, frecuencia inestable)
- 1: Sistema Encendido (voltaje nominal, frecuencia estable)

Significado de las clases:
- Clase "No" (0): Sistema en estado inactivo o defectuoso
- Clase "Sí" (1): Sistema funcionando correctamente
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import seaborn as sns

class Perceptron(object):
    """Clasificador Perceptrón para señales eléctricas industriales"""
    
    def __init__(self, eta=0.1, n_iter=1000, random_state=1):
        """
        Inicializar el perceptrón
        
        Parámetros:
        -----------
        eta : float
            Tasa de aprendizaje (entre 0.0 y 1.0)
        n_iter : int
            Número de épocas de entrenamiento
        random_state : int
            Semilla para el generador de números aleatorios
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Ajuste de los datos de entrenamiento
        
        Parámetros:
        -----------
        X : {array-like}, shape = [n_samples, n_features]
            Vector de entradas de entrenamiento
        y : array-like, shape = [n_samples]
            Vector de salidas (0 o 1)
            
        Retorna:
        --------
        self : object
        """
        # Generador de números aleatorios
        rgen = np.random.RandomState(self.random_state)
        
        # Inicializar pesos: w_0 es el sesgo, w_1...w_n son los pesos de las características
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        
        # Convertir etiquetas 0,1 a -1,1 para el algoritmo del perceptrón
        y_train = np.where(y == 0, -1, 1)
        
        self.errors_ = []  # Lista para almacenar errores por época
        
        # Entrenamiento por épocas
        for epoca in range(self.n_iter):
            errores = 0
            
            # Iterar sobre cada muestra
            for xi, target in zip(X, y_train):
                # Calcular actualización de pesos
                update = self.eta * (target - self.predict(xi))
                
                # Actualizar pesos y sesgo
                self.w_[1:] += update * xi  # Pesos de las características
                self.w_[0] += update        # Sesgo
                
                # Contar errores
                errores += int(update != 0.0)
            
            self.errors_.append(errores)
            
            # Convergencia temprana si no hay errores
            if errores == 0:
                print(f"Convergencia alcanzada en la época {epoca + 1}")
                break
                
        return self
    
    def entrada_neta(self, X):
        """Calcular la entrada neta (combinación lineal)"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        """Retornar la etiqueta de clase después de la función escalón"""
        return np.where(self.entrada_neta(X) >= 0.0, 1, -1)
    
    def predict_binary(self, X):
        """Retornar predicciones en formato binario (0/1)"""
        predictions = self.predict(X)
        return np.where(predictions == -1, 0, 1)
    
    def decision_function(self, X):
        """Retornar el valor de la función de decisión"""
        return self.entrada_neta(X)


class PerceptronClassifier:
    """
    Clasificador Perceptrón con interfaz estándar para señales eléctricas industriales
    """
    
    def __init__(self, eta=0.1, n_iter=1000, random_state=42):
        """
        Inicializar el clasificador
        
        Parámetros:
        -----------
        eta : float
            Tasa de aprendizaje
        n_iter : int
            Número máximo de épocas
        random_state : int
            Semilla para reproducibilidad
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.pipeline = None
        self.perceptron = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        
    def load_and_split_data(self, test_size=0.2):
        """
        Cargar y dividir los datos de señales eléctricas
        
        Parámetros:
        -----------
        test_size : float
            Proporción del conjunto de prueba
            
        Retorna:
        --------
        X_train, X_test, y_train, y_test : arrays
            Datos divididos
        """
        # Generar datos sintéticos de señales eléctricas
        data = self._generar_datos_senales()
        
        # Preparar características y etiquetas
        X = data[['voltaje_promedio', 'frecuencia', 'duracion_señal', 'tasa_cambio']].values
        y = data['estado_sistema'].values
        
        # División entrenamiento/prueba con control de semilla
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Guardar datos de prueba para evaluación
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def _generar_datos_senales(self):
        """Generar datos sintéticos de señales eléctricas industriales"""
        np.random.seed(self.random_state)  # Control de semilla
        n_samples = 1000
        
        # Sistema ENCENDIDO (clase 1): voltaje alto, frecuencia estable
        voltaje_on = np.random.normal(220, 15, n_samples//2)
        frecuencia_on = np.random.normal(50, 2, n_samples//2)
        duracion_on = np.random.normal(500, 50, n_samples//2)
        tasa_cambio_on = np.random.normal(10, 3, n_samples//2)
        
        # Sistema APAGADO (clase 0): voltaje bajo, frecuencia variable
        voltaje_off = np.random.normal(50, 20, n_samples//2)
        frecuencia_off = np.random.normal(30, 10, n_samples//2)
        duracion_off = np.random.normal(100, 30, n_samples//2)
        tasa_cambio_off = np.random.normal(25, 8, n_samples//2)
        
        # Combinar datos
        voltaje = np.concatenate([voltaje_on, voltaje_off])
        frecuencia = np.concatenate([frecuencia_on, frecuencia_off])
        duracion = np.concatenate([duracion_on, duracion_off])
        tasa_cambio = np.concatenate([tasa_cambio_on, tasa_cambio_off])
        labels = np.concatenate([np.ones(n_samples//2), np.zeros(n_samples//2)])
        
        # Crear DataFrame y mezclar
        data = pd.DataFrame({
            'voltaje_promedio': voltaje,
            'frecuencia': frecuencia,
            'duracion_señal': duracion,
            'tasa_cambio': tasa_cambio,
            'estado_sistema': labels
        })
        
        return data.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
    
    def train(self):
        """
        Entrenar el modelo con Pipeline para evitar data leakage
        """
        # Cargar y dividir datos
        X_train, X_test, y_train, y_test = self.load_and_split_data()
        
        # Crear perceptrón
        self.perceptron = Perceptron(eta=self.eta, n_iter=self.n_iter, 
                                   random_state=self.random_state)
        
        # Pipeline con escalado estándar (evita data leakage)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('perceptron', self.perceptron)
        ])
        
        # Entrenar modelo
        print("Entrenando Perceptrón...")
        self.pipeline.fit(X_train, y_train)
        
        # Realizar predicciones en conjunto de prueba usando predict_binary
        X_test_scaled = self.pipeline.named_steps['scaler'].transform(X_test)
        self.y_pred = self.pipeline.named_steps['perceptron'].predict_binary(X_test_scaled)
        
        print("Entrenamiento completado.")
        return self
    
    def evaluate(self):
        """
        Evaluar el modelo y generar métricas e imágenes
        
        Retorna:
        --------
        metrics : dict
            Diccionario con métricas de evaluación
        """
        if self.y_pred is None:
            raise ValueError("Debe entrenar el modelo primero usando train()")
        
        # Calcular métricas
        accuracy = accuracy_score(self.y_test, self.y_pred)
        
        # Reporte de clasificación
        report = classification_report(self.y_test, self.y_pred, 
                                     target_names=['No', 'Sí'], 
                                     output_dict=True)
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Crear métricas organizadas
        metrics = {
            'accuracy': round(accuracy, 4),
            'precision_No': round(report['No']['precision'], 4),
            'precision_Sí': round(report['Sí']['precision'], 4),
            'recall_No': round(report['No']['recall'], 4),
            'recall_Sí': round(report['Sí']['recall'], 4),
            'f1_No': round(report['No']['f1-score'], 4),
            'f1_Sí': round(report['Sí']['f1-score'], 4),
            'support_No': report['No']['support'],
            'support_Sí': report['Sí']['support'],
            'confusion_matrix': cm
        }
        
        # Generar visualizaciones
        self._plot_confusion_matrix(cm)
        self._plot_training_errors()
        
        # Mostrar métricas
        print(f"\n{'='*50}")
        print("MÉTRICAS DE EVALUACIÓN")
        print(f"{'='*50}")
        print(f"Exactitud (Accuracy): {metrics['accuracy']}")
        print(f"\nReporte de Clasificación:")
        print(f"{'Clase':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print(f"{'-'*50}")
        print(f"{'No':<8} {metrics['precision_No']:<10} {metrics['recall_No']:<10} {metrics['f1_No']:<10} {metrics['support_No']:<8}")
        print(f"{'Sí':<8} {metrics['precision_Sí']:<10} {metrics['recall_Sí']:<10} {metrics['f1_Sí']:<10} {metrics['support_Sí']:<8}")
        
        return metrics
    
    def predict_label(self, features, threshold=0.5):
        """
        Predecir etiqueta para nuevas características
        
        Parámetros:
        -----------
        features : array-like
            Vector de características [voltaje, frecuencia, duración, tasa_cambio]
        threshold : float
            Umbral de decisión (no usado en perceptrón, incluido por compatibilidad)
            
        Retorna:
        --------
        label : str
            "Sí" o "No"
        probability : float
            Pseudo-probabilidad basada en la distancia a la frontera de decisión
        """
        if self.pipeline is None:
            raise ValueError("Debe entrenar el modelo primero usando train()")
        
        # Asegurar que features es array 2D
        features = np.array(features).reshape(1, -1)
        
        # Realizar predicción usando predict_binary
        scaler = self.pipeline.named_steps['scaler']
        perceptron = self.pipeline.named_steps['perceptron']
        
        features_scaled = scaler.transform(features)
        prediction = perceptron.predict_binary(features_scaled)[0]
        label = "Sí" if prediction == 1 else "No"
        
        # Calcular pseudo-probabilidad basada en distancia a frontera de decisión
        decision_value = perceptron.decision_function(features_scaled)[0]
        
        # Convertir valor de decisión a pseudo-probabilidad usando sigmoid
        probability = 1 / (1 + np.exp(-decision_value))
        
        return label, round(probability, 4)
    
    def _plot_confusion_matrix(self, cm):
        """Graficar matriz de confusión"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No (Predicho)', 'Sí (Predicho)'],
                   yticklabels=['No (Real)', 'Sí (Real)'])
        plt.title('Matriz de Confusión 2×2\n(Filas=Reales, Columnas=Predichos)')
        plt.ylabel('Valores Reales')
        plt.xlabel('Predicciones')
        plt.tight_layout()
        plt.show()
    
    def _plot_training_errors(self):
        """Graficar evolución de errores durante entrenamiento"""
        if hasattr(self.perceptron, 'errors_'):
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(self.perceptron.errors_) + 1), self.perceptron.errors_, 
                    marker='o', linewidth=2, markersize=6)
            plt.xlabel('Época')
            plt.ylabel('Número de Errores')
            plt.title('Evolución de Errores Durante el Entrenamiento')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


# Funciones auxiliares para compatibilidad y demostración

def generar_datos_senales():
    """Función auxiliar para generar datos (mantiene compatibilidad)"""
    classifier = PerceptronClassifier()
    return classifier._generar_datos_senales()

def visualizar_datos(data):
    """Visualizar la distribución de los datos"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    variables = ['voltaje_promedio', 'frecuencia', 'duracion_señal', 'tasa_cambio']
    titulos = ['Voltaje Promedio (V)', 'Frecuencia (Hz)', 'Duración Señal (ms)', 'Tasa de Cambio (V/s)']
    
    for i, (var, titulo) in enumerate(zip(variables, titulos)):
        ax = axes[i//2, i%2]
        
        # Datos por clase
        encendido = data[data['estado_sistema'] == 1][var]
        apagado = data[data['estado_sistema'] == 0][var]
        
        ax.hist(apagado, alpha=0.7, label='No (Apagado)', bins=30, color='red')
        ax.hist(encendido, alpha=0.7, label='Sí (Encendido)', bins=30, color='green')
        
        ax.set_xlabel(titulo)
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de {titulo}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demo_clasificacion():
    """Demostración completa del clasificador"""
    # Crear y entrenar clasificador
    classifier = PerceptronClassifier(eta=0.1, n_iter=1000, random_state=42)
    classifier.train()
    
    # Evaluar modelo
    metrics = classifier.evaluate()
    
    # Ejemplos de predicción
    print(f"\n{'='*50}")
    print("EJEMPLOS DE PREDICCIÓN")
    print(f"{'='*50}")
    
    # Ejemplo 1: Sistema típicamente encendido
    features1 = [215, 49, 480, 12]  # Voltaje alto, frecuencia estable
    label1, prob1 = classifier.predict_label(features1)
    print(f"Ejemplo 1 - Voltaje: 215V, Frecuencia: 49Hz, Duración: 480ms, Tasa: 12V/s")
    print(f"Predicción: {label1}, Probabilidad: {prob1}")
    
    # Ejemplo 2: Sistema típicamente apagado
    features2 = [45, 25, 80, 30]  # Voltaje bajo, frecuencia variable
    label2, prob2 = classifier.predict_label(features2)
    print(f"\nEjemplo 2 - Voltaje: 45V, Frecuencia: 25Hz, Duración: 80ms, Tasa: 30V/s")
    print(f"Predicción: {label2}, Probabilidad: {prob2}")
    
    # Ejemplo 3: Caso límite
    features3 = [120, 40, 250, 18]  # Valores intermedios
    label3, prob3 = classifier.predict_label(features3)
    print(f"\nEjemplo 3 - Voltaje: 120V, Frecuencia: 40Hz, Duración: 250ms, Tasa: 18V/s")
    print(f"Predicción: {label3}, Probabilidad: {prob3}")
    
    return classifier, metrics

def main():
    """Función principal"""
    print("=" * 70)
    print("PERCEPTRÓN PARA CLASIFICACIÓN DE SEÑALES ELÉCTRICAS INDUSTRIALES")
    print("=" * 70)
    print("\nSignificado de las clases:")
    print("- Clase 'No' (0): Sistema Apagado/Defectuoso")
    print("- Clase 'Sí' (1): Sistema Encendido/Funcionando")
    
    # Generar y visualizar datos
    print("\n1. Generando y visualizando datos...")
    data = generar_datos_senales()
    print(f"Dataset generado con {len(data)} muestras")
    print(f"Distribución de clases:")
    print(data['estado_sistema'].value_counts())
    
    # Mostrar algunas muestras
    print(f"\nPrimeras 5 muestras:")
    print(data.head())
    
    # Visualizar distribución
    visualizar_datos(data)
    
    # Demostración completa
    print("\n2. Entrenamiento y evaluación del modelo...")
    classifier, metrics = demo_clasificacion()
    
    print(f"\n{'='*70}")
    print("ANÁLISIS COMPLETADO")
    print(f"{'='*70}")
    
    return classifier, metrics

if __name__ == "__main__":
    # Ejecutar demostración completa
    clasificador, metricas = main()