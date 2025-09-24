"""
DEMOSTRACIÓN DE INTERFAZ ESTÁNDAR - PERCEPTRÓN
==============================================

Este script demuestra el uso de las funciones expuestas con interfaz estándar:
- evaluate() → retorna métricas (dict) y genera imágenes
- predict_label(features, threshold=0.5) → retorna "Sí"/"No" y probabilidad

Equipo: Perceptrón para Señales Eléctricas Industriales
"""

from perceptron_senales_industriales_mejorado import PerceptronClassifier
import numpy as np

def demo_interfaz_estandar():
    """
    Demostración de la interfaz estándar requerida
    """
    print("=" * 60)
    print("DEMOSTRACIÓN DE INTERFAZ ESTÁNDAR - PERCEPTRÓN")
    print("=" * 60)
    
    # 1. Crear y entrenar el clasificador
    print("\n1. ENTRENAMIENTO DEL MODELO")
    print("-" * 30)
    
    # Inicializar con control de semilla para reproducibilidad
    classifier = PerceptronClassifier(eta=0.1, n_iter=1000, random_state=42)
    
    # Entrenar modelo (incluye carga de datos, split 80/20, preprocesamiento con Pipeline)
    classifier.train()
    
    # 2. Función evaluate() - Retorna métricas y genera imágenes
    print("\n2. FUNCIÓN evaluate() - EVALUACIÓN Y MÉTRICAS")
    print("-" * 50)
    
    metrics = classifier.evaluate()
    
    print(f"\nMétricas retornadas por evaluate():")
    print(f"- Tipo: {type(metrics)}")
    print(f"- Claves disponibles: {list(metrics.keys())}")
    
    # 3. Función predict_label() - Predicción individual
    print(f"\n3. FUNCIÓN predict_label() - PREDICCIÓN INDIVIDUAL")
    print("-" * 55)
    
    # Casos de prueba definidos
    casos_prueba = [
        {
            'nombre': 'Sistema Encendido Típico',
            'features': [220, 50, 500, 10],  # Voltaje alto, frecuencia estable
            'descripcion': 'Voltaje: 220V, Frecuencia: 50Hz, Duración: 500ms, Tasa: 10V/s'
        },
        {
            'nombre': 'Sistema Apagado Típico',
            'features': [50, 30, 100, 25],   # Voltaje bajo, frecuencia variable
            'descripcion': 'Voltaje: 50V, Frecuencia: 30Hz, Duración: 100ms, Tasa: 25V/s'
        },
        {
            'nombre': 'Caso Límite',
            'features': [150, 42, 300, 15],  # Valores intermedios
            'descripcion': 'Voltaje: 150V, Frecuencia: 42Hz, Duración: 300ms, Tasa: 15V/s'
        }
    ]
    
    for i, caso in enumerate(casos_prueba, 1):
        print(f"\nCaso {i}: {caso['nombre']}")
        print(f"Características: {caso['descripcion']}")
        
        # Llamar a predict_label con threshold por defecto
        label, probability = classifier.predict_label(caso['features'])
        
        print(f"Resultado: Etiqueta='{label}', Probabilidad={probability}")
        
        # Demostrar uso con threshold personalizado
        label_custom, prob_custom = classifier.predict_label(caso['features'], threshold=0.6)
        print(f"Con threshold=0.6: Etiqueta='{label_custom}', Probabilidad={prob_custom}")
    
    # 4. Demostrar reproducibilidad y buenas prácticas
    print(f"\n4. BUENAS PRÁCTICAS IMPLEMENTADAS")
    print("-" * 40)
    
    print("✓ Control de semilla (random_state=42)")
    print("✓ Split entrenamiento/prueba (80/20)")
    print("✓ Pipeline para evitar data leakage")
    print("✓ Escalado estándar después del split")
    print("✓ Matriz de confusión 2×2 (filas=reales, columnas=predichos)")
    print("✓ Métricas con 2-4 decimales")
    print("✓ Interfaz estándar: evaluate() y predict_label()")
    
    # 5. Resumen de métricas
    print(f"\n5. RESUMEN DE MÉTRICAS FINALES")
    print("-" * 35)
    
    print(f"Exactitud: {metrics['accuracy']}")
    print(f"Precisión (No): {metrics['precision_No']}")
    print(f"Precisión (Sí): {metrics['precision_Sí']}")
    print(f"Recall (No): {metrics['recall_No']}")
    print(f"Recall (Sí): {metrics['recall_Sí']}")
    print(f"F1-Score (No): {metrics['f1_No']}")
    print(f"F1-Score (Sí): {metrics['f1_Sí']}")
    
    print(f"\nMatriz de Confusión:")
    print(f"[[{metrics['confusion_matrix'][0][0]:3d} {metrics['confusion_matrix'][0][1]:3d}]")
    print(f" [{metrics['confusion_matrix'][1][0]:3d} {metrics['confusion_matrix'][1][1]:3d}]]")
    
    print(f"\n{'='*60}")
    print("DEMOSTRACIÓN COMPLETADA EXITOSAMENTE")
    print(f"{'='*60}")
    
    return classifier, metrics

if __name__ == "__main__":
    # Ejecutar demostración
    clasificador_demo, metricas_demo = demo_interfaz_estandar()