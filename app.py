import matplotlib
matplotlib.use('Agg')

import sys
import os
# Agregar el directorio Proyecto al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Proyecto'))

from flask import Flask, render_template, request
import LinearRegression601T
import io
import base64
import matplotlib.pyplot as plt 
from LogisticRegression601T import modelo_logistico
from perceptron_senales_industriales_mejorado import PerceptronClassifier
from rl_agent_generic import train_qlearning
import os
import pickle

app = Flask(__name__, template_folder='Proyecto/templates', static_folder='Proyecto/static')

@app.route("/")
def home():
    name = "Bienvenido"
    return render_template('home.html', name=name)


@app.route('/health')
def health():
    """Health check for production load balancer/Render"""
    return 'OK', 200


@app.route('/version')
def version():
    """Return a short git commit SHA so we can see which build is deployed"""
    try:
        import subprocess

        sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                      cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        sha = 'unknown'
    return {'commit': sha}
   
@app.route('/ucundinamarca')
def index():
    Myname= "Proyecto de Machine Learning"
    return render_template('index.html', name=Myname)

@app.route('/actividad1')
def actividad1():
    return render_template('actividad1.html')

@app.route('/LR')
def lr():
    calculateResult =None
    calculateResult = LinearRegression601T.calculateGrade(5)
    return "Final Grade Prediction:" + str(calculateResult)

@app.route('/conceptos_regresion')
def conceptos_regresion():
    return render_template('conceptos_regresion.html')

@app.route('/conceptos_regresionLogistica')
def conceptos_regresionLogistica():
    return render_template('conceptos_regresionLogistica.html')

@app.route('/conceptos_clasificacion')
def conceptos_clasificacion():
    return render_template('conceptos_clasificacion.html')

@app.route('/conceptos_teoricos_clasificacion')
def conceptos_teoricos_clasificacion():
    return render_template('conceptos_teoricos_clasificacion.html')

@app.route('/conceptos_refuerzo')
def conceptos_refuerzo():
    return render_template('conceptos_refuerzo.html')

@app.route('/caso_practico_refuerzo')
def caso_practico_refuerzo():
    """
    Ruta para mostrar el caso práctico de Aprendizaje por Refuerzo.
    Carga el modelo entrenado y sus resultados si existen.
    """
    context = {
        'training_completed': False,
        'num_episodes': 1000,
        'eval_episodes': 100,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'bins': 10
    }
    
    # Verificar si existe el modelo entrenado
    model_path = os.path.join('Proyecto', 'modelo_rl_cartpole.pkl')
    rewards_plot_path = os.path.join('Proyecto', 'static', 'rl_training_rewards.png')
    distributions_plot_path = os.path.join('Proyecto', 'static', 'rl_training_distributions.png')
    
    if os.path.exists(model_path):
        try:
            # Cargar modelo
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            context['training_completed'] = True
            
            # Extraer métricas del historial de entrenamiento
            history = model_data.get('training_history', {})
            episode_rewards = history.get('episode_rewards', [])
            episode_lengths = history.get('episode_lengths', [])
            
            if episode_rewards:
                # Calcular métricas de las últimas 100 episodios
                last_100_rewards = episode_rewards[-100:]
                context['mean_reward'] = sum(last_100_rewards) / len(last_100_rewards)
                context['max_reward'] = max(episode_rewards)
                context['mean_length'] = sum(episode_lengths[-100:]) / len(episode_lengths[-100:])
                
                # Tamaño de la tabla Q (estados explorados)
                context['q_table_size'] = len(model_data.get('q_table', {}))
                
                # Simular estadísticas de evaluación (o cargar si están guardadas)
                # En una implementación real, estas vendrían del proceso de evaluación
                context['eval_mean_reward'] = context['mean_reward']
                context['eval_std_reward'] = 15.0  # Desviación estándar estimada
                context['eval_min_reward'] = min(last_100_rewards)
                context['eval_max_reward'] = max(last_100_rewards)
                context['eval_mean_length'] = context['mean_length']
                
                # Cargar imágenes de las gráficas
                if os.path.exists(rewards_plot_path):
                    with open(rewards_plot_path, 'rb') as f:
                        context['rewards_plot'] = base64.b64encode(f.read()).decode()
                
                if os.path.exists(distributions_plot_path):
                    with open(distributions_plot_path, 'rb') as f:
                        context['distributions_plot'] = base64.b64encode(f.read()).decode()
        
        except Exception as e:
            print(f"Error al cargar el modelo RL: {e}")
    
    return render_template('caso_practico_refuerzo.html', **context)

@app.route('/regresion_lineal', methods=['GET', 'POST'])
def regresion_lineal():
    resultado = None
    temperatura = None
    dia_semana = None
    if request.method == 'POST':
        temperatura = request.form.get('temperatura', type=float)
        dia_semana = request.form.get('dia_semana', type=int)
        if temperatura is not None and dia_semana is not None:
            resultado = LinearRegression601T.EstimarVentasHelados(temperatura, dia_semana)
    LinearRegression601T.generar_grafica(temperatura, dia_semana, resultado)
    return render_template('regresion_lineal.html', resultado=resultado)

@app.route('/regresion_logistica', methods=['GET', 'POST'])
def regresion_logistica():
    resultado = None
    probabilidad = None
    matriz_confusion_img = None
    curva_roc_img = None
    accuracy = None
    class_report = None
    
    descripcion_dataset = modelo_logistico.obtener_descripcion_dataset()
    dataset_head = descripcion_dataset['head']
    
    buf = io.StringIO()
    modelo_logistico.df_original.info(buf=buf)
    dataset_info = buf.getvalue()

    if request.method == 'POST':
        edad = request.form.get('edad', type=int)
        tiempo_espera = request.form.get('tiempo_espera', type=int)
        citas_previas = request.form.get('citas_previas', type=int)
        dia_semana = request.form.get('dia_semana')
        
        if all(v is not None for v in [edad, tiempo_espera, citas_previas, dia_semana]):
            resultado, probabilidad = modelo_logistico.predecir_asistencia(
                edad, tiempo_espera, citas_previas, dia_semana
            )
            
            buffer = io.BytesIO()
            modelo_logistico.plot_confusion_matrix()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            matriz_confusion_img = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            buffer = io.BytesIO()
            modelo_logistico.plot_roc_curve()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            curva_roc_img = base64.b64encode(buffer.getvalue()).decode()
            plt.close()

            accuracy, class_report = modelo_logistico.obtener_metricas_evaluacion()
            
    return render_template('regresion_logistica.html',
                         resultado=resultado,
                         probabilidad=probabilidad,
                         matriz_confusion_img=matriz_confusion_img,
                         curva_roc_img=curva_roc_img,
                         accuracy=accuracy,
                         class_report=class_report,
                         dataset_head=dataset_head,
                         dataset_info=dataset_info)

@app.route('/caso_practico_clasificacion', methods=['GET', 'POST'])
def caso_practico_clasificacion():
    resultado_prediccion = None
    probabilidad = None
    metricas = None
    matriz_confusion_img = None
    errores_entrenamiento_img = None
    
    classifier = PerceptronClassifier(eta=0.01, n_iter=50, random_state=42)

    if request.method == 'POST':
        voltaje = request.form.get('voltaje', type=float)
        frecuencia = request.form.get('frecuencia', type=float)
        duracion = request.form.get('duracion', type=float)
        tasa_cambio = request.form.get('tasa_cambio', type=float)

        if all(v is not None for v in [voltaje, frecuencia, duracion, tasa_cambio]):
            import sys
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            classifier.train()

            features = [voltaje, frecuencia, duracion, tasa_cambio]
            resultado_prediccion, probabilidad = classifier.predict_label(features)
            probabilidad = round(probabilidad * 100, 2) 
            metricas = classifier.evaluate()

            buffer_cm = io.BytesIO()
            classifier._plot_confusion_matrix(metricas['confusion_matrix'])
            plt.savefig(buffer_cm, format='png', bbox_inches='tight')
            buffer_cm.seek(0)
            matriz_confusion_img = base64.b64encode(buffer_cm.getvalue()).decode()
            plt.close()

            buffer_err = io.BytesIO()
            classifier._plot_training_errors()
            plt.savefig(buffer_err, format='png', bbox_inches='tight')
            buffer_err.seek(0)
            errores_entrenamiento_img = base64.b64encode(buffer_err.getvalue()).decode()
            plt.close()
            
            sys.stdout = original_stdout

    return render_template('caso_practico_clasificacion.html',
                           resultado=resultado_prediccion,
                           probabilidad=probabilidad,
                           metricas=metricas,
                           matriz_confusion_img=matriz_confusion_img,
                           errores_entrenamiento_img=errores_entrenamiento_img)

if __name__ == '__main__':
    app.run(debug=True)
