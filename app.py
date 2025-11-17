import sys
import os
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

# Agregar el directorio Proyecto al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Proyecto'))

import LinearRegression601T
from LogisticRegression601T import modelo_logistico
from perceptron_senales_industriales_mejorado import PerceptronClassifier
from rl_agent_generic import train_qlearning


app = Flask(__name__,
            template_folder='Proyecto/templates',
            static_folder='Proyecto/static')


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
    """Return git commit SHA for deployment verification"""
    try:
        import subprocess
        sha = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        sha = 'unknown'
    return {'commit': sha}


@app.route('/ucundinamarca')
def index():
    myname = "Proyecto de Machine Learning"
    return render_template('index.html', name=myname)


@app.route('/actividad1')
def actividad1():
    return render_template('actividad1.html')


@app.route('/LR')
def lr():
    calculate_result = None
    calculate_result = LinearRegression601T.calculateGrade(5)
    return "Final Grade Prediction:" + str(calculate_result)


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


@app.route('/caso_practico_refuerzo', methods=['GET', 'POST'])
def caso_practico_refuerzo():
    """
    Ruta interactiva para Aprendizaje por Refuerzo.
    GET: Muestra formulario con parámetros por defecto.
    POST: Entrena un agente Q-Learning con los parámetros del usuario.
    """
    context = {
        'training_completed': False,
        'num_episodes': 500,
        'eval_episodes': 50,
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.01,
        'bins': 10,
        'env_id': 'CartPole-v1'
    }

    if request.method == 'POST':
        try:
            # Obtener parámetros del formulario
            env_id = request.form.get('env_id', 'CartPole-v1')
            num_episodes = int(request.form.get('episodes', 500))
            learning_rate = float(request.form.get('learning_rate', 0.1))
            discount_factor = float(
                request.form.get('discount_factor', 0.99))
            epsilon = float(request.form.get('epsilon', 1.0))
            epsilon_decay = float(
                request.form.get('epsilon_decay', 0.995))
            epsilon_min = float(request.form.get('epsilon_min', 0.01))
            bins = int(request.form.get('bins', 10))

            # Entrenar el agente
            results = train_qlearning(
                env_id=env_id,
                episodes=num_episodes,
                learning_rate=learning_rate,
                discount_factor=discount_factor,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                epsilon_min=epsilon_min,
                bins=bins
            )

            # Extraer evaluación
            eval_stats = results.get('eval', {})
            history = results.get('history', {})

            # Actualizar contexto con resultados
            context.update({
                'training_completed': True,
                'env_id': env_id,
                'num_episodes': num_episodes,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'epsilon': epsilon,
                'epsilon_decay': epsilon_decay,
                'epsilon_min': epsilon_min,
                'bins': bins,
                'mean_reward': history.get(
                    'avg_rewards', [0])[-1] if history.get(
                        'avg_rewards') else 0,
                'max_reward': max(history.get(
                    'episode_rewards', [0])),
                'mean_steps': history.get(
                    'episode_lengths', [0])[0] if history else 0,
                'states_explored': results.get(
                    'q_table_size', 0),
                'eval_mean_reward': eval_stats.get(
                    'mean_reward', 0),
                'eval_std_reward': eval_stats.get(
                    'std_reward', 0),
                'eval_min_reward': eval_stats.get(
                    'min_reward', 0),
                'eval_max_reward': eval_stats.get(
                    'max_reward', 0),
                'eval_mean_steps': eval_stats.get(
                    'mean_length', 0),
                'rewards_plot': results.get(
                    'rewards_plot', ''),
                'distributions_plot': results.get(
                    'distributions_plot', '')
            })

        except Exception as e:
            context['error'] = f"Error al entrenar: {str(e)}"
            import traceback
            traceback.print_exc()

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
            resultado = (
                LinearRegression601T.EstimarVentasHelados(
                    temperatura, dia_semana))
    LinearRegression601T.generar_grafica(
        temperatura, dia_semana, resultado)
    return render_template('regresion_lineal.html', resultado=resultado)


@app.route('/regresion_logistica', methods=['GET', 'POST'])
def regresion_logistica():
    resultado = None
    probabilidad = None
    matriz_confusion_img = None
    curva_roc_img = None
    accuracy = None
    class_report = None

    descripcion_dataset = (
        modelo_logistico.obtener_descripcion_dataset())
    dataset_head = descripcion_dataset['head']

    buf = io.StringIO()
    modelo_logistico.df_original.info(buf=buf)
    dataset_info = buf.getvalue()

    if request.method == 'POST':
        edad = request.form.get('edad', type=int)
        tiempo_espera = request.form.get('tiempo_espera', type=int)
        citas_previas = request.form.get('citas_previas', type=int)
        dia_semana = request.form.get('dia_semana')

        if all(v is not None for v in [edad, tiempo_espera,
                                       citas_previas, dia_semana]):
            resultado, probabilidad = (
                modelo_logistico.predecir_asistencia(
                    edad, tiempo_espera, citas_previas, dia_semana
                ))

            buffer = io.BytesIO()
            modelo_logistico.plot_confusion_matrix()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            matriz_confusion_img = (
                base64.b64encode(buffer.getvalue()).decode())
            plt.close()

            buffer = io.BytesIO()
            modelo_logistico.plot_roc_curve()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            curva_roc_img = (
                base64.b64encode(buffer.getvalue()).decode())
            plt.close()

            accuracy, class_report = (
                modelo_logistico.obtener_metricas_evaluacion())

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

    classifier = PerceptronClassifier(
        eta=0.01, n_iter=50, random_state=42)

    if request.method == 'POST':
        voltaje = request.form.get('voltaje', type=float)
        frecuencia = request.form.get('frecuencia', type=float)
        duracion = request.form.get('duracion', type=float)
        tasa_cambio = request.form.get('tasa_cambio', type=float)

        if all(v is not None for v in [voltaje, frecuencia,
                                       duracion, tasa_cambio]):
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            classifier.train()

            features = [voltaje, frecuencia, duracion, tasa_cambio]
            resultado_prediccion, probabilidad = (
                classifier.predict_label(features))
            probabilidad = round(probabilidad * 100, 2)
            metricas = classifier.evaluate()

            buffer_cm = io.BytesIO()
            classifier._plot_confusion_matrix(
                metricas['confusion_matrix'])
            plt.savefig(buffer_cm, format='png', bbox_inches='tight')
            buffer_cm.seek(0)
            matriz_confusion_img = (
                base64.b64encode(buffer_cm.getvalue()).decode())
            plt.close()

            buffer_err = io.BytesIO()
            classifier._plot_training_errors()
            plt.savefig(buffer_err, format='png', bbox_inches='tight')
            buffer_err.seek(0)
            errores_entrenamiento_img = (
                base64.b64encode(buffer_err.getvalue()).decode())
            plt.close()

            sys.stdout = original_stdout

    return render_template('caso_practico_clasificacion.html',
                           resultado=resultado_prediccion,
                           probabilidad=probabilidad,
                           metricas=metricas,
                           matriz_confusion_img=matriz_confusion_img,
                           errores_entrenamiento_img=(
                               errores_entrenamiento_img))


if __name__ == '__main__':
    app.run(debug=True)
