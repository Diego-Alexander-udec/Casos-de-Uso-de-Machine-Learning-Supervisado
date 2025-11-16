import matplotlib
matplotlib.use('Agg')  

from flask import Flask, render_template, request
import LinearRegression601T
import io
import base64
import matplotlib.pyplot as plt 
from LogisticRegression601T import modelo_logistico
from perceptron_senales_industriales_mejorado import PerceptronClassifier

app = Flask(__name__)

@app.route("/")
def home():
    name = "Bienvenido"
    return render_template('home.html', name=name)
   
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
    return render_template('caso_practico_refuerzo.html')

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
