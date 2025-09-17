from flask import Flask, render_template, request
import LinearRegression601T
import io
import base64
import matplotlib.pyplot as plt 
import matplotlib
from LogisticRegression601T import modelo_logistico

matplotlib.use('Agg')  # Configurar el backend de matplotlib para uso sin interfaz gráfica
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
    
    if request.method == 'POST':
        edad = request.form.get('edad', type=int)
        tiempo_espera = request.form.get('tiempo_espera', type=int)
        citas_previas = request.form.get('citas_previas', type=int)
        dia_semana = request.form.get('dia_semana')
        
        if all(v is not None for v in [edad, tiempo_espera, citas_previas, dia_semana]):
            resultado, probabilidad = modelo_logistico.predecir_asistencia(
                edad, tiempo_espera, citas_previas, dia_semana
            )
            
            # Generar matriz de confusión
            buffer = io.BytesIO()
            modelo_logistico.plot_confusion_matrix()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            matriz_confusion_img = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Generar curva ROC
            buffer = io.BytesIO()
            modelo_logistico.plot_roc_curve()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            curva_roc_img = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
    return render_template('regresion_logistica.html',
                         resultado=resultado,
                         probabilidad=probabilidad,
                         matriz_confusion_img=matriz_confusion_img,
                         curva_roc_img=curva_roc_img)

if __name__ == '__main__':
    app.run(debug=True)
