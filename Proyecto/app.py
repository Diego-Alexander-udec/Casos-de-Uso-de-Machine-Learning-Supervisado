from flask import Flask, render_template, request
import Script_MDA.LinearRegression601T as LinearRegression601T
import Script_MDA.LogisticRegression as LogisticRegression601T

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

if __name__ == '__main__':
    app.run(debug=True)
