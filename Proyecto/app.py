from flask import Flask
from flask import render_template
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


if __name__ == '__main__':
    app.run(debug=True)


