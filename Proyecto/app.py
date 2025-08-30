from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/")
def home():
    name = None
    name = "Flask"
    return f"Hello, {name}!"

   
@app.route('/index')
def index():
    Myname= "Flask"
    return render_template('index.html', name=Myname)

if __name__ == '__main__':
    app.run(debug=True)
    