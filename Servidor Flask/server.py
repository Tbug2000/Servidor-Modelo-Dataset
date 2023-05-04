from flask import Flask,request,jsonify
from flask_cors import CORS
import Mcontroller
app = Flask(__name__)
CORS(app)

@app.route("/predecir_precio", methods=['POST'])
def predecir_precio():
    data = request.json
    if data.get('Estado') == "Aceptable":
        Estado = 1
    else: Estado = 0
    response = jsonify({
        'precio': str(Mcontroller.predict_price(data.get('Estrato'),data.get('m2'),
                                                data.get('bhk'),data.get('Baños'),
                                                Estado))})

    return response
Mcontroller.load_model()

if __name__ == '__main__':
    print("Starting")
    Mcontroller.load_model()
    app.run()