from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pandas as pd
import joblib
from utils import *

app = Flask(__name__)

model = load_model('modelo_rnn.keras')
selector_carregado = joblib.load('./objects/selector.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    df = pd.DataFrame(input_data)

    df = load_scalers(df, ['tempoprofissao', 'renda', 'idade', 'dependentes', 'valorsolicitado', 'valortotalbem', 'propsolicitadototal'])
    df = load_encoders(df, ['profissao', 'tiporesidencia', 'escolaridade', 'score', 'estadocivil', 'produto'])
    df = selector_carregado.transform(df)

    predictions = model.predict(df)

    return jsonify(predictions.tolist())

if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0', por=5000, debug=True)
    