from flask import Flask, request, jsonify
import logging
import pandas as pd
from catboost import CatBoostClassifier, Pool
import configparser

# Настройка приложения Flask
app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Загрузка настроек конфигурации
config = configparser.ConfigParser()
config.read('config.ini')

# Путь до модели
model_path = config["MODEL"]["path"]

# Загрузка модели
model = CatBoostClassifier()
model.load_model(model_path)
log.info(f"Model loaded from {model_path}")

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        if "features" in data:
            try:
                # Преобразование входных данных в DataFrame
                input_data = pd.DataFrame(data["features"])
                
                # Инференс модели
                test_pool = Pool(input_data)
                predictions = model.predict(test_pool)
                
                # Формирование ответа
                response = {"predictions": predictions.tolist()}
                return jsonify(response), 200
            except Exception as e:
                log.error(f"Error in prediction: {e}")
                return jsonify({"error": str(e)}), 500
        else:
            return jsonify({"error": "Request JSON must contain 'features' key"}), 400
    else:
        return jsonify({"error": "Request must be JSON"}), 400

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
