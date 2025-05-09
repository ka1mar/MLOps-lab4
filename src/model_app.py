from flask import Flask, request, jsonify
import logging
import pandas as pd
from catboost import CatBoostClassifier, Pool
import configparser

class ModelManager:    
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.model_path = self.config["MODEL"]["path"]
        self.logger = logging.getLogger('ModelManager')
        
        self.model = self._load_model()
    
    def _load_model(self):
        model = CatBoostClassifier()
        model.load_model(self.model_path)
        self.logger.info(f"Model loaded from {self.model_path}")
        return model
    
    def predict(self, features):
        input_data = pd.DataFrame(features)
        
        test_pool = Pool(input_data)
        predictions = self.model.predict(test_pool)
        
        return predictions.tolist()


class PredictionService:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.logger = logging.getLogger('PredictionService')
    
    def process_request(self, request_data):
        if not request_data.is_json:
            return {"error": "Request must be JSON"}, 400
            
        data = request_data.get_json()
        
        if "features" not in data:
            return {"error": "Request JSON must contain 'features' key"}, 400
            
        try:
            predictions = self.model_manager.predict(data["features"])
            response = {"predictions": predictions}
            return response, 200
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return {"error": str(e)}, 500


class FlaskApp:    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        
        self.app = Flask(__name__)
        
        self.model_manager = ModelManager()
        
        self.prediction_service = PredictionService(self.model_manager)
        
        self._register_routes()
    
    def _register_routes(self):
        self.app.route('/predict', methods=['POST'])(self.predict)
    
    def predict(self):
        response, status_code = self.prediction_service.process_request(request)
        return jsonify(response), status_code
    
    def run(self, host='0.0.0.0', port=5000):
        self.app.run(host=host, port=port)


if __name__ == '__main__':
    app = FlaskApp()
    app.run()
