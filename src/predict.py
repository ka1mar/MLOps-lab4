import argparse
import configparser
import os
import pandas as pd
import logging
from catboost import CatBoostClassifier, Pool
from db_operations import DatabaseOperator

class CatBoostPredictor:
    def __init__(self, config_path='config.ini'):
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

        # Загрузка настроек конфигурации
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Парсер аргументов командной строки
        self.parser = argparse.ArgumentParser(description="CatBoost Predictor")
        self.parser.add_argument("-m", "--model", type=str, help="Path to stored CatBoost model", required=True)
        self.args = self.parser.parse_args()

        self.db_op = DatabaseOperator()

        # Загрузка данных
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"])
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"])
        
        self.log.info("CatBoostPredictor is ready")


    def predict(self):
        self.log.info("Loading model from disk...")
        model = CatBoostClassifier()
        model.load_model(self.args.model)
        self.log.info("Model loaded successfully")

        self.log.info("Performing inference...")
        test_pool = Pool(self.X_test)
        predictions = model.predict(test_pool)

        for i, prediction in enumerate(predictions):
            self.log.info(f"Prediction for test sample {i}: {prediction}")

        predicted_data = self.X_test.copy()
        predicted_data['target'] = predictions

        predicted_data_dict = predicted_data.to_dict('records')
        self.db_op.load_prediction_in_db(predicted_data_dict)
        self.log.info(f"Prediction logged to the database")

# Для запуска этого модуля используйте следующую строку для командной строки:
# python catboost_predictor.py -m catboost_model.cbm
if __name__ == "__main__":
    predictor = CatBoostPredictor('config.ini')
    predictor.predict()
