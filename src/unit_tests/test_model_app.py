import unittest
from unittest.mock import patch
import pandas as pd
import sys
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from model_app import app

class FlaskAppTestCase(unittest.TestCase):

    def setUp(self):
        # Создаем тестовый клиент Flask
        self.app = app.test_client()
        self.app.testing = True

        # Подготовка фиктивных данных для тестов
        self.mock_features = {
            "features": [
                {"feature1": 1.2, "feature2": 3.4},
                {"feature1": 5.6, "feature2": 7.8}
            ]
        }
        
        self.invalid_features = {
            "invalid_key": [
                {"feature1": 1.2, "feature2": 3.4}
            ]
        }

    @patch('catboost.CatBoostClassifier.predict', return_value=pd.Series([1, 0]))
    def test_predict_valid_json(self, mock_predict):
        # Отправка POST запроса с корректным JSON
        response = self.app.post('/predict', json=self.mock_features)
        data = response.get_json()

        # Проверка успешного ответа
        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', data)
        self.assertEqual(data['predictions'], [1, 0])

    def test_predict_invalid_json_structure(self):
        # Отправка POST запроса с некорректным JSON (нет ключа "features")
        response = self.app.post('/predict', json=self.invalid_features)
        data = response.get_json()

        # Проверка на правильность ошибки
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)

    def test_predict_non_json_request(self):
        # Отправка POST запроса с некорректным содержимым (не JSON)
        response = self.app.post('/predict', data="invalid data")
        data = response.get_json()

        # Проверка на правильность ошибки
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)


if __name__ == "__main__":
    unittest.main()
