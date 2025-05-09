import unittest
from unittest.mock import patch
import pandas as pd
import configparser
import sys
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from predict import CatBoostPredictor


class TestCatBoostPredictor(unittest.TestCase):
    def setUp(self):
        # Фиктивные данные
        self.fake_X_test = pd.DataFrame({'feature1': [1.2, 3.4], 'feature2': [5.6, 7.8]})
        self.fake_y_test = pd.Series([0, 1])

        # Фиктивная конфигурация
        self.config = configparser.ConfigParser()
        self.config.read_dict({
            'SPLIT_DATA': {
                'X_test': 'dummy_X_test.csv',
                'y_test': 'dummy_y_test.csv',
            }
        })

        # Патчим методы чтения CSV
        patcher_read_csv = patch('pandas.read_csv', side_effect=[self.fake_X_test, self.fake_y_test])
        self.mock_read_csv = patcher_read_csv.start()
        self.addCleanup(patcher_read_csv.stop)

        # Патчим аргументы командной строки
        patcher_argparse = patch.object(sys, 'argv', ['catboost_predictor.py', '-m', 'dummy_model_path.cbm'])
        self.addCleanup(patcher_argparse.stop)
        patcher_argparse.start()

        patcher_config = patch('configparser.ConfigParser', return_value=self.config)
        self.mock_config = patcher_config.start()

    @patch('catboost.CatBoostClassifier.load_model')
    @patch('catboost.CatBoostClassifier.predict', return_value=[0, 1])
    def test_predict(self, mock_predict, mock_load_model):
        # Инициализация предиктора
        predictor = CatBoostPredictor('dummy_config.ini')

        # Запуск метода predict
        predictor.predict()

        # Проверки
        mock_load_model.assert_called_once_with('dummy_model_path.cbm')
        mock_predict.assert_called_once()

if __name__ == "__main__":
    unittest.main()
