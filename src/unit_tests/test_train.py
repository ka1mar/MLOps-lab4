import unittest
from unittest.mock import patch
import pandas as pd
import configparser
import sys
import os

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from train import CatBoostModel

class TestCatBoostModel(unittest.TestCase):
    
    def setUp(self):
        # Создание мок данных
        self.mock_X_train = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
        self.mock_y_train = pd.Series([0, 1])
        self.mock_X_test = pd.DataFrame({'feature1': [5, 6], 'feature2': [7, 8]})
        self.mock_y_test = pd.Series([1, 0])

        # Создание мок конфигурации
        self.config = configparser.ConfigParser()
        self.config.read_dict({
            'SPLIT_DATA': {
                'X_train': 'dummy_X_train.csv',
                'y_train': 'dummy_y_train.csv',
                'X_test': 'dummy_X_test.csv',
                'y_test': 'dummy_y_test.csv',
            },
            'CATBOOST_PARAMS': {
                'iterations': '100',
                'depth': '6',
                'learning_rate': '0.1',
                'verbose': '0',
                'random_seed': '42',
            },
            'MODEL': {
                'path': 'dummy_model_path.cbm'
            }
        })

        # Патчинг методов
        patcher_config = patch('configparser.ConfigParser', return_value=self.config)
        patcher_read_csv = patch('pandas.read_csv')
        self.mock_read_csv = patcher_read_csv.start()
        self.mock_read_csv.side_effect = [
            self.mock_X_train, self.mock_y_train, self.mock_X_test, self.mock_y_test
        ]
        self.addCleanup(patcher_read_csv.stop)
        self.mock_config = patcher_config.start()

    @patch('pandas.read_csv')
    @patch('catboost.CatBoostClassifier.save_model')
    @patch('catboost.CatBoostClassifier.fit')
    def test_train(self, mock_fit, mock_save_model, mock_read_csv):
        # Настройка мока для чтения CSV
        mock_read_csv.side_effect = [self.mock_X_train, self.mock_y_train, self.mock_X_test, self.mock_y_test]

        catboost_model = CatBoostModel(config_path='dummy_config.ini')
        catboost_model.train()

        mock_fit.assert_called_once()
        mock_save_model.assert_called_once()

    @patch('pandas.read_csv')
    @patch('catboost.CatBoostClassifier.load_model')
    @patch('catboost.CatBoostClassifier.predict', return_value=[1, 0])
    def test_predict(self, mock_predict, mock_load_model, mock_read_csv):
        # Настройка мока для чтения CSV
        mock_read_csv.side_effect = [self.mock_X_train, self.mock_y_train, self.mock_X_test, self.mock_y_test]

        catboost_model = CatBoostModel(config_path='dummy_config.ini')
        catboost_model.predict()

        mock_load_model.assert_called_once_with(self.config['MODEL']['path'])
        mock_predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()