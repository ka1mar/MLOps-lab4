import configparser
import os
import unittest
from unittest.mock import patch
import pandas as pd
import configparser
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from preprocess import DataMaker


class TestDataMaker(unittest.TestCase):

    def setUp(self):
        # Конфигурация для моck'а configparser
        self.config = configparser.ConfigParser()
        self.config.read_dict({
            'DATA': {
                'data': 'path_to_data.csv'
            },
            'SPLIT_DATA': {
                'X_train': 'X_train.csv',
                'X_test': 'X_test.csv',
                'y_train': 'y_train.csv',
                'y_test': 'y_test.csv',
                'test_size': '0.5',
                'random_state': '42'
            }
        })

        # Fake DataFrame для теста
        self.fake_data = pd.DataFrame({
            'Area': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Perimeter': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Compactness': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Kernel.Length': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Kernel.Width': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Asymmetry.Coeff': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Kernel.Groove': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'Type': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        })

        # Патчинг методов и logging
        patcher_config = patch('configparser.ConfigParser', return_value=self.config)
        patcher_read_csv = patch('pandas.read_csv', return_value=self.fake_data)
        patcher_logging = patch('logging.basicConfig')

        self.addCleanup(patcher_config.stop)
        self.addCleanup(patcher_read_csv.stop)
        self.addCleanup(patcher_logging.stop)

        self.mock_config = patcher_config.start()
        self.mock_read_csv = patcher_read_csv.start()

    def test_get_data(self):
        data_maker = DataMaker()
        data_maker.get_data()
        self.assertTrue(hasattr(data_maker, 'X'))
        self.assertTrue(hasattr(data_maker, 'y'))
        self.assertEqual(len(data_maker.X), len(self.fake_data))
        self.assertEqual(len(data_maker.y), len(self.fake_data))

    def test_split_data(self):
        data_maker = DataMaker()
        data_maker.get_data()
        data_maker.split_data()

        # Проверьте, что данные были разделены правильно
        self.assertEqual(len(data_maker.X_train) + len(data_maker.X_test), len(self.fake_data))
        self.assertEqual(len(data_maker.y_train) + len(data_maker.y_test), len(self.fake_data))
        
        # Гарантируем, что правильное количество строк во входных данных
        self.assertTrue(0 < len(data_maker.X_train) < len(self.fake_data))
        self.assertTrue(0 < len(data_maker.X_test) < len(self.fake_data))


    @patch('pandas.DataFrame.to_csv')
    def test_save_splitted_data(self, mock_to_csv):
        data_maker = DataMaker()
        data_maker.get_data()
        data_maker.split_data()
        data_maker.save_splitted_data()

        # Проверяем, что метод to_csv был вызван
        self.assertEqual(mock_to_csv.called, True)


if __name__ == "__main__":
    unittest.main()