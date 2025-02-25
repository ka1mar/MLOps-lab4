import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import configparser
import logging


class DataMaker:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        self.data_path = self.config["DATA"]["data"]
        self.X_train_path = self.config["SPLIT_DATA"]["X_train"]
        self.X_test_path = self.config["SPLIT_DATA"]["X_test"]
        self.y_train_path = self.config["SPLIT_DATA"]["y_train"]
        self.y_test_path = self.config["SPLIT_DATA"]["y_test"]

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.test_size = self.config.getfloat("SPLIT_DATA", "test_size")
        self.random_state = self.config.getint("SPLIT_DATA", "random_state")
        
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)
        self.log.info("DataMaker inited")

    def get_data(self):
        # Читает данные из CSV файла
        data = pd.read_csv(self.data_path)
        self.X = data.iloc[:, :-1]  # Все колонки кроме последней
        self.y = data.iloc[:, -1]   # Последняя колонка

    def split_data(self):
        # Разделяет данные на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state,
            stratify=self.y, shuffle=True,
        )
        self.log.info("Data are splitted")

    def save_splitted_data(self):       
        self.X_train.to_csv(self.X_train_path, index=False)
        self.X_test.to_csv(self.X_test_path, index=False)
        self.y_train.to_csv(self.y_train_path, index=False)
        self.y_test.to_csv(self.y_test_path, index=False)

        self.log.info("Splitted data are saved")

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.get_data()
    data_maker.split_data()
    data_maker.save_splitted_data()