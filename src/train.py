import pandas as pd
from catboost import CatBoostClassifier, Pool
import configparser
import logging
import os
from sklearn.metrics import accuracy_score

class CatBoostModel:
    def __init__(self, config_path='config.ini'):
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)

        # Загрузка настроек конфигурации
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        # Логируем начало загрузки данных
        self.log.info("Loading data...")

        # Загрузка данных
        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"])
        
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"])
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"])
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"])

        # Загрузка параметров модели CatBoost
        self.iterations = self.config.getint("CATBOOST_PARAMS", "iterations")
        self.depth = self.config.getint("CATBOOST_PARAMS", "depth")
        self.learning_rate = self.config.getfloat("CATBOOST_PARAMS", "learning_rate")
        self.verbose = self.config.getint("CATBOOST_PARAMS", "verbose")
        self.random_seed = self.config.getint("CATBOOST_PARAMS", "random_seed")

        self.model_path = self.config["MODEL"]["path"]

        self.log.info("CatBoostModel initialized")

    def train(self):
        self.log.info("Training CatBoost model...")
        
        # Создаем объекты Pool для обучающей и тестовой выборок
        train_pool = Pool(self.X_train, self.y_train)

        # Инициализация и обучение
        model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            verbose=self.verbose,
            random_seed = self.random_seed
        )
        
        try:
            model.fit(train_pool)
            self.log.info("Model training complete")
            model.save_model(self.model_path)
            self.log.info(f"Model saved to {self.model_path}")
        except Exception as e:
            self.log.error(f"Error during model training: {e}")

    def predict(self):
        self.log.info("Making predictions using CatBoost model...")
        model = CatBoostClassifier()
        model.load_model(self.model_path)
        test_pool = Pool(self.X_test)
        predictions = model.predict(test_pool)
        accuracy = accuracy_score(self.y_test, predictions)
        self.log.info(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    catboost_model = CatBoostModel()
    catboost_model.train()
    catboost_model.predict()