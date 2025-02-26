import requests
import json
import os
import configparser

def read_request_data(file_path):
    """Чтение данных запроса из JSON файла."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def test_predict(file_path):
    url = "http://localhost:5000/predict"
    headers = {"Content-Type": "application/json"}
    
    # Чтение данных запроса
    data = read_request_data(file_path)
    
    # Извлечение ожидаемых меток
    expected_labels = data.get("Type", [])
    
    # Отправка POST-запроса к API
    response = requests.post(url, json=data, headers=headers)
    
    if response.status_code == 200:
        predictions = response.json().get("predictions", [])
        
        if predictions == expected_labels:
            print(f"Test Passed for {file_path}. Predictions match expected labels.")
        else:
            print(f"Test Failed for {file_path}. Predictions do not match expected labels.")
            print(f"Predictions: {predictions}")
            print(f"Expected Labels: {expected_labels}")
    else:
        print(f"Test Failed for {file_path}. Error:", response.json())

if __name__ == "__main__":
    # Загрузка конфигурации
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    # Получение пути к папке с тестами из конфигурации
    test_data_directory = config["MODEL"]["smoke_tests_path"]

    # Итерируемся по JSON файлам в указанной директории
    for file_name in os.listdir(test_data_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(test_data_directory, file_name)
            test_predict(file_path)