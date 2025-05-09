import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os
import json

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from model_app import FlaskApp, PredictionService



class PredictionServiceTests(unittest.TestCase):
    """Tests for the PredictionService class"""
    
    def setUp(self):
        self.mock_model_manager = MagicMock()
        self.service = PredictionService(self.mock_model_manager)
        
    def test_process_request_valid_json(self):
        mock_request = MagicMock()
        mock_request.is_json = True
        mock_request.get_json.return_value = {
            "features": [{"feature1": 1.2, "feature2": 3.4}]
        }
        
        self.mock_model_manager.predict.return_value = [1, 0]
        
        response, status_code = self.service.process_request(mock_request)
        
        self.assertEqual(status_code, 200)
        self.assertEqual(response, {"predictions": [1, 0]})
        self.mock_model_manager.predict.assert_called_once_with([{"feature1": 1.2, "feature2": 3.4}])
    
    def test_process_request_invalid_json_structure(self):
        mock_request = MagicMock()
        mock_request.is_json = True
        mock_request.get_json.return_value = {
            "invalid_key": [{"feature1": 1.2, "feature2": 3.4}]
        }
        
        response, status_code = self.service.process_request(mock_request)
        
        self.assertEqual(status_code, 400)
        self.assertIn("error", response)
        self.mock_model_manager.predict.assert_not_called()
    
    def test_process_request_non_json(self):
        mock_request = MagicMock()
        mock_request.is_json = False
        
        response, status_code = self.service.process_request(mock_request)
        
        self.assertEqual(status_code, 400)
        self.assertIn("error", response)
        self.mock_model_manager.predict.assert_not_called()
    
    def test_process_request_exception(self):
        mock_request = MagicMock()
        mock_request.is_json = True
        mock_request.get_json.return_value = {
            "features": [{"feature1": 1.2, "feature2": 3.4}]
        }
        
        self.mock_model_manager.predict.side_effect = Exception("Test error")
        
        response, status_code = self.service.process_request(mock_request)
        
        self.assertEqual(status_code, 500)
        self.assertIn("error", response)
        self.assertEqual(response["error"], "Test error")


class FlaskAppIntegrationTests(unittest.TestCase):
    """Integration tests for the Flask application"""
    
    def setUp(self):
        with patch('model_app.ModelManager') as mock_model_manager_class:
            self.mock_model_manager = MagicMock()
            mock_model_manager_class.return_value = self.mock_model_manager
            
            self.flask_app = FlaskApp()
            self.app = self.flask_app.app.test_client()
            self.app.testing = True
        
        self.mock_features = {
            "features": [
                 {
                    "Area": 12.43,
                    "Perimeter": 13.12,
                    "Compactness": 0.8671,
                    "Kernel.Length": 5.232,
                    "Kernel.Width": 3.002,
                    "Asymmetry.Coeff": 1.243,
                    "Kernel.Groove": 4.988
                }
            ]
        }
        
        self.invalid_features = {
            "invalid_key": [
                {"feature1": 1.2, "feature2": 3.4}
            ]
        }
    
    def test_predict_valid_json(self):
        self.mock_model_manager.predict.return_value = [1, 0]
        
        response = self.app.post('/predict', json=self.mock_features)
        data = response.get_json()
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('predictions', data)
        self.assertEqual(data['predictions'], [1, 0])
    
    def test_predict_invalid_json_structure(self):
        response = self.app.post('/predict', json=self.invalid_features)
        data = response.get_json()
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_predict_non_json_request(self):
        response = self.app.post('/predict', data="invalid data")
        data = response.get_json()
        
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', data)
    
    def test_predict_exception_handling(self):
        self.mock_model_manager.predict.side_effect = Exception("Test error")
        
        response = self.app.post('/predict', json=self.mock_features)
        data = response.get_json()
        
        self.assertEqual(response.status_code, 500)
        self.assertIn('error', data)
        self.assertEqual(data['error'], "Test error")


if __name__ == "__main__":
    unittest.main()