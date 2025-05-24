import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import psycopg2
from io import StringIO
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))
from db_operations import DatabaseOperator

class TestDatabaseOperator(unittest.TestCase):
    
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'HOST': 'test_host',
            'PORT': '5432',
            'DBNAME': 'test_db',
            'USER': 'test_user',
            'PASSWORD': 'test_password'
        })
        self.env_patcher.start()
        
        # Mock configuration
        self.config_patcher = patch('configparser.ConfigParser')
        self.mock_config = self.config_patcher.start()
        
        # Configure mock config
        config_instance = self.mock_config.return_value
        config_instance.__getitem__.return_value = {
            'X_test': 'test_X_test.csv',
            'y_test': 'test_y_test.csv',
            'X_train': 'test_X_train.csv',
            'y_train': 'test_y_train.csv'
        }
        
        # Mock database connection
        self.db_patcher = patch('psycopg2.connect')
        self.mock_connect = self.db_patcher.start()
        self.mock_conn = MagicMock()
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        self.mock_connect.return_value = self.mock_conn
    
    def tearDown(self):
        self.env_patcher.stop()
        self.config_patcher.stop()
        self.db_patcher.stop()
    
    def test_init(self):
        """Test initialization and database setup"""
        db_op = DatabaseOperator()
        
        # Verify database connection and setup
        self.mock_connect.assert_called()
        self.mock_cursor.execute.assert_called()  # Tables were created
        self.mock_conn.commit.assert_called()
    
    @patch('pandas.read_csv')
    def test_load_test_data_to_db(self, mock_read_csv):
        """Test loading test data to database"""
        # Create mock dataframes
        X_test = pd.DataFrame({
            'Area': [15.26, 14.88],
            'Perimeter': [14.84, 14.57],
            'Compactness': [0.871, 0.8811],
            'Kernel.Length': [5.763, 5.554],
            'Kernel.Width': [3.312, 3.333],
            'Asymmetry.Coeff': [2.221, 1.018],
            'Kernel.Groove': [5.22, 4.956]
        })
        y_test = pd.DataFrame({'target': [1, 2]})
        mock_read_csv.side_effect = [X_test, y_test]
        
        db_op = DatabaseOperator()
        db_op.load_test_data_to_db()
        
        # Verify TRUNCATE and 2 rows inserted
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 3)
        self.mock_conn.commit.assert_called()
    
    @patch('pandas.read_csv')
    def test_load_train_data_to_db(self, mock_read_csv):
        """Test loading train data to database"""
        # Create mock dataframes
        X_train = pd.DataFrame({
            'Area': [14.23, 13.2],
            'Perimeter': [15.96, 13.66],
            'Compactness': [0.9486, 0.8883],
            'Kernel.Length': [5.527, 5.236],
            'Kernel.Width': [3.525, 3.232],
            'Asymmetry.Coeff': [2.872, 8.315],
            'Kernel.Groove': [5.443, 5.056]
        })
        y_train = pd.DataFrame({'target': [1, 3]})
        mock_read_csv.side_effect = [X_train, y_train]
        
        db_op = DatabaseOperator()
        db_op.load_train_data_to_db()
        
        # Verify TRUNCATE and 2 rows inserted
        self.assertGreaterEqual(self.mock_cursor.execute.call_count, 3)
        self.mock_conn.commit.assert_called()
    
    def test_load_prediction_in_db(self):
        """Test loading predictions to database"""
        predictions = [
            {'Area': 15.26, 'Perimeter': 14.84, 'Compactness': 0.871,'Kernel.Length': 5.763, 'Kernel.Width': 3.312, 
             'Asymmetry.Coeff': 2.221, 'Kernel.Groove': 5.22, 'target': 1},
            {'Area': 14.88, 'Perimeter': 14.57, 'Compactness': 0.8811, 
             'Kernel.Length': 5.554, 'Kernel.Width': 3.333, 
             'Asymmetry.Coeff': 1.018, 'Kernel.Groove': 4.956, 'target': 2}
        ]
        
        db_op = DatabaseOperator()
        self.mock_cursor.reset_mock()  # Reset call count from initialization
        db_op.load_prediction_in_db(predictions)
        
        # 2 predictions inserted
        self.assertEqual(self.mock_cursor.execute.call_count, 2)
        self.mock_conn.commit.assert_called()
    
    def test_database_error_handling(self):
        """Test error handling during database operations"""
        # Make cursor raise exception
        self.mock_cursor.execute.side_effect = Exception("Database error")
        
        # This should handle the exception without crashing
        db_op = DatabaseOperator()
        
        # Verify rollback was called
        self.mock_conn.rollback.assert_called()

if __name__ == '__main__':
    unittest.main()