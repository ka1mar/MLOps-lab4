import pandas as pd
import psycopg2
import configparser
import logging
import time
import os
import requests
import json

class DatabaseOperator:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)
        
        # Get credentials from Vault
        self._get_db_credentials_from_vault()
        
        self.X_test_path = self.config["SPLIT_DATA"]["X_test"]
        self.y_test_path = self.config["SPLIT_DATA"]["y_test"]

        self.X_train_path = self.config["SPLIT_DATA"]["X_train"]
        self.y_train_path = self.config["SPLIT_DATA"]["y_train"]

        self.log.info("DatabaseOperator initialized")

        self._setup_database()
        self._setup_database()
    
    def _get_db_credentials_from_vault(self):
        """Fetch database credentials from HashiCorp Vault"""
        try:
            vault_addr = os.environ.get('VAULT_ADDR', 'http://localhost:8200')
            vault_token = os.environ.get('VAULT_TOKEN', 'myroot')
            
            headers = {'X-Vault-Token': vault_token}
            url = f"{vault_addr}/v1/db/credentials"

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                self.log.error(f"Failed to fetch secrets from Vault: {response.text}")
                raise Exception("Failed to fetch secrets from Vault")

            secrets = response.json()['data']
            
            self.db_host = secrets['host']
            self.db_port = secrets['port']
            self.db_name = secrets['dbname']
            self.db_user = secrets['user']
            self.db_password = secrets['password']
            
            self.log.info("Successfully retrieved database credentials from Vault")
            
        except Exception as e:
            self.log.error(f"Error retrieving credentials from Vault: {e}")
            raise

    def _test_connection_to_db(self):
        self.log.info("Waiting for Greenplum database to be ready...")
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            conn.close()
            self.log.info("Database connection established successfully")
            return
        except psycopg2.OperationalError:
            self.log.info(f"Error with connection to the database")
            raise ConnectionError("Could not connect to the Greenplum database")
    
    def _setup_database(self):
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()

            # Create table for train data with the correct feature structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS train_data (
                    id SERIAL PRIMARY KEY,
                    area FLOAT,
                    perimeter FLOAT,
                    compactness FLOAT,
                    kernel_length FLOAT,
                    kernel_width FLOAT,
                    asymmetry_coeff FLOAT,
                    kernel_groove FLOAT,
                    target INTEGER
                )
            """)

            # Create table for test data with the correct feature structure
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_data (
                    id SERIAL PRIMARY KEY,
                    area FLOAT,
                    perimeter FLOAT,
                    compactness FLOAT,
                    kernel_length FLOAT,
                    kernel_width FLOAT,
                    asymmetry_coeff FLOAT,
                    kernel_groove FLOAT,
                    target INTEGER
                )
            """)
            
            # Create table for model predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id SERIAL PRIMARY KEY,
                    area FLOAT,
                    perimeter FLOAT,
                    compactness FLOAT,
                    kernel_length FLOAT,
                    kernel_width FLOAT,
                    asymmetry_coeff FLOAT,
                    kernel_groove FLOAT,
                    result INTEGER
                )
            """)
            
            conn.commit()
            self.log.info("Database tables created successfully")
        except Exception as e:
            self.log.error(f"Error setting up database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def load_test_data_to_db(self):
        conn = None
        try:
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path)
            
            test_data = X_test.copy()
            test_data['target'] = y_test.iloc[:, 0]
            
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            cursor.execute("TRUNCATE TABLE test_data CASCADE")
            
            for idx, row in test_data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO test_data 
                    (area, perimeter, compactness, kernel_length, kernel_width, 
                     asymmetry_coeff, kernel_groove, target) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                    RETURNING id
                    """,
                    (
                        row['Area'], 
                        row['Perimeter'], 
                        row['Compactness'], 
                        row['Kernel.Length'], 
                        row['Kernel.Width'], 
                        row['Asymmetry.Coeff'], 
                        row['Kernel.Groove'], 
                        row['target']
                    )
                )
            
            conn.commit()
            self.log.info(f"Successfully loaded {len(test_data)} test records to database")
            
        except Exception as e:
            self.log.error(f"Error loading test data to database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    

    def load_train_data_to_db(self):
        conn = None
        try:
            X_train = pd.read_csv(self.X_train_path)
            y_train = pd.read_csv(self.y_train_path)
            
            train_data = X_train.copy()
            train_data['target'] = y_train.iloc[:, 0]
            
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            cursor.execute("TRUNCATE TABLE train_data CASCADE")
            
            for idx, row in train_data.iterrows():
                cursor.execute(
                    """
                    INSERT INTO train_data 
                    (area, perimeter, compactness, kernel_length, kernel_width, 
                     asymmetry_coeff, kernel_groove, target) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s) 
                    RETURNING id
                    """,
                    (
                        row['Area'], 
                        row['Perimeter'], 
                        row['Compactness'], 
                        row['Kernel.Length'], 
                        row['Kernel.Width'], 
                        row['Asymmetry.Coeff'], 
                        row['Kernel.Groove'], 
                        row['target']
                    )
                )
            
            conn.commit()
            self.log.info(f"Successfully loaded {len(train_data)} train records to database")
            
        except Exception as e:
            self.log.error(f"Error loading train data to database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()


    def load_prediction_in_db(self, predictions):
        conn = None
        try:
            from kafka_producer import KafkaProducer
            
            # Connect to database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()

            # Insert predictions
            for i, pred in enumerate(predictions):
                cursor.execute(
                    """INSERT INTO model_predictions
                    (area, perimeter, compactness, kernel_length, kernel_width, asymmetry_coeff, kernel_groove, result) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        pred['Area'], 
                        pred['Perimeter'], 
                        pred['Compactness'], 
                        pred['Kernel.Length'], 
                        pred['Kernel.Width'],
                        pred['Asymmetry.Coeff'], 
                        pred['Kernel.Groove'], 
                        pred['target']
                    )
                )
            
            conn.commit()
            self.log.info(f"Successfully stored {len(predictions)} predictions in database")
            
            # Send metadata to Kafka
            try:
                kafka_producer = KafkaProducer()
                kafka_producer.send_prediction_metadata(predictions)
                self.log.info("Prediction metadata sent to Kafka")
            except Exception as e:
                self.log.error(f"Failed to send prediction metadata to Kafka: {str(e)}")

        except Exception as e:
            self.log.error(f"Error storing predictions in database: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()


if __name__ == "__main__":
    db_op = DatabaseOperator()
    db_op.load_train_data_to_db()
    db_op.load_test_data_to_db()
