import pandas as pd
import psycopg2
import configparser
import logging
import time

class DatabaseOperator:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Database connection parameters
        self.db_host = self.config.get("DATABASE", "host")
        self.db_port = self.config.get("DATABASE", "port")
        self.db_name = self.config.get("DATABASE", "dbname")
        self.db_user = self.config.get("DATABASE", "user")
        self.db_password = self.config.get("DATABASE", "password")
        
        # Data paths
        self.X_test_path = self.config["SPLIT_DATA"]["X_test"]
        self.y_test_path = self.config["SPLIT_DATA"]["y_test"]
        #self.predictions_path = self.config.get("MODEL", "predictions_path", fallback="predictions.csv")
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.log = logging.getLogger(__name__)
        self.log.info("DatabaseOperator initialized")
        
        # Wait for Greenplum to be ready
        self._wait_for_db()
        
        # Setup database tables
        self._setup_database()
    
    def _wait_for_db(self, max_attempts=30, delay=2):
        """Wait for the database to be available"""
        self.log.info("Waiting for Greenplum database to be ready...")
        attempts = 0
        while attempts < max_attempts:
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
                attempts += 1
                self.log.info(f"Database not ready yet. Attempt {attempts}/{max_attempts}")
                time.sleep(delay)
        
        self.log.error("Failed to connect to database after several attempts")
        raise ConnectionError("Could not connect to the Greenplum database")
    
    def _setup_database(self):
        """Create necessary tables if they don't exist"""
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
                    test_data_id INTEGER REFERENCES test_data(id),
                    prediction INTEGER,
                    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        """Load test data to the database"""
        conn = None
        try:
            # Load test data
            X_test = pd.read_csv(self.X_test_path)
            y_test = pd.read_csv(self.y_test_path)
            
            # Combine features and target
            test_data = X_test.copy()
            test_data['target'] = y_test.iloc[:, 0]
            
            # Connect to database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Clear existing data
            cursor.execute("TRUNCATE TABLE test_data CASCADE")
            
            # Insert test data
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
    
    def load_prediction_in_db(self, prediction):
        """Store model predictions in the database"""
        conn = None
        try:
            # Load predictions
            predictions = pd.read_csv(self.predictions_path)
            
            # Connect to database
            conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                dbname=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            cursor = conn.cursor()
            
            # Get test data IDs from database
            cursor.execute("SELECT id FROM test_data ORDER BY id")
            test_ids = [row[0] for row in cursor.fetchall()]
            
            if len(test_ids) != len(predictions):
                self.log.warning(f"Number of test records ({len(test_ids)}) doesn't match predictions ({len(predictions)})")
            
            # Clear existing predictions
            cursor.execute("TRUNCATE TABLE model_predictions")
            
            # Insert predictions
            for i, pred in enumerate(predictions.iloc[:, 0]):
                if i < len(test_ids):
                    cursor.execute(
                        "INSERT INTO model_predictions (test_data_id, prediction) VALUES (%s, %s)",
                        (test_ids[i], int(pred))
                    )
            
            conn.commit()
            self.log.info(f"Successfully stored {len(predictions)} predictions in database")
            
        except Exception as e:
            self.log.error(f"Error storing predictions in database: {e}")
            if conn:conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def query_data_from_db(self):
        """Query and display data from the database (for demonstration)"""
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
            
            # Example query: Get prediction accuracy
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN td.target = mp.prediction THEN 1 ELSE 0 END) as correct_predictions
                FROM 
                    test_data td
                JOIN 
                    model_predictions mp ON td.id = mp.test_data_id
            """)
            
            result = cursor.fetchone()
            total, correct = result
            
            if total > 0:
                accuracy = (correct / total) * 100
                self.log.info(f"Prediction accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
            else:
                self.log.info("No prediction data found in database")
                
            # Example query: Get distribution of predictions by class
            cursor.execute("""
                SELECT 
                    prediction, 
                    COUNT(*) as count 
                FROM 
                    model_predictions 
                GROUP BY 
                    prediction 
                ORDER BY 
                    prediction
            """)
            
            self.log.info("Prediction distribution:")
            for row in cursor.fetchall():
                pred_class, count = row
                self.log.info(f"  Class {pred_class}: {count} predictions")
            
        except Exception as e:
            self.log.error(f"Error querying data from database: {e}")
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    db_op = DatabaseOperator()
    db_op.load_test_data_to_db()
    db_op.store_predictions_in_db()
    db_op.query_data_from_db()  # Added for demonstration