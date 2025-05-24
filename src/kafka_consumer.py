import json
import logging
import requests
import os
import time
import signal
import threading
from datetime import datetime
from confluent_kafka import Consumer


class PredictionLogConsumer:
    def __init__(self):
        self.logger = logging.getLogger('PredictionConsumer')
        
        # Configuration parameters (from env vars or defaults)
        self.bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')

        self._get_info_from_vault()
        self.topic = os.environ.get('KAFKA_TOPIC')

        # Stats tracking
        self.message_count = 0
        self.start_time = None
        self.running = False
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group,
            'auto.offset.reset': 'earliest',
            'session.timeout.ms': 6000,
            'max.poll.interval.ms': 6000,
            'enable.auto.commit': True,
            'auto.commit.interval.ms': 5000
        }
        
        self.logger.info(f"Initialized consumer for topic '{self.topic}' with group '{self.group}'")

    def _get_info_from_vault(self):
        try:
            vault_addr = os.environ.get('VAULT_ADDR', 'http://localhost:8200')
            vault_token = os.environ.get('VAULT_TOKEN', 'myroot')

            headers = {'X-Vault-Token': vault_token}
            url = f"{vault_addr}/v1/kafka/credentials"

            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                self.logger.error(f"Failed to fetch secrets from Vault: {response.text}")
                raise Exception("Failed to fetch secrets from Vault")

            secrets = response.json()['data']
            self.group = secrets['group']
            
            self.logger.info("Successfully retrieved kafka info from Vault")
            
        except Exception as e:
            self.logger.error(f"Error retrieving credentials from Vault: {e}")
            raise

    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        self.logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
    
    def _print_stats_periodically(self, interval=60):
        while self.running:
            time.sleep(interval)
            if self.message_count > 0 and self.start_time:
                elapsed = time.time() - self.start_time
                rate = self.message_count / elapsed if elapsed > 0 else 0
                self.logger.info(f"Stats: Consumed {self.message_count} messages in {elapsed:.2f}s ({rate:.2f} msgs/sec)")
    
    def start(self):
        self.setup_signal_handlers()
        self.running = True
        self.start_time = time.time()
        
        # Start stats thread
        stats_thread = threading.Thread(target=self._print_stats_periodically)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Create and configure consumer
        try:
            self.logger.info(f"Connecting to Kafka at {self.bootstrap_servers}...")
            consumer = Consumer(self.consumer_config)
            consumer.subscribe([self.topic])
            
            self.logger.info(f"Starting message consumption from topic '{self.topic}'")
            
            # Main consumption loop
            while self.running:
                try:
                    # Poll for messages with a timeout
                    msg = consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        self.logger.error(f"Consumer error: {msg.error()}")
                        continue
                    
                    # Process valid message
                    self._process_message(msg)
                    self.message_count += 1

                except Exception as e:
                    self.logger.error(f"Error during message polling: {str(e)}")
            
            self.logger.info("Consumer loop ended")
            
        except Exception as e:
            self.logger.error(f"Failed to start consumer: {str(e)}")
        finally:
            self.logger.info("Closing consumer")
            if 'consumer' in locals():
                consumer.close()
            
            # Print final stats
            if self.message_count > 0 and self.start_time:
                elapsed = time.time() - self.start_time
                self.logger.info(f"Final stats: Consumed {self.message_count} messages in {elapsed:.2f} seconds")
    
    def _process_message(self, msg):
        try:
            # Parse the message value
            message_data = json.loads(msg.value().decode('utf-8'))
            
            # Extract key information
            timestamp = message_data.get('timestamp')
            num_predictions = message_data.get('num_predictions', 0)
            metadata = message_data.get('metadata', {})
            
            # Format timestamp for display
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'N/A'
            
            # Log the message details
            self.logger.info(
                f"Received prediction log: time={readable_time}, "
                f"count={num_predictions}, producer={metadata.get('producer', 'unknown')}"
            )
            
            # Log sample data if available
            sample_predictions = message_data.get('sample_predictions', [])
            if sample_predictions:
                self.logger.info(f"Sample prediction data ({len(sample_predictions)} of {num_predictions}):")
                for i, pred in enumerate(sample_predictions):
                    # Format to show key features
                    target = pred.get('target', 'unknown')
                    area = pred.get('Area', 'N/A')
                    compactness = pred.get('Compactness', 'N/A')
                    self.logger.info(f"  #{i+1}: Target={target}, Area={area}, Compactness={compactness}")
            
        except json.JSONDecodeError:
            self.logger.warning(f"Received non-JSON message: {msg.value()[:100]}...")
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}")


def wait_for_kafka(bootstrap_servers, max_retries=12, retry_interval=5):
    logger = logging.getLogger('kafka-wait')
    
    from confluent_kafka.admin import AdminClient
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"Checking Kafka availability (attempt {attempt}/{max_retries})...")
        
        try:
            # Try to create an admin client which will connect to Kafka
            admin = AdminClient({'bootstrap.servers': bootstrap_servers})
            
            # List topics as a way to check connectivity
            topics = admin.list_topics(timeout=10)
            if topics:
                logger.info(f"Kafka is available at {bootstrap_servers}")
                return True
                
        except Exception as e:
            logger.warning(f"Kafka not yet available: {str(e)}")
        
        if attempt < max_retries:
            logger.info(f"Waiting {retry_interval} seconds before retry...")
            time.sleep(retry_interval)
    
    logger.error(f"Kafka not available after {max_retries} attempts")
    return False


if __name__ == "__main__":
    # Wait for Kafka to be ready
    bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    
    if wait_for_kafka(bootstrap_servers):
        # Start the consumer
        consumer = PredictionLogConsumer()
        consumer.start()
    else:
        logging.error("Exiting: Could not connect to Kafka")