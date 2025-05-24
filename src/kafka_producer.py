import json
import logging
import time
import os
import socket
import requests
from confluent_kafka import Producer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaProducer:
    def __init__(self):
        # Get configuration from environment
        bootstrap_servers = os.environ.get('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')

        # Configure the producer
        conf = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': socket.gethostname()
        }

        self.topic = os.environ.get('KAFKA_TOPIC')

        # Create the producer
        self.producer = Producer(conf)
        logger.info(f"Kafka producer initialized with bootstrap servers: {bootstrap_servers}")

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")
    
    def send_prediction_metadata(self, predictions):
        try:
            # Create a message with timestamp and metadata
            message = {
                "timestamp": time.time(),
                "num_predictions": len(predictions),
                "metadata": {
                    "producer": socket.gethostname(),
                    "service": "train_model"
                },
                # Include the first 2 predictions as sample (to keep message size reasonable)
                "sample_predictions": predictions[:2] if len(predictions) > 2 else predictions
            }
            
            # Serialize the message to JSON
            json_payload = json.dumps(message)
            
            # Send the message
            self.producer.produce(
                self.topic,
                value=json_payload.encode('utf-8'),
                callback=self.delivery_report
            )
            
            # Flush to ensure delivery
            self.producer.flush()
            
            logger.info(f"Sent metadata for {len(predictions)} predictions to topic {self.topic}")
            
        except Exception as e:
            logger.error(f"Error sending prediction metadata to Kafka: {str(e)}")