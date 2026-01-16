import logging
import json
import os
from datetime import datetime

# Import kafka only if available to prevent crashes in non-docker envs
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False

logger = logging.getLogger(__name__)

class KafkaEventProducer:
    def __init__(self):
        self.producer = None
        self.enabled = False
        
        if not KAFKA_AVAILABLE:
            logger.warning("⚠️ kafka-python not installed. Streaming disabled.")
            return

        # Get Kafka URL from env (default to localhost for dev, but inside docker it should be 'kafka:29092')
        # We need to handle both cases. Inside docker-compose backend sees 'kafka:29092'
        kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=kafka_bootstrap,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                # Short timeout to not block API if Kafka is down
                request_timeout_ms=2000,
                api_version_auto_timeout_ms=2000
            )
            self.enabled = True
            logger.info(f"✅ Kafka Producer connected to {kafka_bootstrap}")
        except Exception as e:
            logger.warning(f"⚠️ Could not connect to Kafka at {kafka_bootstrap}: {e}")
            self.enabled = False

    def send_event(self, topic: str, event_type: str, data: dict):
        """Send an event to Kafka asynchronously."""
        if not self.enabled or not self.producer:
            return
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        
        try:
            # Fire and forget
            self.producer.send(topic, payload)
        except Exception as e:
            logger.error(f"Failed to send Kafka event: {e}")

# Singleton instance
kafka_producer = KafkaEventProducer()
