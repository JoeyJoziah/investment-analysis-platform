"""Kafka client for real-time data streaming"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import BaseModel

from backend.config import settings
from backend.utils.exceptions import DataIngestionException

logger = logging.getLogger(__name__)


class KafkaMessage(BaseModel):
    """Kafka message model"""
    topic: str
    key: Optional[str] = None
    value: dict[str, Any]
    timestamp: datetime = None
    headers: Optional[dict[str, str]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class KafkaConfig(BaseModel):
    """Kafka configuration"""
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "investment-analysis"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    
    # Producer settings
    compression_type: str = "gzip"
    batch_size: int = 16384
    linger_ms: int = 100
    acks: str = "all"
    
    # Topics
    topics = {
        "stock_prices": "stock-prices",
        "stock_fundamentals": "stock-fundamentals",
        "news_sentiment": "news-sentiment",
        "market_events": "market-events",
        "recommendations": "recommendations",
        "alerts": "alerts",
        "audit_logs": "audit-logs"
    }


class KafkaProducerClient:
    """Kafka producer for sending messages"""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig()
        self._producer: Optional[AIOKafkaProducer] = None
        self._started = False
        
    async def start(self):
        """Start the Kafka producer"""
        if self._started:
            return
            
        try:
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                compression_type=self.config.compression_type,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                acks=self.config.acks,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None
            )
            
            await self._producer.start()
            self._started = True
            logger.info("Kafka producer started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise DataIngestionException(f"Kafka producer startup failed: {e}")
            
    async def stop(self):
        """Stop the Kafka producer"""
        if self._producer and self._started:
            await self._producer.stop()
            self._started = False
            logger.info("Kafka producer stopped")
            
    async def send_message(
        self,
        topic: str,
        value: dict[str, Any],
        key: Optional[str] = None,
        headers: Optional[dict[str, str]] = None
    ) -> None:
        """Send a message to Kafka topic"""
        if not self._started:
            await self.start()
            
        try:
            # Convert headers to list of tuples
            kafka_headers = []
            if headers:
                kafka_headers = [(k, v.encode("utf-8")) for k, v in headers.items()]
                
            # Send message
            await self._producer.send(
                topic=topic,
                value=value,
                key=key,
                headers=kafka_headers
            )
            
            logger.debug(f"Message sent to topic '{topic}' with key '{key}'")
            
        except KafkaError as e:
            logger.error(f"Failed to send message to Kafka: {e}")
            raise DataIngestionException(f"Kafka send failed: {e}")
            
    async def send_batch(
        self,
        topic: str,
        messages: list[dict[str, Any]],
        key_field: Optional[str] = None
    ) -> None:
        """Send multiple messages to Kafka topic"""
        if not self._started:
            await self.start()
            
        try:
            batch = self._producer.create_batch()
            
            for msg in messages:
                key = str(msg.get(key_field)) if key_field else None
                value = json.dumps(msg).encode("utf-8")
                key_bytes = key.encode("utf-8") if key else None
                
                # Try to add to batch
                metadata = batch.append(key=key_bytes, value=value, timestamp=None)
                if metadata is None:
                    # Batch is full, send it
                    await self._producer.send_batch(batch, topic)
                    # Create new batch and add message
                    batch = self._producer.create_batch()
                    batch.append(key=key_bytes, value=value, timestamp=None)
                    
            # Send remaining messages
            if batch.record_count() > 0:
                await self._producer.send_batch(batch, topic)
                
            logger.info(f"Sent {len(messages)} messages to topic '{topic}'")
            
        except Exception as e:
            logger.error(f"Failed to send batch to Kafka: {e}")
            raise DataIngestionException(f"Kafka batch send failed: {e}")
            
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class KafkaConsumerClient:
    """Kafka consumer for receiving messages"""
    
    def __init__(
        self,
        topics: list[str],
        config: Optional[KafkaConfig] = None,
        message_handler: Optional[Callable[[KafkaMessage], None]] = None
    ):
        self.config = config or KafkaConfig()
        self.topics = topics
        self.message_handler = message_handler
        self._consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        
    async def start(self):
        """Start the Kafka consumer"""
        try:
            self._consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                max_poll_records=self.config.max_poll_records,
                session_timeout_ms=self.config.session_timeout_ms,
                heartbeat_interval_ms=self.config.heartbeat_interval_ms,
                value_deserializer=lambda v: json.loads(v.decode("utf-8"))
            )
            
            await self._consumer.start()
            self._running = True
            logger.info(f"Kafka consumer started for topics: {self.topics}")
            
        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise DataIngestionException(f"Kafka consumer startup failed: {e}")
            
    async def stop(self):
        """Stop the Kafka consumer"""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped")
            
    async def consume_messages(self):
        """Consume messages from Kafka topics"""
        if not self._consumer:
            await self.start()
            
        try:
            async for msg in self._consumer:
                if not self._running:
                    break
                    
                # Parse message
                kafka_msg = KafkaMessage(
                    topic=msg.topic,
                    key=msg.key.decode("utf-8") if msg.key else None,
                    value=msg.value,
                    timestamp=datetime.fromtimestamp(msg.timestamp / 1000),
                    headers={k: v.decode("utf-8") for k, v in msg.headers} if msg.headers else None
                )
                
                # Handle message
                if self.message_handler:
                    try:
                        await self.message_handler(kafka_msg)
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        
                logger.debug(f"Processed message from topic '{msg.topic}'")
                
        except Exception as e:
            logger.error(f"Error consuming messages: {e}")
            raise DataIngestionException(f"Kafka consume failed: {e}")
            
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class KafkaStreamProcessor:
    """Process streaming data from Kafka"""
    
    def __init__(self, config: Optional[KafkaConfig] = None):
        self.config = config or KafkaConfig()
        self.producer = KafkaProducerClient(self.config)
        self.consumers: dict[str, KafkaConsumerClient] = {}
        
    async def start(self):
        """Start the stream processor"""
        await self.producer.start()
        
    async def stop(self):
        """Stop the stream processor"""
        await self.producer.stop()
        for consumer in self.consumers.values():
            await consumer.stop()
            
    def register_consumer(
        self,
        name: str,
        topics: list[str],
        handler: Callable[[KafkaMessage], None]
    ):
        """Register a consumer for specific topics"""
        consumer = KafkaConsumerClient(
            topics=topics,
            config=self.config,
            message_handler=handler
        )
        self.consumers[name] = consumer
        
    async def process_stock_price(self, ticker: str, price_data: dict[str, Any]):
        """Process and stream stock price data"""
        message = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "open": price_data.get("open"),
            "high": price_data.get("high"),
            "low": price_data.get("low"),
            "close": price_data.get("close"),
            "volume": price_data.get("volume"),
            "source": price_data.get("source", "unknown")
        }
        
        await self.producer.send_message(
            topic=self.config.topics["stock_prices"],
            value=message,
            key=ticker
        )
        
    async def process_news_sentiment(self, ticker: str, sentiment_data: dict[str, Any]):
        """Process and stream news sentiment data"""
        message = {
            "ticker": ticker,
            "timestamp": datetime.utcnow().isoformat(),
            "headline": sentiment_data.get("headline"),
            "sentiment_score": sentiment_data.get("sentiment_score"),
            "confidence": sentiment_data.get("confidence"),
            "source": sentiment_data.get("source"),
            "url": sentiment_data.get("url")
        }
        
        await self.producer.send_message(
            topic=self.config.topics["news_sentiment"],
            value=message,
            key=ticker
        )
        
    async def send_alert(self, alert_type: str, alert_data: dict[str, Any]):
        """Send alert through Kafka"""
        message = {
            "alert_type": alert_type,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": alert_data.get("severity", "info"),
            "title": alert_data.get("title"),
            "message": alert_data.get("message"),
            "metadata": alert_data.get("metadata", {})
        }
        
        await self.producer.send_message(
            topic=self.config.topics["alerts"],
            value=message,
            key=alert_type
        )
        
    async def audit_log(self, action: str, user_id: Optional[int], details: dict[str, Any]):
        """Send audit log through Kafka"""
        message = {
            "action": action,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": details.get("ip_address"),
            "user_agent": details.get("user_agent"),
            "details": details
        }
        
        await self.producer.send_message(
            topic=self.config.topics["audit_logs"],
            value=message,
            key=f"user_{user_id}" if user_id else "system"
        )


# Global Kafka stream processor
kafka_processor = KafkaStreamProcessor()