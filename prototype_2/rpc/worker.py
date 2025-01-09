import functools
import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pika
from pandas.core.window.expanding import Literal
from pika.exchange_type import ExchangeType

from rpc.classifier import PytorchClassifier
from rpc.exceptions import ChannelNotOpenedError, ConnectionNotOpenedError
from rpc.helpers import PublishRequest
from rpc.stats import write_latencies


class Worker:
    EXCHANGE = "packet"
    EXCHANGE_TYPE = ExchangeType.direct
    REQUESTS_QUEUE = "requests_queue"
    ALERTS_QUEUE = "alerts_queue"

    def __init__(self, amqp_url, classifier: PytorchClassifier, name: str):
        # signal.signal(signal.SIGINT, self.handle_interrupt)

        self.should_reconnect = False
        self.was_consuming = False

        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = amqp_url
        self._consuming = False
        self._prefetch_count = 100  # high values for higher consumer throughput
        self.classifier = classifier

        self.processed_packages = 0
        self.latencies: list[float] = []
        self.name = name

    # def handle_interrupt(self, signum, _frame):
    #     print(f"Received signal {signum}. Stopping the worker...")
    #     self.stop()

    def connect(self):
        print(f"Connecting to {self._url}")
        connection = pika.SelectConnection(
            parameters=pika.ConnectionParameters(host=self._url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )
        return connection

    def close_connection(self):
        self._consuming = False
        if self._connection is None:
            raise ConnectionNotOpenedError()
        if self._connection.is_closing or self._connection.is_closed:
            print("Connection is closing or already closed")
        else:
            print("Closing connection")
            self._connection.close()

    def on_connection_open(self, _connection):
        print("Connection opened")
        self.open_channel()

    def on_connection_open_error(self, _connection, err):
        print(f"Connection open failed: {err}")
        self.reconnect()

    def on_connection_closed(self, _connection, reason):
        self._channel = None
        if self._closing:
            if self._connection is None:
                raise ConnectionNotOpenedError()
            self._connection.ioloop.stop()
        else:
            print(f"Connection closed, reconnect necessary: {reason}")
            self.reconnect()

    def reconnect(self):
        self.should_reconnect = True
        self.stop()

    def open_channel(self):
        print("Creating a new channel")
        if self._connection is None:
            raise ChannelNotOpenedError()
        self._connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        print("Channel opened")
        self._channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchanges(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        print("Adding channel close callback")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        print(f"Channel {channel} was closed: {reason}")
        self.close_connection()

    def setup_exchanges(self, exchange_name):
        print(f"Declaring exchange: {exchange_name}")
        cb = functools.partial(self.on_exchange_declareok, userdata=exchange_name)
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.exchange_declare(
            exchange=exchange_name,
            exchange_type=self.EXCHANGE_TYPE,
            callback=cb,
            durable=True,
        )

    def on_exchange_declareok(self, _frame, userdata):
        print(f"Exchange declared: {userdata}")
        self.setup_queues([self.REQUESTS_QUEUE, self.ALERTS_QUEUE])

    def setup_queues(self, queue_names: list[str]):
        for queue_name in queue_names:
            print(f"Declaring queue {queue_name}")
            cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
            if self._channel is None:
                raise ChannelNotOpenedError()
            self._channel.queue_declare(queue=queue_name, callback=cb, durable=True)

    def on_queue_declareok(self, _frame, userdata):
        queue_name = userdata
        print(
            f"Binding {self.EXCHANGE} to {queue_name} with {queue_name} as routing key"
        )
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.queue_bind(
            queue_name, self.EXCHANGE, routing_key=queue_name, callback=cb
        )

    def on_bindok(self, _frame, userdata):
        print(f"Queue bound: {userdata}")
        self.set_qos()

    def set_qos(self):
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.basic_qos(
            prefetch_count=self._prefetch_count, callback=self.on_basic_qos_ok
        )

    def on_basic_qos_ok(self, _frame):
        print(f"QOS set to: {self._prefetch_count}")
        self.start_consuming()

    def start_consuming(self):
        print("Issuing consumer related RPC commands")
        self.add_on_cancel_callback()
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._consumer_tag = self._channel.basic_consume(
            self.REQUESTS_QUEUE, self.on_message
        )
        self.was_consuming = True
        self._consuming = True

    def add_on_cancel_callback(self):
        print("Adding consumer cancellation callback")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.add_on_cancel_callback(self.on_consumer_cancelled)

    def on_consumer_cancelled(self, method_frame):
        print(f"Consumer was cancelled remotely, shutting down: {method_frame}")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.close()

    def on_message(self, channel, basic_deliver, properties, body):
        # print(f"Received message #{basic_deliver.delivery_tag} >> {properties.app_id}")
        processing_start_time = time.time()

        message: PublishRequest = json.loads(body)
        client_timestamp = float(message["send_timestamp"])
        packet = message["packet"]
        metadata = message["metadata"]

        transmission_and_queue_latency = processing_start_time - client_timestamp

        classification_start_time = time.time()
        isMalicious = self.classifier.predict_is_positive_binary(packet)
        classification_end_time = time.time()
        classification_latency = classification_end_time - classification_start_time
        total_latency = transmission_and_queue_latency + classification_latency

        classified_packet = {
            "metadata": metadata,
            "classification_time": classification_latency,
            "latency": total_latency,
            "worker_name": self.name,
            "is_malicious": isMalicious,
            "timestamp": time.time(),
        }

        print(isMalicious)

        def publish_to_alerts():
            if self._channel is not None and self._channel.is_open:
                self._channel.basic_publish(
                    exchange=self.EXCHANGE,
                    routing_key=self.ALERTS_QUEUE,
                    body=json.dumps(classified_packet),
                    properties=pika.BasicProperties(content_type="application/json"),
                )

        if self._connection is not None:
            self._connection.ioloop.add_callback_threadsafe(publish_to_alerts)

        self.latencies.append(total_latency)
        self.processed_packages += 1
        self.acknowledge_message(basic_deliver.delivery_tag)

    def acknowledge_message(self, delivery_tag):
        print(f"Acknowledging message {delivery_tag}")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.basic_ack(delivery_tag)

    def stop_consuming(self):
        if self._channel:
            print("Sending a Basic.Cancel RPC command to RabbitMQ")
            cb = functools.partial(self.on_cancelok, userdata=self._consumer_tag)
            self._channel.basic_cancel(self._consumer_tag, cb)

    def on_cancelok(self, _frame, userdata):
        self._consuming = False
        print(f"RabbitMQ acknowledged the cancellation of the consumer: {userdata}")
        self.close_channel()

    def close_channel(self):
        print("Closing the channel")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.close()

    def run(self):
        try:
            self._connection = self.connect()
            self._connection.ioloop.start()
        except KeyboardInterrupt:
            print("Ctrl+C detected. Stopping worker...")
            self.stop()

    def stop(self):
        if not self._closing:
            self._closing = True
            print("Stopping")
            if self._connection is None:
                raise ConnectionNotOpenedError()
            if self._consuming:
                self.stop_consuming()
                # self._connection.ioloop.start()
            else:
                self._connection.ioloop.stop()
            print("Stopped")


if __name__ == "__main__":
    url = "localhost"
    model_path = "data/fedmedian_model.pt"
    scaler_path = "data/scaler.pkl"
    classifier = PytorchClassifier(model_path=model_path, scaler_path=scaler_path)

    num_workers = 1
    executor = ThreadPoolExecutor(max_workers=num_workers)
    consumers = []
    for _ in range(num_workers):
        worker_name = f"worker_{uuid4()}"
        consumer = Worker(url, classifier, worker_name)
        consumers.append(consumer)

    for consumer in consumers:
        executor.submit(consumer.run)

    def signal_handler(sig, frame):
        print("worker has received shutdown signal")
        for consumer in consumers:
            consumer.stop()
            write_latencies(consumer.name, consumer.latencies)
        executor.shutdown(wait=True)
        print("all workers stopped")

    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
