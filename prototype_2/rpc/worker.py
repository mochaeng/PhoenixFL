import functools
import time
import json
import pika
from pika.exchange_type import ExchangeType
from rpc.exceptions import ConnectionNotOpenedError, ChannelNotOpenedError
from rpc.classifier import PytorchClassifier


class AsyncConsumer:
    EXCHANGE = "packet"
    EXCHANGE_TYPE = ExchangeType.direct
    QUEUE = "requests_queue"
    ROUTING_KEY = QUEUE

    def __init__(self, amqp_url, classifier: PytorchClassifier):
        self.should_reconnect = False
        self.was_consuming = False

        self._connection = None
        self._channel = None
        self._closing = False
        self._consumer_tag = None
        self._url = amqp_url
        self._consuming = False
        # In production, experiment with higher prefetch values
        # for higher consumer throughput
        self._prefetch_count = 1
        self.classifier = classifier

    def connect(self):
        print(f"Connecting to {self._url}")
        return pika.SelectConnection(
            parameters=pika.ConnectionParameters(host=self._url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )

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
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        print("Adding channel close callback")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        print(f"Channel {channel} was closed: {reason}")
        self.close_connection()

    def setup_exchange(self, exchange_name):
        print(f"Declaring exchange: {exchange_name}")
        cb = functools.partial(self.on_exchange_declareok, userdata=exchange_name)
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.exchange_declare(
            exchange=exchange_name, exchange_type=self.EXCHANGE_TYPE, callback=cb
        )

    def on_exchange_declareok(self, _frame, userdata):
        print(f"Exchange declared: {userdata}")
        self.setup_queue(self.QUEUE)

    def setup_queue(self, queue_name):
        print(f"Declaring queue {queue_name}")
        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.queue_declare(queue=queue_name, callback=cb, durable=True)

    def on_queue_declareok(self, _frame, userdata):
        queue_name = userdata
        print(f"Binding {self.EXCHANGE} to {queue_name} with {self.ROUTING_KEY}")
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.queue_bind(
            queue_name, self.EXCHANGE, routing_key=self.ROUTING_KEY, callback=cb
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
        self._consumer_tag = self._channel.basic_consume(self.QUEUE, self.on_message)
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
        print(
            f"Received message # {basic_deliver.delivery_tag} from {properties.app_id}: {body}"
        )
        json_data = json.loads(body)
        prediction = self.classifier.get_prediction(json_data)
        print(f"{prediction == 1}")

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
        self._connection = self.connect()
        self._connection.ioloop.start()

    def stop(self):
        if not self._closing:
            self._closing = True
            print("Stopping")
            if self._connection is None:
                raise ConnectionNotOpenedError()
            if self._consuming:
                self.stop_consuming()
                self._connection.ioloop.start()
            else:
                self._connection.ioloop.stop()
            print("Stopped")


class Worker:
    def __init__(self, url, classifier: PytorchClassifier):
        self._reconnect_delay = 0
        self._amqp_url = url
        self._consumer = AsyncConsumer(self._amqp_url, classifier)

    def run(self):
        while True:
            try:
                self._consumer.run()
            except KeyboardInterrupt:
                self._consumer.stop()
                break
            self._maybe_reconnect()

    def _maybe_reconnect(self):
        if self._consumer.should_reconnect:
            self._consumer.stop()
            reconnect_delay = self._get_reconnect_delay()
            print(f"Reconnecting after {reconnect_delay} seconds")
            time.sleep(reconnect_delay)
            self._consumer = AsyncConsumer(self._amqp_url, classifier)

    def _get_reconnect_delay(self):
        if self._consumer.was_consuming:
            self._reconnect_delay = 0
        else:
            self._reconnect_delay += 1
        if self._reconnect_delay > 30:
            self._reconnect_delay = 30
        return self._reconnect_delay


if __name__ == "__main__":
    model_path = "data/model.pt"
    scaler_path = "data/scaler.pkl"
    classifier = PytorchClassifier(model_path=model_path, scaler_path=scaler_path)

    url = "localhost"
    consumer = Worker(url, classifier)
    consumer.run()