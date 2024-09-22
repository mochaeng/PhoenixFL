import pika
from pika import channel as ch
from uuid import uuid4
import json
import time
from rpc.exceptions import ConnectionNotOpenedError, ChannelNotOpenedError
import functools
from pika.exchange_type import ExchangeType


class ClientRPC(object):
    EXCHANGE = "packet"
    EXCHANGE_TYPE = ExchangeType.direct
    PUBLISH_INTERVAL = 0.001
    QUEUE = "requests_queue"
    ROUTING_KEY = QUEUE

    def __init__(self, amqp_url: str):
        self._connection = None
        self._channel = None

        self._deliveries = {}
        self._acked = 0
        self._nacked = 0
        self._message_number = 0

        self._stopping = False
        self.url = amqp_url

    def connect(self):
        print(f"Connecting to {self.url}")
        return pika.SelectConnection(
            pika.ConnectionParameters(host=self.url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )

    def on_connection_open(self, _unused_connection):
        print("Connection opened")
        self.open_channel()

    def on_connection_open_error(self, _unused_connection, err):
        print(f"Connection open failed, reopening in 5 seconds: {err}")
        if self._connection is None:
            raise ConnectionNotOpenedError()
        self._connection.ioloop.call_later(5, self._connection.ioloop.stop)

    def on_connection_closed(self, _unused_connection, reason):
        self._channel = None
        if self._stopping:
            if self._connection is None:
                raise ConnectionNotOpenedError()
            self._connection.ioloop.stop()
        else:
            print(f"Connection closed, reopening in 5 seconds: {reason}")
            if self._connection is None:
                raise ConnectionNotOpenedError()
            self._connection.ioloop.call_later(5, self._connection.ioloop.stop)

    def open_channel(self):
        print("Creating a new channel")
        if self._connection is None:
            raise ConnectionNotOpenedError()
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
        self._channel = None
        if not self._stopping:
            if self._connection is None:
                raise ConnectionNotOpenedError()
            self._connection.close()

    def setup_exchange(self, exchange_name):
        print(f"Declaring exchange {exchange_name}")
        call_back = functools.partial(
            self.on_exchange_declareok, userdata=exchange_name
        )
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.exchange_declare(
            exchange=exchange_name, exchange_type=self.EXCHANGE_TYPE, callback=call_back
        )

    def on_exchange_declareok(self, _unused_frame, userdata):
        print(f"Exchange declared: {userdata}")
        self.setup_queue(self.QUEUE)

    def setup_queue(self, queue_name):
        print(f"Declaring queue {queue_name}")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.queue_declare(
            queue=queue_name, callback=self.on_queue_declareok, durable=True
        )

    def on_queue_declareok(self, _frame):
        print(f"Binding {self.EXCHANGE} to {self.QUEUE} with {self.ROUTING_KEY}")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.queue_bind(
            self.QUEUE,
            self.EXCHANGE,
            routing_key=self.ROUTING_KEY,
            callback=self.on_bindok,
        )

    def on_bindok(self, _frame):
        print("Queue bound")
        self.start_publishing()

    def start_publishing(self):
        print("Issuing consumer related RPC commands")
        self.enable_delivery_confirmations()
        self.schedule_next_message()

    def enable_delivery_confirmations(self):
        print("Issuing Confirm.Select RPC command")
        if self._channel is None:
            raise ChannelNotOpenedError()
        self._channel.confirm_delivery(self.on_delivery_confirmation)

    def on_delivery_confirmation(self, method_frame):
        confirmation_type = method_frame.method.NAME.split(".")[1].lower()
        ack_multiple = method_frame.method.multiple
        delivery_tag = method_frame.method.delivery_tag

        print(
            f"Received {confirmation_type} for delivery tag: {delivery_tag} (multiple: {ack_multiple})"
        )

        if confirmation_type == "ack":
            self._acked += 1
        elif confirmation_type == "nack":
            self._nacked += 1

        del self._deliveries[delivery_tag]

        if ack_multiple:
            for tmp_tag in list(self._deliveries.keys()):
                if tmp_tag <= delivery_tag:
                    self._acked += 1
                    del self._deliveries[tmp_tag]

        print(
            f"Published {self._message_number} messages, {len(self._deliveries)} have yet to be confirmed, {self._acked} were acked and {self._nacked} were nacked",
        )

    def schedule_next_message(self):
        print(f"Scheduling next message for {self.PUBLISH_INTERVAL} seconds")
        if self._connection is None:
            raise ConnectionNotOpenedError()
        self._connection.ioloop.call_later(self.PUBLISH_INTERVAL, self.publish_message)

    def publish_message(self):
        if self._channel is None or not self._channel.is_open:
            return

        data_path = "data/single_packet.json"
        with open(data_path, "r+") as f:
            data = json.load(f)

        message = json.dumps(data).encode()

        # hdrs = {"مفتاح": " قيمة", "键": "值", "キー": "値"}
        properties = pika.BasicProperties(
            # app_id="example-publisher",
            content_type="application/json",
        )

        # message = "مفتاح قيمة 键 值 キー 値"
        self._channel.basic_publish(
            exchange=self.EXCHANGE,
            routing_key=self.ROUTING_KEY,
            body=message,
            properties=properties,
        )
        self._message_number += 1
        self._deliveries[self._message_number] = True
        print(f"Published message # {self._message_number}")
        self.schedule_next_message()

    def run(self):
        while not self._stopping:
            self._connection = None
            self._deliveries = {}
            self._acked = 0
            self._nacked = 0
            self._message_number = 0

            try:
                self._connection = self.connect()
                self._connection.ioloop.start()
            except KeyboardInterrupt:
                self.stop()
                if self._connection is not None and not self._connection.is_closed:
                    self._connection.ioloop.start()

        print("Stopped")

    def stop(self):
        print("Stopping")
        self._stopping = True
        self.close_channel()
        self.close_connection()

    def close_channel(self):
        if self._channel is not None:
            print("Closing the channel")
            self._channel.close()

    def close_connection(self):
        if self._connection is not None:
            print("Closing connection")
            self._connection.close()


def main():
    example = ClientRPC("localhost")
    example.run()


if __name__ == "__main__":
    main()
