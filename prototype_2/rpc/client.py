import functools
import json
import signal
import time

import pandas as pd
import pika
from pandas.core.interchange.dataframe_protocol import Column
from pandas.core.series import Series
from pika.exchange_type import ExchangeType

from rpc.exceptions import ChannelNotOpenedError, ConnectionNotOpenedError
from rpc.helpers import COLUMNS_TO_REMOVE, Metadata, PublishRequest


class ClientRPC:
    EXCHANGE = "packet"
    EXCHANGE_TYPE = ExchangeType.direct
    PUBLISH_INTERVAL = 0.005
    QUEUE = "requests_queue"
    ROUTING_KEY = QUEUE

    def __init__(self, amqp_url: str, messages: list):
        signal.signal(signal.SIGTERM, self.handle_interrupt)

        self.url = amqp_url
        # self.packets = packets
        self.messages = messages
        self.connection = None
        self.channel = None
        self.deliveries = {}
        self.current_packet = 0
        self.acked = 0
        self.nacked = 0
        self.message_number = 0
        self.stopping = False

    def handle_interrupt(self, signum, _frame):
        print(f"Received signal {signum}. Stopping the client...")
        self.stop()

    def connect(self):
        print(f"Connecting to {self.url}")
        connection = pika.SelectConnection(
            pika.ConnectionParameters(host=self.url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )
        return connection

    def on_connection_open(self, _connection):
        print("Connection opened")
        self.open_channel()

    def on_connection_open_error(self, _connection, err):
        print(f"Connection open failed, reopening in 5 seconds: {err}")
        if self.connection is None:
            raise ConnectionNotOpenedError()
        self.connection.ioloop.call_later(5, self.connection.ioloop.stop)

    def on_connection_closed(self, _connection, reason):
        self.channel = None
        if self.stopping:
            if self.connection is None:
                raise ConnectionNotOpenedError()
            self.connection.ioloop.stop()
        else:
            print(f"Connection closed, reopening in 5 seconds: {reason}")
            if self.connection is None:
                raise ConnectionNotOpenedError()
            self.connection.ioloop.call_later(5, self.connection.ioloop.stop)

    def open_channel(self):
        print("Creating a new channel")
        if self.connection is None:
            raise ConnectionNotOpenedError()
        self.connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        print("Channel opened")
        self.channel = channel
        self.add_on_channel_close_callback()
        self.setup_exchange(self.EXCHANGE)

    def add_on_channel_close_callback(self):
        print("Adding channel close callback")
        if self.channel is None:
            raise ChannelNotOpenedError()
        self.channel.add_on_close_callback(self.on_channel_closed)

    def on_channel_closed(self, channel, reason):
        print(f"Channel {channel} was closed: {reason}")
        self.channel = None
        if not self.stopping:
            if self.connection is None:
                raise ConnectionNotOpenedError()
            self.connection.close()

    def setup_exchange(self, exchange_name):
        print(f"Declaring exchange {exchange_name}")
        call_back = functools.partial(
            self.on_exchange_declareok, userdata=exchange_name
        )
        if self.channel is None:
            raise ChannelNotOpenedError()
        self.channel.exchange_declare(
            exchange=exchange_name,
            exchange_type=self.EXCHANGE_TYPE,
            callback=call_back,
            durable=True,
        )

    def on_exchange_declareok(self, _frame, userdata):
        print(f"Exchange declared: {userdata}")
        self.setup_queue(self.QUEUE)

    def setup_queue(self, queue_name):
        print(f"Declaring queue {queue_name}")
        if self.channel is None:
            raise ChannelNotOpenedError()
        self.channel.queue_declare(
            queue=queue_name, callback=self.on_queue_declareok, durable=True
        )

    def on_queue_declareok(self, _frame):
        print(f"Binding {self.EXCHANGE} to {self.QUEUE} with {self.ROUTING_KEY}")
        if self.channel is None:
            raise ChannelNotOpenedError()
        self.channel.queue_bind(
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
        if self.channel is None:
            raise ChannelNotOpenedError()
        self.channel.confirm_delivery(self.on_delivery_confirmation)

    def on_delivery_confirmation(self, method_frame):
        confirmation_type = method_frame.method.NAME.split(".")[1].lower()
        ack_multiple = method_frame.method.multiple
        delivery_tag = method_frame.method.delivery_tag

        # print(
        #     f"Received {confirmation_type} for delivery tag: {delivery_tag} (multiple: {ack_multiple})"
        # )

        if confirmation_type == "ack":
            self.acked += 1
        elif confirmation_type == "nack":
            self.nacked += 1

        del self.deliveries[delivery_tag]

        if ack_multiple:
            for tmp_tag in list(self.deliveries.keys()):
                if tmp_tag <= delivery_tag:
                    self.acked += 1
                    del self.deliveries[tmp_tag]

        # print(
        #     f"Published {self.message_number} messages, {len(self.deliveries)} have yet to be confirmed, {self.acked} were acked and {self.nacked} were nacked",
        # )

    def schedule_next_message(self):
        # print(f"Scheduling next message for {self.PUBLISH_INTERVAL} seconds")
        if self.connection is None:
            raise ConnectionNotOpenedError()
        # if self.message_number >= 12_000:
        #     self.stop()
        self.connection.ioloop.call_later(self.PUBLISH_INTERVAL, self.publish_message)

    def publish_message(self):
        if self.channel is None or not self.channel.is_open:
            return

        message: PublishRequest = self.messages[
            self.current_packet % len(self.messages)
        ]
        message["send_timestamp"] = str(time.time())
        encoded_message = json.dumps(message).encode()

        self.channel.basic_publish(
            exchange=self.EXCHANGE,
            routing_key=self.ROUTING_KEY,
            body=encoded_message,
            properties=pika.BasicProperties(content_type="application/json"),
        )

        self.current_packet += 1
        self.message_number += 1
        self.deliveries[self.message_number] = True
        print(f"Published message # {self.message_number}")
        self.schedule_next_message()

    def run(self):
        while not self.stopping:
            self.connection = None
            self.deliveries = {}
            self.acked = 0
            self.nacked = 0
            self.message_number = 0

            try:
                self.connection = self.connect()
                self.connection.ioloop.start()
            except KeyboardInterrupt:
                self.stop()
                if self.connection is not None and not self.connection.is_closed:
                    self.connection.ioloop.start()

        print("Stopped")

    def stop(self):
        print("Stopping")
        self.stopping = True
        self.close_channel()
        self.close_connection()

    def close_channel(self):
        if self.channel is not None:
            print("Closing the channel")
            self.channel.close()

    def close_connection(self):
        if self.connection is not None:
            print("Closing connection")
            self.connection.close()


if __name__ == "__main__":
    packets_df = pd.read_csv(
        "data/10_000-raw-packets.csv",
    )

    messages = []
    for idx, row in packets_df.iterrows():
        metadata = row[COLUMNS_TO_REMOVE].to_dict()  # type: ignore
        packet = row.drop(COLUMNS_TO_REMOVE).to_dict()
        data = {
            "metadata": metadata,
            "packet": packet,
        }
        messages.append(data)

    client = ClientRPC("localhost", messages)
    client.run()
