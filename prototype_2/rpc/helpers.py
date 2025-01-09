from typing import TypedDict

COLUMNS_TO_REMOVE = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
]


class Metadata(TypedDict):
    IPV4_SRC_ADDR: str
    L4_SRC_PORT: str
    IPV4_DST_ADDR: str
    L4_DST_PORT: str


class PublishRequest(TypedDict):
    send_timestamp: float
    packet: list
    metadata: Metadata
