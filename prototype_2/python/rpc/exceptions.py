class ConnectionNotOpenedError(ValueError):
    def __init__(self) -> None:
        super().__init__("error: connection has not been opened")


class ChannelNotOpenedError(ValueError):
    def __init__(self) -> None:
        super().__init__("error: channel has not been opened")
