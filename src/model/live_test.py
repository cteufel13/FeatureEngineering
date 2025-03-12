import asyncio
from collections import deque
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream


class LiveStreamPredictor:
    """
    A self-contained class to stream live Alpaca bar data and use a pre-trained
    model to make predictions on a fixed-length sequence of closing prices.

    Attributes:
        model (object): A pre-trained ML model with a predict method.
        symbol (str): The stock symbol to subscribe to.
        seq_length (int): The fixed length of the closing price sequence.
        api_key (str): Alpaca API key.
        secret_key (str): Alpaca API secret key (defaults to empty string if not provided).
        base_url (str): Alpaca API base URL.
        data_feed (str): Data feed to use (e.g., 'iex').
    """

    def __init__(
        self,
        model,
        symbol,
        seq_length,
        api_key,
        secret_key="",
        base_url="https://paper-api.alpaca.markets",
        data_feed="iex",
    ):
        self.model = model
        self.symbol = symbol
        self.seq_length = seq_length
        self.api_key = api_key
        self.secret_key = secret_key  # will be empty string if not provided
        self.base_url = base_url
        self.data_feed = data_feed

        # Buffer to store the latest seq_length closing prices
        self.live_buffer = deque(maxlen=seq_length)

        # Initialize the Alpaca live stream. If your endpoint only requires an API key,
        # providing an empty string for the secret key should work.
        self.stream = Stream(
            self.api_key,
            self.secret_key,
            base_url=self.base_url,
            data_feed=self.data_feed,
        )

    async def handle_bar(self, bar):
        """
        Callback function for each incoming bar.
        Appends the close price to the buffer and makes a prediction when the buffer is full.
        """
        print(f"Received bar at {bar.t}: close = {bar.c}")

        # Add new closing price to the buffer
        self.live_buffer.append(bar.c)

        # If the buffer is full, reshape it and predict
        if len(self.live_buffer) == self.seq_length:
            sequence = np.array(self.live_buffer)
            prediction = self.model.predict(sequence.reshape(1, -1))
            print("Live prediction (1: up, 0: down):", prediction[0])

    async def start_stream(self):
        """
        Subscribes to live bar data for the symbol and starts the event loop.
        """
        # Subscribe to bar updates for the given symbol
        self.stream.subscribe_bars(self.handle_bar, self.symbol)
        print(f"Subscribed to live bars for {self.symbol}.")
        # Run the stream indefinitely
        await self.stream._run_forever()

    def run(self):
        """
        Starts the live stream in an asyncio event loop.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                print("Detected running event loop. Scheduling start_stream as a task.")
                loop.create_task(self.start_stream())
            else:
                asyncio.run(self.start_stream())
        except RuntimeError:
            asyncio.run(self.start_stream())
