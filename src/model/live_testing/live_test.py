import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import wandb
import os
import databento as db


def main():

    symbols = [
        "AMD",
    ]

    live_client = db.Live(key=os.getenv("DATABENTO_KEY"))

    live_client.subscribe(
        dataset="XNAS.ITCH",
        symbols=symbols,
        schema="ohlcv-1m",
        stype_in="continuous",
    )

    live_client.add_stream("data/live_data/live_data.dbn")

    print("Waiting for data...")
    live_client.start()
    live_client.block_for_close(timeout=70)
    print("Close")


if __name__ == "__main__":
    main()
    dbn_store = db.read_dbn("data/live_data/live_data.dbn")
    print(dbn_store.to_df(schema="ohlcv-1s"))
