import databento as db
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path


from src.model.model import *
from src.model.dataset import *


def run_pipeline(args):

    try:
        dataset_class = globals()[args.dataset]
    except Exception as e:
        print("Error loading dataset class:", e)

    try:
        model_class = globals()[args.model]
    except Exception as e:
        print("Error loading model class:", e)

    try:
        featurizer_class = globals()[args.featurizer]
    except Exception as e:
        print("Error loading featurizer class:", e)

    API_KEY = os.environ["DATABENTO_KEY"]
    DATASET = args.DB_Dataset
    SYMBOL = args.symbol
    START_DATE = args.start_date
    END_DATE = args.end_date

    client = db.Historical(API_KEY)

    current_folder = Path.cwd()
    subfolder = current_folder / "data"
    history_file = subfolder / "AAPL_minute_data.csv"

    got_ticker = False

    if history_file.exists():
        print("File exists!")
        got_ticker = True
    else:
        print("File does not exist.")

    if got_ticker:
        print("Reading file...")
        df = pd.read_csv(history_file)
    else:
        print("Fetching data...")
        # Fetch minute-bar data
        df = client.timeseries.get_range(
            dataset=DATASET,
            symbols=SYMBOL,
            schema="ohlcv-1m",
            start=START_DATE,
            end=END_DATE,
        ).to_df()

        df.to_csv(f"data/{SYMBOL}_minute_data.csv")

    df_test = df.copy()
    df_test = df_test.drop(["rtype", "publisher_id", "instrument_id", "symbol"], axis=1)

    featurizer_class = featurizer_class()

    df_test = featurizer_class.featurize(df_test)

    df_test = df_test.drop(["ts_event"], axis=1)
