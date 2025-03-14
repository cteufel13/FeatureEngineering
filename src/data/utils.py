import databento as db
import pandas as pd
from pathlib import Path


def get_data(
    file_exists=True,
    path=None,
    Dataset="XNAS.ITCH",
    Symbol="AAPL",
    start_date="2018-05-01",
    end_date="2025-02-28",
    api_key=None,
):

    client = db.Historical(api_key)

    if file_exists:
        print("Reading file...")
        df = pd.read_csv(path)
        return df
    else:
        print("Fetching data...")
        df = client.timeseries.get_range(
            dataset=Dataset,
            symbols=Symbol,
            schema="ohlcv-1m",
            start=start_date,
            end=end_date,
        ).to_df()

        df.to_csv(f"data/{Symbol}_minute_data.csv")
        return df


def check_file(symbol):

    current_folder = Path.cwd()
    subfolder = current_folder / "data"
    history_file = subfolder / f"{symbol}_minute_data.csv"

    bool_val = False
    if history_file.exists():
        print("File exists!")
        bool_val = True
    else:
        print("File does not exist.")
        bool_val, False

    return bool_val, history_file
