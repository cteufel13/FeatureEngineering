import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.base import DatasetBase
from tqdm import tqdm
import json
import polars as pl

import time

LOG_PATH = "data/raw/log.json"


class Dataset2(DatasetBase):

    def __init__(self, sequence_length=100, predict_horizon=10):
        self.sequence_length = sequence_length
        self.predict_horizon = predict_horizon
        self.log = self.load_log(LOG_PATH)
        self.symbols = list(self.log["symbols"].keys())
        self.dataset = {}

    def get_samples_symbol(self, symbol):
        data_symbol = self.load_data_symbol(symbol)
        samples = self.make_samples(data_symbol)
        return samples

    def load_data_symbol(self, symbol: str) -> pl.DataFrame:
        data = pl.read_parquet(f"data/processed/{symbol}/merged_data.parquet")
        return data

    def load_log(self, path) -> dict:
        with open(path, "r") as f:
            log = json.load(f)
        return log

    def labeler(self, sequence: pl.DataFrame) -> int:

        sequence_std = sequence["std"][0]
        sequence_start_value = sequence["close"][0]
        sequence_end_value = sequence["close"][0]

        threshhold_high = sequence_start_value + sequence_std
        threshhold_higher = sequence_start_value + 2 * sequence_std
        threshhold_low = sequence_start_value - sequence_std
        threshhold_lower = sequence_start_value - 2 * sequence_std

        if sequence_end_value > threshhold_high:
            label = 1

        elif sequence_end_value > threshhold_higher:
            label = 3

        elif sequence_end_value < threshhold_low:
            label = 2

        elif sequence_end_value < threshhold_lower:
            label = 5

        else:
            label = 0

        return label

    def make_samples(self, data_symbol: pl.DataFrame) -> np.ndarray:
        data_symbol.drop_in_place("symbol")
        print(data_symbol.columns)
        data_np = data_symbol.to_numpy()

        total_length = self.sequence_length + self.predict_horizon
        n_samples = len(data_symbol) // total_length
        n_features = len(data_symbol.columns)

        samples = np.zeros((n_samples, total_length, n_features))

        for i in range(n_samples):
            start = i * total_length
            end = start + total_length
            sequence = data_np[start:end]
            samples[i] = sequence

        return samples
