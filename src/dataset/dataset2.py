import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.base import DatasetBase
from tqdm import tqdm
import json
import polars as pl
import joblib

import time

LOG_PATH = "data/raw/log.json"


class Dataset2(DatasetBase):

    def __init__(self, sequence_length=100, predict_horizon=10):
        self.sequence_length = sequence_length
        self.predict_horizon = predict_horizon
        self.log = self.load_log(LOG_PATH)
        self.symbols = list(self.log["symbols"].keys())
        self.dataset = {}

    def get_training_data(self):
        for symbol in tqdm(self.symbols):
            complete_samples, time_samples = self.get_samples_symbol(symbol)
            training_samples, training_time_samples, labels = self.label_all_samples(
                complete_samples, time_samples
            )
            self.dataset[symbol] = {
                "training_samples": training_samples,
                "training_time_samples": training_time_samples,
                "labels": labels,
            }

        trainset = self.dataset[self.symbols[0]]["training_samples"]
        labels = self.dataset[self.symbols[0]]["labels"]
        times = self.dataset[self.symbols[0]]["training_time_samples"]

        for symbol in tqdm(self.symbols[1:]):
            trainset = np.concatenate(
                (trainset, self.dataset[symbol]["training_samples"])
            )
            labels = np.concatenate((labels, self.dataset[symbol]["labels"]))
            times = np.concatenate(
                (times, self.dataset[symbol]["training_time_samples"])
            )

        trainset = np.reshape(trainset, (trainset.shape[0], -1))
        print(trainset.shape, labels.shape, times.shape)

        return trainset, labels, times

    def get_samples_symbol(self, symbol):
        data_symbol = self.load_data_symbol(symbol)
        complete_samples, time_samples = self.make_samples(data_symbol)

        return complete_samples, time_samples

    def load_data_symbol(self, symbol: str) -> pl.DataFrame:
        data = pl.read_parquet(f"data/processed/{symbol}/merged_data.parquet")
        return data

    def load_log(self, path) -> dict:
        with open(path, "r") as f:
            log = json.load(f)
        return log

    def label(self, sequence: np.ndarray) -> int:

        sequence_start_value = sequence[0, 3]
        sequence_end_value = sequence[-1, 3]
        sequence_std = np.std(sequence[:, 3])

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
        data_symbol, aux_data = self.drop_columns(data_symbol)
        data_np = data_symbol.to_numpy()
        time_data_np = aux_data["ts_event"].to_numpy()

        total_length = self.sequence_length + self.predict_horizon
        n_samples = len(data_symbol) // total_length
        n_features = len(data_symbol.columns)

        samples = np.zeros((n_samples, total_length, n_features - 1))
        times_samples = np.empty((n_samples, total_length), dtype="datetime64[ns]")

        for i in range(n_samples):
            start = i * total_length
            end = start + total_length
            sequence = data_np[start:end, 1:]
            date_sequence = time_data_np[start:end]
            samples[i] = sequence
            times_samples[i] = date_sequence

        return samples, times_samples

    def label_all_samples(self, samples: np.array, times_samples: np.array):
        labels = np.zeros(len(samples))
        training_samples = np.zeros(
            (len(samples), self.sequence_length, samples.shape[2])
        )
        training_time_samples = np.zeros((len(samples), self.sequence_length))

        for i in range(len(samples)):
            sequence = samples[i]
            label = self.label(sequence)
            labels[i] = label
            training_samples[i] = sequence[: self.sequence_length]
            training_time_samples[i] = times_samples[i][: self.sequence_length]

        return training_samples, training_time_samples, labels

    def drop_columns(self, data: pl.DataFrame) -> pl.DataFrame:
        removed_data = data.select(
            ["ts_event", "rtype", "publisher_id", "instrument_id", "symbol"]
        )
        data = data.drop(
            ["ts_event", "rtype", "publisher_id", "instrument_id", "symbol"]
        )
        return data, removed_data

    def get_train_test(self):

        trainset, labels, times = self.get_training_data()

        X_train, X_test, y_train, y_test, time_train, time_test = train_test_split(
            trainset, labels, times, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test, time_train, time_test
