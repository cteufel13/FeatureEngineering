import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.base import DatasetBase
from tqdm import tqdm


class Dataset1(DatasetBase):

    def __init__(
        self,
        rawdata,
        n_samples=10000,
        len_sequence=5,
        predict_horizon=10,
    ):
        self.rawdata = rawdata
        self.len_data = len(rawdata)

        self.len_sequence = len_sequence
        self.predict_horizon = predict_horizon

        min_distance = self.len_sequence + self.predict_horizon + 2
        end = self.len_data - self.len_sequence - self.predict_horizon - 10
        self.n_samples = int((end) / min_distance) + 1

        self.n_features = self.rawdata.shape[1]
        self.X = np.zeros((self.n_samples, self.len_sequence, self.n_features))
        self.y = np.zeros((self.n_samples))

        self.aux_data = np.zeros((self.n_samples, 1))
        self.complete_sequence = np.empty(
            (self.n_samples, self.len_sequence + predict_horizon + 20, self.n_features)
        )

    def process(self):

        self.rawdata["std"] = (
            self.rawdata["close"].rolling(window=self.len_sequence).std()
        )
        self.rawdata.dropna(inplace=True)

        random_start_points = np.linspace(
            10,
            self.len_data - 2 * self.len_sequence - self.predict_horizon - 10,
            self.n_samples,
        ).astype(int)

        for i, start_point in tqdm(enumerate(random_start_points)):
            sequence = self.rawdata.iloc[
                start_point : start_point + self.len_sequence + self.predict_horizon
            ]
            std = sequence["std"].values[0]

            self.X[i, :, :] = sequence.drop(["std"], axis=1).values[
                : -self.predict_horizon
            ]
            self.y[i] = self.label(sequence, std)

            self.aux_data[i] = std

            complete_sequence = self.rawdata.iloc[
                start_point
                - 10 : start_point
                + self.len_sequence
                + self.predict_horizon
                + 10
            ]
            self.complete_sequence[i, :, :] = complete_sequence.drop(
                ["std"], axis=1
            ).values

    def label(self, sequence: pd.DataFrame, std: float):
        """
        Label the sequence that is seqlen+predict_horizon long
        """

        sequence_std = std
        sequence_start_value = sequence["close"].values[0]
        sequence_end_value = sequence["close"].values[-1]

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

    def get_data(self):

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        return X_train, X_test, y_train, y_test

    def get_aux_data(self):

        aux_train, aux_test = train_test_split(
            self.aux_data, test_size=0.2, random_state=42
        )
        complete_sequence_train, complete_sequence_test = train_test_split(
            self.complete_sequence, test_size=0.2, random_state=42
        )

        return aux_train, aux_test, complete_sequence_train, complete_sequence_test
