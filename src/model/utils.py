import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)  # Flatten (n_samples, Seq_len * n_features)


def visualize_performance(
    X_test, y_test, y_pred, index, aux_data, complete_sequence_test
):
    """
    Visualize the performance of the model
    """

    sequence = X_test[index]
    complete_sequence = complete_sequence_test[index]
    label = y_test[index]
    pred = y_pred[index]
    std = aux_data[index]

    close = complete_sequence[:, 3]

    fig, ax = plt.subplots()
    ax.plot(close, label="Close")
    ax.axhline(y=close[0] + std, color="r", linestyle="--")
    ax.axhline(y=close[0] - std, color="r", linestyle="--")
