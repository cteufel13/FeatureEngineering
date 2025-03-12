import matplotlib.pyplot as plt
import numpy as np


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
