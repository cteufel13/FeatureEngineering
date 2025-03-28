import matplotlib.pyplot as plt
import polars as pl
import pandas as pd
import numpy as np
import shap


def plot_accuracy_over_time(
    times_test: np.ndarray,
    model_prediction_test: np.ndarray,
    y_test: np.ndarray,
    run_name: str,
):
    df = pl.from_numpy(times_test).cast(pl.Datetime("ns", "UTC"))
    times_index_sorted = np.argsort(times_test, axis=0)[:, 0]
    times_test_newest_datetime = pl.from_numpy(
        times_test[times_index_sorted][:, 0]
    ).cast(pl.Datetime("ns", "UTC"))

    model_prediction_test_sorted = model_prediction_test[times_index_sorted]
    y_test_sorted = y_test[times_index_sorted]

    correctness = np.equal(model_prediction_test_sorted, y_test_sorted).astype(int)
    accuracy_cum = np.cumsum(correctness) / np.arange(1, len(correctness) + 1)
    accuracy_ma_100 = np.expand_dims(
        np.convolve(correctness, np.ones(100) / 100, mode="valid"), axis=1
    )
    accuracy_ma_1000 = np.expand_dims(
        np.convolve(correctness, np.ones(1000) / 1000, mode="valid"), axis=1
    )

    fig = plt.figure(figsize=(20, 5))
    plt.title("Accuracy across time", fontsize=20)
    plt.xlabel("Time", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.plot(times_test_newest_datetime, accuracy_cum, label="Cum")
    plt.plot(
        times_test_newest_datetime[50 : 50 + len(accuracy_ma_100)],
        accuracy_ma_100,
        label="MA_100",
    )
    plt.plot(
        times_test_newest_datetime[500 : 500 + len(accuracy_ma_1000)],
        accuracy_ma_1000,
        label="MA_1000",
    )

    # Summer trading hours (Daylight Time: EDT)\
    plt.legend()
    fig.savefig(f"saved_models/plots/{run_name}_plot1.png")


def plot_accuracy_over_day(
    times_test: np.ndarray,
    model_prediction_test: np.ndarray,
    y_test: np.ndarray,
    run_name: str,
):

    times_test = times_test[:, 0].squeeze()
    times_test_datetime = pd.to_datetime(times_test).time
    times_test_index_sorted = np.argsort(times_test_datetime)
    times_test_sorted = times_test[times_test_index_sorted]
    times = pd.to_datetime(times_test_sorted).time
    times_test_newest_datetime = np.array(
        [(lambda t: t.hour + t.minute / 60.0 + t.second / 3600.0)(t) for t in times]
    )

    model_prediction_test_sorted = model_prediction_test[times_test_index_sorted]
    y_test_sorted = y_test[times_test_index_sorted]
    correctness = np.equal(model_prediction_test_sorted, y_test_sorted)
    accuracy_ma_100 = pd.Series(correctness).rolling(window=100).mean().to_numpy()
    accuracy_ma_1000 = pd.Series(correctness).rolling(window=1000).mean().to_numpy()

    fig = plt.figure(figsize=(20, 5))
    plt.plot(times_test_newest_datetime, accuracy_ma_100, label="Accuracy MA 100")
    plt.plot(times_test_newest_datetime, accuracy_ma_1000, label="Accuracy MA 1000")
    # Winter trading hours (Standard Time: EST)
    plt.axvline(
        x=14.5,
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Winter Open",
    )
    plt.axvline(
        x=21.0,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="Winter Close",
    )

    # Summer trading hours (Daylight Time: EDT)
    plt.axvline(x=13.5, color="red", linestyle="--", linewidth=2, label="Summer Open")
    plt.axvline(x=20.0, color="red", linestyle="-", linewidth=2, label="Summer Close")

    plt.xlabel("Time of day (UTC)")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy over day average")
    plt.legend()

    fig.savefig(f"saved_models/plots/{run_name}_plot2.png")


def plot_sensitivity(sens, run_name):
    indices = sorted(sens.keys())
    original_values = [sens[i]["original"] for i in indices]
    increase_values = [sens[i]["increase"] for i in indices]
    decrease_values = [sens[i]["decrease"] for i in indices]

    max_val = max(max(original_values), max(increase_values), max(decrease_values))
    min_val = min(min(original_values), min(increase_values), min(decrease_values))

    fig = plt.figure(figsize=(30, 6))
    plt.plot(indices, original_values, label="Original", marker="o")
    plt.plot(indices, increase_values, label="Increase", marker="x")
    plt.plot(indices, decrease_values, label="Decrease", marker="s")
    plt.xlabel("Time Step (Index)")
    plt.ylabel("Prediction Value")
    plt.ylim(min_val, max_val)
    plt.title("Feature Sensitivity Analysis Over Time Steps")
    plt.legend()
    plt.grid(True)
    fig.savefig(f"saved_models/plots/{run_name}_plot3.png")


def plot_shap_tree(model, X_test, column_names, run_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    for i in range(3):
        data = pd.DataFrame(X_test[:, :], columns=column_names)
        shap.summary_plot(shap_values[:, :, i], data, show=False)
        plt.savefig(f"saved_models/plots/{run_name}_Shap_Class_{i}.png")
        plt.close()
