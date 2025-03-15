import plotly.graph_objects as go
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from plotly.subplots import make_subplots


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)  # Flatten (n_samples, Seq_len * n_features)


def visualize_performance(
    X_test,
    y_test,
    y_pred,
    index,
    aux_data,
    complete_sequence_test,
    buffer_factor=0.05,
    seq_len=100,
    predict_horizon=10,
    category_preds=None,
):
    """
    Visualize the performance of the model
    """

    sequence = X_test[index]
    complete_sequence = complete_sequence_test[index]
    label = y_test[index]
    pred = y_pred[index]
    std = aux_data[index, 0]

    close = complete_sequence[:, 3]

    y_min, y_max = close.min(), close.max()
    y_min, y_max = min(y_min, close[10] - std), max(y_max, close[10] + std)
    y_min, y_max = (1 - buffer_factor) * y_min, (1 + buffer_factor) * y_max

    # print("ymin", y_min, "ymax", y_max)

    if category_preds is not None:
        category_pred = category_preds[index]
        print(category_pred)
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Stock Forecast", "Prediction"),
            column_widths=[0.66, 0.33],
            shared_yaxes=False,
        )
        categories = ["No Move", "Up", "Down"]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=category_pred,
                name="Category Prediction",
                marker=dict(color="red"),
            ),
            row=1,
            col=2,
        )

    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Stock Forecast"))

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(close)),
            y=close,
            mode="lines",
            marker=dict(color="black", size=10),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(close))[10 : seq_len + 1],
            y=close[10 : seq_len + 1],
            mode="lines",
            name="Close",
            marker=dict(color="blue", size=10),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(close))[seq_len : seq_len + predict_horizon],
            y=close[seq_len : seq_len + predict_horizon],
            mode="lines",
            name="Close (unseen)",
            marker=dict(color="red", size=10),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        title="Close Price with +/- std",
        xaxis_title="Index",
        yaxis_title="Price",
        width=1200,
        height=400,
    )

    fig.add_hline(
        y=close[10] + std,
        line_dash="dash",
        line_color="red",
        annotation_text=f"+std (i={index})",
        annotation_position="top left",
        row=1,
        col=1,
    )
    fig.add_hline(
        y=close[10] - std,
        line_dash="dash",
        line_color="red",
        annotation_text=f"-std (i={index})",
        annotation_position="bottom left",
        row=1,
        col=1,
    )

    fig.update_yaxes(range=[y_min, y_max], row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=2)

    return fig
