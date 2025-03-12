import numpy as np
import pandas as pd

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)  # Flatten (n_samples, Seq_len * n_features)


class BasicXGBOOST1:

    def __init__(self, **kwargs):
        self.xgb_pipeline = Pipeline(
            [
                ("flatten", FlattenTransformer()),
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        objective="multi:softmax",
                        eval_metric="mlogloss",
                        num_class=5,
                        n_estimators=200,
                        max_depth=30,
                        kwargs=kwargs,
                    ),
                ),
            ]
        )

    def fit(self, X, y):
        self.xgb_pipeline.fit(X, y)

    def predict(self, X):
        return self.xgb_pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
