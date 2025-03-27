import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.core.base import ModelBase
from sklearn.base import BaseEstimator, TransformerMixin


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)  # Flatten (n_samples, Seq_len * n_features)


class BasicXGBOOST1(ModelBase):

    job_type = "classification"
    base_library = "xgboost"

    def __init__(self):
        pass

    def fit(self, X, y):
        self.xgb_pipeline.fit(X, y)

    def predict(self, X):
        return self.xgb_pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def init_model(self, callbacks=None, **params):

        if callbacks is not None:

            self.xgb_pipeline = Pipeline(
                [
                    ("flatten", FlattenTransformer()),
                    ("scaler", StandardScaler()),
                    (
                        "xgb",
                        xgb.XGBClassifier(callbacks=[callbacks], **params),
                    ),
                ]
            )
        else:
            self.xgb_pipeline = Pipeline(
                [
                    ("flatten", FlattenTransformer()),
                    ("scaler", StandardScaler()),
                    (
                        "xgb",
                        xgb.XGBClassifier(**params),
                    ),
                ]
            )

    def predict_categories(self, X_test):
        return self.xgb_pipeline.predict_proba(X_test)

    def save(self, path):
        joblib.dump(self.xgb_pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.xgb_pipeline = joblib.load(path)
        print(f"Model loaded from {path}")
