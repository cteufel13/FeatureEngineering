import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.core.base import ModelBase
from src.model.utils import FlattenTransformer


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

    def init_model(self, callbacks, **params):
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
