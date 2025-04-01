import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score

from src.core.base import ModelBase


class FlattenTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)  # Flatten (n_samples, Seq_len * n_features)


class BasicXGBOOST1(ModelBase):

    job_type = "classification"
    base_library = "xgboost"

    def __init__(self, use_kfold=True, n_splits=5, **params):

        self.xgb_pipeline = Pipeline(
            [
                ("flatten", FlattenTransformer()),
                ("scaler", StandardScaler()),
                (
                    "xgb",
                    xgb.XGBClassifier(
                        objective="multi:softprob",  # Set multi-class objective
                        num_class=5,
                        **params,
                    ),
                ),
            ]
        )
        self.use_kfold = use_kfold
        self.kfoldCV = KFold(n_splits=n_splits)

    def fit_kfold(self, X, y):
        if not self.use_kfold:
            raise ValueError("KFold is not enabled. Set use_kfold to True.")
        scores = cross_val_score(
            self.xgb_pipeline, X, y, cv=self.kfoldCV, scoring="accuracy"
        )
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean()}")

    def fit(self, X, y):
        self.xgb_pipeline.fit(X, y)

    def predict(self, X):
        return self.xgb_pipeline.predict(X)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def predict_categories(self, X_test):
        return self.xgb_pipeline.predict_proba(X_test)

    def save(self, path):
        joblib.dump(self.xgb_pipeline, path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.xgb_pipeline = joblib.load(path)
        print(f"Model loaded from {path}")
