from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.model.evaluation.visualisation import (
    plot_sensitivity,
    plot_accuracy_over_time,
    plot_accuracy_over_day,
    plot_shap_tree,
)


class Evaluation:

    def __init__(self, model, X_test, y_test, time_test, run_name, column_names=None):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.time_test = time_test
        self.run_name = run_name
        self.column_names = column_names

    def get_feature_sensitivity(self, ft_idx, X_test, delta=0.05):
        X_preturbed = X_test.copy()

        y_pred_original = self.model.predict(X_test)

        X_preturbed[:, ft_idx] *= 1 + delta
        y_pred_increase = self.model.predict(X_preturbed)

        X_preturbed[:, ft_idx] = X_test[:, ft_idx] * (1 - delta)
        y_pred_decrease = self.model.predict(X_preturbed)

        sensitivity_ft = {
            "original": y_pred_original,
            "increase": y_pred_increase,
            "decrease": y_pred_decrease,
        }

        return sensitivity_ft

    def get_feature_sensitivity_all(
        self, delta: float = 0.05, sequence_length: int = 100
    ):

        unperturbed_X_test = self.X_test.copy()
        n_samples = unperturbed_X_test.shape[0]
        sequence_length = sequence_length
        n_features = unperturbed_X_test.shape[1] // sequence_length

        unperturbed_X_test = np.reshape(
            unperturbed_X_test, (unperturbed_X_test.shape[0], sequence_length, -1)
        )

        sensitivity = {}

        for ft_idx in tqdm(range(n_features)):
            sensitivity[ft_idx] = self.get_feature_sensitivity(
                ft_idx, unperturbed_X_test, delta=delta
            )
            for preds in sensitivity[ft_idx]:
                sensitivity[ft_idx][preds] = self.calculate_accuracy(
                    sensitivity[ft_idx][preds], self.y_test
                )

        plot_sensitivity(sensitivity, self.run_name)

        return sensitivity

    def calculate_accuracy(self, y_pred, y_test):
        return np.mean(y_pred == y_test)

    def get_time_accuracy_year(self):
        return plot_accuracy_over_time(
            self.time_test, self.model.predict(self.X_test), self.y_test, self.run_name
        )

    def get_time_accuracy_day(self):
        return plot_accuracy_over_day(
            self.time_test, self.model.predict(self.X_test), self.y_test, self.run_name
        )

    def get_shap(self):

        extended_columns = []
        for i in range(100):
            extended_columns.extend([f"{name}_{i}" for name in self.column_names])

        if self.model.base_library == "xgboost":
            plot_shap_tree(
                model=self.model.xgb_pipeline.named_steps["xgb"],
                X_test=self.model.xgb_pipeline.named_steps["scaler"].transform(
                    self.X_test
                ),
                column_names=extended_columns,
                run_name=self.run_name,
            )
