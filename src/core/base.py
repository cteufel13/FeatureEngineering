"""
All the function that are required to be implemented by the user are defined here.
"""


class ModelBase:
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def evaluate(self, X, y):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError


class DatasetBase:

    def __init__(self, rawdata, n_samples=10000, len_sequence=5, predict_horizon=10):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def get_data(self):
        raise NotImplementedError


class FeaturizerBase:

    def __init__(self):
        pass

    def featurize(self, data):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
