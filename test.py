from src.features.featurizer import Featurizer2
import polars as pl

featurizer = Featurizer2()
ohlcv_data, bbo_data, imbalance_data = featurizer.process()
