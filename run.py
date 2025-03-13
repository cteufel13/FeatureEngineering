import argparse
import json
from src.run_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset1",
        help="Dataset name, check dataset.py.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="BasicXGBOOST1",
        help="Model name, check model.py.",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2018-05-01",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default="2025-02-28",
        help="Path to the data directory.",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="AAPL",
        help="Symbol to use for training.",
    )

    parser.add_argument(
        "--DB_Dataset",
        type=str,
        default="XNAS.ITCH",
        help="Dataset to use for training.",
    )

    parser.add_argument(
        "--kwargs",
        type=json.loads,
        help="A JSON string representing the keyword arguments.",
    )

    parser.add_argument(
        "--featurizer",
        type=str,
        default="Featurizer1",
        help="Featurizer name, check featurizer.py.",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="Number of samples to use for training.",
    )

    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length for training.",
    )

    parser.add_argument(
        "--predict_horizon",
        type=int,
        default=10,
        help="Prediction horizon.",
    )

    parser.add_argument(
        "run_live",
        type=bool,
        default=False,
        help="Run live prediction.",
    )

    run_pipeline(parser.parse_args())
