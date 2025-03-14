import argparse
import json
from src.run_pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Use wandb for logging.",
    )

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
        "--run_live",
        type=bool,
        default=False,
        help="Run live prediction.",
    )

    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=10,
        help="Stop if eval metric doesn't improve after this many rounds.",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation.",
    )

    #  -- XGBoost hyperparameters --

    parser.add_argument(
        "--xgboost_objective",
        type=str,
        default="multi:softprob",
        help="Objective function for multiclass (e.g. 'multi:softprob').",
    )
    parser.add_argument(
        "--xgboost_eval_metric",
        type=str,
        default="mlogloss",
        help="Evaluation metric (e.g. 'mlogloss', 'merror').",
    )
    parser.add_argument(
        "--xgboost_n_estimators",
        type=int,
        default=100,
        help="Number of boosting rounds (trees).",
    )
    parser.add_argument(
        "--xgboost_learning_rate",
        type=float,
        default=0.1,
        help="Step size shrinkage (lower => slower learning, often better generalization).",
    )
    parser.add_argument(
        "--xgboost_max_depth", type=int, default=6, help="Maximum depth of each tree."
    )
    parser.add_argument(
        "--xgboost_subsample",
        type=float,
        default=1.0,
        help="Row subsampling rate (1.0 => use all rows).",
    )
    parser.add_argument(
        "--xgboost_colsample_bytree",
        type=float,
        default=1.0,
        help="Feature subsampling rate for each tree (1.0 => use all features).",
    )
    parser.add_argument(
        "--xgboost_min_child_weight",
        type=float,
        default=1.0,
        help="Minimum sum of instance weight (hessian) in each leaf.",
    )
    parser.add_argument(
        "--xgboost_gamma",
        type=float,
        default=0.0,
        help="Minimum loss reduction required to make a further partition on a leaf node.",
    )
    parser.add_argument(
        "--xgboost_random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--xgboost_tree_method",
        type=str,
        default="auto",
        help="Tree construction algorithm (e.g. 'auto', 'hist', 'gpu_hist').",
    )

    run_pipeline(parser.parse_args())
