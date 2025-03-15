import databento as db
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import os
from pathlib import Path

from src.data.prep import drop_db_cols
from src.data.utils import get_data, check_file
from src.model.model import *
from src.model.utils import visualize_performance
from src.dataset.dataset1 import *
from src.features.featurizer import *
from src.model.live_test import LiveStreamPredictor

import wandb
from wandb.integration.xgboost import WandbCallback


def run_pipeline(args):
    """
    -------------   0. Load the classes   -------------
    """

    dataset_class = globals()[args.dataset]
    model_class = globals()[args.model]
    featurizer_class = globals()[args.featurizer]

    API_KEY = os.environ["DATABENTO_KEY"]
    DATASET = args.DB_Dataset
    SYMBOL = args.symbol
    START_DATE = args.start_date
    END_DATE = args.end_date

    n_samples = args.n_samples
    len_sequence = args.seq_len
    predict_horizon = args.predict_horizon

    featurizer = featurizer_class()
    model = model_class()

    wandb_params = {
        "dataset": DATASET,
        "symbol": SYMBOL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "n_samples": n_samples,
        "len_sequence": len_sequence,
        "predict_horizon": predict_horizon,
    }

    if args.use_wandb:
        wandb.init(project="Forecasting Market", config=wandb_params)

    if model.base_library == "xgboost":

        model_params = {
            "objective": args.xgboost_objective,
            "eval_metric": args.xgboost_eval_metric,
            "n_estimators": args.xgboost_n_estimators,
            "learning_rate": args.xgboost_learning_rate,
            "max_depth": args.xgboost_max_depth,
            "subsample": args.xgboost_subsample,
            "colsample_bytree": args.xgboost_colsample_bytree,
            "min_child_weight": args.xgboost_min_child_weight,
            "gamma": args.xgboost_gamma,
            "random_state": args.xgboost_random_state,
            "tree_method": args.xgboost_tree_method,
        }

        # wandb.config.update(model_params)
        callback = WandbCallback()

    model.init_model(callback, **model_params)

    """
    -------------   1. Get the data   -------------
    """

    file_exists, history_file = check_file(SYMBOL)

    df = get_data(
        file_exists=file_exists,
        path=history_file,
        Dataset=DATASET,
        Symbol=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        api_key=API_KEY,
    )

    df_test = drop_db_cols(df)

    """
    -------------   2. Featurize the data   -------------
    """

    print("Featurizing data...")

    df_test = featurizer.featurize(df_test)

    df_test = df_test.drop(["ts_event"], axis=1)

    dataset = dataset_class(
        df_test,
        n_samples=n_samples,
        len_sequence=len_sequence,
        predict_horizon=predict_horizon,
    )

    """
    -------------   3. Train the model   -------------
    """

    print("Processing data...")

    dataset.process()

    X_train, X_test, y_train, y_test = dataset.get_data()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = model.evaluate(X_test, y_test)

    print(f"Test Accuracy: {acc:.4f}")

    aux_train, aux_test, complete_sequence_train, complete_sequence_test = (
        dataset.get_aux_data()
    )

    category_preds = None

    if model.job_type == "classification":
        category_preds = model.predict_categories(X_test)

    for i in range(10):
        figure = visualize_performance(
            X_test,
            y_test,
            y_pred,
            i,
            aux_test,
            complete_sequence_test,
            buffer_factor=0.005,
            category_preds=category_preds,
        )
        wandb.log({f"test_set_result_{i}": figure})

    if args.run_live:
        LiveStreamPredictor(
            model,
            SYMBOL,
            len_sequence,
            API_KEY,
            data_feed="iex",
        ).run()
