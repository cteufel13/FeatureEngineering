from src.dataset.dataset1 import Dataset1
from src.dataset.dataset2 import Dataset2
from src.features.featurizer import *
from src.model.models.BasicXGBOOST1 import BasicXGBOOST1
from src.model.evaluation.evaluation import Evaluation
from datetime import datetime
import time


def run_pipeline(args):

    run_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    dataset_class = globals()[args.dataset_version]
    model_class = globals()[args.model]
    featurizer_class = globals()[args.featurizer]

    len_sequence = args.seq_len
    predict_horizon = args.predict_horizon

    use_kfold = args.use_kfold

    if args.make_new_features:
        featurizer = featurizer_class()
        featurizer.process()
        featurizer.process_symbols()

    dataset = dataset_class(
        sequence_length=len_sequence,
        predict_horizon=predict_horizon,
        data_type=args.data_type,
        use_ta=args.use_ta,
    )
    print("Dataset Columns:", dataset.columns)

    X_train, X_test, y_train, y_test, times_train, times_test = dataset.get_train_test()

    model = model_class(use_kfold=use_kfold)

    print("fitting Kfold")
    model.fit_kfold(dataset.X, dataset.y)

    print("Training model")
    model.fit(X_train, y_train)

    print("Evaluating model")
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")

    run_name = f"{model_class.__name__}_{dataset_class.__name__}_{featurizer_class.__name__}_{run_time}"

    y_pred = model.predict(X_test)
    print("Evaluating accuracy over time")

    eval = Evaluation(
        model, X_test, y_test, times_test, run_name, column_names=dataset.columns
    )
    eval.get_feature_sensitivity_all()
    eval.get_time_accuracy_day()
    eval.get_time_accuracy_year()
    eval.get_shap()

    print("Saving model")

    model.save(
        path=f"saved_models/{run_name}",
    )
