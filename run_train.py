from src.train import run_pipeline
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="Dataset2")
    parser.add_argument("--model", type=str, default="BasicXGBOOST1")
    parser.add_argument("--featurizer", type=str, default="Featurizer2")
    parser.add_argument("--seq_len", type=int, default=100)
    parser.add_argument("--predict_horizon", type=int, default=10)
    parser.add_argument("--make_new_features", type=bool, default=False)
    args = parser.parse_args()
    run_pipeline(args)
