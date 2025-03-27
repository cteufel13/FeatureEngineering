from src.dataset.dataset2 import Dataset2
from src.model.models.BasicXGBOOST1 import BasicXGBOOST1


def run():
    dataset = Dataset2()
    X_train, X_test, y_train, y_test, times_train, times_test = dataset.get_train_test()
    model = BasicXGBOOST1()
    model.init_model()
    print("Fitting model")
    model.fit(X_train, y_train)
    accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    model.save("saved_models/model_xgboost1.json")

    return X_test, y_test, times_test


if __name__ == "__main__":
    run()
