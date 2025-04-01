from src.data.datafetcher import DataFetcher
from src.features.featurizer import Featurizer2


def fetch_data():
    # Initialize the DataFetcher
    data_fetcher = DataFetcher()
    print("Data fetching and updating completed.")

    symbols = list(data_fetcher.log.get("symbols", []))
    schemas = list(data_fetcher.log.get("schemas", []))
    print("Symbols:", symbols)
    print("Schemas:", schemas)

    data_fetcher.fetch_data(symbols=symbols, schemas=schemas)
    data_fetcher.join_cache(schemas=schemas)


def featurize():

    featurizer = Featurizer2()
    featurizer.process()
    featurizer.process_symbols()


if __name__ == "__main__":
    # fetch_data()
    featurize()
