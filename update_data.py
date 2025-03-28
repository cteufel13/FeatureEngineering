from src.data.datafetcher import DataFetcher


def main():
    # Initialize the DataFetcher
    data_fetcher = DataFetcher()
    print("Data fetching and updating completed.")

    symbols = list(data_fetcher.log.get("symbols", []))
    schemas = list(data_fetcher.log.get("schemas", []))
    print("Symbols:", symbols)
    print("Schemas:", schemas)

    data_fetcher.fetch_data(symbols=symbols, schemas=schemas)
    data_fetcher.join_cache(schemas=schemas)


if __name__ == "__main__":
    main()
