import os
import polars as pl
import json
from datetime import datetime, timedelta
import databento as db


LOG_PATH = "./data/raw/log.json"
CACHE_PATH = "./data/cache"
RAW_PATH = "./data/raw"


class DataFetcher:
    def __init__(self):

        self.log = self.load_log()

        self.db_datasets = self.log.get("db_datasets", [])
        self.symbols = self.log.get("symbols", {})

        api_key = os.getenv("DATABENTO_KEY")

        if not api_key:
            raise ValueError("API_KEY not found in environment variables.")

        self.client = db.Historical(api_key)

    def load_log(self):
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, "r") as file:
                return json.load(file)
        return {}

    def save_log(self, log):
        with open(LOG_PATH, "w") as file:
            json.dump(log, file, indent=4)

    def fetch_data(self, symbols, schemas):
        actions = self.get_actions(symbols, schemas)
        self.download(actions)

    def get_actions(self, symbols, schemas):
        # check if symbol and schema exists.

        download_actions = {}

        first_db_date = datetime.fromtimestamp(1525132800000000000 / 1e9)
        last_closing_date = (datetime.now() - timedelta(days=1)).replace(
            hour=23, minute=59, second=0, microsecond=0
        )

        for symbol in symbols:

            symbol_actions = []

            # Check if symbol exists in log
            if symbol in self.symbols:
                symbol_info = self.symbols[symbol]
                db_schemas = symbol_info.get("db_schemas", {})

                # Process each target dataset
                for schema in schemas:
                    # If dataset exists in log, download only new data
                    if schema in db_schemas:
                        # last_end_date = db_schemas[dataset].get("end_date")
                        last_end_date = datetime.strptime(
                            db_schemas[schema].get("end_date"),
                            "%Y-%m-%d %H:%M:%S",
                        )

                        # Only add action if there's new data to download
                        if last_end_date and last_end_date < last_closing_date:
                            symbol_actions.append(
                                {
                                    "dataset": "XNAS.ITCH",
                                    "db_schema": schema,
                                    "start_date": last_end_date,
                                    "end_date": last_closing_date,
                                }
                            )
                            print(last_closing_date, last_end_date)
                    # If dataset doesn't exist for this symbol, download all historical data
                    else:
                        start_date = "2018-04-30 20:00:00"  # Default start date based on your log pattern
                        symbol_actions.append(
                            {
                                "dataset": "XNAS.ITCH",
                                "db_schema": schema,
                                "start_date": start_date,
                                "end_date": last_closing_date,
                            }
                        )
            # If symbol doesn't exist in log, download all datasets from beginning
            else:
                start_date = 1525132800000000000  # Default start date

                for schema in schemas:
                    symbol_actions.append(
                        {
                            "dataset": "XNAS.ITCH",
                            "db_schema": schema,
                            "start_date": start_date,
                            "end_date": last_closing_date,
                        }
                    )

            # Add actions to result if any exist
            if symbol_actions:
                download_actions[symbol] = symbol_actions

        return download_actions

    def download(self, actions):

        downloaded_actions = []

        for symbol in actions:
            symbol_actions = actions[symbol]

            for action in symbol_actions:
                dataset = action["dataset"]
                db_schema = action["db_schema"]
                start_date = action["start_date"]
                end_date = action["end_date"]

                print(
                    f"Downloading {db_schema} for {symbol} from {start_date} to {end_date}"
                )
                data = self.client.timeseries.get_range(
                    dataset=dataset,
                    symbols=symbol,
                    schema=db_schema,
                    start=start_date,
                    end=end_date,
                ).to_df()
                data.to_parquet(f"{CACHE_PATH}/{symbol}_{db_schema}.parquet")

                action["symbol"] = symbol
                action["start_date"] = str(start_date)
                action["end_date"] = end_date.strftime("%Y-%m-%d %H:%M:%S")
                downloaded_actions.append(action)

        self.update_log(downloaded_actions)

    def update_log(self, downloaded_actions):
        for action in downloaded_actions:

            symbol = action["symbol"]
            schema = action["db_schema"]
            start_date = action["start_date"]
            end_date = action["end_date"]

            if symbol not in self.symbols:
                self.symbols[symbol] = {"db_dataset": "XNAS.ITCH", "db_schemas": {}}

            if schema not in self.symbols[symbol]["db_schemas"]:
                self.symbols[symbol]["db_schemas"][schema] = {
                    "start_date": start_date,
                    "end_date": end_date,
                }
            else:
                print("Updating end date for", symbol, schema, end_date)
                self.symbols[symbol]["db_schemas"][schema]["end_date"] = end_date

        self.log["schemas"] = list(set(self.log.get("schemas", [])))
        self.save_log(self.log)

    def join_cache(self, schemas):

        cache_files = os.listdir(CACHE_PATH)
        raw_files = [file for file in os.listdir(RAW_PATH) if file.endswith(".parquet")]

        schema_files = {}
        raw_files = dict.fromkeys(raw_files, [])

        for schema in schemas:
            schema_files[schema] = [file for file in cache_files if schema in file]

        for raw_file in raw_files:
            for schema in schemas:
                if schema in raw_file:
                    raw_files[raw_file] = schema_files[schema]

        for raw_file in raw_files:
            print("Working on:", raw_file)
            if raw_files[raw_file]:
                raw_df = pl.read_parquet(f"{RAW_PATH}/{raw_file}")

                for schema in raw_files[raw_file]:
                    print("Joining", schema)
                    cache_df = pl.read_parquet(f"{CACHE_PATH}/{schema}")

                    cache_cols = cache_df.columns
                    raw_cols = raw_df.columns
                    missing_cols = [col for col in raw_cols if col not in cache_cols]
                    extra_cols = [col for col in cache_cols if col not in raw_cols]

                    for col in missing_cols:
                        cache_df = cache_df.with_columns(pl.lit(None).alias(col))

                    for col in extra_cols:
                        cache_df = cache_df.drop(col)

                    cache_df = cache_df.select(raw_cols)

                    if cache_df["symbol"].unique() in raw_df["symbol"].unique():
                        raw_df = pl.concat([raw_df, cache_df], how="vertical")

                    os.remove(f"{CACHE_PATH}/{schema}")

                raw_df = raw_df.sort("ts_event")
                raw_df = raw_df.unique()
                raw_df.write_parquet(f"{RAW_PATH}/{raw_file}")
