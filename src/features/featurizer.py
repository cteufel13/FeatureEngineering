from src.core.base import FeaturizerBase
from src.utils.utils import get_files_folder
import polars as pl
import numpy as np
from tqdm import tqdm
import json

RAW_PATH = "data/raw/"


class Featurizer1(FeaturizerBase):
    """
    Outdated Featurizer class
    """

    def __init__(self):
        pass

    def featurize(self, data):
        return data


class Featurizer2(FeaturizerBase):
    """
    Outdated Featurizer class
    """

    def __init__(self):
        self.raw_data_paths = get_files_folder(RAW_PATH, extension=".parquet")
        log_path = get_files_folder(RAW_PATH, extension=".json")[0]
        self.log = self.load_log(RAW_PATH + log_path)

        self.symbols = list(self.log["symbols"].keys())
        self.schemas = self.log["schemas"]

    def featurize(self, data):
        return data

    def process(self):
        ohlcv_data, bbo_data, imbalance_data = self.load_data()

        ohlcv_data = ohlcv_data.sort("ts_event").select(
            ["ts_event"] + [col for col in ohlcv_data.columns if col != "ts_event"]
        )
        bbo_data = bbo_data.sort("ts_event").select(
            ["ts_event"] + [col for col in bbo_data.columns if col != "ts_event"]
        )
        imbalance_data = imbalance_data.sort("ts_event").select(
            ["ts_event"] + [col for col in imbalance_data.columns if col != "ts_event"]
        )

        time_symbol = ohlcv_data.select(
            ["ts_event", "rtype", "publisher_id", "instrument_id", "symbol"]
        )

        print("time_index", time_symbol)

        bbo_data = self.process_bbos(bbo_data, time_symbol)

        return ohlcv_data, bbo_data, imbalance_data

    def downsampling():

        pass

    def load_data(self):
        ohlcv_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[0])
        bbo_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[1])
        imbalance_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[2])

        return ohlcv_data, bbo_data, imbalance_data

    def process_bbos(
        self, bbo_data: pl.DataFrame, time_symbol: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Process BBO data with optimized Polars operations for large datasets (1.3M entries per symbol).
        """
        # Create empty result DataFrame with schema
        bbo_downsampled = pl.DataFrame(
            schema={
                "ts_event": pl.Datetime("us", time_zone="UTC"),
                "symbol": pl.String,
                "rtype": pl.Int32,
                "publisher_id": pl.Int32,
                "instrument_id": pl.Int32,
                "mean_bid": pl.Float64,
                "mean_ask": pl.Float64,
                "std_bid": pl.Float64,
                "std_ask": pl.Float64,
                "median_ask": pl.Float64,
                "median_bid": pl.Float64,
                "mean_spread": pl.Float64,
                "std_spread": pl.Float64,
                "median_spread": pl.Float64,
                "total_bid_size": pl.UInt32,
                "total_ask_size": pl.UInt32,
                "max_spread": pl.Float64,
                "min_spread": pl.Float64,
                "mean_spread_change": pl.Float64,
                "std_spread_change": pl.Float64,
                "median_spread_change": pl.Float64,
                "volume_avg_price": pl.Float64,
                "time_avg_price": pl.Float64,
                "order_book_imbalance": pl.Float64,
                "order_book_imbalance_change": pl.Float64,
                "order_book_imbalance_std": pl.Float64,
                "order_book_imbalance_median": pl.Float64,
                "order_book_imbalance_avg": pl.Float64,
                "order_book_imbalance_max": pl.Float64,
                "order_book_imbalance_min": pl.Float64,
                "order_book_imbalance_change_std": pl.Float64,
                "order_book_imbalance_change_median": pl.Float64,
                "order_book_imbalance_change_avg": pl.Float64,
                "bid_ask_ratio": pl.Float64,
                "bid_ask_ratio_change": pl.Float64,
                "bid_ask_ratio_std": pl.Float64,
                "bid_ask_ratio_median": pl.Float64,
                "bid_ask_ratio_avg": pl.Float64,
                "bid_ask_ratio_max": pl.Float64,
                "bid_ask_ratio_min": pl.Float64,
                "bid_ask_ratio_change_std": pl.Float64,
                "bid_ask_ratio_change_median": pl.Float64,
                "bid_ask_ratio_change_avg": pl.Float64,
                "amount_of_updates": pl.Int32,
                "updates_per_second": pl.Float64,
                "depth_stability": pl.Float64,
            }
        )

        # Process all symbols with vectorized operations
        results = []

        # Create a dictionary mapping symbols to filtered DataFrames for efficient lookup
        # Use lazy evaluation for better query optimization
        lazy_time_symbol = time_symbol.lazy()
        lazy_bbo_data = bbo_data.lazy()

        # Process each symbol
        for symbol in self.symbols:
            print(f"Processing symbol {symbol}")

            # Get time ranges for this symbol
            symbol_times = (
                lazy_time_symbol.filter(pl.col("symbol") == symbol)
                .select("ts_event")
                .collect()
            )

            if symbol_times.height < 2:
                continue

            # Extract time points and create intervals
            symbol_times_np = symbol_times["ts_event"].to_numpy()
            starts = symbol_times_np[:-1]
            ends = symbol_times_np[1:]

            # Get BBO data for this symbol - use lazy evaluation
            symbol_bbo = lazy_bbo_data.filter(pl.col("symbol") == symbol).collect()

            # Use NumPy for efficient interval assignment
            ts_events = symbol_bbo["ts_event"].to_numpy()
            interval_idx = np.searchsorted(ends, ts_events, side="right")

            # Add interval index to the DataFrame
            symbol_bbo = symbol_bbo.with_columns(
                pl.Series("interval_idx", interval_idx)
            )

            # Process each interval in a vectorized way
            interval_results = []

            # Group by interval_idx - using Polars' native approach
            partitioned_data = symbol_bbo.partition_by("interval_idx", as_dict=True)

            # Convert to a list of (key, value) pairs so tqdm can track progress
            partition_items = list(partitioned_data.items())

            # Now use tqdm with the known length
            interval_results = []
            for key, interval_data in tqdm(
                partition_items, desc=f"Processing {symbol}"
            ):
                # Extract the actual interval_idx from the first row of data
                # since the key might not be directly usable
                interval_idx = interval_data["interval_idx"][0]

                if interval_idx >= len(starts):  # Safety check
                    continue

                end_min = ends[interval_idx]

                # Calculate statistics for this interval
                stats = self.calc_bbo_statistics(interval_data, symbol, end_min)
                interval_results.append(stats)

            # Combine results for this symbol
            if interval_results:
                symbol_results = pl.concat(interval_results)
                results.append(symbol_results)

        # Combine results for all symbols
        if results:
            return pl.concat(results)
        else:
            return bbo_downsampled  # Return empty DataFrame with correct schema

    def calc_bbo_statistics(
        self, bbo_interval: pl.DataFrame, symbol: str, end_min
    ) -> pl.DataFrame:
        """
        Calculate BBO statistics with optimized Polars operations.
        """
        # Extract scalar values once to avoid repeated lookups
        rtype = bbo_interval["rtype"][0]
        publisher_id = bbo_interval["publisher_id"][0]
        instrument_id = bbo_interval["instrument_id"][0]

        # Use expression-based calculations for better performance
        # Calculate derived columns in a single operation
        expr_list = [
            (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread"),
            (
                (pl.col("bid_sz_00") - pl.col("ask_sz_00"))
                / (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
            ).alias("order_book_imbalance"),
            (pl.col("bid_sz_00") / pl.col("ask_sz_00")).alias("bid_ask_ratio"),
        ]

        # Apply expressions in a single operation
        aux_data = bbo_interval.with_columns(expr_list)

        # Sort and calculate differences in a single operation
        aux_data = aux_data.sort("ts_event").with_columns(
            [
                pl.col("spread").diff().alias("spread_change"),
                pl.col("order_book_imbalance")
                .diff()
                .alias("order_book_imbalance_change"),
                pl.col("bid_ask_ratio").diff().alias("bid_ask_ratio_change"),
            ]
        )

        # Calculate time range and updates_per_second
        count = len(aux_data)
        try:
            time_range = aux_data.select(
                (pl.max("ts_event") - pl.min("ts_event")).dt.total_seconds()
            ).item()
            updates_per_second = count / time_range if time_range > 0 else None
        except:
            updates_per_second = None

        # Calculate all statistics at once - use lazy evaluation for complex aggregations
        stats = (
            aux_data.lazy()
            .select(
                [
                    pl.lit(end_min).alias("ts_event"),
                    pl.lit(symbol).alias("symbol"),
                    pl.lit(rtype).alias("rtype"),
                    pl.lit(publisher_id).alias("publisher_id"),
                    pl.lit(instrument_id).alias("instrument_id"),
                    pl.col("bid_px_00").mean().alias("mean_bid"),
                    pl.col("ask_px_00").mean().alias("mean_ask"),
                    pl.col("bid_px_00").std().alias("std_bid"),
                    pl.col("ask_px_00").std().alias("std_ask"),
                    pl.col("ask_px_00").median().alias("median_ask"),
                    pl.col("bid_px_00").median().alias("median_bid"),
                    pl.col("spread").mean().alias("mean_spread"),
                    pl.col("spread").std().alias("std_spread"),
                    pl.col("spread").median().alias("median_spread"),
                    pl.col("bid_sz_00").sum().alias("total_bid_size"),
                    pl.col("ask_sz_00").sum().alias("total_ask_size"),
                    pl.col("spread").max().alias("max_spread"),
                    pl.col("spread").min().alias("min_spread"),
                    pl.col("spread_change").mean().alias("mean_spread_change"),
                    pl.col("spread_change").std().alias("std_spread_change"),
                    pl.col("spread_change").median().alias("median_spread_change"),
                    (
                        (
                            pl.col("bid_px_00") * pl.col("bid_sz_00")
                            + pl.col("ask_px_00") * pl.col("ask_sz_00")
                        )
                        / (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
                    )
                    .mean()
                    .alias("volume_avg_price"),
                    ((pl.col("bid_px_00") + pl.col("ask_px_00")) / 2)
                    .mean()
                    .alias("time_avg_price"),
                    pl.col("order_book_imbalance").mean().alias("order_book_imbalance"),
                    pl.col("order_book_imbalance_change")
                    .mean()
                    .alias("order_book_imbalance_change"),
                    pl.col("order_book_imbalance")
                    .std()
                    .alias("order_book_imbalance_std"),
                    pl.col("order_book_imbalance")
                    .median()
                    .alias("order_book_imbalance_median"),
                    pl.col("order_book_imbalance")
                    .mean()
                    .alias("order_book_imbalance_avg"),
                    pl.col("order_book_imbalance")
                    .max()
                    .alias("order_book_imbalance_max"),
                    pl.col("order_book_imbalance")
                    .min()
                    .alias("order_book_imbalance_min"),
                    pl.col("order_book_imbalance_change")
                    .std()
                    .alias("order_book_imbalance_change_std"),
                    pl.col("order_book_imbalance_change")
                    .median()
                    .alias("order_book_imbalance_change_median"),
                    pl.col("order_book_imbalance_change")
                    .mean()
                    .alias("order_book_imbalance_change_avg"),
                    pl.col("bid_ask_ratio").mean().alias("bid_ask_ratio"),
                    pl.col("bid_ask_ratio_change").mean().alias("bid_ask_ratio_change"),
                    pl.col("bid_ask_ratio").std().alias("bid_ask_ratio_std"),
                    pl.col("bid_ask_ratio").median().alias("bid_ask_ratio_median"),
                    pl.col("bid_ask_ratio").mean().alias("bid_ask_ratio_avg"),
                    pl.col("bid_ask_ratio").max().alias("bid_ask_ratio_max"),
                    pl.col("bid_ask_ratio").min().alias("bid_ask_ratio_min"),
                    pl.col("bid_ask_ratio_change")
                    .std()
                    .alias("bid_ask_ratio_change_std"),
                    pl.col("bid_ask_ratio_change")
                    .median()
                    .alias("bid_ask_ratio_change_median"),
                    pl.col("bid_ask_ratio_change")
                    .mean()
                    .alias("bid_ask_ratio_change_avg"),
                    pl.lit(count).alias("amount_of_updates"),
                    pl.lit(updates_per_second).alias("updates_per_second"),
                    (
                        (
                            pl.col("bid_sz_00").std() / pl.col("bid_sz_00").mean()
                            + pl.col("ask_sz_00").std() / pl.col("ask_sz_00").mean()
                        )
                        / 2
                    ).alias("depth_stability"),
                ]
            )
            .collect()
        )

        return stats

    def load_log(self, log_path):
        with open(log_path, "r") as f:
            log = json.load(f)

        return log
