from src.core.base import FeaturizerBase
from src.utils.utils import get_files_folder
import polars as pl
import numpy as np
from tqdm import tqdm
import json
import os

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

    def __init__(self):
        self.raw_data_paths = get_files_folder(RAW_PATH, extension=".parquet")
        log_path = get_files_folder(RAW_PATH, extension=".json")[0]
        self.log = self.load_log(RAW_PATH + log_path)

        self.symbols = list(self.log["symbols"].keys())

        self.schemas = self.log["schemas"]

        self.make_symbol_folders()

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

        # self.process_bbos(bbo_data, time_symbol)
        self.process_ohlcv(ohlcv_data)
        # self.process_imbalance(imbalance_data, time_symbol)

    def load_data(self):
        ohlcv_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[0])
        bbo_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[1])
        imbalance_data = pl.read_parquet(RAW_PATH + self.raw_data_paths[2])

        return ohlcv_data, bbo_data, imbalance_data

    def process_bbos(self, bbo_data: pl.DataFrame, time_data):

        bbo_data = bbo_data.with_columns(
            pl.when(pl.col("symbol") == "GOOGL")
            .then(pl.lit("GOOG"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        time_data = time_data.with_columns(
            pl.when(pl.col("symbol") == "GOOGL")
            .then(pl.lit("GOOG"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        lazy_bbo_data = bbo_data.lazy()
        lazy_time_symbol = time_data.lazy()

        symbols = bbo_data["symbol"].unique().to_list()
        symbols.sort()
        print(f"Processing symbols: {symbols}")
        results = []
        time_data_collected = lazy_time_symbol.collect()

        symbols = ["GOOG"]
        for symbol in symbols:
            print(f"Processing symbol {symbol}")

            symbol_times = time_data_collected.filter(pl.col("symbol") == symbol)

            if symbol_times.height < 2:
                continue

            # Extract time points and create intervals
            symbol_times_np = symbol_times["ts_event"].to_numpy()
            starts = pl.Series(
                symbol_times_np[:-1], dtype=pl.Datetime("ns", time_zone="UTC")
            )
            ends = pl.Series(
                symbol_times_np[1:], dtype=pl.Datetime("ns", time_zone="UTC")
            )
            symbol_list = [symbol] * len(starts)
            interval_idx = np.arange(0, len(starts))

            intervals_df = pl.DataFrame(
                {
                    "interval_idx": interval_idx,
                    "start_time": starts,
                    "end_time": ends,
                    "symbol": symbol_list,
                }
            ).with_columns(
                [
                    pl.col("start_time")
                    .dt.replace_time_zone("UTC")
                    .alias("start_time"),
                    pl.col("end_time").dt.replace_time_zone("UTC").alias("end_time"),
                ]
            )

            # Get BBO data for this symbol
            symbol_bbo = lazy_bbo_data.filter(pl.col("symbol") == symbol).collect()

            symbol_bbo = symbol_bbo.filter(pl.col("ts_event").is_not_null())
            if symbol_bbo.height == 0:
                continue

            # Use join_asof to efficiently assign intervals
            symbol_bbo = symbol_bbo.join_asof(
                intervals_df.sort("start_time"),
                left_on="ts_event",
                right_on="start_time",
                strategy="forward",
                suffix="_interval",
            )
            symbol_bbo = symbol_bbo.filter(pl.col("ts_event") <= pl.col("end_time"))
            if symbol_bbo.height == 0:
                continue

            interval_results = []

            partitioned_data = symbol_bbo.partition_by("interval_idx", as_dict=True)

            partition_items = list(partitioned_data.items())

            # quarter_len = len(partition_items) // 4
            # subset_partition_items = partition_items[:2]

            for key, interval_data in tqdm(
                partition_items, desc=f"Processing {symbol}"
            ):
                interval_idx = interval_data["interval_idx"][0]

                if interval_idx >= len(ends):  # Safety check
                    continue

                end_min = ends[interval_idx]

                stats = self.calc_bbo_statistics(interval_data, symbol, end_min)
                interval_results.append(stats)

            if interval_results:

                symbol_results = pl.concat(interval_results)
                symbol_results = symbol_results.sort("ts_event")
                symbol_results.write_parquet(
                    f"data/processed/{symbol}/bbo_data.parquet"
                )

    def calc_bbo_statistics(
        self, bbo_interval: pl.DataFrame, symbol: str, end_min
    ) -> pl.DataFrame:
        """
        Calculate BBO statistics with optimized Polars operations.
        Keeping the original function signature but with optimized implementation.
        """
        rtype = bbo_interval["rtype"][0]
        publisher_id = bbo_interval["publisher_id"][0]
        instrument_id = bbo_interval["instrument_id"][0]

        expr_list = [
            (pl.col("ask_px_00") - pl.col("bid_px_00")).alias("spread"),
            (
                (pl.col("bid_sz_00") - pl.col("ask_sz_00"))
                / (pl.col("bid_sz_00") + pl.col("ask_sz_00"))
            ).alias("order_book_imbalance"),
            (pl.col("bid_sz_00") / pl.col("ask_sz_00")).alias("bid_ask_ratio"),
        ]

        aux_data = bbo_interval.with_columns(expr_list)

        aux_data = aux_data.sort("ts_event").with_columns(
            [
                pl.col("spread").diff().alias("spread_change"),
                pl.col("order_book_imbalance")
                .diff()
                .alias("order_book_imbalance_change"),
                pl.col("bid_ask_ratio").diff().alias("bid_ask_ratio_change"),
            ]
        )

        count = len(aux_data)
        try:
            time_range = aux_data.select(
                (pl.max("ts_event") - pl.min("ts_event")).dt.total_seconds()
            ).item()
            updates_per_second = count / time_range if time_range > 0 else None
        except:
            updates_per_second = None

        try:
            depth_stability = (
                aux_data["bid_sz_00"].std(ddof=0) / aux_data["bid_px_00"].mean()
                + aux_data["ask_sz_00"].std(ddof=0) / aux_data["ask_px_00"].mean()
            ) / 2
        except:
            depth_stability = None
        schema = {
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

        stats = pl.DataFrame(
            {
                "ts_event": [end_min],
                "symbol": [symbol],
                "rtype": [rtype],
                "publisher_id": [publisher_id],
                "instrument_id": [instrument_id],
                "mean_bid": [aux_data["bid_px_00"].mean()],
                "mean_ask": [aux_data["ask_px_00"].mean()],
                "std_bid": [aux_data["bid_px_00"].std(ddof=0)],
                "std_ask": [aux_data["ask_px_00"].std(ddof=0)],
                "median_ask": [aux_data["ask_px_00"].median()],
                "median_bid": [aux_data["bid_px_00"].median()],
                "mean_spread": [aux_data["spread"].mean()],
                "std_spread": [aux_data["spread"].std()],
                "median_spread": [aux_data["spread"].median()],
                "total_bid_size": [aux_data["bid_sz_00"].sum()],
                "total_ask_size": [aux_data["ask_sz_00"].sum()],
                "max_spread": [aux_data["spread"].max()],
                "min_spread": [aux_data["spread"].min()],
                "mean_spread_change": [aux_data["spread_change"].mean()],
                "std_spread_change": [aux_data["spread_change"].std(ddof=0)],
                "median_spread_change": [aux_data["spread_change"].median()],
                "volume_avg_price": [
                    (
                        (
                            aux_data["bid_px_00"] * aux_data["bid_sz_00"]
                            + aux_data["ask_px_00"] * aux_data["ask_sz_00"]
                        )
                        / (aux_data["bid_sz_00"] + aux_data["ask_sz_00"])
                    ).mean()
                ],
                "time_avg_price": [
                    ((aux_data["bid_px_00"] + aux_data["ask_px_00"]) / 2).mean()
                ],
                "order_book_imbalance": [aux_data["order_book_imbalance"].mean()],
                "order_book_imbalance_change": [
                    aux_data["order_book_imbalance_change"].mean()
                ],
                "order_book_imbalance_std": [
                    aux_data["order_book_imbalance"].std(ddof=0)
                ],
                "order_book_imbalance_median": [
                    aux_data["order_book_imbalance"].median()
                ],
                "order_book_imbalance_avg": [aux_data["order_book_imbalance"].mean()],
                "order_book_imbalance_max": [aux_data["order_book_imbalance"].max()],
                "order_book_imbalance_min": [aux_data["order_book_imbalance"].min()],
                "order_book_imbalance_change_std": [
                    aux_data["order_book_imbalance_change"].std(ddof=0)
                ],
                "order_book_imbalance_change_median": [
                    aux_data["order_book_imbalance_change"].median()
                ],
                "order_book_imbalance_change_avg": [
                    aux_data["order_book_imbalance_change"].mean()
                ],
                "bid_ask_ratio": [aux_data["bid_ask_ratio"].mean()],
                "bid_ask_ratio_change": [aux_data["bid_ask_ratio_change"].mean()],
                "bid_ask_ratio_std": [aux_data["bid_ask_ratio"].std(ddof=0)],
                "bid_ask_ratio_median": [aux_data["bid_ask_ratio"].median()],
                "bid_ask_ratio_avg": [aux_data["bid_ask_ratio"].mean()],
                "bid_ask_ratio_max": [aux_data["bid_ask_ratio"].max()],
                "bid_ask_ratio_min": [aux_data["bid_ask_ratio"].min()],
                "bid_ask_ratio_change_std": [
                    aux_data["bid_ask_ratio_change"].std(ddof=0)
                ],
                "bid_ask_ratio_change_median": [
                    aux_data["bid_ask_ratio_change"].median()
                ],
                "bid_ask_ratio_change_avg": [aux_data["bid_ask_ratio_change"].mean()],
                "amount_of_updates": [count],
                "updates_per_second": [updates_per_second],
                "depth_stability": [depth_stability],
            },
            schema=schema,
        )

        return stats

    def process_ohlcv(self, ohlcv_data):

        ohlcv_data = ohlcv_data.with_columns(
            pl.when(pl.col("symbol") == "GOOGL")
            .then(pl.lit("GOOG"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        symbols = ohlcv_data["symbol"].unique().to_list()

        for symbol in symbols:
            print(f"Processing symbol {symbol}")
            symbol_data = ohlcv_data.filter(pl.col("symbol") == symbol)
            symbol_data = symbol_data.sort("ts_event")

            symbol_data = self.calc_ta_indicators(symbol_data)
            symbol_data.write_parquet(f"data/processed/{symbol}/ohlcv_data.parquet")

    def calc_ta_indicators(self, ohlcv_data_symbol: pl.DataFrame):
        # ------- SMAs ---------#

        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            pl.col("close").rolling_mean(window_size=5).alias("sma_5"),
            pl.col("close").rolling_mean(window_size=10).alias("sma_10"),
            pl.col("close").rolling_mean(window_size=20).alias("sma_20"),
            pl.col("close").rolling_mean(window_size=30).alias("sma_30"),
            pl.col("close").rolling_mean(window_size=50).alias("sma_60"),
            pl.col("close").rolling_mean(window_size=100).alias("sma_200"),
        )

        # -------- EMAs ---------#
        c_np = ohlcv_data_symbol["close"].to_numpy()
        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [
                pl.Series("ema_12", self.compute_ema(c_np, 12)),
                pl.Series("ema_26", self.compute_ema(c_np, 26)),
                pl.Series("ema_50", self.compute_ema(c_np, 50)),
                pl.Series("ema_200", self.compute_ema(c_np, 200)),
            ]
        )

        # -------- RSI ---------#
        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            pl.Series("rsi_14", self.compute_rsi(c_np, 14)),
            pl.Series("rsi_30", self.compute_rsi(c_np, 30)),
            pl.Series("rsi_50", self.compute_rsi(c_np, 50)),
        )

        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [
                (
                    (
                        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3)
                        * pl.col("volume")
                    ).cum_sum()
                    / pl.col("volume").cum_sum()
                ).alias("vwap_typical_ohlcv")
            ]
        )

        # -------- Stoch. Oscillator --------- #

        period = 14
        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [
                pl.col("low").rolling_min(window_size=period).alias("lowest_low"),
                pl.col("high").rolling_max(window_size=period).alias("highest_high"),
            ]
        )

        # Calculate %K
        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [
                (
                    (
                        (pl.col("close") - pl.col("lowest_low"))
                        / (pl.col("highest_high") - pl.col("lowest_low"))
                    )
                    * 100
                ).alias("stochastic_k")
            ]
        )

        # Calculate %D as a 3-period moving average of %K
        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [pl.col("stochastic_k").rolling_mean(window_size=3).alias("stochastic_d")]
        )

        ohlcv_data_symbol = ohlcv_data_symbol.with_columns(
            [pl.col("close").rolling_std(window_size=100).alias("std")]
        )

        return ohlcv_data_symbol

    def process_imbalance(self, imbalance_data: pl.DataFrame, time_data: pl.DataFrame):
        imbalance_data = imbalance_data.with_columns(
            pl.when(pl.col("symbol") == "GOOGL")
            .then(pl.lit("GOOG"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        time_data = time_data.with_columns(
            pl.when(pl.col("symbol") == "GOOGL")
            .then(pl.lit("GOOG"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        # Get unique symbols and sort
        symbols = imbalance_data["symbol"].unique().to_list()
        symbols.sort()

        print(f"Processing symbols: {symbols}")

        for symbol in symbols:
            print(f"Processing symbol {symbol}")

            # Collect time data for the symbol
            symbol_times = time_data.filter(pl.col("symbol") == symbol)
            print(symbol_times)
            if symbol_times.height < 2:
                continue

            # Extract time points and create intervals
            symbol_times_np = symbol_times["ts_event"].to_numpy()
            starts = pl.Series(
                symbol_times_np[:-1], dtype=pl.Datetime("ns", time_zone="UTC")
            )
            ends = pl.Series(
                symbol_times_np[1:], dtype=pl.Datetime("ns", time_zone="UTC")
            )
            symbol_list = [symbol] * len(starts)
            interval_idx = np.arange(0, len(starts))

            print(len(starts), len(ends), len(symbol_list), len(interval_idx))

            # Create intervals DataFrame
            intervals_df = pl.DataFrame(
                {
                    "interval_idx": interval_idx,
                    "start_time": starts,
                    "end_time": ends,
                    "symbol": symbol_list,
                }
            ).with_columns(
                [
                    pl.col("start_time")
                    .dt.replace_time_zone("UTC")
                    .alias("start_time"),
                    pl.col("end_time").dt.replace_time_zone("UTC").alias("end_time"),
                ]
            )

            # Filter and prepare imbalance data for the symbol
            symbol_data = imbalance_data.filter(pl.col("symbol") == symbol)
            symbol_data = symbol_data.filter(pl.col("ts_event").is_not_null())

            if symbol_data.height == 0:
                continue

            # Use join_asof to efficiently assign intervals
            symbol_data = symbol_data.join_asof(
                intervals_df.sort("start_time"),
                left_on="ts_event",
                right_on="start_time",
                strategy="forward",
                suffix="_interval",
            )
            symbol_data = symbol_data.filter(pl.col("ts_event") <= pl.col("end_time"))

            if symbol_data.height == 0:
                continue

            interval_results = []

            # Partition data by interval
            partitioned_data = symbol_data.partition_by("interval_idx", as_dict=True)
            partition_items = list(partitioned_data.items())

            for key, interval_data in tqdm(
                partition_items, desc=f"Processing {symbol}"
            ):
                interval_idx = interval_data["interval_idx"][0]

                if interval_idx >= len(ends):  # Safety check
                    continue

                end_min = ends[interval_idx]

                # Prep and calculate imbalance indicators for each interval
                interval_data_df = pl.DataFrame(interval_data)
                interval_data_df = self.prep_imbalance_data(interval_data_df)
                interval_data_df = self.calc_imbalance_indicators(
                    interval_data_df, symbol, end_min
                )

                interval_results.append(interval_data_df)

            if interval_results:
                # Concatenate and sort results
                symbol_results = pl.concat(interval_results)
                symbol_results = symbol_results.sort("ts_event")

                # Write processed data to parquet
                symbol_results.write_parquet(
                    f"data/processed/{symbol}/imbalance_data.parquet"
                )

    def calc_imbalance_indicators(
        self, orderbook_interval: pl.DataFrame, symbol: str, end_min
    ) -> pl.DataFrame:

        # Extract key identifiers
        rtype = orderbook_interval["rtype"][0]
        publisher_id = orderbook_interval["publisher_id"][0]
        instrument_id = orderbook_interval["instrument_id"][0]

        # Additional derived metrics
        expr_list = [
            # Core metrics
            ((pl.col("ref_price") - pl.col("ref_price").shift(1)).abs()).alias(
                "price_change"
            ),
            (pl.col("total_imbalance_qty") / pl.col("paired_qty")).alias(
                "imbalance_ratio"
            ),
            # Trading side and auction type characteristics
            pl.col("side").alias("dominant_side"),
            pl.col("auction_type").alias("auction_type"),
        ]

        # Process the data with additional computations
        aux_data = orderbook_interval.with_columns(expr_list)

        # Calculate interval duration
        try:
            time_range = aux_data.select(
                (pl.max("ts_event") - pl.min("ts_event")).dt.total_seconds()
            ).item()
            updates_per_second = len(aux_data) / time_range if time_range > 0 else None
        except:
            time_range = None
            updates_per_second = None

        # Comprehensive schema for detailed statistics
        schema = {
            # Temporal and Identifier Information
            "ts_event": pl.Datetime("us", time_zone="UTC"),
            "symbol": pl.String,
            "rtype": pl.Int32,
            "publisher_id": pl.Int32,
            "instrument_id": pl.Int32,
            # Price Metrics
            "ref_price_mean": pl.Float64,
            "ref_price_std": pl.Float64,
            "ref_price_median": pl.Float64,
            "ref_price_min": pl.Float64,
            "ref_price_max": pl.Float64,
            # Clearing Price Metrics
            "cont_book_clr_price_mean": pl.Float64,
            "auct_interest_clr_price_mean": pl.Float64,
            # Quantity and Imbalance Metrics
            "total_paired_qty": pl.Int64,
            "total_imbalance_qty": pl.Int64,
            "mean_imbalance_qty": pl.Float64,
            "imbalance_ratio_mean": pl.Float64,
            "imbalance_ratio_std": pl.Float64,
            # Auction Characteristics
            "dominant_side": pl.Int32,
            "auction_type": pl.Int32,
            "significant_imbalance_count": pl.Int64,
            # Temporal Metrics
            "interval_duration": pl.Float64,
            "updates_per_second": pl.Float64,
            # Price Change Metrics
            "mean_price_change": pl.Float64,
            "std_price_change": pl.Float64,
            "max_price_change": pl.Float64,
        }

        # Compile statistics into a single-row DataFrame
        stats = pl.DataFrame(
            {
                # Temporal and Identifier Information
                "ts_event": [end_min],
                "symbol": [symbol],
                "rtype": [rtype],
                "publisher_id": [publisher_id],
                "instrument_id": [instrument_id],
                # Reference Price Metrics
                "ref_price_mean": [orderbook_interval["ref_price"].mean()],
                "ref_price_std": [orderbook_interval["ref_price"].std(ddof=0)],
                "ref_price_median": [orderbook_interval["ref_price"].median()],
                "ref_price_min": [orderbook_interval["ref_price"].min()],
                "ref_price_max": [orderbook_interval["ref_price"].max()],
                # Clearing Price Metrics
                "cont_book_clr_price_mean": [
                    orderbook_interval["cont_book_clr_price"].mean()
                ],
                "auct_interest_clr_price_mean": [
                    orderbook_interval["auct_interest_clr_price"].mean()
                ],
                # Quantity and Imbalance Metrics
                "total_paired_qty": [orderbook_interval["paired_qty"].sum()],
                "total_imbalance_qty": [
                    orderbook_interval["total_imbalance_qty"].sum()
                ],
                "mean_imbalance_qty": [
                    orderbook_interval["total_imbalance_qty"].mean()
                ],
                "imbalance_ratio_mean": [aux_data["imbalance_ratio"].mean()],
                "imbalance_ratio_std": [aux_data["imbalance_ratio"].std(ddof=0)],
                # Auction Characteristics
                "dominant_side": [orderbook_interval["side"].mode()[0]],
                "auction_type": [orderbook_interval["auction_type"].mode()[0]],
                "significant_imbalance_count": [
                    orderbook_interval["significant_imbalance"].sum()
                ],
                # Temporal Metrics
                "interval_duration": [time_range],
                "updates_per_second": [updates_per_second],
                # Price Change Metrics
                "mean_price_change": [aux_data["price_change"].mean()],
                "std_price_change": [aux_data["price_change"].std(ddof=0)],
                "max_price_change": [aux_data["price_change"].max()],
            },
            schema=schema,
        )

        return stats

    def prep_imbalance_data(self, imbalance_data_symbol: pl.DataFrame):

        # -------- significant imbalance -------- #

        mapping = {"A": 15, "B": 25, "C": 35, "~": -1, "L": 0}
        imbalance_data_symbol = imbalance_data_symbol.with_columns(
            pl.col("significant_imbalance")
            .replace(mapping)
            .cast(pl.Int32)
            .alias("significant_imbalance")
        )

        imbalance_data_symbol = imbalance_data_symbol.with_columns(
            pl.col("unpaired_side")
            .replace({"N": 0})
            .cast(pl.Int32)
            .alias("unpaired_side")
        )

        mapping2 = {"A": 1, "B": -1, "N": 0}
        imbalance_data_symbol = imbalance_data_symbol.with_columns(
            pl.col("side").replace(mapping2).cast(pl.Int32).alias("side")
        )

        mapping3 = {"O": 1, "C": 2, "H": 3, "A": 4}
        imbalance_data_symbol = imbalance_data_symbol.with_columns(
            pl.col("auction_type")
            .replace(mapping3)
            .cast(pl.Int32)
            .alias("auction_type")
        )

        zero_columns = [
            "auction_time",
            "ssr_filling_price",
            "upper_collar",
            "lower_collar",
            "market_imbalance_qty",
            "unpaired_qty",
            "auction_status",
            "freeze_status",
            "num_extensions",
            "unpaired_side",
        ]

        imbalance_data_symbol = imbalance_data_symbol.drop(zero_columns)

        return imbalance_data_symbol

    def process_symbols(self):

        for symbol in self.symbols:
            print(f"Processing symbol {symbol}")
            ohlcv_data = pl.read_parquet(f"data/processed/{symbol}/ohlcv_data.parquet")

            keys = ["ts_event", "rtype", "instrument_id", "symbol", "publisher_id"]
            suffix_names = {
                col: f"{col}_ohlcv" for col in ohlcv_data.columns if col not in keys
            }

            ohlcv_data = ohlcv_data.rename(suffix_names)

            bbo_data = pl.read_parquet(f"data/processed/{symbol}/bbo_data.parquet")
            imbalance_data = pl.read_parquet(
                f"data/processed/{symbol}/imbalance_data.parquet"
            )
            imbalance_data = imbalance_data.rename(
                {
                    col: f"{col}_imbalance"
                    for col in imbalance_data.columns
                    if col not in keys
                }
            )
            bbo_data = bbo_data.rename(
                {col: f"{col}_bbo" for col in bbo_data.columns if col not in keys}
            )

            ohlcv_data = ohlcv_data.sort("ts_event")

            bbo_data = bbo_data.sort("ts_event").with_columns(
                pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC"))
            )
            imbalance_data = imbalance_data.sort("ts_event").with_columns(
                pl.col("ts_event").cast(pl.Datetime("ns", time_zone="UTC"))
            )

            ohlcv_data = ohlcv_data.join(
                bbo_data,
                on=keys,
                how="left",
                suffix="_bbo",
            )

            ohlcv_data = ohlcv_data.join(
                imbalance_data,
                on=keys,
                how="left",
                suffix="_imbalance",
            )

            keys = {col: f"{col}_general" for col in keys}
            ohlcv_data = ohlcv_data.rename(keys)

            ohlcv_data.write_parquet(f"data/processed/{symbol}/merged_data.parquet")

    def load_log(self, log_path):
        with open(log_path, "r") as f:
            log = json.load(f)

        return log

    def make_symbol_folders(self):
        for symbol in self.symbols:
            os.makedirs(f"data/processed/{symbol}", exist_ok=True)

    @staticmethod
    def compute_ema(series: np.ndarray, span: int) -> np.ndarray:
        """
        Computes the EMA of a numpy array given a span.
        """
        alpha = 2 / (span + 1)
        ema = np.empty_like(series, dtype=np.float64)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def compute_rsi(series: np.ndarray, period: int = 14):
        """
        Computes a causal RSI of a numpy array given a period.
        """
        rsi = np.zeros_like(series)

        # Initialize gains and losses
        gains = np.zeros_like(series)
        losses = np.zeros_like(series)

        # Compute initial delta
        for i in range(1, len(series)):
            delta = series[i] - series[i - 1]
            gains[i] = max(delta, 0)
            losses[i] = max(-delta, 0)

        # Compute initial average gain and loss
        avg_gain = np.mean(gains[1 : period + 1])
        avg_loss = np.mean(losses[1 : period + 1])

        # Compute RSI for each point
        for i in range(period, len(series)):
            # Update averages using exponential moving average concept
            avg_gain = ((period - 1) * avg_gain + gains[i]) / period
            avg_loss = ((period - 1) * avg_loss + losses[i]) / period

            # Avoid division by zero
            if avg_loss == 0:
                rsi[i] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi
