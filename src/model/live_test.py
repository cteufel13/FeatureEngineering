import yfinance as yf
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import wandb


class YahooFinanceTrader:
    def __init__(self, model, symbol="SPY", threshold_upper=0.5, threshold_lower=-0.5):
        """
        Initialize the trader with Yahoo Finance data and XGBoost classifier model.
        Assumes wandb.init() has already been called in the parent script.

        Parameters:
        -----------
        model : object
            Pre-trained XGBoost classifier that takes 100 OHLC data points
            and predicts price movement
        symbol : str
            Trading symbol (default: 'SPY')
        threshold_upper : float
            Upper threshold for price movement prediction (default: 0.5)
        threshold_lower : float
            Lower threshold for price movement prediction (default: -0.5)
        """
        self.model = model
        self.symbol = symbol
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.data_buffer = []
        self.latest_prediction = None
        self.prediction_history = []
        self.prediction_accuracy = []
        self.ticker = yf.Ticker(symbol)

        # Verify wandb is initialized
        # if wandb.run is None:
        #     print("No active wandb run detected. Make sure wandb.init() was called.")
        # else:
        #     print(f"Using existing wandb run: {wandb.run.name}")

        #     # Create wandb Table for detailed prediction tracking
        #     self.wandb_table = wandb.Table(
        #         columns=[
        #             "timestamp",
        #             "prediction",
        #             "actual_direction",
        #             "close_price",
        #             "next_close",
        #             "price_change_pct",
        #         ]
        #     )

    def fetch_initial_data(self):
        """
        Fetch the initial 100 minutes of data to start predictions.
        """
        try:
            # Get data for the last 100 minutes
            # Yahoo Finance interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            # For 1m data, can only fetch 7 days of data
            end_time = datetime.now()
            # Get 7 days back to ensure we have enough minute data
            start_time = end_time - timedelta(days=1)

            # Get 1-minute bars
            historical_data = self.ticker.history(
                start=start_time.strftime("%Y-%m-%d"),
                end=end_time.strftime("%Y-%m-%d"),
                interval="1m",
            )

            # Rename columns to match our expected format
            df = historical_data.reset_index().rename(
                columns={
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            if len(df) < 100:
                print(
                    f"Only retrieved {len(df)} bars. Need at least 100 for prediction."
                )
                # wandb.log({"data_error": f"Only retrieved {len(df)}/100 bars"})
                return False

            # Store the last 100 data points
            self.data_buffer = df.tail(100).copy()
            print(
                f"Successfully fetched initial {len(self.data_buffer)} bars for {self.symbol}"
            )

            # Log data stats to wandb
            # wandb.log(
            #     {
            #         "data_points": len(self.data_buffer),
            #         "price_range": df["close"].max() - df["close"].min(),
            #         "volume_avg": df["volume"].mean(),
            #         "start_time": str(df["timestamp"].min()),
            #         "end_time": str(df["timestamp"].max()),
            #     }
            # )

            return True

        except Exception as e:
            print(f"Error fetching initial data: {str(e)}")
            # wandb.log({"error": f"Initial data fetch: {str(e)}"})
            return False

    def prepare_features(self):
        """
        Prepare the feature matrix from the current data buffer.

        Returns:
        --------
        numpy.ndarray
            Feature matrix for model prediction
        """
        if len(self.data_buffer) < 100:
            print(f"Insufficient data points: {len(self.data_buffer)}/100")
            wandb.log(
                {"data_error": f"Insufficient data points: {len(self.data_buffer)}/100"}
            )
            return None

        # Extract OHLC data and convert to numpy matrix
        features = self.data_buffer[["open", "high", "low", "close"]].values

        # Normalize the data (if needed) - adapt this to match your model's training
        # For example, you might want to use percentage changes or normalization
        # This is just a placeholder; adjust based on your actual preprocessing

        return features.reshape(1, -1)  # Reshape for a single sample with all features

    def make_prediction(self):
        """
        Make a prediction using the loaded model.

        Returns:
        --------
        int
            Prediction: 0 (below lower threshold), 1 (between thresholds), 2 (above upper threshold)
        """
        features = self.prepare_features()
        if features is None:
            return None

        try:
            # Make prediction
            prediction = self.model.predict(features)[0]

            # Get prediction probabilities if available
            try:
                probabilities = self.model.predict_proba(features)[0]
                # Log prediction probabilities to wandb
                class_probs = {
                    f"prob_class_{i}": prob for i, prob in enumerate(probabilities)
                }
                wandb.log(class_probs)
            except:
                probabilities = None

            self.latest_prediction = prediction
            current_time = datetime.now()
            current_price = self.data_buffer["close"].iloc[-1]

            # Log prediction
            prediction_map = {
                0: f"Below {self.threshold_lower}",
                1: f"Between {self.threshold_lower} and {self.threshold_upper}",
                2: f"Above {self.threshold_upper}",
            }

            print(
                f"Prediction: {prediction_map.get(prediction, 'Unknown')} at price {current_price}"
            )

            # Store prediction with timestamp
            prediction_data = {
                "timestamp": current_time,
                "prediction": prediction,
                "current_price": current_price,
                "probabilities": probabilities,
            }
            self.prediction_history.append(prediction_data)

            # Log to wandb
            wandb.log(
                {
                    "prediction": int(prediction),
                    "current_price": current_price,
                    "timestamp": current_time.isoformat(),
                }
            )

            # Create a wandb artifact for visualization
            wandb.log(
                {
                    "prediction_chart": wandb.plot.bar(
                        wandb.Table(
                            data=[[p] for p in [int(prediction)]],
                            columns=["Prediction"],
                        ),
                        "Prediction",
                        "Count",
                    )
                }
            )

            return prediction

        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            wandb.log({"error": f"Prediction error: {str(e)}"})
            return None

    def update_data(self):
        """
        Fetch the latest 1-minute bar and update the data buffer.

        Returns:
        --------
        bool
            True if data was successfully updated, False otherwise
        """
        try:
            # Get the latest data point
            end_time = datetime.now()
            # Get 1-minute bars - need to go back enough to capture at least one new bar
            start_time = end_time - timedelta(minutes=10)

            # Get latest data
            latest_data = self.ticker.history(
                start=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_time.strftime("%Y-%m-%d %H:%M:%S"),
                interval="1m",
            )

            if latest_data.empty:
                print("No new data available")
                return False

            # Convert to DataFrame and format
            latest_df = latest_data.reset_index().rename(
                columns={
                    "Datetime": "timestamp",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                }
            )

            # Check if we have new data
            if len(self.data_buffer) > 0:
                last_timestamp = self.data_buffer["timestamp"].iloc[-1]
                latest_timestamp = latest_df["timestamp"].iloc[-1]

                if last_timestamp >= latest_timestamp:
                    print("No new bar available yet")
                    return False

            # Get the latest bar
            new_data = latest_df.iloc[-1].to_dict()

            # Add new data to buffer and remove oldest
            new_df = pd.DataFrame([new_data])

            # Before updating, check if we can evaluate the previous prediction
            if len(self.prediction_history) > 0:
                self.evaluate_previous_prediction(new_data)

            self.data_buffer = pd.concat([self.data_buffer, new_df]).tail(100)

            # Log data update to wandb
            wandb.log(
                {
                    "latest_close": new_data["close"],
                    "latest_volume": new_data["volume"],
                    "latest_high": new_data["high"],
                    "latest_low": new_data["low"],
                    "update_time": datetime.now().isoformat(),
                }
            )

            print(f"Updated data buffer with new bar at {new_data['timestamp']}")
            return True

        except Exception as e:
            print(f"Error updating data: {str(e)}")
            wandb.log({"error": f"Data update error: {str(e)}"})
            return False

    def evaluate_previous_prediction(self, new_data):
        """
        Evaluate the previous prediction against the new data.

        Parameters:
        -----------
        new_data : dict
            Dictionary containing the new bar data
        """
        # Get the most recent prediction
        last_pred = self.prediction_history[-1]
        prev_price = last_pred["current_price"]
        new_price = new_data["close"]

        # Calculate percent change
        pct_change = (new_price - prev_price) / prev_price * 100

        # Determine actual direction
        if pct_change > self.threshold_upper:
            actual_direction = 2  # Above upper threshold
        elif pct_change < self.threshold_lower:
            actual_direction = 0  # Below lower threshold
        else:
            actual_direction = 1  # Between thresholds

        # Check if prediction was correct
        is_correct = last_pred["prediction"] == actual_direction

        # Add to accuracy tracking
        self.prediction_accuracy.append(
            {
                "timestamp": last_pred["timestamp"],
                "prediction": last_pred["prediction"],
                "actual": actual_direction,
                "correct": is_correct,
                "prev_price": prev_price,
                "new_price": new_price,
                "pct_change": pct_change,
            }
        )

        # Update the wandb table
        self.wandb_table.add_data(
            str(last_pred["timestamp"]),
            int(last_pred["prediction"]),
            int(actual_direction),
            float(prev_price),
            float(new_price),
            float(pct_change),
        )

        # Calculate running accuracy
        correct_count = sum(1 for item in self.prediction_accuracy if item["correct"])
        accuracy = (
            correct_count / len(self.prediction_accuracy)
            if self.prediction_accuracy
            else 0
        )

        # Log evaluation metrics to wandb
        wandb.log(
            {
                "prediction_correct": is_correct,
                "running_accuracy": accuracy,
                "price_change_pct": pct_change,
                "actual_direction": actual_direction,
            }
        )

        print(
            f"Prediction evaluation: {is_correct} (predicted {last_pred['prediction']}, actual {actual_direction})"
        )
        print(f"Running accuracy: {accuracy:.2%}")

        # After 10 predictions, log the full table
        if len(self.prediction_accuracy) % 10 == 0:
            wandb.log({"predictions_table": self.wandb_table})

    def run(self, runtime_minutes=None):
        """
        Run the trading system for the specified number of minutes or indefinitely.

        Parameters:
        -----------
        runtime_minutes : int or None
            Number of minutes to run the system, or None to run indefinitely
        """
        # Fetch initial data
        if not self.fetch_initial_data():
            print("Failed to fetch initial data. Exiting.")
            return

        # Make initial prediction
        self.make_prediction()

        start_time = datetime.now()
        iteration = 0

        try:
            while True:
                iteration += 1
                # Check if runtime exceeded
                if runtime_minutes is not None:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= runtime_minutes:
                        print(
                            f"Runtime of {runtime_minutes} minutes reached. Stopping."
                        )
                        break

                print(f"Iteration {iteration}: Waiting for next minute...")
                wandb.log({"iteration": iteration})

                time.sleep(60)  # Wait for 1 minute

                # Update data and make prediction
                if self.update_data():
                    self.make_prediction()

        except KeyboardInterrupt:
            print("Keyboard interrupt received. Stopping.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            wandb.log({"fatal_error": str(e)})
        finally:
            print("Trading system stopped")

            # Final wandb logs
            final_metrics = {
                "total_iterations": iteration,
                "total_predictions": len(self.prediction_history),
                "final_accuracy": (
                    sum(1 for item in self.prediction_accuracy if item["correct"])
                    / len(self.prediction_accuracy)
                    if self.prediction_accuracy
                    else 0
                ),
            }
            wandb.log(final_metrics)

            # Log final prediction table
            wandb.log({"final_predictions_table": self.wandb_table})

    def get_predictions_summary(self):
        """
        Return a summary of prediction history.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing prediction history
        """
        if not self.prediction_history:
            return pd.DataFrame()

        return pd.DataFrame(self.prediction_history)

    def get_accuracy_summary(self):
        """
        Return a summary of prediction accuracy.

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing prediction accuracy
        """
        if not self.prediction_accuracy:
            return pd.DataFrame()

        return pd.DataFrame(self.prediction_accuracy)
