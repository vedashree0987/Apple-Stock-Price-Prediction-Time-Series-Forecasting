
import warnings
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

FORECAST_HORIZON = 30
LOOKBACK = 20
RANDOM_STATE = 42

DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "P721 DATASET.csv"
)

FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "Daily_Return", "Intraday_Range", "Intraday_Range_Pct",
    "Year", "Month", "DayOfWeek", "Quarter",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3",
    "Close_Lag_5", "Close_Lag_10", "Close_Lag_20",
    "Return_Lag_1", "Return_Lag_2",
    "Return_Lag_5", "Return_Lag_10",
    "Close_RollMean_5", "Close_RollMean_10",
    "Close_RollMean_20", "Close_RollMean_50",
    "Return_RollStd_5", "Return_RollStd_10",
    "Return_RollStd_20", "Return_RollStd_50"
]

TARGET_COLUMNS = [
    f"Target_Close_t_plus_{i}"
    for i in range(1, FORECAST_HORIZON + 1)
]


def load_stock_data():

    data = pd.read_csv(DATA_PATH)

    data.columns = data.columns.str.strip()

    data["Date"] = pd.to_datetime(
        data["Date"], errors="coerce"
    )

    data = data.dropna(subset=["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    numeric_columns = [
        "Open", "High", "Low", "Close", "Volume"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError("The stock dataset is empty.")

    return data


def create_features(data):

    feature_data = data.copy()

    feature_data["Daily_Return"] = (
        feature_data["Close"].pct_change()
    )

    feature_data["Intraday_Range"] = (
        feature_data["High"] - feature_data["Low"]
    )

    feature_data["Intraday_Range_Pct"] = (
        feature_data["Intraday_Range"]
        / feature_data["Open"]
    )

    for lag in [1, 2, 3, 5, 10, 20]:

        feature_data[f"Close_Lag_{lag}"] = (
            feature_data["Close"].shift(lag)
        )

    for lag in [1, 2, 5, 10]:

        feature_data[f"Return_Lag_{lag}"] = (
            feature_data["Daily_Return"].shift(lag)
        )

    for window in [5, 10, 20, 50]:

        feature_data[f"Close_RollMean_{window}"] = (
            feature_data["Close"]
            .rolling(window=window).mean()
        )

        feature_data[f"Return_RollStd_{window}"] = (
            feature_data["Daily_Return"]
            .rolling(window=window).std()
        )

    feature_data["Year"] = (
        feature_data["Date"].dt.year
    )

    feature_data["Month"] = (
        feature_data["Date"].dt.month
    )

    feature_data["DayOfWeek"] = (
        feature_data["Date"].dt.dayofweek
    )

    feature_data["Quarter"] = (
        feature_data["Date"].dt.quarter
    )

    for horizon in range(1, FORECAST_HORIZON + 1):

        feature_data[
            f"Target_Close_t_plus_{horizon}"
        ] = feature_data["Close"].shift(-horizon)

    return feature_data


def forecast_arima(close_prices):

    orders = [
        (1, 1, 0),
        (1, 1, 1),
        (2, 1, 1),
        (2, 1, 2),
        (3, 1, 1)
    ]

    validation_train = close_prices.iloc[:-30]
    validation_actual = close_prices.iloc[-30:].to_numpy()

    best_order = None
    best_mae = float("inf")

    for order in orders:

        try:

            model = ARIMA(
                validation_train.reset_index(drop=True),
                order=order
            ).fit()

            prediction = np.asarray(
                model.forecast(steps=30)
            )

            mae = np.mean(
                np.abs(validation_actual - prediction)
            )

            if mae < best_mae:

                best_mae = mae
                best_order = order

        except Exception:
            continue

    if best_order is None:
        raise ValueError("ARIMA could not fit any candidate order.")

    final_model = ARIMA(
        close_prices.reset_index(drop=True),
        order=best_order
    ).fit()

    return np.asarray(
        final_model.forecast(steps=30)
    )


def forecast_sarima(close_prices):

    options = [
        ((1, 1, 1), (1, 0, 1, 5)),
        ((1, 1, 1), (1, 1, 1, 5)),
        ((2, 1, 1), (1, 0, 1, 5)),
        ((1, 1, 2), (0, 1, 1, 5))
    ]

    validation_train = close_prices.iloc[:-30]
    validation_actual = close_prices.iloc[-30:].to_numpy()

    best_order = None
    best_seasonal_order = None
    best_mae = float("inf")

    for order, seasonal_order in options:

        try:

            model = SARIMAX(
                validation_train.reset_index(drop=True),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            prediction = np.asarray(
                model.forecast(steps=30)
            )

            mae = np.mean(
                np.abs(validation_actual - prediction)
            )

            if mae < best_mae:

                best_mae = mae
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception:
            continue

    if best_order is None:
        raise ValueError("SARIMA could not fit any candidate configuration.")

    final_model = SARIMAX(
        close_prices.reset_index(drop=True),
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    return np.asarray(
        final_model.forecast(steps=30)
    )


def forecast_random_forest(X_train, y_train, X_forecast):

    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    )

    model.fit(X_train, y_train)

    return model.predict(X_forecast).ravel()


def forecast_xgboost(X_train, y_train, X_forecast):

    from xgboost import XGBRegressor

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbosity=0
        )
    )

    model.fit(X_train, y_train)

    return model.predict(X_forecast).ravel()


def forecast_gru(feature_data, last_train_index, forecast_index):

    import tensorflow as tf

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    valid_feature_rows = feature_data[FEATURE_COLUMNS].notna().all(axis=1)
    valid_target_rows = feature_data[TARGET_COLUMNS].notna().all(axis=1)
    valid_rows = valid_feature_rows & valid_target_rows

    valid_positions = np.flatnonzero(valid_rows.to_numpy())
    valid_positions = valid_positions[valid_positions <= last_train_index]

    if len(valid_positions) < 100:
        raise ValueError("Not enough complete training rows for GRU.")

    first_valid_index = int(valid_positions[0])

    training_features = feature_data.iloc[
        first_valid_index:last_train_index + 1
    ][FEATURE_COLUMNS]

    training_targets = feature_data.iloc[
        first_valid_index:last_train_index + 1
    ][TARGET_COLUMNS]

    if training_features.isna().any().any():
        raise ValueError("Missing values found in GRU training features.")

    if training_targets.isna().any().any():
        raise ValueError("Missing values found in GRU training targets.")

    if len(training_features) < 100:
        raise ValueError("Not enough training data for GRU.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    scaled_features = x_scaler.fit_transform(
        training_features
    )

    scaled_targets = y_scaler.fit_transform(
        training_targets
    )

    X_sequences = []
    y_sequences = []

    for end_index in range(LOOKBACK - 1, len(scaled_features)):

        start_index = end_index - LOOKBACK + 1

        X_sequences.append(
            scaled_features[start_index:end_index + 1]
        )

        y_sequences.append(
            scaled_targets[end_index]
        )

    X_sequences = np.asarray(X_sequences)
    y_sequences = np.asarray(y_sequences)

    model = Sequential([
        Input(shape=(LOOKBACK, len(FEATURE_COLUMNS))),
        GRU(32, activation="tanh"),
        Dropout(0.2),
        Dense(30)
    ])

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_sequences,
        y_sequences,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0
    )

    forecast_features = feature_data.iloc[
        forecast_index - LOOKBACK + 1:forecast_index + 1
    ][FEATURE_COLUMNS]

    if len(forecast_features) != LOOKBACK:
        raise ValueError("Not enough rows for the GRU forecast sequence.")

    if forecast_features.isna().any().any():
        raise ValueError("Missing values in the GRU forecast sequence.")

    forecast_scaled = x_scaler.transform(
        forecast_features
    )

    forecast_scaled = forecast_scaled.reshape(
        1, LOOKBACK, len(FEATURE_COLUMNS)
    )

    prediction_scaled = model.predict(
        forecast_scaled,
        verbose=0
    )

    return y_scaler.inverse_transform(
        prediction_scaled
    ).ravel()


def forecast_next_30(data, forecast_date, model_name):

    data = data.copy()

    data["Date"] = pd.to_datetime(data["Date"])

    data = data.sort_values("Date").reset_index(drop=True)

    requested_date = pd.to_datetime(forecast_date)

    available_dates = data.index[
        data["Date"] <= requested_date
    ]

    if len(available_dates) == 0:
        raise ValueError("No stock data exists on or before this date.")

    forecast_index = int(available_dates[-1])

    actual_origin_date = data.loc[
        forecast_index, "Date"
    ]

    if forecast_index < 100:
        raise ValueError("Select a later date with more historical data.")

    feature_data = create_features(data)

    last_train_index = (
        forecast_index - FORECAST_HORIZON
    )

    # Confirm that all expected feature and target columns exist.
    required_columns = FEATURE_COLUMNS + TARGET_COLUMNS
    missing_columns = [
        column for column in required_columns
        if column not in feature_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Generated feature columns are missing: "
            + ", ".join(map(str, missing_columns))
        )

    # Keep only rows where every required feature and target is present.
    training_window = feature_data.iloc[:last_train_index + 1].copy()
    complete_rows = training_window[required_columns].notna().all(axis=1)
    train_data = training_window.loc[complete_rows].copy()

    if len(train_data) < 100:
        raise ValueError("Not enough complete training rows.")

    X_train = train_data[FEATURE_COLUMNS].astype(float)

    y_train = train_data[TARGET_COLUMNS].astype(float)

    X_forecast = feature_data.iloc[
        [forecast_index]
    ][FEATURE_COLUMNS].astype(float)

    if X_forecast.isna().any().any():
        raise ValueError(
            "The selected forecast date does not have all required features. "
            "Choose a later date."
        )

    close_prices = data.iloc[
        :forecast_index + 1
    ]["Close"].astype(float).reset_index(drop=True)

    if model_name == "ARIMA":

        predictions = forecast_arima(close_prices)

    elif model_name == "SARIMA":

        predictions = forecast_sarima(close_prices)

    elif model_name == "Random Forest":

        predictions = forecast_random_forest(
            X_train, y_train, X_forecast
        )

    elif model_name == "XGBoost":

        predictions = forecast_xgboost(
            X_train, y_train, X_forecast
        )

    elif model_name == "GRU":

        predictions = forecast_gru(
            feature_data,
            last_train_index,
            forecast_index
        )

    else:

        raise ValueError("Please select a supported forecasting model.")

    future_dates = pd.bdate_range(
        start=actual_origin_date + pd.offsets.BDay(1),
        periods=FORECAST_HORIZON
    )

    forecast_results = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Close": predictions
    })

    forecast_results.attrs["forecast_origin"] = actual_origin_date
    forecast_results.attrs["model"] = model_name

    return forecast_results
