
import traceback

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from forecast import load_stock_data, forecast_next_30


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Apple Stock Price Prediction",
    page_icon="🍎",
    layout="wide"
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

@st.cache_data
def get_stock_data():
    return load_stock_data()


st.title("🍎 Apple Stock Price Prediction")

st.write(
    "Explore historical Apple stock prices and forecast "
    "the next 30 trading sessions using machine learning "
    "and time-series models."
)

st.info(
    "This project uses historical data from 2012–2019. "
    "Forecasts are based on the selected historical date "
    "and are not live stock-market predictions."
)


try:
    data = get_stock_data()

except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.code(traceback.format_exc())
    st.stop()


# --------------------------------------------------
# DATE RANGE SETTINGS
# --------------------------------------------------

min_date = data["Date"].min().date()
max_date = data["Date"].max().date()


st.sidebar.header("Forecast Settings")

st.sidebar.write(
    "Choose your historical date range."
)


selected_range = st.sidebar.date_input(
    "Select start and end dates",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

model_name = st.sidebar.selectbox(
    "Select forecasting model",
    [
        "ARIMA",
        "SARIMA",
        "Random Forest",
        "XGBoost",
        "GRU"
    ]
)


# --------------------------------------------------
# VALIDATE DATE RANGE
# --------------------------------------------------

if len(selected_range) != 2:

    st.warning(
        "Please select both a start date and an end date."
    )
    st.stop()


start_date, end_date = selected_range


if start_date > end_date:

    st.error(
        "Start date cannot be after end date."
    )
    st.stop()


# --------------------------------------------------
# FILTER HISTORICAL DATA
# --------------------------------------------------

historical_data = data[
    (data["Date"].dt.date >= start_date)
    &
    (data["Date"].dt.date <= end_date)
].copy()


if historical_data.empty:

    st.warning(
        "No stock records found in the selected date range. "
        "Please choose another range."
    )
    st.stop()


# --------------------------------------------------
# HISTORICAL STOCK INFORMATION
# --------------------------------------------------

st.subheader("Historical Stock Information")

col1, col2, col3 = st.columns(3)


col1.metric(
    "Starting Price",
    f"${historical_data['Close'].iloc[0]:.2f}"
)


col2.metric(
    "Latest Price",
    f"${historical_data['Close'].iloc[-1]:.2f}"
)


col3.metric(
    "Trading Sessions",
    len(historical_data)
)


# --------------------------------------------------
# HISTORICAL CLOSING PRICE CHART
# --------------------------------------------------

st.subheader("Historical Closing Price")

historical_chart = go.Figure()

historical_chart.add_trace(
    go.Scatter(
        x=historical_data["Date"],
        y=historical_data["Close"],
        mode="lines",
        name="Historical Close"
    )
)


historical_chart.update_layout(
    xaxis_title="Date",
    yaxis_title="Closing Price ($)",
    template="plotly_white",
    hovermode="x unified"
)


st.plotly_chart(
    historical_chart,
    width="stretch"
)


# --------------------------------------------------
# FORECAST SECTION
# --------------------------------------------------

st.divider()

st.subheader("Generate 30-Session Forecast")


st.write(
    f"Selected model: **{model_name}**"
)


st.write(
    f"Forecast will begin after the selected end date: "
    f"**{end_date}**"
)


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

if st.button(
    "Predict Next 30 Sessions",
    type="primary"
):

    with st.spinner(
        f"Training {model_name} and generating predictions. "
        "Please wait..."
    ):

        try:
            forecast_data = forecast_next_30(
                data,
                end_date,
                model_name
            )


            # ------------------------------------------
            # FORECAST INFORMATION
            # ------------------------------------------

            actual_origin = forecast_data.attrs.get(
                "forecast_origin"
            )

            st.success(
                f"Forecast generated successfully using {model_name}!"
            )


            st.write(
                f"Forecast starting point: "
                f"**{actual_origin.date()}**"
            )


            # ------------------------------------------
            # FORECAST METRICS
            # ------------------------------------------

            st.subheader("Predicted Stock Prices")

            col1, col2, col3 = st.columns(3)


            first_prediction = float(
                forecast_data["Predicted_Close"].iloc[0]
            )

            last_prediction = float(
                forecast_data["Predicted_Close"].iloc[-1]
            )


            col1.metric(
                "First Predicted Price",
                f"${first_prediction:.2f}"
            )


            col2.metric(
                "30th Predicted Price",
                f"${last_prediction:.2f}"
            )


            # Get actual closing price at forecast origin
            origin_rows = data[
                data["Date"] == actual_origin
            ]

            if origin_rows.empty:
                raise ValueError(
                    "Could not find the actual closing price "
                    "for the forecast starting date."
                )


            last_actual_price = float(
                origin_rows["Close"].iloc[-1]
            )


            percentage_change = (
                (
                    last_prediction - last_actual_price
                ) / last_actual_price
            ) * 100


            col3.metric(
                "Change by Session 30",
                f"{percentage_change:.2f}%"
            )


            # ------------------------------------------
            # HISTORICAL PRICE VS FORECAST CHART
            # ------------------------------------------

            st.subheader(
                "Historical Price vs Forecast"
            )


            chart_data = go.Figure()


            chart_history = data[
                (data["Date"].dt.date >= start_date)
                &
                (data["Date"] <= actual_origin)
            ]


            chart_data.add_trace(
                go.Scatter(
                    x=chart_history["Date"],
                    y=chart_history["Close"],
                    mode="lines",
                    name="Historical Close"
                )
            )


            chart_data.add_trace(
                go.Scatter(
                    x=forecast_data["Date"],
                    y=forecast_data["Predicted_Close"],
                    mode="lines+markers",
                    name="Predicted Close",
                    line=dict(dash="dash")
                )
            )


            chart_data.update_layout(
                xaxis_title="Date",
                yaxis_title="Closing Price ($)",
                template="plotly_white",
                hovermode="x unified"
            )


            st.plotly_chart(
                chart_data,
                width="stretch"
            )


            # ------------------------------------------
            # FORECAST TABLE
            # ------------------------------------------

            st.subheader(
                "30-Session Forecast Table"
            )


            display_data = forecast_data.copy()


            display_data["Date"] = (
                display_data["Date"].dt.strftime(
                    "%Y-%m-%d"
                )
            )


            display_data["Predicted_Close"] = (
                display_data["Predicted_Close"].round(2)
            )


            st.dataframe(
                display_data,
                width="stretch",
                hide_index=True
            )


            # ------------------------------------------
            # DOWNLOAD FORECAST
            # ------------------------------------------

            csv_data = forecast_data.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name="apple_stock_forecast.csv",
                mime="text/csv"
            )


        # ----------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------

        except Exception as error:

            st.error(
                f"Forecast could not be generated: {error}"
            )

            st.subheader("Technical Error Details")

            st.code(
                traceback.format_exc()
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Educational project only. Stock-price forecasts are "
    "uncertain and should not be treated as financial advice."
)
