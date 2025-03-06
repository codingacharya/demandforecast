import streamlit as st
import pandas as pd
import plotly.express as px
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

st.title("📦 Supply Chain Demand Forecasting")

# Upload CSV file
st.sidebar.header("Upload Demand Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview")
    st.write(df.head())

    # Ensure correct column names
    if 'date' not in df.columns or 'demand' not in df.columns:
        st.error("CSV must contain 'date' and 'demand' columns")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Plot raw data
        fig = px.line(df, x='date', y='demand', title='Historical Demand')
        st.plotly_chart(fig)
        
        # Prepare data for Prophet
        df_prophet = df.rename(columns={'date': 'ds', 'demand': 'y'})
        
        # Train Prophet Model
        model = Prophet()
        model.fit(df_prophet)
        
        # Future Forecast
        periods = st.sidebar.slider("Forecast Period (Days)", 7, 365, 30)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        # Plot Forecast
        st.write("### Forecasted Demand")
        fig_forecast = px.line(forecast, x='ds', y='yhat', title='Demand Forecast')
        st.plotly_chart(fig_forecast)
        
        # Evaluate Model
        y_true = df_prophet['y'][-30:]  # Last 30 actual values
        y_pred = forecast['yhat'][-30:]  # Last 30 predicted values
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        st.write("### Model Accuracy")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
