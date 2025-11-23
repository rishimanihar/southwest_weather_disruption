import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from xgboost import XGBRegressor

# --- CONFIGURATION ---
st.set_page_config(page_title="Airline Disruption Predictor", layout="wide")

# --- LOAD DATA & MODEL ---
@st.cache_resource
def load_resources():
    # Load the trained model
    model = XGBRegressor()
    model.load_model('data/xgb_airline_model.json')
    
    # Load predictions for the "Insights" tab
    df = pd.read_csv('data/final_model_predictions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return model, df

try:
    model, df = load_resources()
except Exception as e:
    st.error(f"Error loading files. Make sure 'xgb_airline_model.json' and 'final_model_predictions.csv' are in the 'data/' folder. Error: {e}")
    st.stop()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("✈️ Flight Scenarios")
st.sidebar.write("Adjust conditions to predict disruption.")

# User Inputs
selected_airport = st.sidebar.selectbox("Select Airport", df['airport_code'].unique())
selected_month = st.sidebar.slider("Month", 1, 12, 1)
selected_hour = st.sidebar.slider("Hour of Day", 0, 23, 12)

st.sidebar.subheader("Weather Conditions")
temp_c = st.sidebar.slider("Temperature (°C)", -20, 40, 15)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0, 30, 5)
visibility = st.sidebar.slider("Visibility (m)", 0, 16000, 10000)
precip = st.sidebar.number_input("Precipitation (mm)", 0.0, 50.0, 0.0)
is_snow = st.sidebar.checkbox("Snowing?", value=False)
is_thunder = st.sidebar.checkbox("Thunderstorm?", value=False)
is_fog = st.sidebar.checkbox("Foggy?", value=False)

# --- MAIN PAGE ---
st.title("🌩️ Airline Disruption Prediction Dashboard")
st.markdown("Predicting Southwest Airlines flight disruptions (0-100 Score) based on local weather conditions.")

tab1, tab2 = st.tabs(["🔮 Predictor", "📊 Historical Insights"])

# =========================================
# TAB 1: PREDICTION ENGINE
# =========================================
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Current Scenario")
        st.write(f"**Airport:** {selected_airport}")
        st.write(f"**Time:** {selected_hour}:00 (Month: {selected_month})")
        st.write(f"**Wind:** {wind_speed} m/s")
        st.write(f"**Snow:** {'Yes' if is_snow else 'No'}")
        
        predict_btn = st.button("Predict Disruption Score", type="primary")

    with col2:
        if predict_btn:
            # 1. Construct Input Data Frame (Matching Training Columns)
            # Note: For this demo, we assume "rolling" and "lag" features 
            # mirror the current conditions (Simplified logic for UI)
            
            input_data = {
                'temp_c': temp_c,
                'dew_point_c': temp_c - 2, # Approximation
                'wind_speed_ms': wind_speed,
                'visibility_m': visibility,
                'ceiling_m': 30000 if not is_fog else 500,
                'precip_depth_mm': precip,
                'is_fog': int(is_fog),
                'is_rain': int(precip > 0 and not is_snow),
                'is_snow': int(is_snow),
                'is_thunder': int(is_thunder),
                'hour': selected_hour,
                'month': selected_month,
                'day_of_week': 2, # Assume Wednesday (avg day)
                
                # Lags/Rolls (Simplified: Assume consistent weather)
                'wind_speed_ms_lag1': wind_speed,
                'visibility_m_lag1': visibility,
                'precip_depth_mm_lag1': precip,
                'is_thunder_lag1': int(is_thunder),
                'is_snow_lag1': int(is_snow),
                
                'wind_speed_ms_lag3': wind_speed,
                'visibility_m_lag3': visibility,
                'precip_depth_mm_lag3': precip,
                'is_thunder_lag3': int(is_thunder),
                'is_snow_lag3': int(is_snow),
                
                'wind_speed_ms_roll6': wind_speed,
                'precip_depth_mm_roll6': precip,
                'is_snow_roll6': int(is_snow),
                'is_thunder_roll6': int(is_thunder),
                
                'wind_speed_ms_roll12': wind_speed,
                'precip_depth_mm_roll12': precip,
                'is_snow_roll12': int(is_snow),
                'is_thunder_roll12': int(is_thunder)
            }
            
            # 2. One-Hot Encode Airport
            # Get all airport columns from the model's expected features
            # (We need to be careful to match the model's exact feature list)
            model_features = model.get_booster().feature_names
            
            # Create DataFrame initialized with 0s
            df_input = pd.DataFrame([input_data])
            for feature in model_features:
                if feature not in df_input.columns:
                    df_input[feature] = 0
            
            # Set the selected airport to 1
            if f'airport_{selected_airport}' in df_input.columns:
                df_input[f'airport_{selected_airport}'] = 1
                
            # Reorder columns to match model exactly
            df_input = df_input[model_features]

            # 3. Predict
            raw_pred = model.predict(df_input)[0]
            
            # Scale (Approximate based on your training min/max)
            # You might want to save your scaler to a file to be exact, 
            # but for a demo, manual scaling is often okay.
            # Let's assume min=-2, max=15 based on previous outputs.
            min_val, max_val = -2.0, 15.0 
            score_0_100 = ((raw_pred - min_val) / (max_val - min_val)) * 100
            score_0_100 = np.clip(score_0_100, 0, 100)

            # 4. Display Result
            st.metric(label="Predicted Disruption Score (0-100)", value=f"{score_0_100:.1f}")
            
            if score_0_100 < 20:
                st.success("✅ Smooth Operations Expected")
            elif score_0_100 < 60:
                st.warning("⚠️ Moderate Delays Likely")
            else:
                st.error("🚨 SEVERE DISRUPTION PREDICTED")

# =========================================
# TAB 2: ANALYTICS DASHBOARD
# =========================================
with tab2:
    st.header("Historical Data Analysis (2022-2024)")
    
    # Metric 1: Average Disruption by Airport
    st.subheader("Which Airports are the most chaotic?")
    avg_score = df.groupby('airport_code')['Final_Score_0_100'].mean().sort_values().reset_index()
    fig_bar = px.bar(avg_score, x='airport_code', y='Final_Score_0_100', 
                     color='Final_Score_0_100', title="Avg Disruption Score by Airport")
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Metric 2: Disruption over Time
    st.subheader("When do disruptions happen?")
    df['hour'] = df['timestamp'].dt.hour
    hourly_avg = df.groupby('hour')['Final_Score_0_100'].mean().reset_index()
    fig_line = px.line(hourly_avg, x='hour', y='Final_Score_0_100', markers=True,
                       title="Avg Disruption by Hour of Day")
    st.plotly_chart(fig_line, use_container_width=True)
    
    # Metric 3: Raw Data Viewer
    st.subheader("View Raw Predictions")
    st.dataframe(df[['timestamp', 'airport_code', 'Final_Score_0_100', 'Raw_Prediction']].head(100))
