import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Satellite Error Prediction",
    page_icon="🛰️",
    layout="wide"
)

# --- Load Assets ---
# Use Streamlit's caching to load the model and scaler only once
@st.cache_resource
def load_assets():
    model = load_model('lstm_clock_model.h5')
    scaler = joblib.load('data_scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('simulated_satellite_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Load the trained model, scaler, and data
model, scaler = load_assets()
df = load_data()

# --- Helper Function for Prediction ---
def predict_future(satellite_data, prediction_hours):
    # This is a simplified prediction logic for the prototype
    # It takes the last sequence of data and predicts future steps iteratively
    
    # Constants from your training script
    LOOK_BACK = 60 # IMPORTANT: This must be the same sequence length used for training
    
    # Get the last 'LOOK_BACK' data points for the selected satellite
    last_sequence_full = satellite_data['clock_error_ns'].values[-LOOK_BACK:]
    
    # Scale this sequence
    last_sequence_scaled = scaler.transform(last_sequence_full.reshape(-1, 1))
    
    future_predictions_scaled = []
    current_sequence = list(last_sequence_scaled.flatten())

    for _ in range(prediction_hours * 4): # Assuming 15-min intervals
        # Reshape the current sequence for the model
        input_sequence = np.array(current_sequence[-LOOK_BACK:]).reshape(1, LOOK_BACK, 1)
        
        # Predict the next step
        next_step_pred = model.predict(input_sequence)[0][0]
        
        # Append prediction and update the sequence for the next prediction
        future_predictions_scaled.append(next_step_pred)
        current_sequence.append(next_step_pred)
        
    # Inverse transform the predictions to get the real error values
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    
    return future_predictions.flatten()


# --- UI Sidebar ---
st.sidebar.title("🛰️ Prediction Controls")
st.sidebar.markdown("Configure the parameters for the error prediction.")

# Let user select a satellite
satellite_list = df['satellite_id'].unique()
selected_satellite = st.sidebar.selectbox("Select a Satellite:", satellite_list)

# Let user select how many hours to predict
prediction_hours = st.sidebar.slider("Hours to Predict into the Future:", 1, 24, 8)


# --- Main Page ---
st.title("AI-Powered Satellite Error Prediction Prototype")
st.markdown(f"Displaying historical data and future predictions for **{selected_satellite}**.")

# Filter data for the selected satellite
satellite_data = df[df['satellite_id'] == selected_satellite].set_index('timestamp')

# Get recent historical data to display
history_to_show = satellite_data.last('48H') # Show last 48 hours

# --- Run Prediction ---
with st.spinner(f"Running prediction for the next {prediction_hours} hours..."):
    future_predictions = predict_future(history_to_show, prediction_hours)

# Create future timestamps for the prediction plot
last_timestamp = history_to_show.index[-1]
future_timestamps = pd.date_range(start=last_timestamp, periods=len(future_predictions) + 1, freq='15T')[1:]


# --- Display Results ---
st.subheader("Prediction Results")

# Display a key metric
final_predicted_error = future_predictions[-1]
st.metric(
    label=f"Predicted Clock Error in {prediction_hours} Hours",
    value=f"{final_predicted_error:.2f} ns",
    delta=f"{(final_predicted_error - history_to_show['clock_error_ns'][-1]):.2f} ns from now"
)


# --- Plotting ---
fig = go.Figure()

# Plot historical data
fig.add_trace(go.Scatter(
    x=history_to_show.index,
    y=history_to_show['clock_error_ns'],
    mode='lines',
    name='Historical Error',
    line=dict(color='blue')
))

# Plot predicted data
fig.add_trace(go.Scatter(
    x=future_timestamps,
    y=future_predictions,
    mode='lines',
    name='Predicted Error',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title=f"Clock Error: History vs. Prediction for {selected_satellite}",
    xaxis_title="Timestamp",
    yaxis_title="Clock Error (nanoseconds)",
    legend_title="Legend",
    height=500
)

st.plotly_chart(fig, use_container_width=True)