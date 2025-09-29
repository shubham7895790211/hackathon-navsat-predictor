# ==============================================================================
# REFINED, COMPLETE SCRIPT FOR SATELLITE ERROR PREDICTION (v2)
# ==============================================================================
# This script covers the entire pipeline:
# 1. Data Simulation
# 2. Data Preprocessing
# 3. Model Training
# 4. Model Evaluation
# 5. Asset Saving for Deployment
# ==============================================================================

# --- SECTION 0: IMPORTS ---
import os
import pandas as pd
import numpy as np
import datetime
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress TensorFlow logging for cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ==============================================================================
# --- SECTION 1: DATA SIMULATION ---
# ==============================================================================

def generate_data(filename="simulated_satellite_data.csv"):
    if os.path.exists(filename):
        print(f"'{filename}' already exists. Skipping data generation.")
        return

    print("Starting data simulation...")
    
    start_time = datetime.datetime.now() - datetime.timedelta(days=2) # Using a smaller time delta to match your data size
    end_time = datetime.datetime.now()
    data_frequency_minutes = 15
    upload_interval_hours = 24
    num_satellites = 3
    clock_drift_rate = 0.002
    clock_linear_drift = 0.1
    clock_noise_std_dev = 0.5
    
    all_sat_data = []
    for sat_id in range(1, num_satellites + 1):
        print(f"Simulating data for Satellite NAV-0{sat_id}...")
        timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{data_frequency_minutes}min')
        df = pd.DataFrame(index=timestamps)
        df['satellite_id'] = f'NAV-0{sat_id}'
        
        time_since_upload_min = 0
        clock_errors = []
        for _ in df.index:
            if time_since_upload_min >= upload_interval_hours * 60:
                time_since_upload_min = 0
            
            t = time_since_upload_min
            base_clock_error = (clock_drift_rate * (t/60)**2) + (clock_linear_drift * (t/60))
            noise = np.random.normal(0, clock_noise_std_dev)
            clock_errors.append(base_clock_error + noise)
            time_since_upload_min += data_frequency_minutes
            
        df['clock_error_ns'] = clock_errors
        all_sat_data.append(df)
        
    final_df = pd.concat(all_sat_data)
    print("\nSimulation complete!")
    print("Here's a sample of your generated data:")
    print(final_df.head())
    
    final_df.to_csv(filename)
    print(f"\nSaving simulated data to '{filename}'")

# ==============================================================================
# --- SECTION 2: HELPER FUNCTION FOR SEQUENCING ---
# ==============================================================================

def create_sequences(data, look_back=60):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(X), np.array(y)

# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration Parameters ---
    SATELLITE_ID_TO_MODEL = 'NAV-01'
    FEATURE_TO_PREDICT = 'clock_error_ns'
    LOOK_BACK = 24  # REFINED: Changed from 60 to 24 to fix the IndexError
    EPOCHS = 25
    BATCH_SIZE = 32

    # --- Step 1: Generate Data if Needed ---
    generate_data()

    # --- Step 2: Load and Preprocess Data ---
    print("\n--- Starting Data Preprocessing ---")
    df = pd.read_csv("simulated_satellite_data.csv")
    
    data = df[df['satellite_id'] == SATELLITE_ID_TO_MODEL][[FEATURE_TO_PREDICT]]
    
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[0:train_size], data.iloc[train_size:len(data)]
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    
    # --- Step 3: Create Sequences ---
    X_train, y_train = create_sequences(scaled_train_data, LOOK_BACK)
    X_test, y_test = create_sequences(scaled_test_data, LOOK_BACK)
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    print(f"Shape of X_train: {X_train.shape}")
    
    # --- Step 4: Build and Train the LSTM Model ---
    print("\n--- Building the LSTM Model ---")
    model = Sequential([
        LSTM(units=50, return_sequences=False, input_shape=(LOOK_BACK, 1)),
        Dropout(0.2),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()
    
    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test), verbose=1
    )
    
    # --- Step 5: Evaluate the Model -yes--
    print("\n--- Evaluating Model Performance ---")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    y_test
    # --- Step 6: Save the Final Assets ---
print("\n--- Saving Final Assets for Deployment ---")
model.save('lstm_clock_model.h5')
print("Model saved to 'lstm_clock_model.h5'")
joblib.dump(scaler, 'data_scaler.pkl')
print("Scaler saved to 'data_scaler.pkl'")

print("\n--- SCRIPT COMPLETE ---")