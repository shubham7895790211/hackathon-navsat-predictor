def generate_data():
    """
    Generates a simulated satellite data CSV file with columns: 'satellite_id', 'timestamp', 'clock_error_ns'.
    """
    import pandas as pd
    import numpy as np
    import datetime
    # Simulate 7 days of data at 15-minute intervals
    periods = 7 * 24 * 4
    timestamps = pd.date_range(start='2025-09-01', periods=periods, freq='15T')
    satellite_ids = ['NAV-01', 'NAV-02', 'NAV-03']
    data = []
    for sat in satellite_ids:
        # Simulate clock error as a noisy sine wave
        base = np.sin(np.linspace(0, 20 * np.pi, periods)) * 100
        noise = np.random.normal(0, 10, periods)
        clock_error = base + noise
        for t, ts in enumerate(timestamps):
            data.append({
                'satellite_id': sat,
                'timestamp': ts,
                'clock_error_ns': clock_error[t]
            })
    df = pd.DataFrame(data)
    df.to_csv('simulated_satellite_data.csv', index=False)
    print("Simulated satellite data saved to 'simulated_satellite_data.csv'")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
import matplotlib.pyplot as plt
import joblib


# ==============================================================================
# --- SECTION 2: HELPER FUNCTION FOR SEQUENCING ---
# ==============================================================================

def create_sequences(data, look_back=60):
    """
    Converts an array of values into a dataset of sequences for LSTM.
    """
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
    LOOK_BACK = 60  # Use the last 60 data points (15 hours of data if freq=15min)
    EPOCHS = 25
    BATCH_SIZE = 32

    # --- Step 1: Generate Data if Needed ---
    generate_data()

    # --- Step 2: Load and Preprocess Data ---
    print("\n--- Starting Data Preprocessing ---")
    df = pd.read_csv("simulated_satellite_data.csv")
    
    data = df[df['satellite_id'] == SATELLITE_ID_TO_MODEL][[FEATURE_TO_PREDICT]]
    
    # Chronological Train-Test Split
    train_size = int(len(data) * 0.8)
    train_data, test_data = data.iloc[0:train_size], data.iloc[train_size:len(data)]
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
    
    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)
    
    # --- Step 3: Create Sequences ---
    X_train, y_train = create_sequences(scaled_train_data, LOOK_BACK)
    X_test, y_test = create_sequences(scaled_test_data, LOOK_BACK)
    
    # Reshape for LSTM [samples, timesteps, features]
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
        X_train, 
        y_train, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # --- Step 5: Evaluate the Model ---
    print("\n--- Evaluating Model Performance ---")
    
    # Make predictions on the test set
    predictions_scaled = model.predict(X_test)
    
    # Inverse transform the predictions and actual values to their original scale
    predictions = scaler.inverse_transform(predictions_scaled)
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(y_test_original, predictions)
    mae = mean_absolute_error(y_test_original, predictions)
    rmse = np.sqrt(mse)
    
    print(f"Test Set Mean Squared Error (MSE): {mse:.4f}")
    print(f"Test Set Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Test Set Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    # Plotting the results
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Training & Validation Loss
    plt.figure(figsize=(14, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Saved training loss plot to 'training_loss.png'")
    
    # Plot 2: Actual vs. Predicted Values
    plt.figure(figsize=(14, 6))
    plt.plot(y_test_original, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')
    plt.title('Actual vs. Predicted Clock Error on Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Clock Error (ns)')
    plt.legend()
    plt.savefig('prediction_vs_actual.png')
    print("Saved prediction plot to 'prediction_vs_actual.png'")
    #plt.show() # Uncomment to display plots directly if running locally

    # --- Step 6: Save the Final Assets ---
    print("\n--- Saving Final Assets for Deployment ---")
    
    # Save the trained model
    model.save('lstm_clock_model.h5')
    print("Model saved to 'lstm_clock_model.h5'")
    
    # Save the scaler
    joblib.dump(scaler, 'data_scaler.pkl')
    print("Scaler saved to 'data_scaler.pkl'")
    
    print("\n--- SCRIPT COMPLETE ---")