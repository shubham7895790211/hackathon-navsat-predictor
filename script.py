import pandas as pd
import numpy as np
import datetime

# --- Simulation Parameters (You can tweak these!) ---
start_time = datetime.datetime(2025, 9, 26, 0, 0, 0) # A few days before today
end_time = datetime.datetime.now() # Use current time, which is Sunday, September 28, 2025 at 10:16 AM IST
data_frequency_minutes = 15
upload_interval_hours = 24
num_satellites = 3

# Error characteristics for Clock (in nanoseconds, ns)
clock_drift_rate = 0.002  # Quadratic term (error accelerates)
clock_linear_drift = 0.1   # Linear term
clock_noise_std_dev = 0.5  # Random noise

# Error characteristics for Ephemeris (in meters, m)
eph_drift_rate = 0.0005
eph_linear_drift = 0.02
eph_noise_std_dev = 0.2

# --- Data Generation Logic ---
print("Starting data simulation...")

all_sat_data = []

for sat_id in range(1, num_satellites + 1):
    print(f"Simulating data for Satellite NAV-0{sat_id}...")
    
    # Create a timestamp index for this satellite
    timestamps = pd.date_range(start=start_time, end=end_time, freq=f'{data_frequency_minutes}T')
    df = pd.DataFrame(index=timestamps)
    df['satellite_id'] = f'NAV-0{sat_id}'
    
    time_since_upload_min = 0
    clock_errors = []
    eph_errors = []

    for ts in df.index:
        # Check if it's time for an upload/reset
        if time_since_upload_min >= upload_interval_hours * 60:
            time_since_upload_min = 0

        # Calculate error build-up based on time since last upload
        # Formula: error = a*t^2 + b*t + noise
        t = time_since_upload_min
        
        # Clock error
        base_clock_error = (clock_drift_rate * (t/60)**2) + (clock_linear_drift * (t/60))
        noise = np.random.normal(0, clock_noise_std_dev)
        clock_errors.append(base_clock_error + noise)
        
        # Ephemeris error (radial component)
        base_eph_error = (eph_drift_rate * (t/60)**2) + (eph_linear_drift * (t/60))
        noise = np.random.normal(0, eph_noise_std_dev)
        eph_errors.append(base_eph_error + noise)

        time_since_upload_min += data_frequency_minutes

    df['clock_error_ns'] = clock_errors
    df['eph_error_radial_m'] = eph_errors
    all_sat_data.append(df)

# Combine data for all satellites into one DataFrame
final_df = pd.concat(all_sat_data)

print("\nSimulation complete!")
print("Here's a sample of your generated data:")
print(final_df.head())
print("\nSaving simulated data to 'simulated_satellite_data.csv'")
final_df.to_csv('simulated_satellite_data.csv')
df['hour_of_day'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['clock_error_lag_1'] = df['clock_error_ns'].shift(1)
df['rolling_mean_1hr'] = df['clock_error_ns'].rolling(window=12).mean() # Assuming 5-min data
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[0:train_size], df.iloc[train_size:len(df)]

# Use the combined DataFrame for further analysis and plotting
import matplotlib.pyplot as plt
df = final_df.copy()
df.index = pd.to_datetime(df.index)
plt.figure(figsize=(15, 6))
plt.plot(df['clock_error_ns'])
plt.title('Clock Error Build-up Over Time')
plt.show()

# Feature engineering
df['hour_of_day'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['clock_error_lag_1'] = df['clock_error_ns'].shift(1)
df['rolling_mean_1hr'] = df['clock_error_ns'].rolling(window=4).mean() # 15-min data, 1hr=4 samples
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[0:train_size], df.iloc[train_size:len(df)]