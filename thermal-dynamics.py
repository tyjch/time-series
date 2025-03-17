# Thermal Response Dynamics for DS18B20 Sensor State Detection
# =========================================================

# This notebook demonstrates how to use thermal response characteristics
# to detect when the DS18B20 temperature sensor is connected or disconnected.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from influxdb_client import InfluxDBClient
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from datetime import datetime, timedelta
import joblib
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Set Matplotlib parameters for better readability
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 1. Introduction to Thermal Response Dynamics
# ==========================================
print("""
# Thermal Response Dynamics Explained
# =================================

Thermal Response Dynamics focuses on how temperature sensors respond to their environment 
over time. The key insight is that a DS18B20 sensor will show very different 
thermal behavior when it's:

1. **Connected to a human body**: 
   - Slower temperature changes (thermal inertia of the body)
   - Less responsive to ambient temperature changes
   - Temperature variations primarily driven by physiological processes

2. **Disconnected (ambient)**: 
   - More rapid temperature changes
   - Directly responsive to environmental fluctuations
   - Closely correlated with room temperature changes

This approach uses time-series analysis techniques to detect these characteristic 
patterns, including:

- **Rate of temperature change** (temperature derivatives)
- **Frequency analysis** to identify dominant fluctuation patterns
- **Smoothness/variability metrics** to detect the damping effect of human body
- **Response to environmental changes** to measure coupling with ambient conditions

Let's implement these techniques to create a robust sensor state detector!
""")

# 2. Connect to InfluxDB and Query Data
# ====================================
# Replace these with your actual InfluxDB credentials
url = "https://your-influxdb-url"
token = "your-generated-token"
org = "your-organization"
bucket = "your-bucket"

print("\nConnecting to InfluxDB...")
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# Define the time range for data retrieval (e.g., last 7 days)
time_range = "7d"

print(f"Retrieving data from the last {time_range}...")

# Query for DS18B20 temperature data
ds18b20_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "DS18B20")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
'''

# Query for Si7021 temperature data (room temperature)
si7021_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "SI7021")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
'''

# Query for RPi CPU temperature
rpi_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "RPi Zero 2W")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
'''

# Execute queries and convert to DataFrame
ds18b20_result = query_api.query_data_frame(ds18b20_query)
si7021_result = query_api.query_data_frame(si7021_query)
rpi_result = query_api.query_data_frame(rpi_query)

# Check if we have data
if ds18b20_result.empty or si7021_result.empty or rpi_result.empty:
    print("One or more queries returned no data. Please check your InfluxDB queries.")
else:
    print(f"Retrieved {len(ds18b20_result)} data points for DS18B20")
    print(f"Retrieved {len(si7021_result)} data points for Si7021")
    print(f"Retrieved {len(rpi_result)} data points for RPi CPU")

# 3. Process and Merge Data
# =========================
# Keep only necessary columns and rename for clarity
ds18b20_df = ds18b20_result[['_time', '_value']].rename(columns={'_value': 'ds18b20_temp'})
si7021_df = si7021_result[['_time', '_value']].rename(columns={'_value': 'room_temp'})
rpi_df = rpi_result[['_time', '_value']].rename(columns={'_value': 'cpu_temp'})

# Make time the index for easier merging
ds18b20_df.set_index('_time', inplace=True)
si7021_df.set_index('_time', inplace=True)
rpi_df.set_index('_time', inplace=True)

# Resample data to ensure consistent timestamps (1-minute intervals)
# For thermal dynamics analysis, we want higher resolution data
ds18b20_df = ds18b20_df.resample('1T').mean()
si7021_df = si7021_df.resample('1T').mean()
rpi_df = rpi_df.resample('1T').mean()

# Merge all data into a single DataFrame
merged_df = pd.concat([ds18b20_df, si7021_df, rpi_df], axis=1)

# Handle missing values (if any)
# Here we use forward fill followed by backward fill for any remaining NaNs
merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

print("\nMerged data sample:")
print(merged_df.head())
print("\nDataset information:")
print(merged_df.info())
print("\nSummary statistics:")
print(merged_df.describe())

# For initial state estimation, we'll use a temperature threshold
body_temp_threshold = 90.0
merged_df['initial_label'] = (merged_df['ds18b20_temp'] > body_temp_threshold).astype(int)

# 4. Exploratory Data Analysis
# ============================
print("\nPerforming exploratory data analysis...")

plt.figure(figsize=(14, 10))

# Plot temperature data
plt.subplot(3, 1, 1)
plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
plt.title('DS18B20 Temperature Over Time')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(merged_df.index, merged_df['room_temp'], 'g-', label='Si7021 (Room)')
plt.title('Room Temperature Over Time')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(merged_df.index, merged_df['cpu_temp'], 'r-', label='RPi CPU')
plt.title('CPU Temperature Over Time')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.tight_layout()
plt.savefig('thermal_dynamics_time_series.png')
plt.show()

# 5. Thermal Response Feature Engineering
# =====================================
print("\nEngineering thermal response features...")

# EXPLANATION: Thermal Response Features
# ====================================
print("""
# Thermal Response Features Explained
# =================================

We're extracting features that characterize how the DS18B20 sensor responds to 
its thermal environment. These features will help distinguish between the 
connected and disconnected states:

1. **Temperature Derivatives**: How rapidly temperature changes
   - First derivative (dT/dt): Rate of change
   - Second derivative (d²T/dt²): Acceleration of change

2. **Variability Metrics**: How much temperature fluctuates
   - Rolling standard deviation: Short-term variability
   - Rolling IQR (interquartile range): Robust measure of variability
   
3. **Frequency Domain Features**: Periodic patterns in temperature
   - Spectral energy in different frequency bands
   - Wavelet coefficients capturing time-localized frequency information
   
4. **Coupling with Environment**: How DS18B20 relates to ambient conditions
   - Correlation with room temperature
   - Response delay to room temperature changes
   
5. **Moving Averages**: To capture different timescales of variation
   - Short-term vs long-term temperature trends
   - Differences between moving averages

A human body has thermal inertia and acts as a low-pass filter, dampening 
high-frequency temperature variations. These features help us detect this effect.
""")

# 5.1 Calculate temperature derivatives
# First derivative: Rate of temperature change
merged_df['ds18b20_deriv1'] = merged_df['ds18b20_temp'].diff() / (1/60)  # °F per hour
merged_df['room_deriv1'] = merged_df['room_temp'].diff() / (1/60)        # °F per hour

# Second derivative: Acceleration of temperature change
merged_df['ds18b20_deriv2'] = merged_df['ds18b20_deriv1'].diff() / (1/60)  # °F per hour^2

# Drop rows with NaN derivatives (first two rows)
merged_df = merged_df.dropna(subset=['ds18b20_deriv2']).copy()

# 5.2 Calculate rolling statistics to capture variability
# Define window sizes
short_window = 5   # 5 minutes
medium_window = 15  # 15 minutes
long_window = 60   # 1 hour

# Rolling standard deviation (variability)
merged_df['ds18b20_std_short'] = merged_df['ds18b20_temp'].rolling(window=short_window).std()
merged_df['ds18b20_std_medium'] = merged_df['ds18b20_temp'].rolling(window=medium_window).std()
merged_df['ds18b20_std_long'] = merged_df['ds18b20_temp'].rolling(window=long_window).std()

# Rolling IQR (interquartile range) - robust measure of dispersion
def rolling_iqr(x):
    q75 = np.percentile(x, 75)
    q25 = np.percentile(x, 25)
    return q75 - q25

merged_df['ds18b20_iqr_medium'] = merged_df['ds18b20_temp'].rolling(window=medium_window).apply(rolling_iqr)

# 5.3 Calculate moving averages at different timescales
merged_df['ds18b20_ma_short'] = merged_df['ds18b20_temp'].rolling(window=short_window).mean()
merged_df['ds18b20_ma_medium'] = merged_df['ds18b20_temp'].rolling(window=medium_window).mean()
merged_df['ds18b20_ma_long'] = merged_df['ds18b20_temp'].rolling(window=long_window).mean()

# Calculate differences between moving averages
merged_df['ds18b20_ma_short_medium_diff'] = merged_df['ds18b20_ma_short'] - merged_df['ds18b20_ma_medium']
merged_df['ds18b20_ma_medium_long_diff'] = merged_df['ds18b20_ma_medium'] - merged_df['ds18b20_ma_long']

# 5.4 Calculate coupling with room temperature
# Correlation between DS18B20 and room temperature (rolling window)
def rolling_correlation(x, y):
    return np.corrcoef(x, y)[0, 1] if len(x) > 1 else np.nan

# Create a function to apply rolling correlation
def apply_rolling_corr(series1, series2, window):
    result = []
    for i in range(len(series1) - window + 1):
        result.append(rolling_correlation(
            series1.iloc[i:i+window].values, 
            series2.iloc[i:i+window].values
        ))
    
    # Pad with NaNs at the beginning
    result = [np.nan] * (window - 1) + result
    return pd.Series(result, index=series1.index)

# Calculate rolling correlation with different windows
merged_df['temp_corr_medium'] = apply_rolling_corr(
    merged_df['ds18b20_temp'], 
    merged_df['room_temp'], 
    medium_window
)

# Calculate the ratio of DS18B20 to room temperature derivatives
# This captures how much the DS18B20 responds to room temperature changes
merged_df['deriv_ratio'] = np.abs(merged_df['ds18b20_deriv1']) / (np.abs(merged_df['room_deriv1']) + 1e-6)

# 5.5 Calculate frequency domain features
# We'll compute these for segments of the data to avoid NaN issues
window_size = 60  # 1 hour for frequency analysis

# Function to compute frequency domain features for a segment
def compute_freq_features(series, window_size):
    """Compute frequency domain features for a time series segment."""
    # Ensure we have enough data
    if len(series) < window_size:
        return pd.Series({
            'energy_high': np.nan,
            'energy_medium': np.nan, 
            'energy_low': np.nan,
            'wavelet_high': np.nan,
            'wavelet_medium': np.nan,
            'wavelet_low': np.nan
        })
    
    # Compute FFT
    segment = series.iloc[-window_size:].values
    segment = segment - np.mean(segment)  # Remove DC component
    
    # Apply window function to reduce spectral leakage
    segment = segment * signal.windows.hann(len(segment))
    
    # Compute FFT and frequencies
    fft_vals = fft(segment)
    fft_freq = fftfreq(len(segment), d=1/60)  # 1-minute sampling
    
    # Use only positive frequencies
    pos_mask = fft_freq > 0
    fft_vals = fft_vals[pos_mask]
    fft_freq = fft_freq[pos_mask]
    
    # Calculate power spectrum
    power = np.abs(fft_vals) ** 2
    
    # Define frequency bands (cycles per hour)
    # Low: < 0.5 cph (slower than 2-hour cycle)
    # Medium: 0.5-2 cph (between 30-minute and 2-hour cycles)
    # High: > 2 cph (faster than 30-minute cycle)
    low_mask = fft_freq < 0.5/60
    medium_mask = (fft_freq >= 0.5/60) & (fft_freq < 2/60)
    high_mask = fft_freq >= 2/60
    
    # Calculate energy in different frequency bands
    energy_low = np.sum(power[low_mask]) if np.any(low_mask) else 0
    energy_medium = np.sum(power[medium_mask]) if np.any(medium_mask) else 0
    energy_high = np.sum(power[high_mask]) if np.any(high_mask) else 0
    
    # Total energy for normalization
    total_energy = energy_low + energy_medium + energy_high
    if total_energy > 0:
        energy_low /= total_energy
        energy_medium /= total_energy
        energy_high /= total_energy
    
    # Wavelet transform for time-frequency analysis
    wavelet = 'db4'  # Daubechies wavelet
    coeffs = pywt.wavedec(segment, wavelet, level=3)
    
    # Calculate energy in wavelet coefficients at different levels
    # Level 1: High frequency, Level 3: Low frequency
    wavelet_high = np.sum(coeffs[1] ** 2) if len(coeffs) > 1 else 0
    wavelet_medium = np.sum(coeffs[2] ** 2) if len(coeffs) > 2 else 0
    wavelet_low = np.sum(coeffs[3] ** 2) if len(coeffs) > 3 else 0
    
    # Normalize wavelet energies
    wavelet_total = wavelet_high + wavelet_medium + wavelet_low
    if wavelet_total > 0:
        wavelet_high /= wavelet_total
        wavelet_medium /= wavelet_total
        wavelet_low /= wavelet_total
    
    return pd.Series({
        'energy_high': energy_high,
        'energy_medium': energy_medium, 
        'energy_low': energy_low,
        'wavelet_high': wavelet_high,
        'wavelet_medium': wavelet_medium,
        'wavelet_low': wavelet_low
    })

# Calculate frequency features for each segment
freq_features = []
for i in range(window_size, len(merged_df) + 1):
    segment = merged_df['ds18b20_temp'].iloc[i-window_size:i]
    features = compute_freq_features(segment, window_size)
    freq_features.append(features)

# Create frequency features dataframe
freq_df = pd.DataFrame(freq_features, index=merged_df.index[window_size-1:])

# Join with main dataframe
merged_df = merged_df.join(freq_df, how='left')

# Drop rows with NaN frequency features
merged_df = merged_df.dropna(subset=['energy_high']).copy()

# 5.6 Create a difference ratio feature
# This captures the contrast between device temperatures
merged_df['ds18b20_room_diff_ratio'] = (merged_df['ds18b20_temp'] - merged_df['room_temp']) / merged_df['room_temp']

# 5.7 Calculate smoothness metrics
# The "jitter" of the temperature signal
merged_df['ds18b20_jitter'] = np.abs(
    merged_df['ds18b20_deriv1'].diff()
).rolling(window=5).mean()

# Drop any remaining NaN values
merged_df = merged_df.dropna().copy()

print(f"\nFeature engineering complete. Dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns.")

# 6. Visualize and Analyze Engineered Features
# ==========================================
print("\nVisualizing and analyzing engineered features...")

# 6.1 Visualize derivatives and variability features
plt.figure(figsize=(14, 12))

# Plot 1: Original temperature data with labels
plt.subplot(4, 1, 1)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], '-', color=color, label=label)
plt.title('DS18B20 Temperature by Initial Label (Threshold-Based)')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: First derivative
plt.subplot(4, 1, 2)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_deriv1'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20 Temperature Rate of Change (°F/hour)')
plt.ylabel('dT/dt')
plt.legend()

# Plot 3: Short-term variability
plt.subplot(4, 1, 3)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_std_short'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20 Short-term Variability (5-min Rolling Std)')
plt.ylabel('Std Dev')
plt.legend()

# Plot 4: Jitter
plt.subplot(4, 1, 4)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_jitter'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20 Temperature Jitter (Variability in Rate of Change)')
plt.ylabel('Jitter')
plt.legend()

plt.tight_layout()
plt.savefig('thermal_dynamics_features1.png')
plt.show()

# 6.2 Visualize frequency domain and coupling features
plt.figure(figsize=(14, 12))

# Plot 1: High-frequency energy
plt.subplot(4, 1, 1)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['energy_high'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20 High-Frequency Energy Component')
plt.ylabel('Energy Ratio')
plt.legend()

# Plot 2: Correlation with room temperature
plt.subplot(4, 1, 2)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['temp_corr_medium'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20-Room Temperature Correlation (15-min Window)')
plt.ylabel('Correlation')
plt.ylim(-1.1, 1.1)
plt.legend()

# Plot 3: Derivative ratio
plt.subplot(4, 1, 3)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['deriv_ratio'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('Ratio of DS18B20 to Room Temp Derivatives')
plt.ylabel('Ratio')
plt.yscale('log')  # Log scale to better visualize ratios
plt.legend()

# Plot 4: Temperature difference ratio
plt.subplot(4, 1, 4)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_room_diff_ratio'][mask], '.', color=color, alpha=0.5, label=label)
plt.title('DS18B20-Room Temperature Difference Ratio')
plt.ylabel('Diff Ratio')
plt.legend()

plt.tight_layout()
plt.savefig('thermal_dynamics_features2.png')
plt.show()

# 6.3 Feature distributions by state
# Let's see how these features distribute differently between states
feature_cols = [
    'ds18b20_deriv1', 'ds18b20_std_short', 'ds18b20_jitter',
    'energy_high', 'temp_corr_medium', 'deriv_ratio'
]

plt.figure(figsize=(14, 10))
for i, feature in enumerate(feature_cols, 1):
    plt.subplot(3, 2, i)
    for state in [0, 1]:
        label = 'Connected' if state == 1 else 'Disconnected'
        data = merged_df[merged_df['initial_label'] == state][feature]
        sns.kdeplot(data, label=label, fill=True, alpha=0.5)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()
    plt.legend()

plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.show()

# 7. Build Classification Model
# ===========================
print("\nBuilding classification model based on thermal dynamics features...")

# EXPLANATION: Classification Approach
# =================================
print("""
# Classification Approach for Sensor State Detection
# ===============================================

We're using a machine learning approach to classify the sensor state based on 
thermal dynamics features. Here's the process:

1. **Feature Selection**: We've engineered features that capture different aspects
   of the sensor's thermal behavior:
   - Derivatives (rate of change)
   - Variability metrics
   - Frequency domain characteristics
   - Correlation with ambient conditions

2. **Random Forest Classifier**: We're using a Random Forest because it:
   - Handles non-linear relationships well
   - Works with different feature scales
   - Provides feature importance rankings
   - Is robust to outliers and noise
   
3. **Temporal Smoothing**: We'll apply post-processing to reduce state flipping,
   ensuring more stable state detection.

4. **Feature Importance Analysis**: We'll examine which features are most useful
   for distinguishing connected vs. disconnected states.

By combining multiple thermal dynamics features, we can create a more robust 
detection method than simple thresholding or time series models alone.
""")

# 7.1 Prepare features and target for classification
# Select relevant features for classification
selected_features = [
    # Derivative-based features
    'ds18b20_deriv1', 'ds18b20_deriv2', 
    
    # Variability features
    'ds18b20_std_short', 'ds18b20_std_medium', 'ds18b20_iqr_medium',
    
    # Moving average features
    'ds18b20_ma_short_medium_diff', 'ds18b20_ma_medium_long_diff',
    
    # Frequency domain features
    'energy_high', 'energy_medium', 'energy_low',
    'wavelet_high', 'wavelet_medium', 'wavelet_low',
    
    # Coupling features
    'temp_corr_medium', 'deriv_ratio', 'ds18b20_room_diff_ratio',
    
    # Smoothness metrics
    'ds18b20_jitter'
]

# Create feature matrix and target
X = merged_df[selected_features].values
y = merged_df['initial_label'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7.2 Train a Random Forest classifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# Fit the model
clf.fit(X_scaled, y)

# 7.3 Evaluate the model on the training data
# (In production, you'd want to use cross-validation or a separate test set)
y_pred = clf.predict(X_scaled)
y_pred_proba = clf.predict_proba(X_scaled)[:, 1]  # Probability of connected state

# Add predictions to DataFrame
merged_df['thermal_state'] = y_pred
merged_df['thermal_connected_prob'] = y_pred_proba

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['Disconnected', 'Connected']))

# 7.4 Analyze feature importance
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Thermal Dynamics Classification')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# 7.5 Apply temporal smoothing to reduce state flipping
window_size = 5  # 5-minute window for smoothing
merged_df['thermal_state_smoothed'] = merged_df['thermal_connected_prob'].rolling(
    window=window_size, center=True
).mean().fillna(merged_df['thermal_connected_prob']) > 0.5

# 7.6 Visualize the classification results
plt.figure(figsize=(14, 12))

# Plot 1: Original temperature data
plt.subplot(4, 1, 1)
plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
plt.plot(merged_df.index, merged_df['room_temp'], 'g-', label='Room', alpha=0.5)
plt.title('Temperature Data')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: Initial labels (threshold-based)
plt.subplot(4, 1, 2)
plt.plot(merged_df.index, merged_df['initial_label'], 'k-', label='Threshold Label')
plt.title('Initial State Labels (Threshold-Based)')
plt.ylabel('State (1=Connected)')
plt.ylim(-0.1, 1.1)
plt.legend()

# Plot 3: Thermal dynamics classification
plt.subplot(4, 1, 3)
plt.plot(merged_df.index, merged_df['thermal_state'], 'b-', label='Thermal Dynamics')
plt.title('Thermal Dynamics Classification')
plt.ylabel('State (1=Connected)')
plt.ylim(-0.1, 1.1)
plt.legend()

# Plot 4: Connection probability
plt.subplot(4, 1, 4)
plt.plot(merged_df.index, merged_df['thermal_connected_prob'], 'r-', label='Connection Probability')
plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
plt.fill_between(merged_df.index, 0, merged_df['thermal_connected_prob'], alpha=0.3, color='salmon')
plt.title('Connection Probability from Thermal Dynamics Model')
plt.ylabel('Probability')
plt.ylim(-0.1, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig('thermal_dynamics_classification.png')
plt.show()

# 7.7 Compare smoothed vs. unsmoothed classification
plt.figure(figsize=(14, 10))

# Plot 1: Original temperature data
plt.subplot(3, 1, 1)
plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
plt.title('DS18B20 Temperature')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: Unsmoothed classification
plt.subplot(3, 1, 2)
plt.plot(merged_df.index, merged_df['thermal_state'], 'r-', label='Unsmoothed')
plt.title('Unsmoothed Thermal Dynamics Classification')
plt.ylabel('State (1=Connected)')
plt.ylim(-0.1, 1.1)
plt.legend()

# Plot 3: Smoothed classification
plt.subplot(3, 1, 3)
plt.plot(merged_df.index, merged_df['thermal_state_smoothed'], 'g-', label='Smoothed')
plt.title(f'Smoothed Classification ({window_size}-minute window)')
plt.ylabel('State (1=Connected)')
plt.ylim(-0.1, 1.1)
plt.legend()

plt.tight_layout()
plt.savefig('smoothed_vs_unsmoothed.png')
plt.show()

# 7.8 Zoom in on transition periods
# Find state transitions in the smoothed classification
transitions = merged_df['thermal_state_smoothed'].diff().abs() > 0
transition_times = merged_df.index[transitions]

# If we have transitions, let's zoom in on one
if len(transition_times) > 0:
    # Get the first transition
    transition_time = transition_times[0]
    
    # Define a window around the transition
    window_start = transition_time - pd.Timedelta(minutes=30)
    window_end = transition_time + pd.Timedelta(minutes=30)
    
    # Filter data for the transition window
    transition_df = merged_df.loc[window_start:window_end].copy()
    
    if len(transition_df) > 0:
        # Plot the transition
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Temperature during transition
        plt.subplot(3, 1, 1)
        plt.plot(transition_df.index, transition_df['ds18b20_temp'], 'b-', label='DS18B20')
        plt.plot(transition_df.index, transition_df['room_temp'], 'g-', label='Room', alpha=0.5)
        plt.axvline(x=transition_time, color='r', linestyle='--', label='Transition')
        plt.title('Temperature During Transition')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        
        # Plot 2: Key features during transition
        plt.subplot(3, 1, 2)
        plt.plot(transition_df.index, transition_df['ds18b20_deriv1'], 'r-', label='Rate of Change')
        plt.plot(transition_df.index, transition_df['ds18b20_std_short'], 'g-', label='Short-term Variability')
        plt.axvline(x=transition_time, color='k', linestyle='--', label='Transition')
        plt.title('Key Features During Transition')
        plt.ylabel('Feature Value')
        plt.legend()
        
        # Plot 3: Connection probability during transition
        plt.subplot(3, 1, 3)
        plt.plot(transition_df.index, transition_df['thermal_connected_prob'], 'b-', label='Connection Probability')
        plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold')
        plt.axvline(x=transition_time, color='r', linestyle='--', label='Transition')
        plt.title('Connection Probability During Transition')
        plt.ylabel('Probability')
        plt.ylim(-0.1, 1.1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('transition_analysis.png')
        plt.show()

# 8. Create a Detection Function
# ============================
print("\nCreating a sensor state detection function...")

def detect_connection_thermal(ds18b20_temp, room_temp, cpu_temp, 
                              feature_history=None, new_sample=True):
    """
    Detect if the DS18B20 sensor is connected using thermal dynamics features.
    
    Parameters:
    -----------
    ds18b20_temp : float
        Current DS18B20 temperature reading (°F)
    room_temp : float
        Current room temperature from Si7021 (°F)
    cpu_temp : float
        Current CPU temperature (°F)
    feature_history : dict, optional
        Dictionary containing history of measurements and computed features
        for calculating time-dependent features
    new_sample : bool, default=True
        Whether to add the current readings to the history
        
    Returns:
    --------
    dict
        Dictionary containing detection results and diagnostics
    """
    # Initialize feature history if needed
    if feature_history is None:
        feature_history = {
            'ds18b20_temps': [],
            'room_temps': [],
            'cpu_temps': [],
            'timestamps': [],
            'features': {},  # Computed features
            'states': [],    # Past state predictions
            'probs': []      # Past state probabilities
        }
    
    # Add current readings to history if this is a new sample
    current_time = datetime.now()
    if new_sample:
        feature_history['ds18b20_temps'].append(ds18b20_temp)
        feature_history['room_temps'].append(room_temp)
        feature_history['cpu_temps'].append(cpu_temp)
        feature_history['timestamps'].append(current_time)
    
    # Ensure history doesn't grow too large (keep last hour for feature calculation)
    max_history = 60  # 1 hour at 1-minute sampling
    if len(feature_history['ds18b20_temps']) > max_history:
        feature_history['ds18b20_temps'] = feature_history['ds18b20_temps'][-max_history:]
        feature_history['room_temps'] = feature_history['room_temps'][-max_history:]
        feature_history['cpu_temps'] = feature_history['cpu_temps'][-max_history:]
        feature_history['timestamps'] = feature_history['timestamps'][-max_history:]
        feature_history['states'] = feature_history['states'][-max_history:]
        feature_history['probs'] = feature_history['probs'][-max_history:]
    
    # Check if we have enough history for feature calculation
    if len(feature_history['ds18b20_temps']) < 5:
        # Not enough history, use a simple threshold approach
        is_connected = ds18b20_temp > 90.0
        prob_connected = 1.0 if is_connected else 0.0
        
        # Save state and probability
        if new_sample:
            feature_history['states'].append(int(is_connected))
            feature_history['probs'].append(prob_connected)
        
        return {
            'ds18b20_temp': ds18b20_temp,
            'room_temp': room_temp,
            'cpu_temp': cpu_temp,
            'is_connected': is_connected,
            'connected_probability': prob_connected,
            'method': 'threshold',  # Indicate we used simple thresholding
            'feature_history': feature_history
        }
    
    # Calculate basic features from history
    temps = np.array(feature_history['ds18b20_temps'])
    room_temps = np.array(feature_history['room_temps'])
    
    # Calculate derivatives (rate of change)
    if len(temps) >= 2:
        ds18b20_deriv1 = (temps[-1] - temps[-2]) * 60  # °F per hour
        room_deriv1 = (room_temps[-1] - room_temps[-2]) * 60  # °F per hour
    else:
        ds18b20_deriv1 = 0
        room_deriv1 = 0
    
    # Calculate second derivative
    if len(temps) >= 3:
        ds18b20_deriv2 = ((temps[-1] - temps[-2]) - (temps[-2] - temps[-3])) * 60  # °F per hour^2
    else:
        ds18b20_deriv2 = 0
    
    # Calculate short-term variability (standard deviation)
    ds18b20_std_short = np.std(temps[-5:]) if len(temps) >= 5 else 0
    
    # Calculate medium-term variability
    ds18b20_std_medium = np.std(temps[-15:]) if len(temps) >= 15 else ds18b20_std_short
    
    # Calculate IQR (interquartile range)
    if len(temps) >= 15:
        q75 = np.percentile(temps[-15:], 75)
        q25 = np.percentile(temps[-15:], 25)
        ds18b20_iqr_medium = q75 - q25
    else:
        ds18b20_iqr_medium = 0
    
    # Calculate moving averages
    ds18b20_ma_short = np.mean(temps[-5:]) if len(temps) >= 5 else temps[-1]
    ds18b20_ma_medium = np.mean(temps[-15:]) if len(temps) >= 15 else ds18b20_ma_short
    ds18b20_ma_long = np.mean(temps[-60:]) if len(temps) >= 60 else ds18b20_ma_medium
    
    # Calculate differences between moving averages
    ds18b20_ma_short_medium_diff = ds18b20_ma_short - ds18b20_ma_medium
    ds18b20_ma_medium_long_diff = ds18b20_ma_medium - ds18b20_ma_long
    
    # Calculate coupling with room temperature
    # Correlation between DS18B20 and room temperature
    if len(temps) >= 15 and len(room_temps) >= 15:
        try:
            temp_corr_medium = np.corrcoef(temps[-15:], room_temps[-15:])[0, 1]
        except:
            temp_corr_medium = 0
    else:
        temp_corr_medium = 0
    
    # Calculate derivative ratio
    deriv_ratio = abs(ds18b20_deriv1) / (abs(room_deriv1) + 1e-6)
    
    # Calculate temperature difference ratio
    ds18b20_room_diff_ratio = (ds18b20_temp - room_temp) / (room_temp + 1e-6)
    
    # Calculate jitter (variability in rate of change)
    if len(temps) >= 6:
        derivs = []
        for i in range(len(temps) - 1, max(0, len(temps) - 6), -1):
            if i > 0:
                derivs.append(temps[i] - temps[i-1])
        ds18b20_jitter = np.std(derivs) if derivs else 0
    else:
        ds18b20_jitter = 0
    
    # Calculate frequency domain features if we have enough data
    if len(temps) >= 30:
        segment = temps[-30:]
        segment = segment - np.mean(segment)  # Remove DC component
        
        # Apply window function
        segment = segment * signal.windows.hann(len(segment))
        
        # Compute FFT
        fft_vals = fft(segment)
        fft_freq = fftfreq(len(segment), d=1/60)  # 1-minute sampling
        
        # Use only positive frequencies
        pos_mask = fft_freq > 0
        fft_vals = fft_vals[pos_mask]
        fft_freq = fft_freq[pos_mask]
        
        # Calculate power spectrum
        power = np.abs(fft_vals) ** 2
        
        # Define frequency bands
        low_mask = fft_freq < 0.5/60
        medium_mask = (fft_freq >= 0.5/60) & (fft_freq < 2/60)
        high_mask = fft_freq >= 2/60
        
        # Calculate energy in different frequency bands
        energy_low = np.sum(power[low_mask]) if np.any(low_mask) else 0
        energy_medium = np.sum(power[medium_mask]) if np.any(medium_mask) else 0
        energy_high = np.sum(power[high_mask]) if np.any(high_mask) else 0
        
        # Normalize
        total_energy = energy_low + energy_medium + energy_high
        if total_energy > 0:
            energy_low /= total_energy
            energy_medium /= total_energy
            energy_high /= total_energy
    else:
        energy_low = 0.33
        energy_medium = 0.33
        energy_high = 0.33
    
    # Calculate wavelet features if we have enough data
    if len(temps) >= 30:
        try:
            segment = temps[-30:]
            segment = segment - np.mean(segment)  # Remove DC component
            
            # Wavelet transform
            wavelet = 'db4'  # Daubechies wavelet
            coeffs = pywt.wavedec(segment, wavelet, level=3)
            
            # Calculate energy in wavelet coefficients
            wavelet_high = np.sum(coeffs[1] ** 2) if len(coeffs) > 1 else 0
            wavelet_medium = np.sum(coeffs[2] ** 2) if len(coeffs) > 2 else 0
            wavelet_low = np.sum(coeffs[3] ** 2) if len(coeffs) > 3 else 0
            
            # Normalize
            wavelet_total = wavelet_high + wavelet_medium + wavelet_low
            if wavelet_total > 0:
                wavelet_high /= wavelet_total
                wavelet_medium /= wavelet_total
                wavelet_low /= wavelet_total
        except:
            wavelet_high = 0.33
            wavelet_medium = 0.33
            wavelet_low = 0.33
    else:
        wavelet_high = 0.33
        wavelet_medium = 0.33
        wavelet_low = 0.33
    
    # Create feature vector
    features = [
        ds18b20_deriv1, ds18b20_deriv2,
        ds18b20_std_short, ds18b20_std_medium, ds18b20_iqr_medium,
        ds18b20_ma_short_medium_diff, ds18b20_ma_medium_long_diff,
        energy_high, energy_medium, energy_low,
        wavelet_high, wavelet_medium, wavelet_low,
        temp_corr_medium, deriv_ratio, ds18b20_room_diff_ratio,
        ds18b20_jitter
    ]
    
    # Scale features using the pre-trained scaler
    features_scaled = scaler.transform([features])[0]
    
    # Predict connection state using the Random Forest model
    connected_prob = clf.predict_proba([features_scaled])[0][1]
    is_connected = connected_prob > 0.5
    
    # Apply temporal smoothing if we have enough history
    if len(feature_history['probs']) >= 5:
        # Average the last 5 probabilities (including the current one)
        all_probs = feature_history['probs'][-4:] + [connected_prob]
        smoothed_prob = np.mean(all_probs)
        is_connected = smoothed_prob > 0.5
    else:
        smoothed_prob = connected_prob
    
    # Save state and probability to history
    if new_sample:
        feature_history['states'].append(int(is_connected))
        feature_history['probs'].append(connected_prob)
    
    # Prepare result
    result = {
        'ds18b20_temp': ds18b20_temp,
        'room_temp': room_temp,
        'cpu_temp': cpu_temp,
        'is_connected': bool(is_connected),
        'connected_probability': float(smoothed_prob),
        'raw_probability': float(connected_prob),
        'method': 'thermal_dynamics',
        'feature_history': feature_history,
        'features': {
            'ds18b20_deriv1': float(ds18b20_deriv1),
            'ds18b20_std_short': float(ds18b20_std_short),
            'ds18b20_jitter': float(ds18b20_jitter),
            'temp_corr_medium': float(temp_corr_medium),
            'energy_high': float(energy_high)
        }
    }
    
    return result

# Example usage with the last data point
if len(merged_df) > 0:
    last_point = merged_df.iloc[-1]
    
    # Initialize feature history from existing data
    feature_history = {
        'ds18b20_temps': merged_df['ds18b20_temp'].tolist()[-60:],
        'room_temps': merged_df['room_temp'].tolist()[-60:],
        'cpu_temps': merged_df['cpu_temp'].tolist()[-60:],
        'timestamps': merged_df.index.tolist()[-60:],
        'features': {},
        'states': merged_df['thermal_state'].tolist()[-60:],
        'probs': merged_df['thermal_connected_prob'].tolist()[-60:]
    }
    
    # Test the detection function
    detection_result = detect_connection_thermal(
        last_point['ds18b20_temp'],
        last_point['room_temp'],
        last_point['cpu_temp'],
        feature_history=feature_history,
        new_sample=False  # Don't add this to history since it's already there
    )
    
    print("\nDetection result for the most recent data point:")
    for key, value in detection_result.items():
        if key != 'feature_history':  # Skip printing the entire history
            print(f"{key}: {value}")

# 9. Save the Model for Later Use
# =============================
# Save the classifier and scaler
joblib.dump({
    'classifier': clf,
    'scaler': scaler,
    'feature_names': selected_features
}, 'ds18b20_thermal_dynamics_model.pkl')

print("\nSaved model and parameters to 'ds18b20_thermal_dynamics_model.pkl'")

# 10. Final Recommendations
# =======================
print("""
# Summary and Recommendations
# =========================

This notebook demonstrates how to use thermal response dynamics to detect 
when the DS18B20 temperature sensor is connected to a human body versus 
when it's disconnected and measuring ambient temperature.

Key findings:
1. The thermal dynamics of the DS18B20 differ significantly when connected to a human body
2. Key distinguishing features include:
   - Temperature rate of change and variability
   - Frequency domain characteristics
   - Correlation with room temperature
   - Response to environmental changes

Advantages of the thermal dynamics approach:
1. Uses multiple complementary features for more robust detection
2. Captures physical principles of heat transfer and thermal inertia
3. Less dependent on specific temperature thresholds
4. Can detect subtle changes in sensor behavior

Recommendations for implementation:
1. Combine with other detection methods for higher reliability
2. Use temporal smoothing to prevent rapid state flipping
3. Retrain the model periodically with known connected/disconnected periods
4. Consider implementing an adaptive version that updates feature distributions

Next steps:
1. Evaluate with labeled ground truth data
2. Explore additional features like humidity effects
3. Implement real-time detection in your monitoring system
4. Consider fusion with the residual analysis and HMM approaches
""")

# Close the InfluxDB client
client.close()
print("\nAnalysis complete!")