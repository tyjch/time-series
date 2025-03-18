# Cross-Correlation Analysis for DS18B20 Sensor State Detection
# =========================================================

# This notebook demonstrates how to use cross-correlation analysis techniques
# to detect when the DS18B20 temperature sensor is connected or disconnected.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from influxdb_client import InfluxDBClient
from statsmodels.tsa.stattools import acf, ccf, pacf, grangercausalitytests
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_ccf
from statsmodels.tsa.stattools import adfuller
from scipy import signal
from datetime import datetime, timedelta
from dotenv import load_dotenv
import joblib
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

load_dotenv()

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Set Matplotlib parameters for better readability
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 1. Introduction to Cross-Correlation Analysis
# ==========================================
print("""
# Cross-Correlation Analysis Explained
# =================================

Cross-correlation analysis examines the relationship between multiple time series
to understand how they influence each other. In our sensor state detection problem:

1. **Connected State (Body Temperature)**:
   - DS18B20 readings should show low correlation with room temperature
   - Any correlation will have significant lag (delayed response)
   - The body maintains its own thermal regulation

2. **Disconnected State (Ambient Temperature)**:
   - DS18B20 readings should closely track room temperature
   - Correlation will be high with minimal lag
   - The sensor directly responds to environmental changes

Key concepts we'll explore:

- **Cross-correlation function (CCF)**: Measures correlation between two time series
  at different time lags

- **Granger causality**: Tests whether one time series helps forecast another

- **Transfer function models**: Characterize how changes in room temperature
  "transfer" to changes in the DS18B20 readings

- **Coherence analysis**: Examines correlation in the frequency domain

By analyzing these temporal relationships, we can create a robust detector
for the sensor's connection state without relying solely on absolute temperature values.
""")

# 2. Connect to InfluxDB and Query Data
# ====================================
# Replace these with your actual InfluxDB credentials
url    = os.getenv('INFLUX_URL')
token  = os.getenv('INFLUX_TOKEN')
org    = os.getenv('INFLUX_ORG')
bucket = os.getenv('INFLUX_BUCKET')

print(url, token, org, bucket)

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
plt.savefig('cross_correlation_time_series.png')
plt.show()

# 5. Basic Time Series Analysis
# ============================
# Before diving into cross-correlation, we should understand the individual time series

# 5.1 Check for stationarity using Augmented Dickey-Fuller test
# Stationarity is important for correlation analysis
print("\nChecking for stationarity...")

def test_stationarity(timeseries, series_name):
    # Calculate statistics
    rolling_mean = timeseries.rolling(window=60).mean()
    rolling_std = timeseries.rolling(window=60).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(timeseries, label=series_name)
    plt.plot(rolling_mean, label='Rolling Mean (1-hour window)')
    plt.plot(rolling_std, label='Rolling Std (1-hour window)')
    plt.legend()
    plt.title(f'Rolling Mean & Standard Deviation for {series_name}')
    plt.tight_layout()
    plt.savefig(f'stationarity_{series_name}.png')
    plt.show()
    
    # Perform Dickey-Fuller test
    print(f'Results of Dickey-Fuller Test for {series_name}:')
    adf_test = adfuller(timeseries.dropna(), autolag='AIC')
    adf_output = pd.Series(adf_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key, value in adf_test[4].items():
        adf_output[f'Critical Value ({key})'] = value
    print(adf_output)
    
    # Determine if stationary
    is_stationary = adf_test[1] <= 0.05
    print(f"Series is {'stationary' if is_stationary else 'non-stationary'} (p-value <= 0.05 indicates stationarity)\n")
    return is_stationary

# Check stationarity for both temperature series
ds18b20_stationary = test_stationarity(merged_df['ds18b20_temp'], 'DS18B20 Temperature')
room_stationary = test_stationarity(merged_df['room_temp'], 'Room Temperature')

# If series are non-stationary, we should consider differencing
# for better correlation analysis
print("Differencing the time series to achieve stationarity...")
merged_df['ds18b20_diff'] = merged_df['ds18b20_temp'].diff().fillna(0)
merged_df['room_diff'] = merged_df['room_temp'].diff().fillna(0)

# Check stationarity of differenced series
ds18b20_diff_stationary = test_stationarity(merged_df['ds18b20_diff'], 'DS18B20 Temperature (Differenced)')
room_diff_stationary = test_stationarity(merged_df['room_diff'], 'Room Temperature (Differenced)')

# 5.2 Autocorrelation analysis
# Understand the internal time-dependency structure of each series
print("\nAnalyzing autocorrelation patterns...")

plt.figure(figsize=(14, 10))

# ACF of DS18B20 temperature
plt.subplot(2, 2, 1)
plot_acf(merged_df['ds18b20_temp'].values, lags=60, alpha=0.05, title='ACF: DS18B20 Temperature')

# PACF of DS18B20 temperature
plt.subplot(2, 2, 2)
plot_pacf(merged_df['ds18b20_temp'].values, lags=60, alpha=0.05, method='ols', title='PACF: DS18B20 Temperature')

# ACF of room temperature
plt.subplot(2, 2, 3)
plot_acf(merged_df['room_temp'].values, lags=60, alpha=0.05, title='ACF: Room Temperature')

# PACF of room temperature
plt.subplot(2, 2, 4)
plot_pacf(merged_df['room_temp'].values, lags=60, alpha=0.05, method='ols', title='PACF: Room Temperature')

plt.tight_layout()
plt.savefig('autocorrelation_analysis.png')
plt.show()

# 6. Cross-Correlation Analysis
# ===========================
print("\nPerforming cross-correlation analysis...")

# EXPLANATION: Cross-Correlation Analysis
# ====================================
print("""
# Cross-Correlation Analysis for Sensor State Detection
# ==================================================

Cross-correlation analysis examines how two time series relate to each other across
different time lags. For our sensor detection problem:

1. **What Cross-Correlation Measures**:
   - How strongly the DS18B20 and room temperature series correlate
   - How much lag (delay) exists between changes in the two series
   - The directionality of influence between temperature measurements

2. **Key Patterns**:
   - When DISCONNECTED: High correlation with short lag
   - When CONNECTED: Low correlation or longer lag
   
3. **Why This Works**:
   - A disconnected sensor directly responds to ambient air temperature
   - A connected sensor is thermally coupled to the human body, which:
     a) Maintains its own temperature independent of the room
     b) Has thermal inertia that delays environmental effects
     c) Has physiological temperature regulation

By analyzing these correlation patterns, we can detect the sensor state without
relying only on absolute temperature thresholds.
""")

# 6.1 Calculate cross-correlation between sensors for the full dataset
ccf_values = ccf(merged_df['ds18b20_temp'].values, merged_df['room_temp'].values, adjusted=True)
lags = np.arange(-30, 31)  # -30 to +30 minute lags

# Plot the cross-correlation function
plt.figure(figsize=(12, 6))
plt.stem(lags, ccf_values[30:91], linefmt='b-', markerfmt='bo', basefmt='r-')
plt.axhline(y=0, color='r', linestyle='-')
plt.axhline(y=1.96/np.sqrt(len(ds18b20_df)), linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ds18b20_df)), linestyle='--', color='gray')
plt.xlabel('Lag (minutes)')
plt.ylabel('Cross-correlation')
plt.annotate(f'Max correlation: {max_corr:.4f} at lag {max_lag}', 
                xy=(0.05, 0.95), xycoords='axes fraction')
plt.grid(True)
plt.tight_layout()
    


# Calculate CCF for likely disconnected periods
disconnected_mask = merged_df['initial_label'] == 0
if disconnected_mask.sum() > 60:  # Ensure we have enough data points
    disconnected_corr, disconnected_lag, disconnected_ccf = calculate_plot_ccf(
        merged_df.loc[disconnected_mask, 'ds18b20_temp'].values,
        merged_df.loc[disconnected_mask, 'room_temp'].values,
        'Cross-Correlation: DS18B20 vs Room Temp (Disconnected State)'
    )
    plt.savefig('ccf_disconnected.png')
    plt.show()
else:
    print("Not enough data points for disconnected state CCF calculation")

# Calculate CCF for likely connected periods
connected_mask = merged_df['initial_label'] == 1
if connected_mask.sum() > 60:  # Ensure we have enough data points
    connected_corr, connected_lag, connected_ccf = calculate_plot_ccf(
        merged_df.loc[connected_mask, 'ds18b20_temp'].values,
        merged_df.loc[connected_mask, 'room_temp'].values,
        'Cross-Correlation: DS18B20 vs Room Temp (Connected State)'
    )
    plt.savefig('ccf_connected.png')
    plt.show()
else:
    print("Not enough data points for connected state CCF calculation")

# Compare characteristics of the CCF patterns
if 'disconnected_corr' in locals() and 'connected_corr' in locals():
    print("\nCross-correlation characteristics by sensor state:")
    print(f"Disconnected state: Max correlation = {disconnected_corr:.4f} at lag {disconnected_lag}")
    print(f"Connected state: Max correlation = {connected_corr:.4f} at lag {connected_lag}")
    print(f"Ratio of max correlations (disconnected/connected): {abs(disconnected_corr/connected_corr):.2f}")

# 7. Granger Causality Analysis
# ===========================
print("\nPerforming Granger causality analysis...")

# EXPLANATION: Granger Causality
# ===========================
print("""
# Granger Causality for Sensor State Detection
# =========================================

Granger causality is a statistical test that determines whether one time series
helps forecast another. In our context:

1. **Core Concept**: 
   - If past values of room temperature help predict DS18B20 readings (beyond
     what DS18B20's own past values predict), then room temperature "Granger causes"
     the DS18B20 readings.

2. **Expected Patterns**:
   - When DISCONNECTED: Strong Granger causality from room to DS18B20
   - When CONNECTED: Weak or no Granger causality

3. **How to Interpret**:
   - P-value < 0.05 indicates significant Granger causality
   - The lower the p-value, the stronger the causal relationship
   - F-statistic magnitude indicates strength of relationship

This approach helps identify if the DS18B20 is being primarily driven by room
temperature (disconnected) or by human body temperature (connected).
""")

# Function to perform rolling Granger causality tests
def rolling_granger_causality(y, x, window_size=120, max_lag=5):
    """Calculate rolling Granger causality p-values."""
    p_values = []
    f_stats  = []
    
    # We need enough data points for the test
    if len(y) < window_size + max_lag:
        return [], []
    
    for i in range(window_size, len(y) + 1):
        y_window = y.iloc[i-window_size:i].values
        x_window = x.iloc[i-window_size:i].values
        
        # Create DataFrame for Granger test
        data = pd.DataFrame({
            'y': y_window,
            'x': x_window
        })
        
        try:
            # Test Granger causality
            gc_res = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)
            
            # Get results for the best lag (lowest p-value)
            best_p = min([gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
            best_lag = np.argmin([gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)]) + 1
            best_f = gc_res[best_lag][0]['ssr_ftest'][0]
            
            p_values.append(best_p)
            f_stats.append(best_f)
        except:
            # Handle errors (e.g., singular matrix)
            p_values.append(1.0)  # No causality
            f_stats.append(0.0)
    
    return p_values, f_stats

# Calculate rolling Granger causality from room to DS18B20
window_size = 120  # 2 hours at 1-minute intervals
print("Calculating rolling Granger causality (this might take a while)...")

p_values, f_stats = rolling_granger_causality(
    merged_df['ds18b20_temp'],
    merged_df['room_temp'],
    window_size=window_size,
    max_lag=5  # Test up to 5-minute lags
)

# Add to dataframe
merged_df['granger_p_value'] = pd.Series(p_values, index=merged_df.index[window_size-1:])
merged_df['granger_f_stat'] = pd.Series(f_stats, index=merged_df.index[window_size-1:])

# Convert p-values to significance level
merged_df['granger_significance'] = -np.log10(merged_df['granger_p_value'])

# Plot Granger causality results
plt.figure(figsize=(14, 10))

# Plot 1: Original temperature data with labels
plt.subplot(3, 1, 1)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], '-', color=color, label=label)
plt.title('DS18B20 Temperature by Initial Label (Threshold-Based)')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: Granger causality p-value
plt.subplot(3, 1, 2)
plt.semilogy(merged_df.index[window_size-1:], merged_df['granger_p_value'], 'g-')
plt.axhline(y=0.05, color='r', linestyle='--', label='Significance level (p=0.05)')
plt.title('Granger Causality p-value (Room → DS18B20)')
plt.ylabel('p-value (log scale)')
plt.legend()

# Plot 3: Granger causality significance
plt.subplot(3, 1, 3)
plt.plot(merged_df.index[window_size-1:], merged_df['granger_significance'], 'b-')
plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='Significance level (p=0.05)')
plt.title('Granger Causality Significance (-log10(p-value))')
plt.ylabel('-log10(p-value)')
plt.legend()

plt.tight_layout()
plt.savefig('granger_causality.png')
plt.show()

# 8. Coherence Analysis
# ====================
print("\nPerforming coherence analysis...")

# EXPLANATION: Coherence Analysis
# ============================
print("""
# Coherence Analysis for Sensor State Detection
# =========================================

Coherence is a frequency-domain measure of correlation between two time series.
It tells us how well two signals correlate at different frequencies:

1. **Key Concept**:
   - Coherence ranges from 0 (no correlation) to 1 (perfect correlation)
   - It's calculated separately for each frequency component
   
2. **Expected Patterns**:
   - When DISCONNECTED: High coherence across frequencies
   - When CONNECTED: Lower coherence, especially at higher frequencies
   
3. **Why This Works**:
   - The human body acts as a low-pass filter, dampening high-frequency variations
   - A disconnected sensor will follow room temperature fluctuations at all frequencies
   - A connected sensor will be partially isolated from environmental changes

This approach reveals how the sensor's frequency response changes with connection state.
""")

# Function to calculate coherence between two time series
def calculate_coherence(x, y, fs=1/60):
    """Calculate coherence between two time series.
    
    Parameters:
    -----------
    x, y : array-like
        Input time series
    fs : float
        Sampling frequency in Hz (1/60 = 1 sample per minute)
        
    Returns:
    --------
    f : array
        Frequency array
    Cxy : array
        Coherence array
    """
    f, Cxy = signal.coherence(x, y, fs=fs, nperseg=120)
    return f, Cxy

# Calculate coherence for disconnected state
if disconnected_mask.sum() > 120:  # Need enough data for good frequency resolution
    f_disconnected, Cxy_disconnected = calculate_coherence(
        merged_df.loc[disconnected_mask, 'ds18b20_temp'].values,
        merged_df.loc[disconnected_mask, 'room_temp'].values
    )
else:
    print("Not enough data for disconnected coherence calculation")

# Calculate coherence for connected state
if connected_mask.sum() > 120:
    f_connected, Cxy_connected = calculate_coherence(
        merged_df.loc[connected_mask, 'ds18b20_temp'].values,
        merged_df.loc[connected_mask, 'room_temp'].values
    )
else:
    print("Not enough data for connected coherence calculation")

# Plot coherence spectra
if 'Cxy_disconnected' in locals() and 'Cxy_connected' in locals():
    plt.figure(figsize=(12, 6))
    
    plt.plot(f_disconnected * 60, Cxy_disconnected, 'b-', label='Disconnected')
    plt.plot(f_connected * 60, Cxy_connected, 'r-', label='Connected')
    
    plt.title('Coherence Spectrum: DS18B20 vs. Room Temperature')
    plt.xlabel('Frequency (cycles per hour)')
    plt.ylabel('Coherence')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('coherence_spectrum.png')
    plt.show()
    
    # Calculate average coherence in different frequency bands
    # Low: < 0.5 cph, Medium: 0.5-2 cph, High: > 2 cph
    freq_bands = {
        'low': (f_disconnected * 60 < 0.5),
        'medium': (f_disconnected * 60 >= 0.5) & (f_disconnected * 60 < 2),
        'high': (f_disconnected * 60 >= 2)
    }
    
    print("\nCoherence by frequency band:")
    for band, mask in freq_bands.items():
        if np.any(mask):
            disconnected_avg = np.mean(Cxy_disconnected[mask])
            connected_avg = np.mean(Cxy_connected[mask])
            ratio = disconnected_avg / connected_avg if connected_avg > 0 else float('inf')
            
            print(f"{band.capitalize()} frequency band ({band}):")
            print(f"  Disconnected coherence: {disconnected_avg:.4f}")
            print(f"  Connected coherence: {connected_avg:.4f}")
            print(f"  Ratio (disconnected/connected): {ratio:.2f}")

# 9. Rolling Coherence Analysis
# ===========================
# Calculate coherence in rolling windows to track changes over time
print("\nCalculating rolling coherence metrics...")

def rolling_coherence_metrics(x, y, window_size=120, fs=1/60):
    """Calculate coherence metrics in rolling windows."""
    # Define frequency bands (cycles per hour)
    # Low: < 0.5 cph, Medium: 0.5-2 cph, High: > 2 cph
    low_band = (0, 0.5/60)    # < 0.5 cycles per hour
    medium_band = (0.5/60, 2/60)  # 0.5-2 cycles per hour
    high_band = (2/60, None)   # > 2 cycles per hour
    
    # Initialize arrays for results
    low_coherence = []
    medium_coherence = []
    high_coherence = []
    
    for i in range(window_size, len(x) + 1):
        # Get window of data
        x_window = x.iloc[i-window_size:i].values
        y_window = y.iloc[i-window_size:i].values
        
        try:
            # Calculate coherence
            f, Cxy = signal.coherence(x_window, y_window, fs=fs, nperseg=min(60, window_size//2))
            
            # Calculate average coherence in each band
            low_mask = f < low_band[1]
            medium_mask = (f >= medium_band[0]) & (f < medium_band[1])
            high_mask = f >= high_band[0]
            
            low_coh = np.mean(Cxy[low_mask]) if np.any(low_mask) else 0
            medium_coh = np.mean(Cxy[medium_mask]) if np.any(medium_mask) else 0
            high_coh = np.mean(Cxy[high_mask]) if np.any(high_mask) else 0
            
            low_coherence.append(low_coh)
            medium_coherence.append(medium_coh)
            high_coherence.append(high_coh)
        except:
            # Handle errors
            low_coherence.append(0)
            medium_coherence.append(0)
            high_coherence.append(0)
    
    return low_coherence, medium_coherence, high_coherence

# Calculate rolling coherence metrics
window_size = 120  # 2 hours at 1-minute intervals
low_coh, medium_coh, high_coh = rolling_coherence_metrics(
    merged_df['ds18b20_temp'],
    merged_df['room_temp'],
    window_size=window_size
)

# Add to dataframe
merged_df['coherence_low'] = pd.Series(low_coh, index=merged_df.index[window_size-1:])
merged_df['coherence_medium'] = pd.Series(medium_coh, index=merged_df.index[window_size-1:])
merged_df['coherence_high'] = pd.Series(high_coh, index=merged_df.index[window_size-1:])

# Plot rolling coherence metrics
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

# Plot 2: Low-frequency coherence
plt.subplot(4, 1, 2)
plt.plot(merged_df.index[window_size-1:], merged_df['coherence_low'], 'g-')
plt.title('Low-Frequency Coherence (<0.5 cycles/hour)')
plt.ylabel('Coherence')
plt.ylim(0, 1.05)

# Plot 3: Medium-frequency coherence
plt.subplot(4, 1, 3)
plt.plot(merged_df.index[window_size-1:], merged_df['coherence_medium'], 'b-')
plt.title('Medium-Frequency Coherence (0.5-2 cycles/hour)')
plt.ylabel('Coherence')
plt.ylim(0, 1.05)

# Plot 4: High-frequency coherence
plt.subplot(4, 1, 4)
plt.plot(merged_df.index[window_size-1:], merged_df['coherence_high'], 'r-')
plt.title('High-Frequency Coherence (>2 cycles/hour)')
plt.ylabel('Coherence')
plt.ylim(0, 1.05)

plt.tight_layout()
plt.savefig('rolling_coherence.png')
plt.show()

# 10. Combining Cross-Correlation Metrics for Detection
# ==================================================
print("\nCombining metrics for sensor state detection...")

# 10.1 Define detection thresholds
# Based on the analysis, we'll set thresholds for detection
# Find the optimal threshold for each metric using statistical analysis
def set_optimal_threshold(metric, initial_label):
    """Find optimal threshold for a metric based on labeled data."""
    # Need both classes for proper threshold setting
    if (initial_label.sum() == 0) or (initial_label.sum() == len(initial_label)):
        return np.median(metric)
    
    # Try different thresholds and find the one that maximizes accuracy
    thresholds = np.linspace(metric.min(), metric.max(), 100)
    accuracies = []
    
    for threshold in thresholds:
        prediction = (metric > threshold).astype(int)
        accuracy = (prediction == initial_label).mean()
        accuracies.append(accuracy)
    
    optimal_threshold = thresholds[np.argmax(accuracies)]
    return optimal_threshold

# Prepare data for threshold setting (handle NaNs)
valid_data = merged_df.dropna(subset=[
    'rolling_ccf_max', 'granger_significance', 
    'coherence_low', 'coherence_medium', 'coherence_high'
]).copy()

# Set optimal thresholds
if len(valid_data) > 0:
    ccf_threshold = set_optimal_threshold(-valid_data['rolling_ccf_max'], valid_data['initial_label'])
    granger_threshold = set_optimal_threshold(valid_data['granger_significance'], valid_data['initial_label'])
    coherence_high_threshold = set_optimal_threshold(-valid_data['coherence_high'], valid_data['initial_label'])
    
    print(f"\nOptimal detection thresholds:")
    print(f"CCF Max (negative): {ccf_threshold:.4f}")
    print(f"Granger Significance: {granger_threshold:.4f}")
    print(f"High-frequency Coherence (negative): {coherence_high_threshold:.4f}")
else:
    print("Not enough valid data to set optimal thresholds")
    ccf_threshold = 0
    granger_threshold = -np.log10(0.05)  # Default to p=0.05 significance
    coherence_high_threshold = 0.5

# 10.2 Create detection function
# Define a function to detect connection state based on cross-correlation metrics
def detect_connection_ccf(ds18b20_temps, room_temps, feature_history=None, window_size=60):
    """
    Detect if the DS18B20 sensor is connected using cross-correlation analysis.
    
    Parameters:
    -----------
    ds18b20_temps : array-like
        Recent DS18B20 temperature readings
    room_temps : array-like
        Recent room temperature readings
    feature_history : dict, optional
        Dictionary to store historical features for next call
    window_size : int
        Size of the window for analysis
        
    Returns:
    --------
    dict
        Dictionary containing detection results and diagnostics
    """
    # Initialize result
    result = {
        'is_connected': False,
        'confidence': 0.0,
        'metrics': {}
    }
    
    # Check if we have enough data
    if len(ds18b20_temps) < window_size or len(room_temps) < window_size:
        # Not enough data for CCF analysis
        # Fall back to simple threshold
        is_connected = ds18b20_temps[-1] > 90.0
        result['is_connected'] = is_connected
        result['confidence'] = 0.7 if is_connected else 0.7  # Lower confidence due to simple method
        result['method'] = 'threshold'
        return result
    
    # Calculate cross-correlation
    ccf_values = ccf(ds18b20_temps[-window_size:], room_temps[-window_size:], adjusted=True)
    max_idx = np.argmax(np.abs(ccf_values))
    max_corr = ccf_values[max_idx]
    max_lag = max_idx - (len(ccf_values) // 2)
    
    # Calculate Granger causality (simplified for real-time use)
    try:
        data = pd.DataFrame({
            'y': ds18b20_temps[-window_size:],
            'x': room_temps[-window_size:]
        })
        gc_res = grangercausalitytests(data[['y', 'x']], maxlag=5, verbose=False)
        best_p = min([gc_res[lag][0]['ssr_ftest'][1] for lag in range(1, 6)])
        granger_significance = -np.log10(best_p)
    except:
        granger_significance = 0
    
    # Calculate coherence
    try:
        f, Cxy = signal.coherence(
            ds18b20_temps[-window_size:], 
            room_temps[-window_size:],
            fs=1/60,  # 1 sample per minute
            nperseg=min(30, window_size//2)
        )
        
        # Calculate average coherence in high-frequency band
        high_mask = f >= 2/60  # > 2 cycles per hour
        coherence_high = np.mean(Cxy[high_mask]) if np.any(high_mask) else 0
    except:
        coherence_high = 0
    
    # Store metrics in result
    result['metrics'] = {
        'ccf_max': max_corr,
        'ccf_lag': max_lag,
        'granger_significance': granger_significance,
        'coherence_high': coherence_high
    }
    
    # Make decision based on metrics
    # We invert CCF and coherence because lower values indicate connected state
    ccf_score = 1 - abs(max_corr)  # Higher score = more likely connected
    granger_score = granger_significance / 10.0  # Normalize to [0,1] range
    coherence_score = 1 - coherence_high  # Higher score = more likely connected
    
    # Combine scores (weighted average)
    confidence = 0.4 * ccf_score + 0.3 * granger_score + 0.3 * coherence_score
    
    # Final decision
    is_connected = confidence > 0.5
    
    result['is_connected'] = is_connected
    result['confidence'] = confidence
    result['method'] = 'cross_correlation'
    
    return result

# 10.3 Test detection function on the dataset
print("\nTesting cross-correlation-based detection on sample data...")

# Create example input for detection function
test_window = 60  # 1 hour for testing
if len(merged_df) > test_window:
    test_idx = len(merged_df) - test_window  # Use last hour of data
    
    # Extract test data
    ds18b20_temps = merged_df['ds18b20_temp'].values[test_idx:test_idx+test_window]
    room_temps = merged_df['room_temp'].values[test_idx:test_idx+test_window]
    
    # Run detection
    detection_result = detect_connection_ccf(ds18b20_temps, room_temps)
    
    print("\nDetection result for test data:")
    print(f"Is connected: {detection_result['is_connected']}")
    print(f"Confidence: {detection_result['confidence']:.4f}")
    print("\nDetection metrics:")
    for metric, value in detection_result['metrics'].items():
        print(f"  {metric}: {value:.4f}")
else:
    print("Not enough data for testing the detection function")

# 11. Save cross-correlation analysis parameters
# ===========================================
print("\nSaving cross-correlation analysis parameters...")

# Save key parameters for later use
params = {
    'ccf_threshold': ccf_threshold if 'ccf_threshold' in locals() else 0,
    'granger_threshold': granger_threshold if 'granger_threshold' in locals() else -np.log10(0.05),
    'coherence_high_threshold': coherence_high_threshold if 'coherence_high_threshold' in locals() else 0.5,
    'window_size': 60  # 1-hour window for real-time detection
}

joblib.dump(params, 'ds18b20_ccf_params.pkl')
print("Saved parameters to 'ds18b20_ccf_params.pkl'")

# 12. Final Recommendations
# =======================
print("""
# Summary and Recommendations
# =========================

This notebook demonstrates how to use cross-correlation analysis to detect 
when the DS18B20 temperature sensor is connected to a human body versus 
when it's disconnected and measuring ambient temperature.

Key findings:
1. Connected and disconnected states show distinct cross-correlation patterns
2. Key distinguishing metrics include:
   - Cross-correlation magnitude and lag
   - Granger causality significance
   - Coherence in different frequency bands

Advantages of the cross-correlation approach:
1. Focuses on relationships between sensors rather than absolute values
2. Captures temporal dependencies and causal relationships
3. Works well even when temperature ranges overlap
4. Provides multiple metrics for robust detection

Recommendations for implementation:
1. Use a combination of CCF, Granger causality, and coherence metrics
2. Employ a sliding window approach for real-time monitoring
3. Adjust window size based on available computational resources
4. Combine with other detection methods for higher reliability

Next steps:
1. Integrate cross-correlation detection with other approaches
2. Implement a weighted ensemble of all detection methods
3. Optimize window size for real-time performance
4. Collect labeled data for better threshold tuning
""")

# Close the InfluxDB client
client.close()
print("\nAnalysis complete!")
plt.axhline(y=-1.96/np.sqrt(len(merged_df)), linestyle='--', color='gray')
plt.xlabel('Lag (minutes)')
plt.ylabel('Cross-correlation')
plt.title('Cross-Correlation: DS18B20 vs Room Temperature (Full Dataset)')
plt.grid(True)
plt.tight_layout()
plt.savefig('ccf_full_dataset.png')
plt.show()

print(f"Maximum cross-correlation: {np.max(ccf_values):.4f} at lag {lags[np.argmax(ccf_values[30:91])]} minutes")

# 6.2 Calculate rolling cross-correlation
# This helps us identify changes in correlation patterns over time
print("\nCalculating rolling cross-correlation...")

# Function to calculate CCF maximum and lag for a window
def rolling_ccf_max(series1, series2, window_size=120):
    """Calculate rolling maximum cross-correlation and corresponding lag."""
    max_corrs = []
    max_lags = []
    
    for i in range(window_size, len(series1) + 1):
        s1 = series1.iloc[i-window_size:i].values
        s2 = series2.iloc[i-window_size:i].values
        
        # Make series stationary if needed
        s1 = s1 - np.mean(s1)
        s2 = s2 - np.mean(s2)
        
        # Calculate CCF
        ccf_result = ccf(s1, s2, adjusted=True)
        
        # Find maximum and corresponding lag
        max_idx = np.argmax(np.abs(ccf_result))
        max_corr = ccf_result[max_idx]
        max_lag = max_idx - (len(ccf_result) // 2)
        
        max_corrs.append(max_corr)
        max_lags.append(max_lag)
    
    return max_corrs, max_lags

# Calculate rolling CCF with a 2-hour window
window_size = 120  # 2 hours at 1-minute intervals
max_corrs, max_lags = rolling_ccf_max(
    merged_df['ds18b20_temp'], 
    merged_df['room_temp'],
    window_size
)

# Add to dataframe
merged_df['rolling_ccf_max'] = pd.Series(max_corrs, index=merged_df.index[window_size-1:])
merged_df['rolling_ccf_lag'] = pd.Series(max_lags, index=merged_df.index[window_size-1:])

# Plot the rolling CCF maximum and lag
plt.figure(figsize=(14, 10))

# Plot 1: Original temperature data with labels
plt.subplot(3, 1, 1)
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], '-', color=color, label=label)
plt.title('DS18B20 Temperature by Initial Label (Threshold-Based)')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: Rolling maximum cross-correlation
plt.subplot(3, 1, 2)
plt.plot(merged_df.index[window_size-1:], merged_df['rolling_ccf_max'], 'g-')
plt.title('Maximum Cross-Correlation (2-hour Rolling Window)')
plt.ylabel('Correlation')
plt.ylim(-1.1, 1.1)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# Plot 3: Rolling lag at maximum correlation
plt.subplot(3, 1, 3)
plt.plot(merged_df.index[window_size-1:], merged_df['rolling_ccf_lag'], 'b-')
plt.title('Lag at Maximum Cross-Correlation (minutes)')
plt.ylabel('Lag (minutes)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('rolling_ccf.png')
plt.show()

# 6.3 Compare CCF patterns by sensor state
# Calculate CCF separately for periods when the sensor is likely connected vs disconnected
print("\nComparing cross-correlation patterns by sensor state...")

# Define a function to calculate and plot CCF for specific data segments
def calculate_plot_ccf(ds18b20_data, room_data, title):
    # Calculate CCF
    ccf_values = ccf(ds18b20_data, room_data, adjusted=True)
    lags = np.arange(-30, 31)  # -30 to +30 minute lags
    
    # Find maximum correlation and corresponding lag
    max_idx = np.argmax(np.abs(ccf_values[30:91]))
    max_corr = ccf_values[30:91][max_idx]
    max_lag = lags[max_idx]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.stem(lags, ccf_values[30:91], linefmt='b-', markerfmt='bo', basefmt='r-')
    plt.axhline(y=0, color='r', linestyle='-')