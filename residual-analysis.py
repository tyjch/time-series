# Residual Analysis for DS18B20 Sensor State Detection
# =====================================================

# This notebook demonstrates how to use residual analysis techniques to detect 
# when the DS18B20 temperature sensor is connected or disconnected.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from influxdb_client import InfluxDBClient
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime, timedelta

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

# Set Matplotlib parameters for better readability
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12

# 1. Connect to InfluxDB
# ======================
# Replace these with your actual InfluxDB credentials
url = "https://your-influxdb-url"
token = "your-generated-token"
org = "your-organization"
bucket = "your-bucket"

print("Connecting to InfluxDB...")
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

# 2. Query sensor data
# ====================
# Define the time range for data retrieval (e.g., last 7 days)
# Adjust this based on how much data you want to analyze
time_range = "7d"

print(f"Retrieving data from the last {time_range}...")

# Query for DS18B20 temperature data
ds18b20_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "DS18B20")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
'''

# Query for Si7021 temperature data (room temperature)
si7021_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "SI7021")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
'''

# Query for RPi CPU temperature
rpi_query = f'''
    from(bucket: "{bucket}")
        |> range(start: -{time_range})
        |> filter(fn: (r) => r["_measurement"] == "RPi Zero 2W")
        |> filter(fn: (r) => r["dimension"] == "temperature")
        |> filter(fn: (r) => r["_field"] == "value")
        |> aggregateWindow(every: 5m, fn: mean, createEmpty: false)
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

# 3. Process and merge data
# =========================
# Keep only necessary columns and rename for clarity
ds18b20_df = ds18b20_result[['_time', '_value']].rename(columns={'_value': 'ds18b20_temp'})
si7021_df = si7021_result[['_time', '_value']].rename(columns={'_value': 'room_temp'})
rpi_df = rpi_result[['_time', '_value']].rename(columns={'_value': 'cpu_temp'})

# Make time the index for easier merging
ds18b20_df.set_index('_time', inplace=True)
si7021_df.set_index('_time', inplace=True)
rpi_df.set_index('_time', inplace=True)

# Resample data to ensure consistent timestamps (5-minute intervals)
ds18b20_df = ds18b20_df.resample('5T').mean()
si7021_df = si7021_df.resample('5T').mean()
rpi_df = rpi_df.resample('5T').mean()

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
plt.savefig('temperature_time_series.png')
plt.show()

# Correlation analysis
correlation = merged_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Between Temperature Sensors')
plt.tight_layout()
plt.savefig('temperature_correlation.png')
plt.show()

print("\nCorrelation matrix:")
print(correlation)

# 5. Residual Analysis Approach
# =============================
print("\nStarting residual analysis...")

# EXPLANATION: Residual Analysis Theory
# ====================================
# 
# The core idea of residual analysis for sensor state detection is:
# 
# 1. Build a model that predicts DS18B20 readings based on room temperature and CPU temperature
#    when the sensor is DISCONNECTED (ambient mode)
# 
# 2. When the sensor is CONNECTED to a body, its temperature will deviate from this model's
#    predictions, resulting in large residuals (errors)
# 
# 3. By monitoring these residuals, we can detect the sensor state:
#    - Small residuals → sensor is disconnected (follows ambient temperature model)
#    - Large positive residuals → sensor is connected (warmer than ambient model predicts)

# 5.1. Visualization to identify connected vs disconnected periods
# The first step is to visually inspect the data to find periods when the sensor
# was likely connected vs disconnected

plt.figure(figsize=(14, 8))
plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
plt.plot(merged_df.index, merged_df['room_temp'], 'g-', label='Room (Si7021)')
plt.title('DS18B20 vs Room Temperature')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.axhline(y=90, color='r', linestyle='--', label='90°F Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('ds18b20_vs_room.png')
plt.show()

# Based on the visualization, we can roughly identify connected vs disconnected periods
# Let's assume temperatures above 90°F indicate the sensor is likely connected to a body
# (adjust this threshold based on your visual inspection)

threshold = 90.0
merged_df['likely_connected'] = merged_df['ds18b20_temp'] > threshold

print(f"\nIdentified {merged_df['likely_connected'].sum()} data points where sensor is likely connected")
print(f"Identified {(~merged_df['likely_connected']).sum()} data points where sensor is likely disconnected")

# 5.2. Build a regression model for DISCONNECTED state
# We'll use data points where the sensor is likely disconnected to train our model

# Create training data from periods when the sensor is likely disconnected
disconnected_df = merged_df[~merged_df['likely_connected']].copy()

if len(disconnected_df) < 10:  # Need sufficient data for modeling
    print("Warning: Not enough data points identified as disconnected for modeling.")
    # You might need to adjust your threshold or use a different period of data
else:
    # Prepare features (X) and target (y) for the disconnected state model
    X_disconnected = disconnected_df[['room_temp', 'cpu_temp']]
    y_disconnected = disconnected_df['ds18b20_temp']
    
    # Fit linear regression model
    model = LinearRegression()
    model.fit(X_disconnected, y_disconnected)
    
    # Model coefficients and statistics
    print("\nDisconnected State Model:")
    print(f"Intercept: {model.intercept_:.4f}")
    print(f"Coefficients: Room Temp: {model.coef_[0]:.4f}, CPU Temp: {model.coef_[1]:.4f}")
    
    # Predict DS18B20 temperatures using the model
    disconnected_df['predicted_temp'] = model.predict(X_disconnected)
    
    # Calculate residuals for the training data
    disconnected_df['residuals'] = disconnected_df['ds18b20_temp'] - disconnected_df['predicted_temp']
    
    # Model performance metrics
    r2 = r2_score(y_disconnected, disconnected_df['predicted_temp'])
    rmse = np.sqrt(mean_squared_error(y_disconnected, disconnected_df['predicted_temp']))
    print(f"Model R-squared: {r2:.4f}")
    print(f"Model RMSE: {rmse:.4f}°F")
    
    # Visualize model fit for disconnected periods
    plt.figure(figsize=(14, 8))
    plt.plot(disconnected_df.index, disconnected_df['ds18b20_temp'], 'b-', label='Actual DS18B20')
    plt.plot(disconnected_df.index, disconnected_df['predicted_temp'], 'r-', label='Predicted DS18B20')
    plt.title('Disconnected State Model: Actual vs Predicted')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('disconnected_model_fit.png')
    plt.show()
    
    # Visualize residuals for disconnected periods
    plt.figure(figsize=(14, 6))
    plt.plot(disconnected_df.index, disconnected_df['residuals'], 'g-')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals for Disconnected State Model')
    plt.ylabel('Residual (°F)')
    plt.tight_layout()
    plt.savefig('disconnected_residuals.png')
    plt.show()
    
    # Calculate residual statistics for the disconnected state
    residual_mean = disconnected_df['residuals'].mean()
    residual_std = disconnected_df['residuals'].std()
    print(f"\nResidual statistics for disconnected state:")
    print(f"Mean: {residual_mean:.4f}°F")
    print(f"Standard Deviation: {residual_std:.4f}°F")
    
    # 5.3. Apply the model to ALL data and analyze residuals
    # This will show how residuals behave differently when the sensor is connected
    
    # Prepare features for all data
    X_all = merged_df[['room_temp', 'cpu_temp']]
    
    # Predict temperatures for all data using the disconnected state model
    merged_df['predicted_temp'] = model.predict(X_all)
    
    # Calculate residuals for all data
    merged_df['residuals'] = merged_df['ds18b20_temp'] - merged_df['predicted_temp']
    
    # Visualize residuals for all data, colored by likely connection state
    plt.figure(figsize=(14, 8))
    
    # Plot all residuals
    plt.scatter(merged_df.index[~merged_df['likely_connected']], 
                merged_df.loc[~merged_df['likely_connected'], 'residuals'],
                c='blue', alpha=0.6, label='Likely Disconnected')
    
    plt.scatter(merged_df.index[merged_df['likely_connected']], 
                merged_df.loc[merged_df['likely_connected'], 'residuals'],
                c='red', alpha=0.6, label='Likely Connected')
    
    # Add reference lines
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Calculate control limits (e.g., ±3 standard deviations of disconnected residuals)
    control_limit_upper = residual_mean + 3 * residual_std
    control_limit_lower = residual_mean - 3 * residual_std
    
    plt.axhline(y=control_limit_upper, color='g', linestyle='--', 
                label=f'Upper Control Limit (+3σ): {control_limit_upper:.2f}°F')
    plt.axhline(y=control_limit_lower, color='g', linestyle='--', 
                label=f'Lower Control Limit (-3σ): {control_limit_lower:.2f}°F')
    
    plt.title('Residuals Analysis: DS18B20 Connection Detection')
    plt.ylabel('Residual (°F)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('all_residuals_analysis.png')
    plt.show()
    
    # 5.4. Develop a CUSUM (Cumulative Sum) Control Chart
    # This is a powerful tool for detecting small but persistent shifts in the residuals
    
    # Initialize CUSUM variables
    k = 0.5 * residual_std  # Reference value (often set to half of the standard deviation)
    h = 5 * residual_std    # Decision threshold
    
    # Calculate CUSUM values
    merged_df['cusum_plus'] = 0.0   # Cumulative sum for detecting positive shifts
    merged_df['cusum_minus'] = 0.0  # Cumulative sum for detecting negative shifts
    
    # First values
    merged_df.loc[merged_df.index[0], 'cusum_plus'] = max(0, merged_df.loc[merged_df.index[0], 'residuals'] - k)
    merged_df.loc[merged_df.index[0], 'cusum_minus'] = max(0, -merged_df.loc[merged_df.index[0], 'residuals'] - k)
    
    # Calculate remaining CUSUM values
    for i in range(1, len(merged_df)):
        # CUSUM+ detects upward shifts (connected state - higher temperatures)
        merged_df.loc[merged_df.index[i], 'cusum_plus'] = max(
            0, 
            merged_df.loc[merged_df.index[i-1], 'cusum_plus'] + merged_df.loc[merged_df.index[i], 'residuals'] - k
        )
        
        # CUSUM- detects downward shifts (rarely useful for this application, but included for completeness)
        merged_df.loc[merged_df.index[i], 'cusum_minus'] = max(
            0, 
            merged_df.loc[merged_df.index[i-1], 'cusum_minus'] - merged_df.loc[merged_df.index[i], 'residuals'] - k
        )
    
    # Define connection state based on CUSUM
    # If CUSUM+ exceeds threshold h, sensor is likely connected
    merged_df['cusum_connected'] = merged_df['cusum_plus'] > h
    
    # Visualize CUSUM chart
    plt.figure(figsize=(14, 10))
    
    # Plot CUSUM+ values
    plt.subplot(2, 1, 1)
    plt.plot(merged_df.index, merged_df['cusum_plus'], 'b-', label='CUSUM+')
    plt.axhline(y=h, color='r', linestyle='--', label=f'Threshold (h={h:.2f})')
    plt.title('CUSUM+ Chart for Detecting Connected State')
    plt.ylabel('CUSUM+ Value')
    plt.legend()
    
    # Plot original DS18B20 temperatures with CUSUM-based classification
    plt.subplot(2, 1, 2)
    plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20 Temperature')
    plt.plot(merged_df.index, merged_df['room_temp'], 'g-', label='Room Temperature', alpha=0.5)
    
    # Color the background based on detected connection state
    for i in range(len(merged_df)-1):
        if merged_df['cusum_connected'].iloc[i]:
            plt.axvspan(merged_df.index[i], merged_df.index[i+1], alpha=0.2, color='red')
    
    plt.title('DS18B20 Temperature with Connection State Detection')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cusum_analysis.png')
    plt.show()
    
    # 5.5. Compare Threshold-Based vs. CUSUM Detection
    
    # Calculate agreement between simple threshold and CUSUM methods
    agreement = (merged_df['likely_connected'] == merged_df['cusum_connected']).mean() * 100
    print(f"\nAgreement between threshold-based and CUSUM detection: {agreement:.2f}%")
    
    # Confusion matrix-like comparison
    confusion = pd.crosstab(
        merged_df['likely_connected'], 
        merged_df['cusum_connected'],
        rownames=['Threshold'],
        colnames=['CUSUM']
    )
    
    print("\nComparison of detection methods:")
    print(confusion)
    
    # 6. Implement Detection Function
    # ==============================
    
    # This function can be used to detect the sensor state for new data
    def detect_ds18b20_connection(ds18b20_temp, room_temp, cpu_temp, 
                                 model=model, k=k, h=h, cusum_plus_prev=0):
        """
        Detect if the DS18B20 sensor is connected using the residual-based CUSUM approach.
        
        Parameters:
        -----------
        ds18b20_temp : float
            Current DS18B20 temperature reading (°F)
        room_temp : float
            Current room temperature from Si7021 (°F)
        cpu_temp : float
            Current CPU temperature (°F)
        model : sklearn.linear_model
            Trained regression model for disconnected state
        k : float
            CUSUM reference value
        h : float
            CUSUM decision threshold
        cusum_plus_prev : float
            Previous CUSUM+ value (for continuous monitoring)
            
        Returns:
        --------
        dict
            Dictionary containing detection results and diagnostics
        """
        # Predict expected DS18B20 temperature if it were disconnected
        predicted_temp = model.predict([[room_temp, cpu_temp]])[0]
        
        # Calculate residual
        residual = ds18b20_temp - predicted_temp
        
        # Update CUSUM+ value
        cusum_plus = max(0, cusum_plus_prev + residual - k)
        
        # Determine connection state
        is_connected = cusum_plus > h
        
        # Prepare results
        results = {
            'ds18b20_temp': ds18b20_temp,
            'predicted_temp': predicted_temp,
            'residual': residual,
            'cusum_plus': cusum_plus,
            'is_connected': is_connected,
            'confidence': min(1.0, cusum_plus / h) if is_connected else min(1.0, (h - cusum_plus) / h)
        }
        
        return results
    
    # Example of using the detection function with the last data point
    last_point = merged_df.iloc[-1]
    detection_result = detect_ds18b20_connection(
        last_point['ds18b20_temp'],
        last_point['room_temp'],
        last_point['cpu_temp'],
        model=model,
        k=k,
        h=h,
        cusum_plus_prev=merged_df['cusum_plus'].iloc[-2] if len(merged_df) > 1 else 0
    )
    
    print("\nDetection result for the most recent data point:")
    for key, value in detection_result.items():
        print(f"{key}: {value}")

# 7. Save parameters for later use
# ===============================
import joblib

# Only save if we successfully created a model
if 'model' in locals():
    # Save model and parameters
    model_params = {
        'model': model,
        'residual_mean': residual_mean,
        'residual_std': residual_std,
        'k': k,
        'h': h
    }
    
    joblib.dump(model_params, 'ds18b20_residual_model.pkl')
    print("\nSaved model and parameters to 'ds18b20_residual_model.pkl'")

# 8. Final Recommendations
# =======================
print("""
# Summary and Recommendations
# ==========================

This notebook demonstrates how to use residual analysis to detect when the DS18B20 temperature
sensor is connected to a human body versus when it's disconnected and measuring ambient temperature.

Key findings:
1. The disconnected DS18B20 temperature can be modeled as a function of room temperature and CPU temperature
2. When connected to a body, the sensor's temperature deviates significantly from this model
3. CUSUM (Cumulative Sum) control charts provide a robust way to detect this deviation

Recommendations for implementation:
1. In production, implement the 'detect_ds18b20_connection' function with the trained model
2. Periodically recalibrate the model using known disconnected periods
3. Consider using a moving window for CUSUM to prevent long-term drift
4. Add additional features like temperature derivatives (rate of change) to improve accuracy

Next steps:
1. Evaluate detection accuracy with proper ground truth data
2. Explore other detection methods (Hidden Markov Models, thermal response dynamics)
3. Implement online learning to adapt the model over time
""")

# Close the InfluxDB client
client.close()
print("\nAnalysis complete!")
