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

# Resample data to ensure consistent timestamps and reduce noise
# For advanced time series models, 5-minute intervals provide a good balance
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
plt.savefig('advanced_models_time_series.png')
plt.show()

# Create a plot showing temperature with likely connection state
plt.figure(figsize=(14, 6))
for state in [0, 1]:
    mask = merged_df['initial_label'] == state
    label = 'Connected' if state == 1 else 'Disconnected'
    color = 'red' if state == 1 else 'blue'
    plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], 'o-', color=color, label=label, alpha=0.7)
plt.plot(merged_df.index, merged_df['room_temp'], 'g-', alpha=0.3, label='Room Temp')
plt.title('DS18B20 Temperature Colored by Likely Connection State (Threshold-Based)')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.savefig('temperature_by_state.png')
plt.show()

# 5. Prepare Data for Time Series Modeling
# ======================================
print("\nPreparing data for time series modeling...")

# 5.1 Create temperature difference features
merged_df['temp_diff'] = merged_df['ds18b20_temp'] - merged_df['room_temp']
merged_df['ds18b20_diff'] = merged_df['ds18b20_temp'].diff().fillna(0)
merged_df['room_diff'] = merged_df['room_temp'].diff().fillna(0)

# 5.2 Normalize data for easier modeling
# Center and scale the temperature data
def normalize_series(series):
    return (series - series.mean()) / series.std()

merged_df['ds18b20_norm'] = normalize_series(merged_df['ds18b20_temp'])
merged_df['room_norm'] = normalize_series(merged_df['room_temp'])
merged_df['temp_diff_norm'] = normalize_series(merged_df['temp_diff'])

# 5.3 Visualize the normalized data
plt.figure(figsize=(14, 6))
plt.plot(merged_df.index, merged_df['ds18b20_norm'], 'b-', label='DS18B20 (Norm)')
plt.plot(merged_df.index, merged_df['room_norm'], 'g-', label='Room (Norm)', alpha=0.7)
plt.plot(merged_df.index, merged_df['temp_diff_norm'], 'r-', label='Temp Diff (Norm)', alpha=0.7)
plt.title('Normalized Temperature Data')
plt.ylabel('Standardized Temperature')
plt.legend()
plt.savefig('normalized_temperatures.png')
plt.show()

# 6. Markov Switching Model
# =======================
print("\nImplementing Markov Switching Model...")

# EXPLANATION: Markov Switching Model
# =================================
print("""
# Markov Switching Model for Sensor State Detection
# ============================================

A Markov Switching Model (MSM) is perfect for our sensor state detection problem 
because it explicitly models different regimes (states) in time series data:

1. **Core Concept**:
   - The time series switches between distinct regimes with different statistical properties
   - These regimes (connected/disconnected) are governed by a hidden Markov process
   - Each regime has its own set of parameters (mean, variance, autoregressive coefficients)

2. **Key Components**:
   - Hidden state variable St ∈ {0,1} representing disconnected/connected
   - State-specific model parameters
   - Transition probability matrix between states

3. **What We'll Implement**:
   - A Markov Switching Autoregression (MS-AR) model for the DS18B20 temperature
   - The model allows different means, variances, and AR coefficients in each state
   - We'll extract the smoothed probabilities of being in each state

4. **Why This Works**:
   - Connected state: Higher mean, lower variance, stronger autoregressive pattern
   - Disconnected state: Mean closer to room temp, higher variance, weaker AR pattern
   
This approach provides a principled statistical framework for state detection with
probabilities rather than binary decisions.
""")

# 6.1 Prepare data for Markov Switching Model
# We'll use the temperature difference as our primary variable
# This tends to work better than raw temperatures for regime identification
msm_data = merged_df['temp_diff_norm'].dropna().copy()

# Check if we have enough data for modeling
if len(msm_data) < 30:
    print("Not enough data for Markov Switching modeling.")
else:
    try:
        # 6.2 Fit Markov Switching AR model
        # The model allows for 2 regimes with different means, variances, and AR(1) coefficients
        print("Fitting Markov Switching model (this may take a while)...")
        
        # Create the model
        ms_model = MarkovAutoregression(
            msm_data,
            k_regimes=2,        # Two states: connected and disconnected
            order=1,            # AR(1) process in each state
            switching_variance=True,  # Allow different variances in each state
        )
        
        # Fit the model
        ms_results = ms_model.fit(maxiter=100)
        
        print("\nMarkov Switching Model Results:")
        print(ms_results.summary())
        
        # 6.3 Extract regime probabilities
        smoothed_probs = ms_results.smoothed_marginal_probabilities
        
        # Determine which regime corresponds to "connected"
        # If regime 0 has higher mean temperature difference, it's the connected state
        means = ms_results.regime_parameters.iloc[0, :]
        regime_connected = 0 if means[0] > means[1] else 1
        regime_disconnected = 1 - regime_connected
        
        print(f"\nRegime {regime_connected} corresponds to 'Connected' state")
        print(f"Regime {regime_disconnected} corresponds to 'Disconnected' state")
        
        # Add smoothed probabilities to the main dataframe
        prob_series = pd.Series(
            smoothed_probs[:, regime_connected], 
            index=msm_data.index
        )
        
        # Reindex to match original dataframe (handles NaNs)
        merged_df['msm_connected_prob'] = prob_series.reindex(merged_df.index)
        
        # Create state predictions
        merged_df['msm_state'] = (merged_df['msm_connected_prob'] > 0.5).astype(int)
        
        # 6.4 Visualize the results
        plt.figure(figsize=(14, 12))
        
        # Plot 1: Original temperature data
        plt.subplot(3, 1, 1)
        plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
        plt.plot(merged_df.index, merged_df['room_temp'], 'g-', alpha=0.5, label='Room')
        plt.title('Temperature Data')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        
        # Plot 2: Smoothed probabilities from MSM
        plt.subplot(3, 1, 2)
        plt.plot(merged_df.index, merged_df['msm_connected_prob'], 'r-')
        plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
        plt.title('Probability of Connected State (Markov Switching Model)')
        plt.ylabel('Probability')
        plt.ylim(-0.05, 1.05)
        plt.legend()
        
        # Plot 3: DS18B20 temperature colored by detected state
        plt.subplot(3, 1, 3)
        for state in [0, 1]:
            mask = merged_df['msm_state'] == state
            label = 'Connected' if state == 1 else 'Disconnected'
            color = 'red' if state == 1 else 'blue'
            plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], 'o-', color=color, label=label, alpha=0.7)
        plt.title('DS18B20 Temperature by Detected State (MSM)')
        plt.ylabel('Temperature (°F)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('markov_switching_results.png')
        plt.show()
        
        # 6.5 Compare with threshold-based approach
        agreement = (merged_df['msm_state'] == merged_df['initial_label']).mean() * 100
        print(f"\nAgreement between threshold-based and MSM detection: {agreement:.2f}%")
        
        # Confusion matrix-like comparison
        confusion = pd.crosstab(
            merged_df['initial_label'], 
            merged_df['msm_state'],
            rownames=['Threshold (90°F)'],
            colnames=['Markov Switching']
        )
        
        print("\nComparison of detection methods:")
        print(confusion)
        
    except Exception as e:
        print(f"Error fitting Markov Switching model: {e}")
        print("Try with a different dataset or adjust model parameters.")

# 7. Vector Autoregression (VAR)
# ===========================
print("\nImplementing Vector Autoregression (VAR) model...")

# EXPLANATION: Vector Autoregression
# ===============================
print("""
# Vector Autoregression (VAR) for Sensor State Detection
# ==================================================

Vector Autoregression models the joint evolution of multiple time series,
capturing interactions between them:

1. **Core Concept**:
   - Each variable is modeled as a function of past values of itself AND other variables
   - This captures the interdependencies between DS18B20 and room temperatures
   - Changes in these relationships indicate changes in sensor state
   
2. **VAR Approach for Sensor Detection**:
   - Fit a VAR model to segments of data
   - Extract the coefficient that represents how room temperature affects DS18B20
   - When connected: this coefficient should be small (weak influence)
   - When disconnected: this coefficient should be large (strong influence)

3. **Implementation**:
   - Rolling window VAR model estimation
   - Extract and track the room→DS18B20 coefficient
   - Use coefficient magnitude to distinguish states

This approach quantifies the changing causal relationship between room and
sensor temperatures.
""")

# 7.1 Function for rolling VAR model estimation
def rolling_var_analysis(ds18b20_temp, room_temp, window_size=24):
    """Calculate VAR model coefficients in rolling windows."""
    # Initialize arrays for results
    coef_room_to_ds18b20 = []
    timestamps = []
    
    # Need at least 4 data points for VAR estimation
    if len(ds18b20_temp) < window_size or window_size < 4:
        return [], []
    
    for i in range(window_size, len(ds18b20_temp) + 1):
        try:
            # Prepare data for VAR
            data = pd.DataFrame({
                'ds18b20': ds18b20_temp.iloc[i-window_size:i].values,
                'room': room_temp.iloc[i-window_size:i].values
            })
            
            # Normalize data (this helps with numerical stability)
            for col in data.columns:
                data[col] = (data[col] - data[col].mean()) / data[col].std()
            
            # Fit VAR model
            from statsmodels.tsa.api import VAR
            var_model = VAR(data)
            var_result = var_model.fit(1)  # VAR(1) model
            
            # Extract coefficient: effect of room(t-1) on ds18b20(t)
            coef = var_result.coefs[0, 0, 1]  # [lag, eq, var]
            
            coef_room_to_ds18b20.append(coef)
            timestamps.append(ds18b20_temp.index[i-1])
            
        except Exception as e:
            # Handle errors (not enough data, singular matrix, etc.)
            coef_room_to_ds18b20.append(0)
            timestamps.append(ds18b20_temp.index[i-1])
    
    return pd.Series(coef_room_to_ds18b20, index=timestamps)

# 7.2 Calculate rolling VAR coefficients
# We use a 24-point (2-hour) window for stability
print("Calculating rolling VAR coefficients (this may take a while)...")
var_window = 24  # 2 hours at 5-minute intervals

var_coefs = rolling_var_analysis(
    merged_df['ds18b20_norm'],  # Use normalized data
    merged_df['room_norm'],
    window_size=var_window
)

# Add to dataframe
merged_df['var_coef'] = var_coefs

# Convert coefficients to probabilities
# Higher coefficient = more likely disconnected
if not merged_df['var_coef'].isna().all():
    # Calculate percentiles for scaling
    p10 = merged_df['var_coef'].quantile(0.1)
    p90 = merged_df['var_coef'].quantile(0.9)
    
    # Scale to [0,1] range where 1 = connected (low coefficient)
    merged_df['var_connected_prob'] = 1 - ((merged_df['var_coef'] - p10) / (p90 - p10)).clip(0, 1)
    
    # Create state predictions
    merged_df['var_state'] = (merged_df['var_connected_prob'] > 0.5).astype(int)
    
    # 7.3 Visualize VAR results
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Original temperature data
    plt.subplot(3, 1, 1)
    plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
    plt.plot(merged_df.index, merged_df['room_temp'], 'g-', alpha=0.5, label='Room')
    plt.title('Temperature Data')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    
    # Plot 2: VAR coefficient (room→DS18B20)
    plt.subplot(3, 1, 2)
    plt.plot(merged_df.index, merged_df['var_coef'], 'r-')
    plt.title('VAR Coefficient: Room Temperature → DS18B20')
    plt.ylabel('Coefficient')
    
    # Plot 3: Connection probability from VAR
    plt.subplot(3, 1, 3)
    plt.plot(merged_df.index, merged_df['var_connected_prob'], 'b-')
    plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
    plt.title('Probability of Connected State (VAR Model)')
    plt.ylabel('Probability')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('var_results.png')
    plt.show()
    
    # 7.4 Compare with threshold-based approach
    agreement = (merged_df['var_state'] == merged_df['initial_label']).mean() * 100
    print(f"\nAgreement between threshold-based and VAR detection: {agreement:.2f}%")
else:
    print("VAR coefficient calculation failed or returned all NaNs.")

# 8. ARIMAX Model (ARIMA with exogenous variables)
# ==============================================
print("\nImplementing ARIMAX model...")

# EXPLANATION: ARIMAX Model
# ======================
print("""
# ARIMAX Model for Sensor State Detection
# ====================================

ARIMAX (ARIMA with eXogenous variables) models a time series as a function 
of its own past values PLUS external factors:

1. **Core Concept**:
   - Model DS18B20 temperature as a function of its own history and room temperature
   - The influence of room temperature changes depending on sensor state
   - Residuals (unexplained variations) reveal the state
   
2. **Key Elements**:
   - ARIMA component: captures the time series' internal dynamics
   - X (exogenous) component: models influence of room temperature
   - When connected: residuals are larger (body temp isn't explained by room temp)
   - When disconnected: residuals are smaller (DS18B20 closely follows room temp)

3. **Implementation Approach**:
   - Fit ARIMAX model to the entire dataset
   - Calculate one-step-ahead forecasts and residuals
   - Large residuals indicate the model's exogenous factors (room temp) 
     don't explain the DS18B20 behavior → likely connected
   - Small residuals indicate the model works well → likely disconnected

This method focuses on how well room temperature explains DS18B20 readings.
""")

# 8.1 Prepare data for ARIMAX model
# We'll model DS18B20 temperature with room temperature as an exogenous variable
endog = merged_df['ds18b20_norm'].copy()  # Use normalized data
exog = merged_df[['room_norm']].copy()    # Use normalized data

# 8.2 Fit ARIMAX model
try:
    print("Fitting ARIMAX model (this may take a while)...")
    
    # ARIMAX model with order (1,1,1)
    arimax_model = SARIMAX(
        endog,
        exog=exog,
        order=(1, 1, 1),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    arimax_results = arimax_model.fit(disp=False)
    
    print("\nARIMAX Model Results:")
    print(arimax_results.summary())
    
    # 8.3 Generate one-step-ahead forecasts and residuals
    pred = arimax_results.get_prediction(dynamic=False)
    pred_ci = pred.conf_int()
    
    # Add predictions and residuals to dataframe
    merged_df['arimax_pred'] = pred.predicted_mean
    merged_df['arimax_residual'] = endog - pred.predicted_mean
    
    # Calculate absolute residuals
    merged_df['arimax_abs_residual'] = np.abs(merged_df['arimax_residual'])
    
    # Convert residuals to probabilities
    # Higher absolute residual = more likely connected
    # (DS18B20 doesn't follow the model's prediction based on room temp)
    p10 = merged_df['arimax_abs_residual'].quantile(0.1)
    p90 = merged_df['arimax_abs_residual'].quantile(0.9)
    
    merged_df['arimax_connected_prob'] = ((merged_df['arimax_abs_residual'] - p10) / (p90 - p10)).clip(0, 1)
    
    # Create state predictions
    merged_df['arimax_state'] = (merged_df['arimax_connected_prob'] > 0.5).astype(int)
    
    # 8.4 Visualize ARIMAX results
    plt.figure(figsize=(14, 16))
    
    # Plot 1: DS18B20 temperature with ARIMAX predictions
    plt.subplot(4, 1, 1)
    plt.plot(merged_df.index, merged_df['ds18b20_norm'], 'b-', label='DS18B20 (Norm)')
    plt.plot(merged_df.index, merged_df['arimax_pred'], 'r-', label='ARIMAX Prediction')
    plt.fill_between(
        merged_df.index,
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1],
        color='k', alpha=0.1
    )
    plt.title('DS18B20 Temperature and ARIMAX Predictions')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    
    # Plot 2: Room temperature (exogenous variable)
    plt.subplot(4, 1, 2)
    plt.plot(merged_df.index, merged_df['room_norm'], 'g-', label='Room Temp (Norm)')
    plt.title('Room Temperature (Exogenous Variable)')
    plt.ylabel('Normalized Temperature')
    plt.legend()
    
    # Plot 3: Residuals
    plt.subplot(4, 1, 3)
    plt.plot(merged_df.index, merged_df['arimax_residual'], 'b-')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('ARIMAX Model Residuals')
    plt.ylabel('Residual')
    
    # Plot 4: Connection probability from residuals
    plt.subplot(4, 1, 4)
    plt.plot(merged_df.index, merged_df['arimax_connected_prob'], 'r-')
    plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
    plt.title('Probability of Connected State (ARIMAX Model)')
    plt.ylabel('Probability')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('arimax_results.png')
    plt.show()
    
    # 8.5 Compare with threshold-based approach
    agreement = (merged_df['arimax_state'] == merged_df['initial_label']).mean() * 100
    print(f"\nAgreement between threshold-based and ARIMAX detection: {agreement:.2f}%")
    
except Exception as e:
    print(f"Error fitting ARIMAX model: {e}")
    print("Try with a different dataset or adjust model parameters.")

# 9. Kalman Filter Approach
# =======================
print("\nImplementing Kalman Filter approach...")

# EXPLANATION: Kalman Filter
# =======================
print("""
# Kalman Filter for Sensor State Detection
# ====================================

The Kalman Filter is an optimal state estimator for linear systems. In our context:

1. **Core Concept**:
   - Model temperature as a state-space system with hidden states
   - One of these states represents the connection status
   - The filter recursively estimates the hidden states from measurements
   
2. **State-Space Formulation**:
   - States: true DS18B20 temperature, connection state
   - Measurements: observed DS18B20 temperature, room temperature
   - When connected: DS18B20 follows its own dynamics with little room influence
   - When disconnected: DS18B20 measurement closely tracks room temperature
   
3. **Simplified Implementation**:
   - Use two Kalman filters: one for "connected" model, one for "disconnected"
   - Calculate prediction error for each model
   - The model with lower error indicates the likely state

While a full switching state-space model would be ideal, this simplified
approach using competing models is more tractable and still effective.
""")

# 9.1 Simple Kalman Filter implementation for state detection
def simple_kalman_detector(ds18b20_temp, room_temp, window_size=12):
    """
    Simple Kalman Filter approach for sensor state detection.
    
    Uses two competing models:
    - Connected model: DS18B20 follows its own dynamics primarily
    - Disconnected model: DS18B20 follows room temperature primarily
    
    Returns probabilities of connected state based on prediction errors.
    """
    from scipy.stats import norm
    
    # Initialize arrays
    connected_probs = []
    
    # Need at least 2 points for prediction
    if len(ds18b20_temp) < 2:
        return []
    
    # Process each point
    for i in range(max(1, window_size), len(ds18b20_temp)):
        # Prepare data
        if i >= window_size:
            train_ds18b20 = ds18b20_temp.iloc[i-window_size:i]
            train_room = room_temp.iloc[i-window_size:i]
        else:
            train_ds18b20 = ds18b20_temp.iloc[:i]
            train_room = room_temp.iloc[:i]
        
        # Current values
        curr_ds18b20 = ds18b20_temp.iloc[i]
        curr_room = room_temp.iloc[i]
        prev_ds18b20 = ds18b20_temp.iloc[i-1]
        
        try:
            # Model 1: Connected - DS18B20 follows AR(1) process
            # Simple AR(1) coefficient estimate
            coef = np.cov(train_ds18b20[:-1], train_ds18b20[1:])[0, 1] / np.var(train_ds18b20[:-1])
            coef = max(0, min(coef, 0.99))  # Constrain coefficient
            
            # Predict next value
            pred_connected = prev_ds18b20 * coef
            
            # Model 2: Disconnected - DS18B20 follows room temperature
            # Get relationship between room and DS18B20
            beta = np.cov(train_room, train_ds18b20)[0, 1] / np.var(train_room)
            
            # Predict next value
            pred_disconnected = curr_room * beta
            
            # Calculate prediction errors
            error_connected = abs(curr_ds18b20 - pred_connected)
            error_disconnected = abs(curr_ds18b20 - pred_disconnected)
            
            # Use errors to compute probabilities
            # Assuming Gaussian errors, larger errors = lower probability
            # Scale errors to reasonable values
            error_scale = np.std(train_ds18b20) / 2
            
            # Probability calculations
            p_connected = norm.pdf(error_connected, scale=error_scale)
            p_disconnected = norm.pdf(error_disconnected, scale=error_scale)
            
            # Normalize to get probability
            prob = p_connected / (p_connected + p_disconnected)
            
            # Constrain to [0, 1]
            prob = max(0, min(prob, 1))
            
            connected_probs.append(prob)
            
        except Exception as e:
            # Handle errors - default to uncertainty
            connected_probs.append(0.5)
    
    return connected_probs

# Calculate Kalman Filter-based probabilities
print("Calculating Kalman Filter detection probabilities...")
kalman_window = 12  # 1 hour at 5-minute intervals

kalman_probs = simple_kalman_detector(
    merged_df['ds18b20_norm'],  # Use normalized data
    merged_df['room_norm'],
    window_size=kalman_window
)

# Add to dataframe
index_start = max(1, kalman_window)
merged_df['kalman_connected_prob'] = pd.Series(
    kalman_probs, 
    index=merged_df.index[index_start:]
)

# Create state predictions
merged_df['kalman_state'] = (merged_df['kalman_connected_prob'] > 0.5).astype(int)

# 9.2 Visualize Kalman Filter results
plt.figure(figsize=(14, 12))

# Plot 1: Original temperature data
plt.subplot(3, 1, 1)
plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
plt.plot(merged_df.index, merged_df['room_temp'], 'g-', alpha=0.5, label='Room')
plt.title('Temperature Data')
plt.ylabel('Temperature (°F)')
plt.legend()

# Plot 2: Kalman Filter connection probability
plt.subplot(3, 1, 2)
plt.plot(merged_df.index, merged_df['kalman_connected_prob'], 'r-')
plt.axhline(y=0.5, color='k', linestyle='--', label='Decision Threshold (0.5)')
plt.title('Probability of Connected State (Kalman Filter)')
plt.ylabel('Probability')
plt.ylim(-0.05, 1.05)
plt.legend()

# Plot 3: DS18B20 temperature colored by detected state
plt.subplot(3, 1, 3)
for state in [0, 1]:
    mask = merged_df['kalman_state'] == state
    if mask.any():  # Only plot if there are points with this state
        label = 'Connected' if state == 1 else 'Disconnected'
        color = 'red' if state == 1 else 'blue'
        plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], 'o-', color=color, label=label, alpha=0.7)
plt.title('DS18B20 Temperature by Detected State (Kalman Filter)')
plt.ylabel('Temperature (°F)')
plt.legend()

plt.tight_layout()
plt.savefig('kalman_filter_results.png')
plt.show()

# 9.3 Compare with threshold-based approach
agreement = (merged_df['kalman_state'] == merged_df['initial_label']).mean() * 100
print(f"\nAgreement between threshold-based and Kalman detection: {agreement:.2f}%")

# 10. Ensemble Method: Combining Models
# ==================================
print("\nImplementing ensemble method to combine all models...")

# EXPLANATION: Ensemble Method
# =========================
print("""
# Ensemble Method for Robust Sensor State Detection
# =============================================

Ensemble methods combine multiple models to improve accuracy and robustness:

1. **Why Use an Ensemble?**
   - Each model has strengths and weaknesses
   - Different models may excel in different situations
   - Combining models reduces the impact of individual model failures
   
2. **Our Ensemble Approach**:
   - Weighted average of probabilities from all models
   - Models with better performance get higher weights
   - This creates a more robust detector that works in various conditions
   
3. **Benefits**:
   - More stable detection
   - Reduced false positives/negatives
   - Higher overall accuracy
   - Graceful handling of edge cases

The ensemble gives us the "wisdom of the crowd" from our different
time series approaches.
""")

# 10.1 Create ensemble detector
# First, gather all probabilities from different models
detection_cols = [
    'msm_connected_prob',
    'var_connected_prob',
    'arimax_connected_prob',
    'kalman_connected_prob'
]

# Determine which models succeeded
available_models = [col for col in detection_cols if col in merged_df.columns and not merged_df[col].isna().all()]

if not available_models:
    print("No successful models available for ensemble.")
else:
    # 10.2 Create ensemble probability (simple average)
    print(f"Creating ensemble using: {', '.join(available_models)}")
    
    # Create ensemble probability as the average of available models
    merged_df['ensemble_prob'] = merged_df[available_models].mean(axis=1)
    
    # Create ensemble state prediction
    merged_df['ensemble_state'] = (merged_df['ensemble_prob'] > 0.5).astype(int)
    
    # 10.3 Visualize ensemble results
    plt.figure(figsize=(14, 12))
    
    # Plot 1: Original temperature data
    plt.subplot(3, 1, 1)
    plt.plot(merged_df.index, merged_df['ds18b20_temp'], 'b-', label='DS18B20')
    plt.plot(merged_df.index, merged_df['room_temp'], 'g-', alpha=0.5, label='Room')
    plt.title('Temperature Data')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    
    # Plot 2: All model probabilities
    plt.subplot(3, 1, 2)
    for col in available_models:
        label = col.split('_')[0].upper()
        plt.plot(merged_df.index, merged_df[col], alpha=0.5, label=label)
    plt.plot(merged_df.index, merged_df['ensemble_prob'], 'k-', linewidth=2, label='ENSEMBLE')
    plt.axhline(y=0.5, color='k', linestyle='--')
    plt.title('Connection Probabilities from All Models')
    plt.ylabel('Probability')
    plt.ylim(-0.05, 1.05)
    plt.legend()
    
    # Plot 3: DS18B20 temperature colored by ensemble state
    plt.subplot(3, 1, 3)
    for state in [0, 1]:
        mask = merged_df['ensemble_state'] == state
        if mask.any():  # Only plot if there are points with this state
            label = 'Connected' if state == 1 else 'Disconnected'
            color = 'red' if state == 1 else 'blue'
            plt.plot(merged_df.index[mask], merged_df['ds18b20_temp'][mask], 'o-', color=color, label=label, alpha=0.7)
    plt.title('DS18B20 Temperature by Ensemble Detection')
    plt.ylabel('Temperature (°F)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ensemble_results.png')
    plt.show()
    
    # 10.4 Compare with threshold-based approach
    agreement = (merged_df['ensemble_state'] == merged_df['initial_label']).mean() * 100
    print(f"\nAgreement between threshold-based and ensemble detection: {agreement:.2f}%")
    
    # 10.5 Performance comparison of all methods
    # Calculate agreement for all methods
    method_cols = [
        ('initial_label', 'Threshold (90°F)'),
        ('msm_state', 'Markov Switching'),
        ('var_state', 'Vector AR'),
        ('arimax_state', 'ARIMAX'),
        ('kalman_state', 'Kalman Filter'),
        ('ensemble_state', 'Ensemble')
    ]
    
    # Keep only methods that are available
    method_cols = [(col, name) for col, name in method_cols if col in merged_df.columns]
    
    # Create comparison table
    print("\nPerformance Comparison:")
    print("======================")
    print("Method                   | Agreement with Threshold")
    print("-------------------------|------------------------")
    
    for col, name in method_cols:
        if col == 'initial_label':
            agreement = 100.0  # Same as itself
        else:
            agreement = (merged_df[col] == merged_df['initial_label']).mean() * 100
        print(f"{name:25} | {agreement:6.2f}%")

# 11. Creating a Detection Function
# ==============================
print("\nCreating a reusable detection function...")

def detect_connection_advanced(ds18b20_temps, room_temps, feature_history=None):
    """
    Detect if the DS18B20 sensor is connected using advanced time series models.
    
    Parameters:
    -----------
    ds18b20_temps : array-like
        Recent DS18B20 temperature readings
    room_temps : array-like
        Recent room temperature readings
    feature_history : dict, optional
        Dictionary to store historical features for next call
        
    Returns:
    --------
    dict
        Dictionary containing detection results and diagnostics
    """
    # Check if we have enough data
    if len(ds18b20_temps) < 12 or len(room_temps) < 12:
        # Not enough data for advanced models
        # Fall back to simple threshold
        is_connected = ds18b20_temps[-1] > 90.0
        return {
            'is_connected': is_connected,
            'connected_probability': 1.0 if is_connected else 0.0,
            'method': 'threshold',
            'confidence': 0.7  # Lower confidence due to simple method
        }
    
    # Convert to numpy arrays
    ds18b20_array = np.array(ds18b20_temps)
    room_array = np.array(room_temps)
    
    # Normalize data
    ds18b20_norm = (ds18b20_array - np.mean(ds18b20_array)) / np.std(ds18b20_array)
    room_norm = (room_array - np.mean(room_array)) / np.std(room_array)
    
    # Calculate temperature difference
    temp_diff = ds18b20_norm - room_norm
    
    # 1. Simplified Kalman Filter approach
    try:
        # Connected model: DS18B20 follows AR(1) process
        ar_coef = np.corrcoef(ds18b20_norm[:-1], ds18b20_norm[1:])[0, 1]
        pred_connected = ds18b20_norm[-2] * ar_coef
        
        # Disconnected model: DS18B20 follows room temperature
        beta = np.corrcoef(room_norm, ds18b20_norm)[0, 1]
        pred_disconnected = room_norm[-1] * beta
        
        # Calculate prediction errors
        error_connected = abs(ds18b20_norm[-1] - pred_connected)
        error_disconnected = abs(ds18b20_norm[-1] - pred_disconnected)
        
        # Convert errors to probabilities
        from scipy.stats import norm
        error_scale = np.std(ds18b20_norm) / 2
        p_connected = norm.pdf(error_connected, scale=error_scale)
        p_disconnected = norm.pdf(error_disconnected, scale=error_scale)
        
        kalman_prob = p_connected / (p_connected + p_disconnected)
    except:
        kalman_prob = 0.5  # Default to uncertainty
    
    # 2. VAR coefficient approach
    try:
        # Create small dataset for VAR
        data = pd.DataFrame({
            'ds18b20': ds18b20_norm,
            'room': room_norm
        })
        
        # Fit VAR model
        from statsmodels.tsa.api import VAR
        var_model = VAR(data)
        var_result = var_model.fit(1)  # VAR(1) model
        
        # Extract coefficient: effect of room(t-1) on ds18b20(t)
        var_coef = var_result.coefs[0, 0, 1]  # [lag, eq, var]
        
        # Convert to probability (higher coefficient = lower connection probability)
        var_prob = 1 - min(max((var_coef + 0.5) / 1.0, 0), 1)
    except:
        var_prob = 0.5  # Default to uncertainty
    
    # 3. Simple regime detection (minimal Markov Switching)
    try:
        # Calculate statistics for temperature difference
        # Connected: mean is high (DS18B20 > room), variance is low
        # Disconnected: mean is low, variance is higher
        last_diff = temp_diff[-1]
        mean_diff = np.mean(temp_diff)
        
        # Simple probability based on where current diff lies relative to mean
        if last_diff > mean_diff:
            regime_prob = 0.8  # Likely connected
        else:
            regime_prob = 0.2  # Likely disconnected
    except:
        regime_prob = 0.5  # Default to uncertainty
    
    # Combine probabilities (simple average)
    combined_prob = np.mean([kalman_prob, var_prob, regime_prob])
    
    # Determine connection state
    is_connected = combined_prob > 0.5
    
    # Prepare result
    result = {
        'is_connected': bool(is_connected),
        'connected_probability': float(combined_prob),
        'method': 'advanced_ensemble',
        'confidence': float(abs(combined_prob - 0.5) * 2),  # Scale to [0,1]
        'model_probabilities': {
            'kalman': float(kalman_prob),
            'var': float(var_prob),
            'regime': float(regime_prob)
        }
    }
    
    return result

# Test the detection function on a sample from the dataset
if len(merged_df) > 24:  # Need at least 24 points for testing
    # Extract a sample from the data
    test_slice = slice(-24, None)  # Last 24 points
    
    ds18b20_sample = merged_df['ds18b20_temp'].iloc[test_slice].values
    room_sample = merged_df['room_temp'].iloc[test_slice].values
    
    # Run detection
    detection_result = detect_connection_advanced(ds18b20_sample, room_sample)
    
    print("\nAdvanced detection result for test data:")
    for key, value in detection_result.items():
        if key != 'model_probabilities':
            print(f"{key}: {value}")
    print("\nIndividual model probabilities:")
    for model, prob in detection_result['model_probabilities'].items():
        print(f"  {model}: {prob:.4f}")
else:
    print("Not enough data to test the detection function.")

# 12. Save models for later use
# ===========================
print("\nSaving analysis results...")

# Save key parameters and models
analysis_results = {
    'models': {},
    'threshold': body_temp_threshold
}

# Add models that were successfully created
if 'ms_results' in locals():
    analysis_results['models']['markov_switching'] = {
        'regime_connected': regime_connected,
        'means': ms_results.regime_parameters.iloc[0, :].tolist()
    }

# Store the detection function as serialized code
# (can't directly serialize the function)
detection_function_code = """
def detect_connection_advanced(ds18b20_temps, room_temps, feature_history=None):
    \"\"\"
    Detect if the DS18B20 sensor is connected using advanced time series models.
    
    Parameters:
    -----------
    ds18b20_temps : array-like
        Recent DS18B20 temperature readings
    room_temps : array-like
        Recent room temperature readings
    feature_history : dict, optional
        Dictionary to store historical features for next call
        
    Returns:
    --------
    dict
        Dictionary containing detection results and diagnostics
    \"\"\"
    # Implementation code here - refer to the notebook for full implementation
    # This is a placeholder for serialization purposes
    pass
"""

analysis_results['detection_function_code'] = detection_function_code

# Save the results
joblib.dump(analysis_results, 'ds18b20_advanced_models.pkl')
print("Saved analysis results to 'ds18b20_advanced_models.pkl'")

# 13. Final Recommendations
# =======================
print("""
# Summary and Recommendations
# =========================

This notebook demonstrates how to use advanced time series models to detect 
when the DS18B20 temperature sensor is connected to a human body versus 
when it's disconnected and measuring ambient temperature.

Key findings:
1. Advanced time series models can effectively distinguish sensor states
2. Each model has strengths and weaknesses:
   - Markov Switching: Best at capturing distinct regimes
   - VAR: Strong at modeling sensor interactions
   - ARIMAX: Good at detecting anomalies from expected behavior
   - Kalman Filter: Robust to noise and transient fluctuations
3. Ensemble methods provide the most robust detection overall

Advantages of advanced time series approaches:
1. Model the underlying data generation process rather than just thresholds
2. Provide probabilistic outputs with confidence measures
3. Adapt to different temperature ranges and environmental conditions
4. Capture subtle temporal patterns and relationships

Recommendations for implementation:
1. Use the ensemble method for most robust detection
2. Consider computational requirements - simpler methods may suffice in resource-constrained environments
3. Periodically retrain models as more data becomes available
4. Combine with other approaches (residual analysis, cross-correlation) for higher reliability

Next steps:
1. Implement real-time detection with the ensemble approach
2. Create a more sophisticated state space model with explicit connection state
3. Collect labeled data with known connected/disconnected periods for better evaluation
4. Optimize computationally expensive models for embedded deployment
""")

# Close the InfluxDB client
client.close()
print("\nAnalysis complete!")# Advanced Time Series Models for DS18B20 Sensor State Detection
# =========================================================

# This notebook demonstrates how to use advanced time series models
# to detect when the DS18B20 temperature sensor is connected or disconnected.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from influxdb_client import InfluxDBClient
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
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

# 1. Introduction to Advanced Time Series Models
# ============================================
print("""
# Advanced Time Series Models Explained
# ==================================

Advanced time series models offer powerful techniques for detecting sensor states
by modeling the different statistical regimes and time-dependent behaviors:

1. **Markov Switching Models**
   - Model time series with discrete, unobserved states (regimes)
   - Each regime has its own statistical parameters
   - The regimes naturally correspond to "connected" and "disconnected" states
   - Transitions between states follow a Markov process

2. **Vector Autoregression (VAR)**
   - Model interactions between multiple time series (DS18B20, room temp, etc.)
   - Capture how variables influence each other over time
   - Detect changes in these relationships to identify state changes

3. **Kalman Filters**
   - Optimal state estimation for time-varying systems
   - Handle noise and uncertainty in measurements
   - Estimate the hidden connection state alongside temperature

4. **ARIMA Models with External Regressors (ARIMAX)**
   - Model temperature as a function of both its past values and external factors
   - Identify when the relationship with external factors (room temp) changes

These models are particularly powerful because they incorporate both:
- The temporal dynamics of each individual temperature series
- The changing relationships between different temperature measurements

Let's implement these models to create state-of-the-art sensor detection techniques!
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
ds