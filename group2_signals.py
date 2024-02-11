# Import necessary libraries
from statsmodels.tsa.stattools import adfuller
import keras
import tensorflow as tf
import numpy as np
import pandas as pd
from multiprocessing import Pool
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

def calculate_macd(prices, short_window, long_window, signal_window):
    """
    Calculate MACD and signal line.

    Args:
        prices (pd.Series): Historical prices.
        short_window (int): Short-term window for MACD calculation.
        long_window (int): Long-term window for MACD calculation.
        signal_window (int): Window for signal line calculation.

    Returns:
        pd.Series: MACD values.
        pd.Series: Signal line values.
    """
    exp12 = prices.ewm(span=short_window, adjust=False).mean()
    exp26 = prices.ewm(span=long_window, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

def create_dataset(dataset, time_step=1):
    """
    Converts an array of prices into a dataset matrix.

    Parameters:
    dataset (array-like): Input array of prices.
    time_step (int): Time step to create sequences of input features and output labels. Default is 1.

    Returns:
    tuple: A tuple containing two arrays: dataX, the input features, and dataY, the corresponding output labels.
    """

    dataX, dataY = [], []

    # Iterate through the dataset to create sequences of input features and output labels
    for i in range(len(dataset) - time_step - 1):
        # Slice the dataset to create a sequence of input features with length equal to time_step
        a = dataset[i:(i + time_step), 0]
        # Append the sequence of input features to dataX
        dataX.append(a)
        # Append the corresponding output label to dataY
        dataY.append(dataset[i + time_step, 0])
    
    # Convert dataX and dataY to numpy arrays
    return np.array(dataX), np.array(dataY)



# Initialize MinMaxScaler object with feature range (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))

# Define the model architecture
model = Sequential()

# Input shape: (number of time steps, number of features)
model.add(LSTM(50,return_sequences=True,input_shape=(200,1)))
# LSTM with 50 units, return sequences to pass output to the next layer

model.add(LSTM(50,return_sequences=True))  # Another  LSTM layer

model.add(LSTM(50))  # Final LSTM layer without returning sequences

# Add a Dense output layer to produce single output value
model.add(Dense(1))

# Compile the model with mean squared error loss and Adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam')

# Add early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

def group2_ensemble_model_signals(hist):
    """
    Outputs a signal indicating whether to buy, sell, or hold based on various technical indicators.

    Args:
        hist (pd.Series or pd.DataFrame): Price history data.

    Returns:
        int: Signal for the last day; -1 (sell), 0 (hold), or 1 (buy).
    """
    
    # Check if historical standard deviation is zero
    if hist.std() == 0:
        return 0  
    
    # Reshape the DataFrame into a 1D array and apply MinMax scaling
    # MinMaxScaler expects input shape (n_samples, n_features), so reshape the 1D array to (n_samples, 1)
    df_reshaped = np.array(hist).reshape(-1, 1)
    df_scaled = scaler.fit_transform(df_reshaped)
    
    # Set parameters for model and indicators
    time_step = 200
    short_window = 5
    long_window = 17
    signal_window = 5
    bollinger_window = 5
    bollinger_dev = 1.5

    # Prepare data for model training
    X_train, y_train = create_dataset(df_scaled, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, batch_size=64,callbacks=[early_stopping],verbose=0)
    
    # Predict prices using trained model
    predicted_price = model.predict(X_train)[-1]
    #predicted_price = scaler.inverse_transform(predicted_price)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))  # Reshaping to 2D array
    predicted_price = predicted_price[0, 0]  # Extracting the scalar value from the 2D array
    
    # Calculate Bollinger Bands
    sma = hist[-bollinger_window:].mean()
    rolling_std = hist[-bollinger_window:].std()
    upper_band = sma + (bollinger_dev * rolling_std)
    lower_band = sma - (bollinger_dev * rolling_std)
  
    # Calculate MACD
    close_prices = hist[-long_window:]
    macd, signal = calculate_macd(close_prices, short_window, long_window, signal_window)
    
    # Calculate Simple Moving Average (SMA)
    sma_short = hist.rolling(window=short_window).mean()
    sma_long = hist.rolling(window=long_window).mean()

    # Perform Augmented Dickey-Fuller Test
    p_value = adfuller(hist, autolag='AIC')[1]

    # Generate signals based on Bollinger Bands, MACD, and SMA
    if p_value >= 0.05:
        # Trend-Following
        if (predicted_price > upper_band and
            macd.iloc[-1] > signal.iloc[-1] and
            sma_short.iloc[-1] > sma_long.iloc[-1]):
            return 1  # Buy signal
        elif (predicted_price < lower_band and
              macd.iloc[-1] < signal.iloc[-1] and
              sma_short.iloc[-1] < sma_long.iloc[-1]):
            return -1  # Sell signal
        else:
            return 0  # Hold signal
    else:
        # Mean Reversion
        if (predicted_price < lower_band and
            macd.iloc[-1] > signal.iloc[-1] and
            sma_short.iloc[-1] < sma_long.iloc[-1]):
            return 1  # Buy signal
        elif (predicted_price > upper_band and
              macd.iloc[-1] < signal.iloc[-1] and
              sma_short.iloc[-1] > sma_long.iloc[-1]):
            return -1  # Sell signal
        else:
            return 0  # Hold signal
