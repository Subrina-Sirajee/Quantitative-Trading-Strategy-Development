# Signal Generation Module 

This module provides functionality for quantitative analysis of financial time series data. It includes methods for calculating technical indicators such as
Moving Average Convergence Divergence (MACD), Bollinger Bands, Simple Moving Average (SMA), and generating signals based on these indicators for trading 
strategies.  Additionally, it incorporates a Long Short-Term Memory (LSTM) model for time series forecasting.

## Installation

To use this module, follow these steps:

 Install the required libraries by running:
    ```
    pip install statsmodels keras tensorflow numpy pandas scikit-learn
    ```
## Constraint    
In the `group2_ensemble_model_signals` function,the length of the DataFrame passed to the function should be at least `201`.
