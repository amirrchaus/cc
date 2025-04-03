import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset from a local CSV file
file_path = "example_air_passengers.csv"  # Replace with the actual filename
data = pd.read_csv(file_path, header=0, parse_dates=[0], index_col=0)

# Splitting into train and test sets (last 12 months for testing)
train = data.iloc[:-12] 
test = data.iloc[-12:]

# ARIMA Model
arima_model = ARIMA(train, order=(5,1,0))
arima_result = arima_model.fit()
arima_forecast = arima_result.forecast(steps=12)

# SARIMAX Model
sarimax_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)) 
sarimax_result = sarimax_model.fit()
sarimax_forecast = sarimax_result.forecast(steps=12)

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data', color='orange')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', linestyle='dashed')
plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast', linestyle='dashed')
plt.legend()
plt.title('ARIMA and SARIMAX Forecasting')
plt.show()

# Evaluate Performance
arima_mae = mean_absolute_error(test, arima_forecast)
sarimax_mae = mean_absolute_error(test, sarimax_forecast)

arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
sarimax_rmse = np.sqrt(mean_squared_error(test, sarimax_forecast))

print(f"ARIMA MAE: {arima_mae:.2f}, SARIMAX MAE: {sarimax_mae:.2f}" )
print(f"ARIMA RMSE: {arima_rmse:.2f}, SARIMAX RMSE: {sarimax_rmse:.2f}")


#run on cmd
