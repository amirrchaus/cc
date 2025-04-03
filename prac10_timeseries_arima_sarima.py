import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"
data = pd.read_csv(url, header=0, parse_dates=[0], index_col=0,)
train = data.iloc[:-12] 
test = data.iloc[-12:]

arima_model = ARIMA(train, order=(5,1,0))
arima_result = arima_model.fit()

arima_forecast = arima_result.forecast(steps=12)

sarimax_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)) 
sarimax_result = sarimax_model.fit()
sarimax_forecast = sarimax_result.forecast(steps=12)

plt.figure(figsize=(10, 5))
plt.plot(train, label='Train')
plt.plot(test, label='Test')
plt.plot(arima_forecast, label='ARIMA Forecast')
plt.plot(sarimax_forecast, label='SARIMAX Forecast')
plt.legend()
plt.title('ARIMA and SARIMAX Forecasting')
plt.show()

#MAE
arima_mae = mean_absolute_error(test, arima_forecast)
sarimax_mae = mean_absolute_error(test, sarimax_forecast)
#RMSE
arima_mse = mean_squared_error(test, arima_forecast)
arima_rmse = np.sqrt(arima_mse)
sarimax_mse = mean_squared_error(test, sarimax_forecast)
sarimax_rmse = np.sqrt(sarimax_mse)
print(f"ARIMA MAE: {arima_mae:.2f}, SARIMAX MAE: {sarimax_mae:.2f}" )
print(f"ARIMA RMSE: {arima_rmse:.2f}, SARIMAX RMSE: {sarimax_rmse:.2f}")





##### run on cmd cd C:\Users\Amirrchaus\Desktop\DS Pracs
##
##C:\Users\Amirrchaus\Desktop\DS Pracs>python prac10_timeseries_arima_sarima.py
