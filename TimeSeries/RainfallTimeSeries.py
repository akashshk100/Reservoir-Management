import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')
data = df[['date', 'rainfall']]
series_data = pd.Series(df['level'].values,
                        index=pd.DatetimeIndex(
                        data=(tuple(pd.date_range('01/01/2004',
                                                  periods=len(df),
                                                  freq='D'))),
                        freq='D'))

# Plotting time series data
plt.figure(figsize=(12, 8))
plt.plot(series_data)
plt.xlabel('date')
plt.ylabel('rainfall')
plt.show()
# AD Fuller test for stationarity
print('Results of ad fuller test:')
df_test = adfuller(series_data, autolag="AIC")
df_output = pd.Series(df_test[0:4],
                      index=["Test statistics", "p-value",
                             "Number of lags",
                             "Number of observation used"])
print(df_output)

# Finding correlation
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(series_data, lags=730, ax=ax1)
plot_pacf(series_data, lags=730, ax=ax2)
plt.show()

# ARIMA model training
model = ARIMA(series_data, order=(15, 0, 0))
result = model.fit()
plt.figure(figsize=(12, 8))
plt.plot(series_data)
plt.plot(result.fittedvalues, color='red')
plt.show()
fitted_values = result.fittedvalues
for i in range(len(fitted_values)):
    print(series_data[i], '\t', fitted_values[i])

# forecast = result.predict(start='2004-10-26', end='2004-10-31')
# print(forecast)
