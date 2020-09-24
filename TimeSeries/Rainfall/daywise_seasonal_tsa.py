import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA


def model_evaluation(series_data, fitted_values):
    plt.figure(figsize=(12, 8))
    plt.plot(series_data, label='Actual')
    plt.plot(fitted_values, color='red', label='Predicted')
    plt.ylabel('Rainfall in mm')
    plt.xlabel('Date')
    plt.title('Time series rainfall ARIMA')
    plt.legend()
    plt.show()


def tune_rainfall(rainfall):
    rainfall = np.array(rainfall)
    replace_value = []
    replace_index = []
    for i in range(len(rainfall) - 2):
        if rainfall[i] != 0 and rainfall[i + 2] != 0:
            if rainfall[i + 1] == 0:
                replace_index.append(i + 1)
                replace_value.append((rainfall[i] + rainfall[i + 2]) / 2)
    j = 0
    for i in replace_index:
        rainfall[i] = replace_value[j]
        j += 1
    return list(rainfall)


def tune_output(fitted_value, series_data):
    ratios = []
    non_zero_indices = []
    for i in range(len(fitted_value)):
        if series_data[i] != 0:
            non_zero_indices.append(i)
            ratios.append(series_data[i]/fitted_value[i])
    ratio_sum = 0
    for i in range(len(ratios)):
        ratio_sum += ratios[i]
    avg_ratio = ratio_sum/len(ratios)
    for val in non_zero_indices:
        fitted_value[val] = fitted_value[val]*(avg_ratio/1.5)
    model_evaluation(series_data, fitted_value)
    return avg_ratio


def forecast_rainfall(start, end):
    df = pd.read_csv('../../Datasets/reservoir_5.csv')
    data = df['rainfall'].values
    # data = tune_rainfall(data)
    series_data = pd.Series(data, index=pd.DatetimeIndex(
        data=(tuple(pd.date_range('01/01/2004',
                                  periods=len(df),
                                  freq='D'))),
        freq='D'))

    # ARIMA model training
    model = ARIMA(series_data, order=(10, 0, 0))
    result = model.fit()
    fitted_values = result.fittedvalues
    avg_ratio = tune_output(fitted_values, series_data)
    forecast = result.predict(start=start, end=end)
    tune_forecast_list = []
    for val in forecast:
        if val > 1.5:
            tune_forecast_list.append(avg_ratio*val)
        else:
            tune_forecast_list.append(0)
    return tune_forecast_list


rain_fore = forecast_rainfall('2018-10-06', '2018-10-10')
for val in rain_fore:
    print(val)
