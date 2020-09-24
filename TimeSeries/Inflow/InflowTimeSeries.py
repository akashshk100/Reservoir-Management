import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt


def forecast_inflow(start_date, end_date):
    df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')
    series_data = pd.Series(df['inflow'].values,
                            index=pd.DatetimeIndex(
                            data=(tuple(pd.date_range('01/01/2004',
                                                      periods=len(df),
                                                      freq='D'))),
                            freq='D'))

    # ARIMA model training
    model = ARIMA(series_data, order=(5, 0, 0))
    result = model.fit()
    fitted_values = result.fittedvalues
    plt.figure(figsize=(12, 8))
    plt.plot(series_data, label='actual')
    plt.plot(result.fittedvalues, color='red', label='predicted')
    plt.ylabel('Inflow in cusecs')
    plt.title('Time Series Inflow ARIMA')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.show()
    for i in range(len(fitted_values)):
        print(series_data[i], '\t', fitted_values[i])
    forecast = result.predict(start_date, end_date)
    return forecast


gen_forecast = forecast_inflow('2018-10-06', '2018-10-10')
print(gen_forecast)
