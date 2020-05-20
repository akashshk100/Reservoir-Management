import pandas as pd
from csv import writer
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def find_sum(temp):
    sum = 0
    for i in range(len(temp)):
        sum += temp[i]
    return sum


def monthly_decompose():
    df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')
    rainfall = df['rainfall'].values
    date = df['date'].values
    date_obj = []
    for i in range(len(date)):
        date_obj.append(datetime.strptime(date[i], '%d/%m/%Y').date())

    # monthly decomposition of rainfall
    acum_rainfall_1 = [[], [], [], [], [], [], [], [], [], [], [], []]
    mon = date_obj[0].month
    temp = []
    i = 0
    while i != len(rainfall):
        if date_obj[i].month != mon:
            acum_rainfall_1[date_obj[i-1].month-1].append(find_sum(temp))
            temp = []
            mon = date_obj[i].month
        temp.append(rainfall[i])
        i += 1
    for val in acum_rainfall_1:
        print(val)
    collected_list = []
    for i in range(15):
        for j in range(12):
            collected_list.append(acum_rainfall_1[j][i])
    for i in range(len(collected_list)):
        append_list_as_row('/home/akash/PycharmProjects/Reservoir/Datasets/monthly_rainfall.csv', [collected_list[i]])
    return collected_list


def model_evaluation(series_data, fitted_values):
    plt.figure(figsize=(12, 8))
    plt.plot(series_data, label='Actual')
    plt.plot(fitted_values, color='red', label='Predicted')
    plt.ylabel('Rainfall in mm')
    plt.xlabel('Date')
    plt.title('Time series rainfall')
    plt.legend()
    plt.show()


def predict_rainfall(start, end):
    df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/monthly_rainfall.csv')
    collected_list = df['total_rainfall'].values
    series_data = pd.Series(collected_list, index=pd.DatetimeIndex(
                                data=(tuple(pd.date_range('01/01/2004',
                                                          periods=len(collected_list),
                                                          freq='M'))),
                                freq='M'))
    model = SARIMAX(series_data, order=(0, 1, 0), seasonal_order=(1, 1, 1, 12))
    result = model.fit()
    fitted_values = result.fittedvalues
    model_evaluation(series_data, fitted_values)
    return result.predict(start=start, end=end)


# monthly_decompose()
print(predict_rainfall(start=177, end=178))
