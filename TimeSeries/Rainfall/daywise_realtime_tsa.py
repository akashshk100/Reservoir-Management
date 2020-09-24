import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import BatchNormalization

df = pd.read_csv('../../Datasets/class4.csv')
model = Sequential()


def fit_model(X_train, y_train):
    model.add(Dense(1000, activation='relu', input_dim=5))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # model.add(BatchNormalization())
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=200, verbose=1)


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


def supervised_data(lag):
    rainfall = df['rainfall'].values
    # rainfall = tune_rainfall(rainfall)
    supervised_input = []
    supervised_output = []
    for i in range(len(rainfall) - lag):
        row = []
        for j in range(lag):
            row.append(rainfall[i + j])
        supervised_output.append(rainfall[i + 5])
        supervised_input.append(row)
    # for i in range(len(supervised_output)):
    #     print(supervised_input[i], '\t', supervised_output[i])
    return supervised_input, supervised_output


def forecast_rainfall(input_list):
    ip, op = supervised_data(5)
    ip = np.array(ip)
    op = np.array(op)
    X_train, X_test, y_train, y_test = train_test_split(ip, op, test_size=0.3)
    fit_model(X_train, y_train)
    verify_model(X_test, y_test)
    output_list = []
    for i in range(5):
        input_list = input_list[i:]
        for val in output_list:
            input_list.append(val)
        input_array = np.array([input_list])
        y_pred = model.predict(input_array)
        output_list.append(y_pred[0][0])
    return output_list


def verify_model(X_test, y_test):
    # verification code for test dataset
    y_pred = model.predict(X_test)
    print('predicted', '\t', 'actual')
    for i in range(len(y_pred)):
        print(y_pred[i][0], '\t', y_test[i])
    n_groups = len(y_pred)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, y_test, bar_width,
                     alpha=opacity,
                     color='b',
                     label='actual')

    rects2 = plt.bar(index + bar_width, y_pred[:, 0], bar_width,
                     alpha=opacity,
                     color='r',
                     label='Predicted')
    plt.ylabel('Rainfall in mm')
    plt.title('Time series rainfall')
    plt.legend()

    plt.tight_layout()
    plt.show()


print(forecast_rainfall([0.0, 0.0, 0.0, 16, 27]))
