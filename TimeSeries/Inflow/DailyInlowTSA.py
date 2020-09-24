import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')
model = Sequential()


def fit_model(X_train, y_train):
    model.add(Dense(100, activation='relu', input_dim=5))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=500, verbose=1)


def supervised_data(lag):
    inflow = df['inflow'].values
    supervised_input = []
    supervised_output = []
    for i in range(len(inflow) - lag):
        row = []
        for j in range(lag):
            row.append(inflow[i + j])
        supervised_output.append(inflow[i + 3])
        supervised_input.append(row)
    return supervised_input, supervised_output


def forecast_inflow(input_list):
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
    plt.ylabel('Inflow in cusecs')
    plt.title('Time Series Inflow')
    plt.legend()

    plt.tight_layout()
    plt.show()


print(forecast_inflow([520, 620, 620, 620, 801]))
