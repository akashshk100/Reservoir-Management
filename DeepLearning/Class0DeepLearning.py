import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization


def Class0StoragePrediction(storage, inflow, rainfall, outflow):
    df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/class0.csv')
    X = df[['storage', 'inflow', 'rainfall', 'outflow']]
    y = np.array(df['next_storage'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # designing architecture
    model = Sequential()
    model.add(Dense(15, input_dim=4, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.add(BatchNormalization())

    # training
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, verbose=1,
                        batch_size=len(X_train))

    # Verification
    input_array = np.array(X_test)
    y_pred = model.predict(input_array)
    for i in range(len(y_pred)):
        print(y_test[i], '\t', y_pred[i])

    input_array = np.array([[storage, inflow, rainfall, outflow]])
    y_pred = model.predict(input_array)
    return y_pred


result = Class0StoragePrediction(1613.0, 663.0, 0.0, 337.0)
print(result)
