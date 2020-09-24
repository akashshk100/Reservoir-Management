import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

model = Sequential()


def fit_model(X_train, y_train, X_test, y_test):
    # designing architecture
    model.add(Dense(15, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.add(BatchNormalization())
    # training
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, verbose=1,
                        batch_size=len(X_train))
    # evaluate the model
    train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=len(X_train))
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=len(X_test))
    print(model.metrics_names)
    print('Train: %.6f, Test: %.6f' % (train_loss, test_loss))
    print('Train: %.6f, Test: %.6f' % (train_acc, test_acc))
    plt.title('Loss / Mean Squared Error')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def verify_model(X_test, y_test):
    input_array = np.array(X_test)
    y_pred = model.predict(input_array)
    for i in range(len(y_pred)):
        print(y_test[i], '\t', y_pred[i])

    n_groups = len(y_pred)
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, y_test, bar_width, alpha=opacity, color='b', label='actual')

    rects2 = plt.bar(index + bar_width, y_pred[:, 0], bar_width, alpha=opacity, color='r', label='Predicted')
    plt.ylabel('Storage in mcft')
    plt.title('Deep Learning Storage')
    plt.legend()

    plt.tight_layout()
    plt.show()


def Class4StoragePrediction(storage, inflow, rainfall):
    df = pd.read_csv('../Datasets/class0.csv')
    X = df[['storage', 'inflow', 'rainfall']]
    y = np.array(df['next_storage'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    fit_model(X_train, y_train, X_test, y_test)
    verify_model(X_test, y_test)
    input_array = np.array([[storage, inflow, rainfall]])
    y_pred = model.predict(input_array)
    return y_pred


result = Class4StoragePrediction(2142.0, 1001.0, 16.0)
print(result)
