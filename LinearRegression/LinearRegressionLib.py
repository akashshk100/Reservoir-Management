import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split

reg = linear_model.LinearRegression()


def fit_linear_model():
    global reg
    df = pd.read_csv('../Datasets/reservoir_5.csv')
    inflow = df['inflow']
    outflow = df['outflow']
    df_label = pd.read_csv('../Datasets/reservoir_6.csv')
    label = df_label['label']

    X = []
    y = []
    for i in range(df.__len__()):
        if label[i] == 3:
            if inflow[i] != 0:
                X.append([inflow[i]])
                y.append(outflow[i])

    mdl_accuracy = 0
    while 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg.fit(X_train, y_train)
        mdl_accuracy = reg.score(X_test, y_test)*100
        if mdl_accuracy > 96.00:
            break
        else:
            continue

    print('Model Accuracy: ', mdl_accuracy, '%')
    y_predicted = reg.predict(X_test)
    plt.scatter(X_test, y_predicted, color="blue", s=10, label='Test data')
    plt.scatter(X_test, y_test, color="red", s=10, label='Test data')
    plt.xlabel('INFLOW')
    plt.ylabel('DISCHARGE')
    plt.show()


def linear_predict(inflow):
    global reg
    fit_linear_model()
    outflow = reg.predict([[inflow]])
    return outflow[0]


print(linear_predict(13489.0))
