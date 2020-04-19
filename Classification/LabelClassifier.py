import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_6.csv')
X = df[['level', 'storage', 'inflow', 'rainfall']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


def find_k():
    k_range = range(1, 20)
    accuracy_list = []
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_predicted = knn.predict(X_test)
        accuracy_list.append(metrics.accuracy_score(y_test, y_predicted))

    plt.plot(k_range, accuracy_list)
    plt.xlabel('k value')
    plt.ylabel('accuracy')
    plt.show()


def find_label(data):
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(data)
    return y_predicted


def predict_class(data):
    classes = {0: 'complex', 1: 'complex', 2: 'flood alarm', 3: 'linear', 4: 'complex'}
    # print(find_label([data]))
    return classes[find_label([data])[0]]


# print(predict_class([138.95, 2807.0, 30918, 189.4]))
