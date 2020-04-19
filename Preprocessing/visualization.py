import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_5.csv')

X = np.array(df[['level', 'storage', 'inflow', 'rainfall']])
df_label = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/reservoir_6.csv')
label = df_label['label']
y = df['outflow']

for i in range(5):
    t_list = []
    for j in range(len(df)):
        if label[j] == i:
            t_list.append(j)
    if i == 0:
        plt.scatter(list(X[i, 1] for i in t_list), list(y[i] for i in t_list), marker='*')
    else:
        plt.scatter(list(X[i, 1] for i in t_list), list(y[i] for i in t_list))
plt.show()
