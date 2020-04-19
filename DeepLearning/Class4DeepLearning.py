import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

df = pd.read_csv('/home/akash/PycharmProjects/Reservoir/Datasets/class4.csv')
X = df[['level', 'inflow', 'rainfall']]
y = np.array(df['storage'])
scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train.reshape(len(y_train), 1))[:, 0]
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test.reshape(len(y_test), 1))[:, 0]


# designing architecture
model = Sequential()
model.add(Dense(15, input_dim=3, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(15, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))
model.add(BatchNormalization())

# training
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000, verbose=1, batch_size=len(X_train))


# evaluate the model
train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=len(X_train))
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=len(X_train))
print(model.metrics_names)
print('Train: %.6f, Test: %.6f' % (train_loss, test_loss))
print('Train: %.6f, Test: %.6f' % (train_acc, test_acc))
plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# Verification
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)
for i in range(len(y_pred)):
    print(y_pred[i], '\t', y_test[i])
