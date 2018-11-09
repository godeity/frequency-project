from get_data import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
import numpy as np
from keras import regularizers

path = 'Data'
steps = 3
data, batch_size, n_batch = get_data(path)
print('length of data', len(data))

'''
data processing
'''

# split into train and test sets
values = data.values
train = values[:n_batch, :]
validate = values[n_batch:2 * n_batch, :]
test = values[-n_batch:, :]

# split into input and outputs
train_X, train_y = np.split(train, [3, ], axis=1)
validate_X, validate_y = np.split(validate, [3, ], axis=1)
test_X, test_y = np.split(test, [3, ], axis=1)


# reshape input to be 3D [samples, timesteps, features]
var_names = ['train_X', 'validate_X', 'test_X']
names = locals()                    # get all local variable names!

for n, data_X in enumerate([train_X, validate_X, test_X]):
    reshape_X = []
    for i in range(0, len(data_X)-steps+1):
        reshape_X.append(data_X[i:i+steps])
    names[var_names[n]] = np.array(reshape_X)         #cyclic assignment , like: train_X = np.array(reshape_X)

y_names = ['train_y', 'validate_y', 'test_y']
for n, data_y in enumerate([train_y, validate_y, test_y]):
    names[y_names[n]] = data_y[steps-1:]

'''
design network
'''

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(256, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(tf.keras.layers.Dropout(0.7))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(512, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.7))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.LSTM(512))
model.add(tf.keras.layers.Dropout(0.7))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
model.add(tf.keras.layers.Dropout(0.7))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2, kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.01)))


# sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse', optimizer=RMSprop)
# fit network
history = model.fit(train_X, train_y, epochs=15, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# # make a prediction
y_hat = model.predict(test_X)
# # calculate MSE
mse = mean_squared_error(test_y, y_hat)
print('Test MSE: %.3f' % mse)
