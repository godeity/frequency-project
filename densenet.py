from get_data import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf
import numpy as np
from keras import regularizers

def NMSE(pred, actual):
    NMSE = 10*math.log10(np.sum(np.power((pred.reshape(-1, 1) - actual.reshape(-1, 1)), 2))/np.sum(
        np.power(actual.reshape(-1, 1), 2)))
    return NMSE


path = 'Data'


data, batch_size, n_batch = get_data(path)
print('length of data', len(data))

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
# train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
# test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))



# design network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))

# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))  # 256 个神经元的全连接层,from keras import regularizers
# model.add(LeakyReLU(alpha=0.3))
model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(2, activation='tanh', kernel_regularizer=regularizers.l2(0.01),
                                activity_regularizer=regularizers.l1(0.01)))


# sgd = tf.keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
RMSprop = tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mse', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=15, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=True)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()


# # make a prediction
y_hat = model.predict(test_X)
# # calculate MSE
mse = mean_squared_error(test_y, y_hat)
nmse = NMSE(y_hat, test_y)
print('Test mse:', mse)
print('Test NMSE:', nmse)

train_y_hat = model.predict(train_X)
train_mse = mean_squared_error(train_y_hat, train_y)
train_nmse = NMSE(train_y_hat, train_y)     #keep in mind prediction go first
print('train_mse:', train_mse)
print('train_nmse:', train_nmse)
