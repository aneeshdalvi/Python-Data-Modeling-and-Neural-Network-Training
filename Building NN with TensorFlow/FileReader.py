from __future__ import absolute_import, division, print_function
import pandas as pd
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

train_data = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\564_Assignments\\Assignment 4\\VC_Input.csv')


column_names = ['', 'Velocity', 'LanePos', 'SpeedLimit', 'Steer', 'Accel', 'Brake', 'LongAccel', 'LatAccel', 'HeadwayTime', 'HeadwayDist']

order = np.argsort(np.random.random(train_data.shape))


print("Training set: {}".format(train_data.shape))

#print(train_data[0:10])

df = pd.DataFrame(train_data, columns=column_names)
# print(df.head())


# Test data is *not* used when calculating the mean and std

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std


# print(train_data[0:0])

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model


model = build_model()
model.summary()


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 500

# Store training stats
history = model.fit(train_data, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[PrintDot()])
