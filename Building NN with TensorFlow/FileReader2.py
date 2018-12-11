

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import pandas as pd

train = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\564_Assignments\\Assignment 4\\VC_Input.csv')
#train = pd.read_csv('E:\\ASU USA\\Subjects\\CSE 564 Software Design\\Assignment\\Assignment-4\\VC_output.csv')

n = len(train)
n = n * 0.8
n = int(n)

train_data = train[0:n]
test_data = train[n + 1:]

output = pd.read_csv('C:\\Users\\Aneesh Dalvi\\Desktop\\564_Assignments\\Assignment 4\\Vh_Output.csv')
#output = pd.read_csv('E:\\ASU USA\\Subjects\\CSE 564 Software Design\\Assignment\\Assignment-4\\VH_output.csv')
train_labels = output[0:n]
test_labels = output[n + 1:]

# Shuffle the training set
print("Training set: {}".format(train_data.shape))
print("Testing set:  {}".format(test_data.shape))


column_names = ['Velocity', 'LanePos', 'Steer', 'SpeedLimit', 'Accel', 'Brake', 'LongAccel', 'LatAccel', 'HeadwayTime', 'HeadwayDist']
#column_names = ['Velocity','LanePos','Steer','SpeedLimit','Accel','Brake','LongAccel','HeadwayTime','HeadwayDist']

df = pd.DataFrame(train_data, columns=column_names)
df.head()

print(df.head())
print(train_data[0:1])

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# model - hidden layers


def build_model():
  model = keras.Sequential([
      keras.layers.Dense(256, activation=tf.nn.relu,
                         input_shape=(df.shape[1],)),
      keras.layers.Dropout(0.4),
      keras.layers.Dense(128, activation=tf.nn.relu),
      keras.layers.Dropout(0.4),
      keras.layers.Dense(64, activation=tf.nn.relu),
      keras.layers.Dropout(0.4),
      keras.layers.Dense(4, activation=tf.nn.relu)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)
  model.compile(loss='mse', optimizer='adam', metrics=['mae'])
  return model


# Display training progress
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0:
      print('')
      print(epoch, logs, end='')
    # print('')


EPOCHS = 500
model = build_model()

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.1, verbose=0,
                    callbacks=[PrintDot()])


[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
print(model.evaluate(test_data, test_labels, verbose=0))

print("Testing set Mean Abs Error: {:7.2f}".format(mae))
