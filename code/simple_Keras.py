# This script is to try some simple keras CNN model for stock data for classification of 0, -1, 1;
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras.utils import plot_model

import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
image = pd.read_csv('../data/image_matrix.csv')
data = pd.read_csv('../data/preprocessed_sp500.csv', sep='\t')
np_data = np.array(data)[:,1:]
np_image = np.array(image)[:,1:] # Turn all data into np.array data structure
X = np_image.reshape([-1, 128, 128])
Y = np_data[:,-2] # This is column to classification in CNN model

batch_size = 128
num_classes = 3
epochs = 5

# input image dimensions
img_rows, img_cols = 128, 128

# the data, split between train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=0.25)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=input_shape))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3), padding ='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# add another CNN layer in this step
model.add(Conv2D(64,kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Full connected layer
model.add(Flatten())
model.add(Dense(512)) # add full connected layer
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# Change optimization method in this step
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
plot_model(model, to_file='model.png')

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Finish Train data in simple CNN model



