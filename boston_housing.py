from __future__ import print_function
import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
 
batch_size = 404
num_classes = 6
epochs = 10000

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
plt.hist(y_train)
plt.hist(y_test)
plt.show()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train /= 10
y_test /= 10

y_train = y_train.round()
y_test = y_test.round()

plt.hist(y_train)
plt.hist(y_test)
plt.show()

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(13,)))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
 
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tb_cb = keras.callbacks.TensorBoard(log_dir="tflog4/", histogram_freq=1)
cbks = [tb_cb]

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(x_test, y_test),
                    callbacks=cbks)
score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])