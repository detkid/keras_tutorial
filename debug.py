from __future__ import print_function
import keras
from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 
batch_size = 128
num_classes = 13
epochs = 2 # 実行時間が長いのでここだけ 12 から 2 に変更しました
 
# input image dimensions
img_rows, img_cols = 32, 32
 
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
