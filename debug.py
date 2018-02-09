from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
 
batch_size = 128
num_classes = 10
epochs = 2 # 実行時間が長いのでここだけ 12 から 2 に変更しました
 
# input image dimensions
img_rows, img_cols = 32, 32
 
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

a = x_train.shape[1:]
b = 0