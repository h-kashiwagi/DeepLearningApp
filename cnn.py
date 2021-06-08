import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Con2D,MaxPooling2D
from keras.optimizers import SDG
from keras.utils import np_utils

classes = ["data","clothes"]
num_classes = len(classes)
image_size = 150 

X_train,X_test,y_train,y_test = np.load("./imagefile.npy") #ファイルをロード
y_train = np_utils.to_categorical(y_train,num_classes)#ワンホットベクトル形式に変換
y_test