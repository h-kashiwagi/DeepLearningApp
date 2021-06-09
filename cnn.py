#モデルの定義とトレーニングのコード

import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.layers import Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.utils import np_utils ←バージョン変更で廃止
from tensorflow.keras import utils

classes = ["data","clothes"]
num_classes = len(classes)
image_size = 150 

X_train,X_test,y_train,y_test = np.load("./imagefile.npy", allow_pickle=True) #ファイルをロード
#numpy v1.16.3 より、numpy.load()関数の挙動が変更されたためallow_pickle=Trueがないとエラーになる
y_train = utils.to_categorical(y_train,num_classes) #ワンホットベクトル形式に変換
y_test = utils.to_categorical(y_test,num_classes)

#モデル定義
model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(image_size,image_size,3)))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0,25))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

opt = SGD(Ir=0.01) #rmsprop,adam
model.compile(loss='categorical_crossentropy',optimizer=opt)

model.fit(X_train,y_train,batch_size=32,epochs=10)

score = model.evaluate(X_test,y_test,batch_size=32)