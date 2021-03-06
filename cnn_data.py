#モジュールのインポート
from PIL import Image #Imageスラス
import os
import glob
import numpy as np
from sklearn import model_selection #データをスプリットする

#パラメーターの初期化
classes = ["data","clothes"]  #リストを作成
num_classes = len(classes) #clothesの文字の長さ
image_size = 150


#画像の読み込みとNumPy配列への変換
X = []
Y = []

for index,classlabel in enumerate(classes):
    photos_dir = "./" + classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i,file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size,image_size))
        data = np.asarray(image) / 255.0 #255.0で割ったので浮動小数点数になり、ファイルのサイズが大きくなっている

        X.append(data)
        Y.append(index)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y)  #サーキットラーンを使ってトレーニングしている
xy = (X_train, X_test, y_train, y_test) #タプル型
np.save("./imagefile.npy", xy) #バイナリーファイルに変換する


