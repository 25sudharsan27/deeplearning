import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from tensorflow import keras

data=pd.read_csv('iris.csv')
lable_encoder=preprocessing.LabelEncoder()
data["Species"]=lable_encoder.fit_transform(data["Species"])
data1=data.to_numpy()

X_data=data1[:,0:4]
Y_data=data1[:,4]
scalar=StandardScaler().fit(X_data)
X_data=scalar.transform(X_data)
Y_data=tf.keras.utils.to_categorical(Y_data,3)


X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_data,test_size=0.10)

modal=tf.keras.models.Sequential()

modal.add(keras.layers.Dense(128,input_shape=(4,),name='Hidden-Layer-1',activation='relu'))

modal.add(keras.layers.Dense(128,name='Hidden-Layer-2',activation='relu'))

modal.add(keras.layers.Dense(3,name='Output-Layer',activation='softmax'))

modal.compile(loss='categorical_crossentropy',metrics=['accuracy'])

modal.summary()
VERBOSE=1
VALIDATION_SPLIT=0.20
CLASS=3
EPOCHES=10
BRANCH_SIZE=16

history=modal.fit(X_train,Y_train,verbose=VERBOSE,validation_split=VALIDATION_SPLIT,epochs=EPOCHES,batch_size=BRANCH_SIZE)

modal.evaluate(X_test,Y_test)

modal.save('Sudharsan_save')


