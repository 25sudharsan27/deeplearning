import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn import preprocessing


data=pd.read_csv('iris.csv')
label_encoder=preprocessing.LabelEncoder()
data['Species']=label_encoder.fit_transform(data['Species'])
data1=data.to_numpy()
X_data=data1[:,0:4]
Y_data=data1[:,4]
scalar=StandardScaler().fit(X_data)
X_data=scalar.transform(X_data)
Y_data=tf.keras.utils.to_categorical(Y_data,3)

loaded=keras.models.load_model('Sudharsan_save')
input=[[2.1,4.1,3.4,1.4]]
trans=scalar.transform(input)
prediction=loaded.predict(trans)
print("Prediction: ",label_encoder.inverse_transform([np.argmax(prediction)]))