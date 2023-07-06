import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds


data,info=tfds.load('yelp_polarity_reviews',with_info=True,as_supervised=True)

train,test=data['train'],data['test']

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]
print("comp")
for X,Y in train:
    X_train.append(str(X.numpy().decode('utf8')))
    Y_train.append(Y.numpy())

for X,Y in test:
    X_test.append(str(X.numpy().decode('utf8')))
    Y_test.append(Y.numpy())

Y_train=np.array(Y_train)
Y_test=np.array(Y_test)
print('comp')
vocab_size=10000
max_length=120
oov_tok='<oov>'
padding_type='post'
trunc_type='post'
embedding_dim=16

tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index

X_train_seq=tokenizer.texts_to_sequences(X_train)
X_train_pad=pad_sequences(X_train_seq)

X_test_seq=tokenizer.texts_to_sequences(X_test)
X_test_pad=pad_sequences(X_test_seq)
print('comp')

model=tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='adam')

model.summary()
print("comp")