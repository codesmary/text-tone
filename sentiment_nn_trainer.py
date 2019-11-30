import dictionary_creator

import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing import sequence
from keras.optimizers import Adam
import sqlite3
import string
import re
import os

#create training and testing sets for imdb data
x_train, y_train, x_test, y_test = [], [], [], []


dirs = [os.path.join('data', 'movie-sentiment', 'train', 'neg'),\
        os.path.join('data', 'movie-sentiment', 'train', 'pos')]
ct = 0

for dir in dirs:
    for filename in os.listdir(dir):
        f = open(os.path.join(dir, filename), 'r')
        contents = f.read()
        sentence_embedding = dictionary_creator.get_mapping(contents)    
        x_train.append(sentence_embedding)
        y_train.append(ct)
    ct += 1


dirs = [os.path.join('data', 'test', 'neg'),\
        os.path.join('data', 'test', 'pos')]
ct = 0

for dir in dirs:
    for filename in os.listdir(dir):
        f = open(os.path.join(dir, filename), 'r')
        contents = f.read()
        sentence_embedding = dictionary_creator.get_mapping(contents)
        x_test.append(sentence_embedding)
        y_test.append(ct)
    ct += 1

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

x_train = sequence.pad_sequences(x_train, maxlen=500)
x_test = sequence.pad_sequences(x_test, maxlen=500)

model = Sequential()
model.add(Embedding(5000, 32, input_length=500))
model.add(Flatten())
model.add(Dense(250, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

opt = Adam(lr=0.0005)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=3, shuffle=True)

scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save_weights('sent_nn.h5')
