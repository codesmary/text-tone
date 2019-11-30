import dictionary_creator

import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing import sequence
from keras.optimizers import Adam
import sqlite3
import string
import csv
import re
import os

#create indexed sets for our reddit data
original_phrases = []
x = []
conn = sqlite3.connect("data/reddit-comments.sqlite")
c = conn.cursor()
ct = 0

for row in c.execute('SELECT body FROM May2015'):
    row = row[0]
    sentence_embedding = dictionary_creator.get_mapping(row)
    original_phrases.append(row)
    x.append(sentence_embedding)
    ct += 1
    print(ct)
    if ct == 1000000:
        break

x = np.array(x)
x = sequence.pad_sequences(x, maxlen=500)

#load the sentiment model back in
model = Sequential()
model.add(Embedding(5000, 32, input_length=500))
model.add(Flatten())
model.add(Dense(250, activation = "relu"))
model.add(Dense(50, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))

model.load_weights('models/sent_nn.h5')

#create the csv file with the predicted sentiments
with open(os.path.join('data','reddit-comments-sentiment.csv'), 'w') as csvfile:
    filewriter = csv.writer(csvfile)
    filewriter.writerow(['Sentiment', 'Text'])
    for i in range(0, 1000000 - 32, 32):
        print(i)
        prediction = model.predict(x[i:32+i])
        for j in range(32):
            sentiment = 1 if prediction[j] >= 0.5 else 0
            text = original_phrases[i+j].replace('\n', ' ')
            filewriter.writerow([sentiment, text])
