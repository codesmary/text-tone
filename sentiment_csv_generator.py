import dictionary_creator

import numpy as np
from keras.layers.embeddings import Embedding
from keras.models import Model, Sequential
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Activation
from keras.preprocessing import sequence
from keras.optimizers import Adam
import sqlite3
import csv
import os

chunk_size = 1000000

#load the sentiment model back in
def load_model():
    model = Sequential()
    model.add(Embedding(5001, 32, input_length=500))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=32, kernel_size=7, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(Conv1D(filters=64, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(os.path.join('models', 'sent_nn.h5'))
    print("Model weights loaded.")

    return model

#create indexed sets for our reddit data
def prepare_data(chunk_offset: int):
    print("Preparing data.")
    original_phrases = []
    x = []
    conn = sqlite3.connect(os.path.join("data","reddit-comments.sqlite"))
    c = conn.cursor()

    for row in c.execute('SELECT body FROM May2015 LIMIT ? OFFSET ?',\
                        (chunk_size, chunk_offset*chunk_size)):
        row = row[0]
        sentence_embedding = dictionary_creator.get_mapping(row)
        original_phrases.append(row)
        x.append(sentence_embedding)

    x = np.array(x)
    x = sequence.pad_sequences(x, maxlen=500)

    print("Original phrases and x aggregated.")
    return original_phrases, x

#create the csv file with the predicted sentiments
def write_to_file(initial_write: bool, model, original_phrases, x):
    print("Beginning writing to file.")
    with open(os.path.join('data','reddit-comments-sentiment.csv'),\
                           'w' if initial_write else 'a') as csvfile:
        filewriter = csv.writer(csvfile)
        if initial_write:
            filewriter.writerow(['Sentiment', 'Text'])
        for i in range(0, chunk_size - 32, 32):
            prediction = model.predict(x[i:32+i])
            for j in range(32):
                sentiment = 1 if prediction[j] >= 0.5 else 0
                text = original_phrases[i+j].replace('\n', ' ')
                filewriter.writerow([sentiment, text])
    print("File " + ("written to." if initial_write else "appended to."))

model = load_model()
been_first = True
for i in range(5):
    original_phrases, x = prepare_data(i)
    write_to_file(been_first, model, original_phrases, x)
    been_first = False