from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import numpy as np
import sqlite3
import string
import csv
import re
import os

MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 20000

#create vocab from first 1,000,000 reddit comments
conn = sqlite3.connect("data/reddit-comments.sqlite")
c = conn.cursor()

texts = []
ct = 0
for row in c.execute('SELECT body FROM May2015'):
    row = row[0].split()
    text = ''
    for word in row:
        if len(text) + len(word) > MAX_SEQUENCE_LENGTH:
            break
        text += word
    texts.append(text)
    ct += 1
    if ct == 1000000:
        break

#add imdb vocab to our vocabulary
ct = 0
f = open(os.path.join('data', 'imdb.vocab'), 'r')
for word in f:
    texts.append(word)

#create Tokenizer
tokenizer = Tokenizer(MAX_NB_WORDS + 1, oov_token='unk')
tokenizer.fit_on_texts(texts)
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS}
tokenizer.word_index[tokenizer.oov_token] = MAX_NB_WORDS + 1

def get_mapping(words: str):
    return tokenizer.texts_to_sequences(words)
