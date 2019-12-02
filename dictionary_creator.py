from keras.preprocessing.text import Tokenizer
import numpy as np
import sqlite3
import csv
import os
import string
import re

MAX_SEQUENCE_LENGTH = 25
MAX_NB_WORDS = 5000

#create vocab from first 2,000,000 reddit comments
conn = sqlite3.connect("data/reddit-comments.sqlite")
c = conn.cursor()

texts = []
ct = 0
for row in c.execute('SELECT body FROM May2015'):
    row = row[0].split()
    '''
    text = ''
    for word in row:
        if len(text) + len(word) > MAX_SEQUENCE_LENGTH:
            break
        text += word
    texts.append(text)
    '''
    texts.append(row)
    ct += 1
    if ct == 1000000:
        break

#add imdb vocab to our vocabulary
ct = 0
f = open(os.path.join('data', 'imdb.vocab'), 'r')
for word in f:
    texts.append(word)

#create Tokenizer
tokenizer = Tokenizer(MAX_NB_WORDS, oov_token='unk')
tokenizer.fit_on_texts(texts)
tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= MAX_NB_WORDS}
tokenizer.word_index[tokenizer.oov_token] = 0

def get_mapping(words: str):
    mappings = []
    words = words.lower()
    words = re.sub('[^%s]' % (string.ascii_lowercase + '\'\- '), ' ', words)
    for word in words.split(' '):
        if not word:
            continue

        try:
            mapping = tokenizer.word_index[word]
        except:
            mapping = tokenizer.word_index[tokenizer.oov_token]

        mappings.append(mapping)

    return mappings
