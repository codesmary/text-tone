from pymongo import MongoClient
import numpy as np
import sqlite3
import string
import csv
import re
import os

#create vocab from first 1,000,000 reddit comments
conn = sqlite3.connect("data/reddit-comments.sqlite")
c = conn.cursor()

vocab = dict()
ct = 0
for row in c.execute('SELECT body FROM May2015'):
    row = row[0].lower()
    row = re.sub("[^%s]" % (string.ascii_lowercase + '\'\- '), ' ', row)
    for word in row.split(' '):
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1
    ct += 1
    if ct == 1000000:
        break

#add imdb vocab to our vocabulary
ct = 0
f = open(os.path.join('data', 'imdb.vocab'), 'r')
for word in f:
    word = word.lower()
    word = re.sub('[^%s]' % (string.ascii_lowercase + '\'\- '), '', word)
    if word not in vocab:
        vocab[word] = 1
    else:
        vocab[word] += 1

#create mapping from word: index for training
words_to_cts = [(word, count) for word, count in vocab.items()]
words_to_cts.sort(key=lambda x: x[1], reverse=True)

word_to_idxes = []
ct = 1

for word, count in words_to_cts:
    word_to_idxes.append({'word': word, 'index': ct})
    ct += 1

client = MongoClient()
db = client["word-idx-database"]
collection = db["word-idx-collection"]
db.posts.insert_many(word_to_idxes)


#helper function to return an array of mappings given an unedited
#string of words, for only top 5000 words
def get_mapping(words: str):
    client = MongoClient()
    db = client['word-idx-database']
    posts = db.posts

    mappings = []
    words = words.lower()
    words = re.sub('[^%s]' % (string.ascii_lowercase + '\'\- '), ' ', words)
    for word in words.split(' '):
        if not word:
            continue

        entry = posts.find_one({'word': word})        
        if entry:
            mapping = entry['index']
            if mapping >= 5000:
                mapping = 0
        else:
            mapping = 0

        mappings.append(mapping)
    
    return mappings
