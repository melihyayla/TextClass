# -*- coding: utf-8 -*-
"""
https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input, Flatten, Conv1D, MaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dropout
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from keras.preprocessing import text, sequence

STOPWORDS = set(stopwords.words('turkish'))
from bs4 import BeautifulSoup

from keras.models import Model

import cufflinks
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff

InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

from keras.preprocessing.text import Tokenizer

import pandas as pd
from keras.models import Sequential

from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
plt.style.use('ggplot')

import os

from TurkishStemmer import TurkishStemmer

from nltk.corpus import stopwords
from unicode_tr import unicode_tr

stemmer = TurkishStemmer()


def findfiles(path, flist):
    dirs = os.listdir(path)
    for df in dirs:
        if os.path.isdir(path + "/" + df):
            findfiles(path + "/" + df, flist)
        else:
            flist.append(path + "/" + df)


# path = './42bin_haber/news/'
# path = 'D:/Dropbox/Current Research/0_Turkish News Classification/42bin_haber/news

path = '/home/mes/Desktop/news'

# path = './TTC-3600_Orj/'
# path = 'C:/Users/Velid/Desktop/bin_haber/'
# categories = os.listdir(path)[1:]
#categories = ['ekonomi','kultur-sanat','magazin','saglik','siyaset','spor','teknoloji']
categories = ['siyaset']
news_files = {}
for cat in categories:
    flist = []
    findfiles(path + "/" + cat, flist)
    news_files[cat] = flist[1:]

myleng = len(categories)

num_class = len(categories)


def preprocess(words, stop_words=stopwords.words('turkish'), url_regex=None):
    # Remove URLS
    if url_regex:
        words = [word for word in words if not re.match(url_regex, word)]

    # Remove trash characters
    words = [re.sub("\xad|\x95|\x80|\x82|\x93|\x94|\x91|\x92|\x96|^\'+|^\*+|^-+|\'+$", "", word) for word in words]

    # Remove nonalphanumeric
    words = [word for word in words if not re.match("\W", word)]

    # Lower all words
    words = [unicode_tr(word.strip()).lower() for word in words if word.strip() != ""]

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    return words


category_corpus = {}
test_files = {}
all_words = []
X = []
y = []

for cat in categories:
    for f in news_files[cat]:
        with open(f, encoding="utf8") as file:
            lines = file.readlines()
            lines = ''.join(lines)

            words = word_tokenize(lines)
            # self.encoder = json.load(open(vocab_file, 'r', encoding='utf-8'))
            words = preprocess(words)

            X.append(words)
            y.append(cat)

X = pd.Series(X).astype(str)
y = pd.Series(y).astype(str)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

Y = pd.get_dummies(y).values
print('Shape of label tensor:', Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

op = 1


if op == 1:
    # word level tf-idf
    tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=70)
    tfidf_vect.fit(X_train)
    X_train = tfidf_vect.transform(X_train)
    X_test = tfidf_vect.transform(X_test)
elif op == 2:
    # ngram level tf-idf
    tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), max_features=70)
    tfidf_vect_ngram.fit(X_train)
    X_train = tfidf_vect_ngram.transform(X_train)
    X_test = tfidf_vect_ngram.transform(X_test)
elif op == 3:
    # characters level tf-idf
    tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2, 3),
                                             max_features=70)
    tfidf_vect_ngram_chars.fit(X_train)
    X_train = tfidf_vect_ngram_chars.transform(X_train)
    X_test = tfidf_vect_ngram_chars.transform(X_test)
elif op == 4:
    # load the pre-trained word-embedding vectors
    embeddings_index = {}
    for i, line in enumerate(open('/home/mes/Desktop/cc.tr.300.vec', encoding="utf8")):
        values = line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    # create a tokenizer
    token = text.Tokenizer()
    token.fit_on_texts(X_train)
    word_index = token.word_index

    # convert text to sequence of tokens and pad them to ensure equal length vectors
    X_train = sequence.pad_sequences(token.texts_to_sequences(X_train), maxlen=70)
    X_test = sequence.pad_sequences(token.texts_to_sequences(X_test), maxlen=70)

    # create token-embedding mapping
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


if op == 1 or op == 2 or op == 3 or op == 4:
    # create a tokenizer
    token = text.Tokenizer()
    token.fit_on_texts(X_train)
    word_index = token.word_index

use_LSTM = 0
if use_LSTM == 1:

    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

elif use_LSTM == 0:
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(35))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

epochs = 30
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test, Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0], accr[1]))

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();
