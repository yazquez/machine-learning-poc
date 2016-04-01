import gensim
import nltk
from gensim.models.doc2vec import LabeledSentence
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# shuffle
from random import shuffle

# logging
import logging
import os.path
import sys




def load_data():
    # Descargamos los datos, los descomprimimos en la carpeta ./data/txt_sentoken
    # "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz")
    dataset = load_files('./data/txt_sentoken', shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    return dataset

dataset = load_data()

# 1. El analisis de sentimiento es un proceso por el que se intenta determinar si un texto tiene una carga
#    positiva o negativa. El dataset cargado contiene opiniones sobre varias películas. Estás opiniones han sido
#    categorizadas como positivas (1) o negativas (0).
#    Se pide crear un modelo que sea capaz de determinar el sentimiento de la crítica de una película.
print(dataset.data[2])
print()
print("Sentimiento : %d" % dataset.target[2])

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
       self.labels_list = labels_list
       self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(doc.decode("utf-8").split(),["SENT_" + '_%s' % self.labels_list[idx]])

    def to_array(self):
        sentences_to_array = []
        for idx, doc in enumerate(self.doc_list):
            sentences_to_array.append(LabeledSentence(doc.decode("utf-8").split(),["SENT_" + '_%s' % self.labels_list[idx]]))
        return sentences_to_array;
    def suffle(self):
        shuffle(self.sentences)
        return self.sentences



model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025) # use fixed learning rate

sentences = LabeledLineSentence(dataset.data, dataset.target)

# model.build_vocab(sentences)
model.build_vocab(sentences.to_array())

model.train(sentences)

# Training Doc2Vec
# Now we train the model. The model is better trained if in each training epoch,
# the sequence of sentences fed to the model is randomized.
# This is important: missing out on this steps gives you really shitty results.
# This is the reason for the suffle method in our LabeledLineSentences class.

for epoch in range(10):
    model.train(sentences.suffle())


# for epoch in range(10):
#     model.train(sentences)
#     model.alpha -= 0.002  # decrease the learning rate
#     model.min_alpha = model.alpha  # fix the learning rate, no decay