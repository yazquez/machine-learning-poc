""" Example using GenSim's LDA and sklearn. """
import gensim
import nltk
import numpy as np

from gensim import corpora
from gensim import matutils
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from sklearn import linear_model
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer


nltk.download("stopwords")
stop_words_english = stopwords.words("english")


def load_data():
    # Descargamos los datos, los descomprimimos en la carpeta ./data/txt_sentoken
    # "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz")

    dataset = load_files('./data/txt_sentoken', shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    return dataset

dataset = load_data()

texts = [[word for word in document.decode("utf-8").lower().split() if word not in stop_words_english] for document in dataset.data]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, update_every=1, chunksize=10000, passes=1)

lda.print_topics(12)

new_doc = "Amazon is a new acquisition"
new_text = [word for word in new_doc.lower().split() if word not in stop_words_english]
doc_bow  = dictionary.doc2bow(new_text)

lda[doc_bow]

lda.show_topics(15, 15)




def print_features(clf, vocab, n=10):
    """ Print sorted list of non-zero features/weights. """
    coef = clf.coef_[0]
    print ('positive features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[::-1][:n] if coef[j] > 0])))
    print ('negative features: %s' % (' '.join(['%s/%.2f' % (vocab[j], coef[j]) for j in np.argsort(coef)[:n] if coef[j] < 0])))


def fit_classifier(X, y, C=0.1):
    """ Fit L1 Logistic Regression classifier. """
    # Smaller C means fewer features selected.
    clf = linear_model.LogisticRegression(penalty='l1', C=C)
    clf.fit(X, y)
    return clf


def fit_lda(X, vocab, num_topics=5, passes=20):
    """ Fit LDA from a scipy CSR matrix (X). """
    print ('fitting lda...')
    return LdaModel(matutils.Sparse2Corpus(X), num_topics=num_topics,
                    passes=passes,
                    id2word=dict([(i, s) for i, s in enumerate(vocab)]))


def print_topics(lda, vocab, n=10):
    """ Print the top words for each topic. """
    # topics = lda.show_topics(topics=-1, topn=n, formatted=False)
    # for ti, topic in enumerate(topics):
    #     print ('topic %d: %s' % (ti, ' '.join('%s/%.2f' % (t[1], t[0]) for t in topic)))


if (__name__ == '__main__'):
    # Load data.
    rand = np.random.mtrand.RandomState(8675309)
    cats = ['rec.sport.baseball', 'sci.crypt']
    data = fetch_20newsgroups(subset='train',
                              categories=cats,
                              shuffle=True,
                              random_state=rand)
    vec = CountVectorizer(min_df=10, stop_words='english')
    X = vec.fit_transform(data.data)
    vocab = vec.get_feature_names()

    # Fit classifier.
    clf = fit_classifier(X, data.target)
    print_features(clf, vocab)

    # Fit LDA.
    lda = fit_lda(X, vocab)
    print_topics(lda, vocab)