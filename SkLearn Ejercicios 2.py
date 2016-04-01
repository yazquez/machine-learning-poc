import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics



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

#
# The case study on movie-reviews has been considered and repository of movie reviews is stored in unstructured
# textual format. This unstructured data need to be converted in to meaningful data in order to apply machine learning
# algorithms.  The  processing  of  unstructured  data  includes  removal  of  vague  information,  removal  of  unnecessary
# blank spaces. This processed data is converted to numerical vectors where each vector corresponds to a review and
# entries of each vector represent the presence of feature in that particular review.

# Para el procesamiento de los textos vamos a excluir las llamadas "Stop words"
nltk.download("stopwords")
stop_words_english = stopwords.words("english")

# Tambien vamos a usar Stemming, es un método para reducir una palabra a su raíz
# Es decir las palabras fool y foolish ylas consideraría una coincidencia.
# El proceso de stemming o lemmatizaci´on consiste en quitar la desinencia de las diferentes
# t´erminos, para llevarlos a su ra´ız. Esta etapa se lleva a cabo con el objetivo de que el algoritmo
# detecte cuando se est´an tratando los mismos conceptos, dado que se utilizan las mismas
# palabras. Por ejemplo, las palabras “canta”, “cantaba”, “cantando”, pueden ser asociadas al
# verbo “cantar”.

english_stemmer = nltk.stem.SnowballStemmer('english')

# Vamos a crear una clase que hereda de CountVectorizer pero que aplica el proceso de stemming.
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


# Creamos el vectorizador usando la lista de "Stop Words" Creada
vectorizer = StemmedCountVectorizer(min_df=1, stop_words=stop_words_english)

# Una vez creado el vectorizador lo aplicamos sobre el conjunto de documentos (las opiniones) que tenemos
vectorized_data = vectorizer.fit_transform(dataset.data)

# Mostramos alguna informacion
# Diccionario
print(vectorizer.get_feature_names())
# Matriz generada.
# print(vectorized_data.toarray())
# Tamaño de la matriz
# El tamaño de la matriz sería 2000x25570, es decir 2000 filas (Correspondientes a las 2000 criticas
# y 25570 columnas (correspondientes a las 25570 palabras distintas que tiene el diccionario).
print(vectorized_data.shape)



# The supervised machine learning algorithm is applicable where the labeled dataset is available. The dataset used
# in this study is labeled dataset and each review in the corpus is either labeled as positive or negative. Two diơerent
# machine learning algorithms considered in this study this is going to be Stochastic Gradient Descent (SGD) is a simple
# yet very efficient approach to discriminative learning of linear classifiers

# split the dataset in training and test set:
X_train, X_test, y_train, y_test = train_test_split(vectorized_data, dataset.target, test_size=0.25, random_state=3948)

classifier = SGDClassifier().fit(X_train, y_train)

# Mostramos algunas metricas
y_predict = classifier.predict(X_test)
print("Score    :", classifier.score(X_test, y_test))
print("Success  :", sum(y_predict == y_test))
print("Errors   :", sum(y_predict != y_test))
print("Accuracy :", metrics.accuracy_score(y_test, y_predict))
print("Confusion Matrix\n", metrics.confusion_matrix(y_test, y_predict))


# 2. Además de los modelos de procesamiento de texto vistos en clase existen otros modelos basados en topics.
#    Gensim es una librería orientada al procesamiento del lenguage natural. Aquí podeís encontrar un tutoral: 
#    https://radimrehurek.com/gensim/tutorial.html. Scikit-learn también tiene una implementación de  Usar alguno de los modelos de gensim por ejemplo LDA o LSA para
#    realizar la tarea de clasificación anterior.

# Gensim utiliza un paradigma muy poderoso en el ´area de procesamiento de lenguaje
# natural, denominado VSM (modelo de espacio de vectores), en el cual se representan los
# documentos de un corpus como vectores de un espacio multidimensional. Esta representaci´on
# explota la idea de que los textos en el ´area del modelado de t´opicos se pueden expresar seg´un
# un n´umero de conceptos, lo que aumenta la eficiencia y tambi´en ayuda a eliminar ruido.


# LSI	&	LDA
# • Métodos	de	Reducción	de	Dimensionalidad,
# principalmente usados	en	textos,	pero ya
# desde hace varios años	con	otras aplicaciones
# LSI: Latent	SemanDc	Indexing)
# LDA	 Latent	Dirichlet	AllocaDon


from gensim.corpora import TextCorpus, MmCorpus, Dictionary

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy

# random
from random import shuffle

# classifier
from sklearn.linear_model import LogisticRegression

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences







