import gensim
import nltk
from gensim import corpora
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics





def load_data2():
    # Descargamos los datos, los descomprimimos en la carpeta ./data/txt_sentoken
    # "http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz")
    dataset = load_files('./data/txt_sentoken', shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    return dataset

def load_data():
    try:
        from urllib import urlopen
    except ImportError:
        from urllib.request import urlopen
    import os
    import tarfile
    from contextlib import closing
    try:
        from urllib import urlopen
    except ImportError:
        from urllib.request import urlopen


    URL = ("http://www.cs.cornell.edu/people/pabo/"
           "movie-review-data/review_polarity.tar.gz")

    ARCHIVE_NAME = URL.rsplit('/', 1)[1]
    DATA_FOLDER = "txt_sentoken"


    if not os.path.exists(DATA_FOLDER):

        if not os.path.exists(ARCHIVE_NAME):
            print("Downloading dataset from %s (3 MB)" % URL)
            opener = urlopen(URL)
            with open(ARCHIVE_NAME, 'wb') as archive:
                archive.write(opener.read())

        print("Decompressing %s" % ARCHIVE_NAME)
        with closing(tarfile.open(ARCHIVE_NAME, "r:gz")) as archive:
            archive.extractall(path='.')
        os.remove(ARCHIVE_NAME)

    from sklearn.datasets import load_files
    return load_files('txt_sentoken', shuffle=False)



dataset = load_data()

# 1. El analisis de sentimiento es un proceso por el que se intenta determinar si un texto tiene una carga
#    positiva o negativa. El dataset cargado contiene opiniones sobre varias películas. Estás opiniones han sido
#    categorizadas como positivas (1) o negativas (0).
#    Se pide crear un modelo que sea capaz de determinar el sentimiento de la crítica de una película.
print(dataset.data[2])
print()
print("Sentimiento : %d" % dataset.target[2])


#
# El caso de estudio de caso sobre las críticas de películas se ha considerado y el repositorio de reseñas de películas se almacenan en
# un formato desestructurado#. Estos datos no estructurados deben ser convertidos a datos significativos con el fin de aplicar el algoritmo
# de aprendizaje automático. El tratamiento de los datos no estructurados incluye la eliminación de la información vaga o la eliminación de innecesarios
# espacios en blanco. Estos datos procesados se convierte a vectores numéricos donde cada columna corresponde a una de las palabras del corpus
# y el valor corresponde a la presencia de esa palabra en el documento en cuestión

# Para el procesamiento de los textos vamos a excluir las llamadas "Stop words"
nltk.download("stopwords")
stop_words_english = stopwords.words("english")

# Tambien vamos a usar Stemming, es un método para reducir una palabra a su raíz
# Es decir las palabras fool y foolish ylas consideraría una coincidencia.
# El proceso de stemming o lemmatización consiste en quitar la desinencia de las diferentes
# téerminos, para llevarlos a su raíz. Esta etapa se lleva a cabo con el objetivo de que el algoritmo
# detecte cuando se están tratando los mismos conceptos, dado que se utilizan las mismas
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


# El algoritmo de aprendizaje es aplicable cuando el conjunto de datos de las etiqueta está disponible. El conjunto de datos utilizados
# En este estudio se etiqueta todo conjunto de datos y cada revisión en el corpus se etiqueta, ya sea positivo o negativo.


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

# Gensim utiliza un paradigma muy poderoso en el área de procesamiento de lenguaje
# natural, denominado VSM (modelo de espacio de vectores), en el cual se representan los
# documentos de un corpus como vectores de un espacio multidimensional. Esta representación
# explota la idea de que los textos en el área del modelado de tópicos se pueden expresar según
# un número de conceptos, lo que aumenta la eficiencia y también ayuda a eliminar ruido.

# LSI y	LDA son métodos	de	Reducción	de	Dimensionalidad, principalmente usados	en	textos,
# pero ya desde hace varios año con	otras aplicaciones
# LSI : Latent	SemanDc	Indexing)
# LDA : Latent	Dirichlet	AllocaDon

# Creamos el modelo LDA

docs = [[english_stemmer.stem(word) for word in document.decode("utf-8").lower().split() if word not in stop_words_english] for document in dataset.data]
dictionary = corpora.Dictionary(docs)
corpus = [dictionary.doc2bow(doc) for doc in docs]

lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=2, update_every=1, chunksize=10000, passes=1)
