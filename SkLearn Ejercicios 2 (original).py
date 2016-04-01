

try:
    from urllib import urlopen
except ImportError:
    from urllib.request import urlopen


# URL = ("http://www.cs.cornell.edu/people/pabo/"
#        "movie-review-data/review_polarity.tar.gz")
#
# ARCHIVE_NAME = URL.rsplit('/', 1)[1]
# DATA_FOLDER = "txt_sentoken"
#
#
# if not os.path.exists(DATA_FOLDER):
#     if not os.path.exists(ARCHIVE_NAME):
#         print("Downloading dataset from %s (3 MB)" % URL)
#         opener = urlopen(URL)
#         with open(ARCHIVE_NAME, 'wb') as archive:
#             archive.write(opener.read())
#
#     print("Decompressing %s" % ARCHIVE_NAME)
#     with closing(tarfile.open(ARCHIVE_NAME, "r:gz")) as archive:
#         archive.extractall(path='.')
#     os.remove(ARCHIVE_NAME)
    
#load data
from sklearn.datasets import load_files
dataset = load_files('./data/txt_sentoken', shuffle=False)
print("n_samples: %d" % len(dataset.data))

#create text and train
from sklearn.cross_validation import train_test_split
# split the dataset in training and test set:
X_train, X_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=3948)

X_train[2]
# In[14]:

# 1. El analisis de sentimiento es un proceso por el que se intenta determinar si un texto tiene una carga 
#    positiva o negativa. El dataset cargado contiene opiniones sobre varias películas. Estás opiniones han sido
#    categorizadas como positivas (1) o negativas (0).
print(dataset.data[2])
print()
print("Sentimiento : %d" % dataset.target[2])


# In[ ]:

# Se pide crear un modelo que sea capaz de determinar el sentimiento de la crítica de una película.


# In[ ]:

# 2. Además de los modelos de procesamiento de texto vistos en clase existen otros modelos basados en topics. 
#    Gensim es una librería orientada al procesamiento del lenguage natural. Aquí podeís encontrar un tutoral: 
#    https://radimrehurek.com/gensim/tutorial.html. Scikit-learn también tiene una implementación de  Usar alguno de los modelos de gensim por ejemplo LDA o LSA para
#    realizar la tarea de clasificación anterior.

