# Clustering
#
# Técnicas de agrupamiento o clustering
# Se trata de dividir un conjunto de datos de entrada en
# subconjuntos (clusters), de tal manera que los elementos
# de cada subconjunto compartan cierto patrón o
# características a priori desconocidas
# Aprendizaje no supervisado: no tenemos información
# sobre qué cluster corresponde a cada dato.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets


# ----------------------------------------------------------------------------------------------------------------------
# 1) Ejemplo - Clusterización de los datos de IRIS

np.random.seed(5)
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Podemos aplicar k -medias, con k = 3 y distancia euclídea,
# ignorando el último atributo (como si no se conociera):
# • En 6 iteraciones se estabiliza
# • De los tres clusters obtenidos, el primero incluye
# justamente a las 50 instancias que originalmente estaban
# clasificadas como iris setosa
# • El segundo cluster incluye a 47 versicolor y a 3 virgínicas
# • El tercero incluye 14 versicolor y 36 virgínicas
# • No ha sido capaz de discriminar correctamente entre
# versicolor y virgínica

# En este ejemplo partimos de que conocemos el número de cluster
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Mostramos el resultado
labels = kmeans.labels_
print(labels)
# Dado que en tenemos el resultado real de la clasificación podemos compararlo
print(y)


# Pintamos la representación frafica
fig = plt.figure(1, figsize=(10, 10))
plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=28, azim=134)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')


# ----------------------------------------------------------------------------------------------------------------------
# 2) Ejemplo - Búsqueda de documentos similares usando clustering.


# En esta segunda parte vamos a usar las herramientas vistas en la primera
# parte de la práctica, junto con k-medias, para experimentar sobre el
# conjunto de datos "20newsgroups". La idea es que seamos capaces de mostrar
# los posts más relacionados con un nuevo post que se escriba. Para ello
# usamos el esquema tfidf para la extracción de características de un texto, y
# en lugar de buscar los más cercanos de entre todos el conjunto de posts,
# busco el más cercano de entre los del mismo cluster. 

import nltk.stem
import scipy as sp
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = nltk.stem.SnowballStemmer('english')

# Ejemplo de nuevo post
new_post = """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""


# ---------------------
# 1. Carga de los posts
# --------------------

# Al descargar 20_newsgroup mediante fetch_20newsgroup la primera vez, da un
# mensaje que NO es de error. Simplemente hay que esperar, NO CORTAR (si se
# corta, al cargarlo después da un error, y habría que colocar a mano
# (entrando en la web de MLCOMP), el fichero 20news-bydate.tar.gz)

datos_20newsgroup = sklearn.datasets.fetch_20newsgroups(subset="all")

# Podemos ver los grupos de post
print(datos_20newsgroup.target_names)

# Por simplificar, nos quedamos sólo con los grupos de informática:

target_groups = [
    'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']


# Para esos grupos, cargamos los posts de train y de test
train_data = sklearn.datasets.fetch_20newsgroups(subset="train", categories=target_groups)
test_data = sklearn.datasets.fetch_20newsgroups(subset="test", categories=target_groups)


# ---------------------------
# 2. Vectorizado de los posts
# ----------------------------

# Usamos stop_words, stemming y tfidf, con min_df=10 y max_df=0.5
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (stemmer.stem(w) for w in analyzer(doc))


vectorizer20 = StemmedTfidfVectorizer(min_df=10, max_df=0.5, stop_words='english', decode_error='ignore')

vectorized20=vectorizer20.fit_transform(train_data.data)

# Consultamos la matriz obtenida
vectorized20.shape  # 3529 posts, con 4712 características

# Mostramos algunos terminos del vocabulario obtenido, Por ejemplo, 10 términos
print(vectorizer20.get_feature_names()[1000:1010])

# --------------------------------
# 3. K-medias sobre los documentos
# --------------------------------

from sklearn.cluster import KMeans

centers = [[1, 1], [-1, -1], [1, -1]]

# El número de clusters lo ponemos a 50, pero deberíamos experimentar.
clusters_number=50

km20 = KMeans(n_clusters=clusters_number, n_init=1, verbose=1, random_state=3)

# Entrenamos (Nota: Esto tardar unos segundos)
clustered20 = km20.fit(vectorized20)

# Explicar
print(clustered20.cluster_centers_)
# Verificamos que efectivamente se han creado 50 clusters
len(clustered20.cluster_centers_)

clustered20.labels_


# ----------------------------
# 4. Búsqueda de los similares
# ----------------------------

# Un ejemplo sobre cómo clasificar un nuevo post y quedarse con los del mismo grupo:

nuevo_post_vec=vectorizer20.transform([new_post])
nuevo_post_label=km20.predict(nuevo_post_vec)[0]
indices_de_los_similares=(km20.labels_ == nuevo_post_label).nonzero()[0]

indices_de_los_similares

# La siguiente función recibe:
# - Un texto nuevo (string)
# - Todos los textos del conjunto de entrenamiento (por ejemplo, el
#   train_data.data de arriba) (array de strings)
# - Un vectorizador vec (de la clase TfidfVectorizer) y además supondremos que
#   ya hemos hecho fit_transform sobre los textos del conjunto
#   de entrenamiento (el segundo argumento) 
# - Los textos del conjunto de entrenamiento vectorizados con el vectorizador
#   anterior. 
# - Un kmedias (objeto de la clase KMeans), en el que se supone
#   que ya se ha hecho fit sobre el vectorizado de los textos del conjunto de
#   entrenamiento (el tercer argumento) [es decir, lo que hemos hecho con el km de arriba].


def similares_ordenados_por_distancia(nuevo_texto,textos_entrenamiento,vec,texto_entr_vec,km):
    nuevo_texto_vec=vec.transform([nuevo_texto])
    etiqueta_nuevo_texto=km.predict(nuevo_texto_vec)[0] 
    indices_de_los_similares=(km.labels_==etiqueta_nuevo_texto).nonzero()[0]
    similares = []
    for i in indices_de_los_similares:
        disti = sp.linalg.norm((nuevo_texto_vec[0] - texto_entr_vec[i]).toarray())
        similares.append((disti, textos_entrenamiento[i]))
    return sorted(similares)


# In[28]:

# La siguiente función llama a la anterior e imprime textos del clúster más
# cercano a nuevo_texto. En concreto imprime los tres más cercanos, el primer
# decil y el menos cercano.

def imprime_5_similares(nuevo_texto,textos_entrenamiento,vec,texto_entr_vec,km):

    similares=similares_ordenados_por_distancia(nuevo_texto,textos_entrenamiento,vec,texto_entr_vec,km)

    print("Número de textos similares: {}".format(len(similares)))

    primero = similares[0]
    segundo = similares[1]
    tercero = similares[2]
    
    decil = similares[int(len(similares) / 10)]
    ultimo = similares[int(len(similares) -1)]


    print("=== Primero ===")
    print(primero[0])
    print(primero[1])
    print()

    print("=== Segundo ===")
    print(segundo[0])
    print(segundo[1])
    print()

    print("=== Tercero ===")
    print(tercero[0])
    print(tercero[1])
    print()

    print("=== Decil ===")
    print(decil[0])
    print(decil[1])
    print()


    print("=== Último ===")
    print(ultimo[0])
    print(ultimo[1])
    print()


# In[29]:

print(new_post)
imprime_5_similares(new_post, train_data.data, vectorizer20, vectorized20, km20)


# In[ ]:



