# El objetivo de este código es, dado un texto, buscar aquellos
# que sean más similares

# import numpy as np
# import math
#
# A = np.array([1, 2, 3, 1])
# B = np.array([4, -1, 3, 2])
#
# d= A-B
#
# sp.linalg.norm(d)
#
# math.sqrt( (A[0]-B[0])**2 + (A[1]-B[1])**2 + (A[2]-B[2])**2 + (A[3]-B[3])**2 )

import nltk.stem
import nltk
import scipy as sp
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Teoría de las distancias
# http://turing.iimas.unam.mx/~ivanvladimir/es/post/intro_distancias_texto/
# http://www.sc.ehu.es/sbweb/energias-renovables/MATLAB/simbolico/geometria/geometria_1.html
# http://wwwae.ciemat.es/~cardenas/docs/lessons/MedidasdeDistancia.pdf

# https://es.wikipedia.org/wiki/Distancia_euclidiana
# La distancia entre dos puntos es el módulo del vector diferencia de los vectores v1-v2
def euclidean_distance(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())


# La distancia del coseno no es propiamente una distancia sino una medida de similaridad entre dos
# vectores en un espacio que tiene definido un producto interior.
# La similitud coseno es una medida del parecido entre dos vectores en un espacio que
# posee un producto interior, y resulta de evaluar el valor del coseno del ´angulo comprendido
# entre ellos. Para poder hacer uso de esta propiedad se utiliza el modelo vectorial, modelando
# a los documentos como vectores.
# Este modelo algebraico es utilizado para representar documentos en lenguaje natural de
# una manera formal, y tiene la caracter´ıstica de que favorece la direcci´on a la cu´al apuntan
# los documentos independientemente de su longitud. Consecuentemente, textos que hablan de
# los mismos temas en distinta cantidad de palabras pueden tener una gran similitud con esta
# m´etrica
# http://web.fi.uba.ar/~fbarrios/tprofesional/informe.pdf
def cos_distance(v1, v2):
    return 1 - (v1 * v2.transpose()).sum() / (sp.linalg.norm(v1.toarray()) * sp.linalg.norm(v2.toarray()))


def closest(X_train, docs, new_doc, vectorizer, distance_function=euclidean_distance):
    new_doc_vect = vectorizer.transform([new_doc])[0]
    closest_distance = float("inf")
    best_doc_index = None
    print("Vector de la consulta: %s" % new_doc_vect.toarray())

    for i, (doc, doc_vect) in enumerate(zip(docs, X_train)):
        d = distance_function(doc_vect, new_doc_vect)
        print("=== Documento {0} con distancia={1:.2f}: {2}".format(i, d, doc))

        if d < closest_distance:
            closest_distance = d
            best_doc_index = i

    print("El documento más cercano es {0} con distancia={1:.2f}".format(best_doc_index, closest_distance))



# Cargamos el corpus, esto es el conjunto de elementos con los que compararemos
# docs = open("./data/corpus-sample.txt", "r").readlines()

docs = open("/home/yazquez/Dev/Python/machine-learning-poc/classifier/text_classifier/data/corpus-sample.txt", "r").readlines()

docs = [l.strip() for l in docs]

# Antes de empezar a analizar las entradas tenemos que extraer y estructurar la
# información contenida en el corpus.

# CountVectorizer creará una matriz en la que cada columna representa una de las palabras
# del corpus (es decir todas las palabras posibles), y cada fila corresponde con uno de los
# documentos y en las celdas se indica el numero de veces que la palabra correspondiente a esa columna
# aparece en el documento.


vectorizer0 = CountVectorizer(min_df=1)
X_train0 = vectorizer0.fit_transform(docs)

# Mostramos el diccionario
print(vectorizer0.get_feature_names())
# Mostramos la matriz generada.
print(X_train0.toarray())
# El tamaño de la matriz sería 5x28, es decir 5 filas (Correspondientes a los 5 docuemntos
# y 28 columnas (correspondientes a las 28 palabras distintas que tiene el diccionario).
print(X_train0.shape)

new_doc = "camión de plata y oro"

# 1º Versión : Usamos el vectorizador creado y distancia euclidea
closest(X_train0, docs, new_doc, vectorizer0)

# 2º Versión : Si simplemente consideramos la distancia euclídea, puede haber mucha diferencia
#              entre textos de distinta longitud, aunque tuvieran la misma distribución relativa de
#              los términos.
#              Usamos el mismo vectorizador pero calcularemos las distancias usando la distancia coseno.
closest(X_train0, docs, new_doc, vectorizer0, distance_function=cos_distance)

# 3º Versión : En esta ocasión vamos a usar un vectorizador que excluya de las comparaciones las
#              palabras comunes del lenguaje (articulos, determinantes, etc.)
#              Asumimos que los textos estarán en castellano,
nltk.download("stopwords")
stop_words_spanish = stopwords.words("spanish")

vectorizer1 = CountVectorizer(min_df=1, stop_words=stop_words_spanish)

X_train1 = vectorizer1.fit_transform(docs)
# Mostramos el diccionario, podremos comprobar que han desaparecido las palabras del lenguaje
print(vectorizer1.get_feature_names())
# Mostramos la matriz generada.
print(X_train1.toarray())
# El tamaño de la matriz se ha reducido a 5x15
print(X_train1.shape)

closest(X_train1, docs, new_doc, vectorizer1, distance_function=cos_distance)

# 4º Versión : En esta versión vamos a usar Stemming.
#              Stemming es un método para reducir una palabra a su raíz o (en inglés) a un stem.
#              Es decir las palabras camión y camiones las consideraría una coincidencia.

spanish_stemmer = nltk.stem.SnowballStemmer('spanish')


# Vamos a crear una clase que hereda de CountVectorizer pero que aplica el proceso de stem.
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))


vectorizer2 = StemmedCountVectorizer(min_df=1, stop_words=stop_words_spanish)

X_train2 = vectorizer2.fit_transform(docs)
# Mostramos el diccionario, podremos comprobar que ahora está compuesto por las raices de los documentos
print(vectorizer2.get_feature_names())
# Mostramos la matriz generada.
print(X_train2.toarray())
# El tamaño de la matriz se ha reducido a 5x13
print(X_train2.shape)

closest(X_train2, docs, new_doc, vectorizer2, distance_function=cos_distance)


# 5º Versión : El valor aumenta tf-idf proporcionalmente al número de veces que una palabra aparece en el
#              documento, pero se compensa con la frecuencia de la palabra en el corpus, que ayuda a ajustar
#              por el hecho de que algunas palabras aparecen con mayor frecuencia en general.
#              En el modelo vectorial es usual considerar una variante que no da a todos los términos la misma
#              importancia.El esquema de pesos más comúnmente usado se denomina frecuencia documental inversa
#              La idea es restar importancia a términos que aparecen en muchos documentos

# Vamos a crear una clase para aplicar el stem, solo que en esta ocasión vamos a heredar de TfidfVectorizer
class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))


vectorizer3 = StemmedTfidfVectorizer(min_df=1, stop_words=stop_words_spanish)
X_train3 = vectorizer3.fit_transform(docs)

# Mostramos el diccionario, podremos comprobar que ahora está compuesto por las raices de los documentos
print(vectorizer3.get_feature_names())
print(vectorizer3.vocabulary)
# Mostramos la matriz generada.
print(X_train3.toarray())
# El tamaño de la matriz se ha reducido a 5x13
print(X_train3.shape)

closest(X_train3, docs, new_doc, vectorizer3, distance_function=cos_distance)


# Cambiamos el corpus para que la palabra "plata" aparezca muchas veces, esto la penalizará
# por ser muy general.
docs[0] = 'Éste texto no tiene nada que ver ocn los demás'
docs[1] = 'La plata fue entregada en camiones color plata'
docs[
    2] = 'El cargamento de oro y plata llegó en un camión El cargamento de oro y plata llegó en un camión El cargamento de oro y plata llegó en un camión'
docs[3] = 'Cargamentos de oro y plata dañados por el fuego'
docs[4] = 'El cargamento de oro y plata llegó en un camión'

X_train3 = vectorizer3.fit_transform(docs)
closest(X_train3, docs, new_doc, vectorizer3, distance_function=cos_distance)

zz = vectorizer3.transform([new_doc])[0]

zz.toarray()
