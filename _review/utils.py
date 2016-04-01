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
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


def similares_ordenados_por_distancia(nuevo_texto,textos_entrenamiento,vec,texto_entr_vec,km):
    nuevo_texto_vec=vec.transform([nuevo_texto])
    etiqueta_nuevo_texto=km.predict(nuevo_texto_vec)[0] 
    indices_de_los_similares=(km.labels_==etiqueta_nuevo_texto).nonzero()[0]
    similares = []
    for i in indices_de_los_similares:
        disti = sp.linalg.norm((nuevo_texto_vec[0] - texto_entr_vec[i]).toarray())
        similares.append((disti, textos_entrenamiento[i]))
    return sorted(similares)


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

class StemmedTfidfVectorizer(TfidfVectorizer):
    def build_analyzer(self):
        spanish_stemmer = nltk.stem.SnowballStemmer("spanish")
        analyzer = super().build_analyzer()
        return lambda doc: (spanish_stemmer.stem(w) for w in analyzer(doc))
