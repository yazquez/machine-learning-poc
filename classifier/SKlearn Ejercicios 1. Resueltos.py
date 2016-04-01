
# coding: utf-8

# In[1]:

#matplotlib: http://matplotlib.org/
import matplotlib

get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt

def representacion_grafica(datos,caracteristicas,objetivo,clases,c1,c2):

    for tipo,marca,color in zip(range(len(clases)),"soD","rgb"):
        plt.scatter(datos[objetivo == tipo,c1],
                    datos[objetivo == tipo,c2],marker=marca,c=color)

        plt.xlabel(caracteristicas[c1])
    plt.ylabel(caracteristicas[c2])
    


# In[2]:

# Me define que dos variables voy a usar para clasificar
c1 = 2
c2 = 3


# In[4]:

from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

#leemos el dataset
iris = load_iris()

#dividimos en test y training
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.75, random_state=342)
X_names = iris.feature_names
clases = iris.target_names


# In[5]:

# 1. Repetir el proceso visto en clase (sin normalizar) para clasificar la especie del iris setosa pero tomado las 
#    características longitud y ancho del pétalo. Representar la distribución de las características. 
#    Probar los resultados normalizando y sin normalizar. ¿ Influye la normalización sobre este clasificador ? 
#    Obtener las métricas de precisión, exactitud, ...

from sklearn.linear_model import SGDClassifier

#representación de las variables
representacion_grafica(X, X_names, y, clases, c1, c2)




# In[58]:

# se aprecia en la representación que este caso es mejor "separable" -> podemos encontrar dos rectas que nos 
# separen las tres clases

# entrenamos sin normalizar
def train_and_score(X_train, y_train, X_test, y_test, c1, c2):
    sgd = SGDClassifier().fit(X_train[:,[c1,c2]], y_train)
    nscore = sgd.score(X_test[:,[c1,c2]], y_test)
    print('SGD score: %f' % nscore)

train_and_score(X_train, y_train, X_test, y_test, c1, c2)


# In[59]:

# normalizamos
from sklearn.preprocessing import StandardScaler

#creamos el normalizador y entrenamos
normalizador = StandardScaler().fit(X_train)

#normalizamos los datos
X_train_norm = normalizador.transform(X_train)
X_test_norm = normalizador.transform(X_test)

train_and_score(X_train_norm, y_train, X_test_norm, y_test, c1, c2)


# In[ ]:

# Si se ejecuta en varias ocasiones ambos entrenamientos (sin normalizar y normalizando) podemos 
# observar que el rendimiento normalizando es superior en la mayoría de las ocasiones. 


# In[63]:

# 2. Cada vez que lanzamos el entrenamiento, el modelo resultante tendrá un rendimiento diferente. Crear un
#    algoritmo que entrene el modelo 10000 veces motrando por consola cada vez que se obtiene un modelo mejor
#    que el mejor obtenido hasta el momento. Usad el método score de SGDClassifier para obtener la medida de 
#    rendimiento. Mostrar por consola el rendimiento sobre los datos de entrenamiento y sobre los datos de test.
#   ¿ Que implica que el rendimiento sobre el test sean mejor que sobre los de entrenamiento ? ¿ y al contrario ? 
#   De todas las salidas obtenidas, ¿ cuál considerarías que es el mejor modelo ?

class MultiTrainer():
    
    def __init__(self, alg, iterations=10000):
        self.alg = alg
        self.iterations = 10000
        
    def fit(self, X_train, y_train, X_test, y_test):
        
        score = 0
        self.best_sgd = None
        for i in range(0,self.iterations):
            sgd = self.alg.fit(X_train, y_train)
            nscore = sgd.score(X_test, y_test)
            train_score = sgd.score(X_train, y_train)

            if nscore > score:
                print('SGD score: %f (train: %f)' % (nscore, train_score) )
                score = nscore
                self.best_sgd = sgd
                
        return self
                
mt = MultiTrainer(SGDClassifier())
mt.fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)
                


# In[ ]:

# Al estar tomando el modelo que mejor rendimiento obtiene sobre el test estamos sobre-ajustando el modelo a los datos
# del test. Esto se observa viendo como el rendimiento en el test deja a aumentar mientras que el rendimiento en el test
# continua aumentando. 

# Lo ideal es que el rendimiento sobre el test y sobre el train sean los más similares posibles. De esta forma tenemos
# cierta fiabilidad del rendimiento de nuestro modelo sobre nuevos datos. 

# En la prueba anterior, el modelo a seleccionar sería el 4º: SGD score: 0.946903 (train: 0.891892), 
# al existir una menor diferencia entre el rendimiento de test y training.


# In[88]:

# 3. SGDClassifier tiene multitud de parámetros de entrada que pueden influir sobre el rendimiento del modelo.
#    Revisar los parámetros en http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#    e intentar manipular estos parámetros para aumentar el rendimiento del modelo anterior.

                
mt = MultiTrainer(SGDClassifier(average=True, alpha=0.35)).fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)
              


# In[ ]:

# Probando con varios valores de alpha progresivamente: 0.1, 0.2, 0.3, 0.4, 0.6 (disminuye el rendimiento) -> 0.35, 
# llegamos a consguir el modelo que mejora tanto en trainig como en test el anterior: SGD score: 0.964602 (train: 0.945946)
# además, la diferencia de rendimiento entre test y training es menor.


# In[90]:

# 4. Representar graficamente las líneas de clasificación del mejor clasificador obtenido

plt.clf()

def representacion_separador(X,X_names,y,clases, c1, c2, classifier):
    representacion_grafica(X,X_names,y,clases,c1,c2)
    plt.legend(clases)

    xmin, xmax = X[:,c1].min(), X[:,c1].max()
    ymin, ymax = X[:,c2].min(), X[:,c2].max()

    xs = np.arange(xmin,xmax,0.5)
    plt.xlim(xmin-0.5,xmax+0.5)
    plt.ylim(ymin-0.5,ymax+0.5)


    for i,c in zip(range(3),"rgb"):
        m = -mt.alg.coef_[i,0] / mt.alg.coef_[i,1]
        n = -mt.alg.intercept_[i] / mt.alg.coef_[i,1]

        ys = (xs*m + n) 
        plt.plot(xs,ys,hold=True,color=c)
        
plt.figure(2, figsize=(20, 6))
plt.subplot(121)
representacion_separador(X_train, X_names, y_train, clases, c1, c2, mt.best_sgd)
plt.subplot(122)
representacion_separador(X_test, X_names, y_test, clases, c1, c2, mt.best_sgd)


# In[96]:

# 5. En http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html podeis encontrar
# una revisión de varios clasificadores. Probad algunos de ellos...

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


mt = MultiTrainer(KNeighborsClassifier()).fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)


# In[97]:

mt = MultiTrainer(GaussianNB()).fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)


# In[95]:

mt = MultiTrainer(DecisionTreeClassifier()).fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)


# In[94]:

mt = MultiTrainer(RandomForestClassifier()).fit(X_train[:,[c1,c2]], y_train, X_test[:,[c1,c2]], y_test)


# In[118]:

# 6. Repetir los ejercicios anteriores normalizando los datos. ¿Se aprecia alguna mejora? 

mt = MultiTrainer(SGDClassifier(average=True, alpha=0.8)).fit(X_train_norm[:,[c1,c2]], 
                                                               y_train, X_test_norm[:,[c1,c2]], y_test)
       


# In[ ]:



