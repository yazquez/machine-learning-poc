# Cargamos los datos de Iris desde Scikit-learn

# Graficamos

# Importamos las librerias necesarias

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris

# Un aspecto sutil, es que al establecer una métrica entre frases, puede resultar que una frase es
# repetida más de dos ocasiones, ejemplo “Cómo estas, cómo estas, cómo estas”. Dentro de la bolsa de palabras
# lo que uno obtendrá será un vector como el de la frase “Cómo estas” pero multiplicado por 3. Es decir, un
# vector de la forma (3,0,0,3,0,0,0,0,), entonces lo que se hace para evitar este tipo de anomalías es algo
# usual en álgebra lineal, normalizar los vectores. Es decir, se cambia el vector origina (3,0,0,3,0,0,0,0)
# dividido entre raíz de 2, lo cual hace que el vector sea igual en la norma al obtenido de (1,0,0,1,0,0,0,0).
# Así esas dos frases ya pueden considerarse iguales en el espacio de frases.




# # Cargamos los datos y graficamos

datos = load_iris()
caract = datos.data
caract_names = datos.feature_names
tar = datos.target

# Graficamos los datos con colores distintos y tipo de marcas distintos
for t, marca, c in zip(range(3), ">ox", "rgb"):
    plt.scatter(caract[tar == t, 0], caract[tar == t, 1], marker=marca, c=c)

plt.show()

# Importamos las librerias necesarias

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import mlpy
from sklearn.cluster import KMeans

#Cargamos los datos y graficamos

datos=load_iris()
dat=datos.data
caract_names=datos.feature_names
tar=datos.target

#Calculamos los cluster

cls, means, steps = mlpy.kmeans(dat, k=3, plus=True)

#steps
#Esta variable permite conocer los pasos que realizó el algoritmo para terminar

#Construimos las gráficas correspondiente

plt.subplot(2,1,1)
fig = plt.figure(1)
fig.suptitle("Ejemplo de k-medias",fontsize=15)
plot1 = plt.scatter(dat[:,0], dat[:,1], c=cls, alpha=0.75)
#Agregamos las Medias a las gráficas

plot2 = plt.scatter(means[:,0], means[:,1],c=[1,2,3], s=128, marker='d')
#plt.show()

#Calculamos lo mismo mediante la librería scikit-lean

KM=KMeans(init='random',n_clusters=5).fit(dat)

#Extraemos las medias

L=KM.cluster_centers_

#Extraemos los valores usados para los calculos

Lab=KM.labels_

#Generamos las gráfica

plt.subplot(2,1,2)
fig1= plt.figure(1)
fig.suptitle("Ejemplo de k-medias",fontsize=15)
plot3= plt.scatter(dat[:,0], dat[:,1], c=Lab, alpha=0.75)

#Agregamos las Medias a las gráficas

plot4= plt.scatter(L[:,0], L[:,1],c=[1,2,3,4,5], s=128, marker='d')

#Mostramos la gráfica con los dos calculos

plt.show()
