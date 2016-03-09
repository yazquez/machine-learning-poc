import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def load_titanic():
    with open('./data/titanic.csv') as csvfile:
        lector = csv.reader(csvfile, delimiter=",", quotechar='"')
        feature_names = np.array(lector.__next__())
        data = []
        target = []
        for linea in lector:
            data.append(linea)
            target.append(linea[2])
    return np.array(data), np.array(target), feature_names

X_titanic, y_titanic, X_names = load_titanic()

# ----------------------------------------------------------------------------
# Fase 1) Preparación de datos

# Nos quedamos con: "pclass", "age" and "sex"
X_titanic = X_titanic[:, [1, 4, 10]]
X_names = X_names[[1, 4, 10]]

# En los casos en los que la edad no sea conocida tomaremos la media
edades = X_titanic[X_titanic[:, 1] != 'NA', 1]
media = np.mean(edades.astype(float))

X_titanic[X_titanic[:, 1] == 'NA', 1] = media
X_titanic[:, 1]

# Asignamos un valor numérico a la variable sex (0:female, 1:male)
# X_titanic[X_titanic[:, 2] == 'female', 2] = 0
# X_titanic[X_titanic[:, 2] == 'male', 2] = 1
codificadorSexo = LabelEncoder()
codificadorSexo.fit(X_titanic[:, 2])
codificadorSexo.classes_
codificadorSexo.transform(X_titanic[:, 2])
X_titanic[:, 2] = codificadorSexo.transform(X_titanic[:, 2])


# Asignamos un valor numérico a la variable pclass ('1st': 1, '2nd': 2, '3rd': 3)
# X_titanic[X_titanic[:, 0] == '1st', 0] = 1
# X_titanic[X_titanic[:, 0] == '2nd', 0] = 2
# X_titanic[X_titanic[:, 0] == '3rd', 0] = 3
codificadorClase = LabelEncoder()
codificadorClase.fit(X_titanic[:, 0])
codificadorClase.classes_
codificadorClase.transform(X_titanic[:, 0])
X_titanic[:, 0] = codificadorClase.transform(X_titanic[:, 0])


clases = [[x] for x in X_titanic[:, 0].astype(int)]
modificador = OneHotEncoder()
modificador.fit(clases)
nuevas_clases = modificador.transform(clases).toarray()

X_titanic = np.delete(X_titanic,[0],1)
X_titanic.shape
X_titanic = np.concatenate((X_titanic, nuevas_clases),1)
X_titanic.shape

X_names = ["Edad", "Sexo", "Primera Clase", "Segunda Clase", "Tercera_Clase"]

X_titanic = X_titanic.astype("float")
y_titanic = y_titanic.astype("float")

X_train, X_test, y_train, y_test = train_test_split(X_titanic, y_titanic,test_size=0.25)
clasificador = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_leaf=3)
clasificador.fit(X_train, y_train)

y_predict = clasificador.predict(X_test)
print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))