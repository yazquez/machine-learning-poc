import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_predict


def load_titanic():
    with open("titanic.csv") as csvfile:
        lector = csv.reader(csvfile, delimiter=",", quotechar='"')
        feature_names = np.array(lector.__next__())
        data = []
        target = []
        for linea in lector:
            data.append(linea)
            target.append(linea[2])
    return np.array(data), np.array(target), feature_names


X_titanic, y_titanic, X_names = load_titanic()

X_titanic = X_titanic[:, [1, 4, 10]]
X_names = X_names[[1, 4, 10]]

edades = X_titanic[X_titanic[:, 1] != 'NA', 1]

media = np.mean(edades.astype(float))

X_titanic[X_titanic[:, 1] == 'NA', 1] = media

X_titanic[:, 1]
