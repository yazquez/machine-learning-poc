import numpy as np
# import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def load_titanic():
    with open('../data/titanic.csv', "r") as csvfile:
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




# # References:
# # http://pandas.pydata.org/pandas-docs/stable/missing_data.html
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# df = pd.read_csv('./data/titanic.csv')
# # df = pd.read_csv('./data/titanic-only10.csv')
#
# # With "info" we get a insight of how are our data,
# # for example, how many data we have in each variable.
# # For example we can see that there aren't too many data in
# # room and boat variable, and we have just half of ages.
# df.info()
# # <class 'pandas.core.frame.DataFrame'>
# # Int64Index: 1313 entries, 0 to 1312
# # Data columns (total 11 columns):
# # row.names    1313 non-null int64
# # pclass       1313 non-null object
# # survived     1313 non-null int64
# # name         1313 non-null object
# # age          633 non-null float64
# # embarked     821 non-null object
# # home.dest    754 non-null object
# # room         77 non-null object
# # ticket       69 non-null object
# # boat         347 non-null object
# # sex          1313 non-null object
#
# # Stage 1) Cleaning the data
# #
#
#
# # A lot of Machine learning algorithms are going to need numeric
# # values, so we converted string values to int
# df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)
# # df['embarked'] = df['embarked'].fillna('Unknown')
# # df['embarked'] = df['embarked'].map({'Cherbourg': 1, 'Queenstown': 2, 'Southampton': 3, 'Unknown': 4}).astype(int)
# df['age'] = df['age'].fillna(df['age'].mean())
# df['pclass'] = df['pclass'].dropna().map({'1st': 1, '2nd': 2, '3rd': 3}).astype(int)
#
#
# # We could do a deeper analysis and figure out where we have more data with NaN age.
# # If we do that, it come out that a better value for this NaN should be the age mean
# # of male passenger of third class.
# # 1) First we figure out how many NaN we have taking car of class and sex
# # df[np.isnan(df['age']) == True].groupby(['pclass', 'sex'])['survived'].count()
# # pclass  sex
# # 1st     0       42
# #         1       54
# # 2nd     0       22
# #         1       46
# # 3rd     0      156
# #         1      360
# # 2) We figure out the mean for each group and for the whole dataset, in order
# #    to check it if there are relevant differences between them
# # df[np.isnan(df['age']) == False].groupby(['pclass', 'sex'])['age'].mean()
# # pclass  sex
# # 1st     0      37.772277
# #         1      41.199334
# # 2nd     0      27.388235
# #         1      28.910761
# # 3rd     0      22.564328
# #         1      25.327294
# # df[np.isnan(df['age']) == False]['age'].mean()
# # 31.19418104265403
#
#
# # If age variable were of String type, we could do something like
# # m = df['age'].dropna().astype(float).mean()
# # df['age'] = df['age'].fillna(m).astype(float)
#
#
# # Stage 2) Analyzing the data
# #
#
# # A interesting data could have been the room number, but we don't have enough information.
#
# _ = sns.pairplot(df, vars=['pclass', 'sex'], hue="survived", palette='Set1')
# _ = sns.lmplot('sex', 'pclass', df, x_jitter=.15, y_jitter=.15, hue="survived", palette="Set1", fit_reg=False)
# # sns.pairplot(df, vars=['embarked', 'sex'], hue="survived", palette='Set1')
# # sns.lmplot('sex', 'embarked', df, x_jitter=.15, y_jitter=.15, hue="survived", palette="Set1", fit_reg=False);
# plt.figure(figsize=(12, 10))
# _ = sns.corrplot(df, annot=False)
#
#
#
