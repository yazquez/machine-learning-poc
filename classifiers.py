# coding: utf-8
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
# Import some classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier


def repeat_fit(model, normalized_data, repetitions):
    logger.write("\n\n-----------------------------------------------------------------------------------");
    logger.write(
        "Entrenando modelo durante {0} veces. Usando datos {1}.\nResultados sobre datos de Test y datos de Entrenamiento.\n".format(
            repetitions, "Normalizados" if normalized_data else "Reales"))
    best_score = float("-inf")
    best_model = None

    for i in range(1, repetitions):
        score = model.fit(normalized=normalized_data).score()
        if score > best_score:
            best_score = score
            best_model = i
            logger.write("Test:           ", end="")
            model.print_metrics(show_extended=False)
            logger.write("Entrenamiento:  ", end="")
            model.print_metrics(show_extended=False, use_train_data=True)
            logger.write("\n")

    return best_model

def get_best_sgdmodel_parametrization(normalized_data):
    logger.write("\n\n-----------------------------------------------------------------------------------")
    logger.write("Analizando diferentes configuraciones del clasificador SGDClassifier. Datos {}\n".format("Normalizados" if normalized_data else "Reales"))
    best_score = float("-inf")
    best_model = None
    for i, model in enumerate(sgd_models):
        score = model.fit(normalized=normalized_data).print_metrics().score()
        if score > best_score:
            best_score = score
            best_model = model

    # We show statistics of variation with highest score
    logger.write("\n\nLa parametrización del modelo SGDClassifier con mejores resultados ha sido:", best_model.name)
    logger.write("Estadísticas:\n");
    best_model.print_metrics(show_extended=True)

    return best_model

def try_different_model(normalized_data):
    logger.write("\n\n-----------------------------------------------------------------------------------")
    logger.write("Resultados con diferentes tipos de clasificadores.Datos {}\n".format("Normalizados" if normalized_data else "Reales"))

    for i, model in enumerate(models):
        model.fit(normalized=normalized_data).print_metrics()

    # Dibujamos la gráfica
    figure = plt.figure(figsize=(30, 10))
    figure.canvas.set_window_title("Graficas comparativas de los distintos Clasificadores. Datos " + ("Normalizados" if normalized_data else "Reales"))
    for i, model in enumerate(models):
        plt.subplot(2, 5, i + 1)
        model.draw_graph(showlegend=True, showplot=False)
    figure.subplots_adjust(left=.02, right=.98)
    plt.show()


class DataContainer:
    """Esta clase encapsulará toda la gestión de los datos ofreciendo métodos para
       obtener cada uno de los datos en sus versiones normalizadas o sin normalizar"""

    def __init__(self, X, y, feature_names, y_classes, feature_1, feature_2, test_size=0.25):
        self.X = X[:, [feature_1, feature_2]]
        self.X_names = [feature_names[feature_1], feature_names[feature_2]]

        self.y = y
        self.y_classes = y_classes

        self.test_size = test_size

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=self.test_size, random_state=342)

        # Obtenemos los datos normalizados por si luego son necesarios
        # se podría obtimizar y no normalizarlos hasta el primer uso, pero dado que
        # lo más lógico es usar estos datos para el entrenamiento se hace siempre.
        normalizador = StandardScaler().fit(self.X_train)
        self.Xn_train = normalizador.transform(self.X_train)
        self.Xn_test = normalizador.transform(self.X_test)
        self.mean = normalizador.mean_
        self.scale = normalizador.scale_

    def get_traindata(self, data_normalized=True):
        """Devuelve los datos de entranamiento, pueden ser normalizados o sin normalizar"""
        if data_normalized:
            return self.Xn_train, self.y_train
        else:
            return self.X_train, self.y_train

    def get_testdata(self, data_normalized=True, use_train_data=False):
        """ Devuelve los datos para testear el modelo, pueden ser los de test o los de
            entrenamiento, y dentro de cada grupo los normalizados o los sin normalizar
        """
        if use_train_data:
            if data_normalized:
                return self.Xn_train, self.y_train
            else:
                return self.X_train, self.y_train
        else:
            if data_normalized:
                return self.Xn_test, self.y_test
            else:
                return self.X_test, self.y_test


class Model:
    def __init__(self, classifier, data, name="Sin nombre"):
        self.classifier = classifier
        self.data = data
        self.name = name
        self.data_normalized = True
        self.fit_duration = 0
        self.logger = Logger("console")

    def set_name(self, label):
        self.name = label
        return self

    def show_log(self):
        self.logger.write("Log\n--------------------------")
        if self.data_normalized:
            self.logger.write("Usando datos normalizados")
            self.logger.write("Media: ", self.data.mean)
            self.logger.write("Escala: ", self.data.scale)
        else:
            self.logger.write("Usando datos sin normalizar")
        self.logger.write("Intercept:\n", self.classifier.intercept_)
        self.logger.write("Coef:\n", self.classifier.coef_)

        return self

    def fit(self, normalized=True):
        self.data_normalized = normalized
        X, y = self.data.get_traindata(self.data_normalized)

        start = time.clock()
        self.classifier.fit(X, y)
        self.fit_duration = time.clock() - start

        return self

    def draw_graph(self, showlegend=False, showplot=True):

        def draw_mesh():
            h = .02  # step size in the mesh
            xx, yy = np.meshgrid(np.arange(xmin, xmax, h), np.arange(ymin, ymax, h))
            Z = self.classifier.predict(np.c_[xx.ravel(), yy.ravel()])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
            plt.axis('tight')

        def draw_lines():
            try:
                # Obtenemos los valores de x
                xs = np.arange(xmin, xmax, 0.5)
                for i, c in zip(range(3), "rgb"):
                    m = -self.classifier.coef_[i, 0] / self.classifier.coef_[i, 1]
                    n = -self.classifier.intercept_[i] / self.classifier.coef_[i, 1]
                    ys = (xs * m + n)
                    plt.plot(xs, ys, color=c, ls="--", )
            except:
                pass

        def configure_draw():
            plt.title("{} [{:.3f}]".format(self.name, self.score()))
            plt.xlabel(self.data.X_names[0])
            plt.ylabel(self.data.X_names[1])

        # Creamos los mapas de colores para los puntos y para el fondo
        cmap_light = ListedColormap(['#FFEEEE', '#EEFFEE', '#EEEEFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Reiniciamos solo si vamos a pintar esta grafica de manera autonoma,
        # o sea no pensamos mostrar más de una grafica distinta en un solo plot.
        if showplot: plt.clf()

        configure_draw()

        X, y = self.data.get_traindata(self.data_normalized)

        d = 0.5
        xmin, xmax = X[:, 0].min()-d, X[:, 0].max()+d
        ymin, ymax = X[:, 1].min()-d, X[:, 1].max()+d

        # Centramos en cuanto a los eje x e y, para x no es necesario puesto que el rango de valores
        # que puede tomar x está definido entre sus minimos y maximos
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        # Pintamos la malla
        draw_mesh()

        # Pintamos los puntos
        #plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        for tipo, marca, color in zip(range(len(self.data.y_classes)), "soD", "rgb"):
             plt.scatter(X[y == tipo, 0], X[y == tipo, 1], marker=marca, c=color)
        # if showlegend: plt.legend(self.data.y_classes)

        # Pintamos las lineas
        draw_lines()

        if showplot: plt.show()

        return self

    def score(self, use_train_data=False):
        # Podemos obtener los resultados contra los datos de test (opción por defecto) o contra
        # los mismos con los que se entrenó el modelo. Adicionalmente indicamos si queremos los
        # datos normalizados o sin normalizar
        X, y = self.data.get_testdata(self.data_normalized, use_train_data)

        return self.classifier.score(X, y)

    def get_metrics(self, show_extended=True, use_train_data=False):
        # Podemos obtener los resultados contra los datos de test (opción por defecto) o contra
        # los mismos con los que se entrenó el modelo. Adicionalmente indicamos si queremos los
        # datos normalizados o sin normalizar
        X, y = self.data.get_testdata(self.data_normalized, use_train_data)

        y_predict = self.classifier.predict(X)

        results = dict()
        results.update({"success": sum(y_predict == y)})
        results.update({"errors": sum(y_predict != y)})
        results.update({"score": self.score(use_train_data)})
        if show_extended:
            results.update({"accuracy": metrics.accuracy_score(y, y_predict)})
            results.update({"confusion_matrix": metrics.confusion_matrix(y, y_predict)})
            results.update({"classification_report":
                                metrics.classification_report(y, y_predict, target_names=self.data.y_classes)})

        return results

    def print_metrics(self, show_extended=False, use_train_data=False):
        m = self.get_metrics(show_extended=show_extended, use_train_data=use_train_data)

        norm_str = "Datos Normalizados" if self.data_normalized else "Datos Reales"
        self.logger.write("{:.<35} {:<18} - Puntuación: {:<18}  [Aciertos/Errores: {:<3}/{:<3}].[Duración:{:.3f}]".
                            format(self.name, norm_str, m["score"], m["success"], m["errors"], self.fit_duration))

        if show_extended:
            self.logger.write("Exactitud: %f" % m["accuracy"])
            self.logger.write("Informe de clasificación\n %s" % m["classification_report"])
            self.logger.write("Matriz de confusión\n %s" % m["confusion_matrix"])

        return self

    def set_window_title(self, title):
        plt.figure().canvas.set_window_title(title)
        return self


class Logger:
    def __init__(self, out):
        self.out = out

    def write(self, message, end=None):
        if end:
            print(message,end=end)
        else:
            print(message)

# ----------------------------------------------------------------------------------------------------------------------

iris = load_iris()

# ----------------------------------------------------------------------------------------------------------------------

# declaramos constantes para acceder a las distinas variables
sepal_length = 0
sepal_width = 1
petal_length = 2
petal_width = 3
# Establecemos el numero de repeticiones a realizar (cuando proceda)
repetitions = 10
# Establecemos el tamaño del conjunto de datos de test
test_size = 0.75

# ----------------------------------------------------------------------------------------------------------------------
# Creamos una instancia de la clasa DataContainer con los datos seleccionados en ellas
# almacenaremos los datos para pasarlos facilmente a los distintos clasificadores
data_container = DataContainer(iris.data, iris.target, iris.feature_names, iris.target_names,
                               petal_length, petal_width, test_size=test_size)



logger = Logger("console")

# ----------------------------------------------------------------------------------------------------------------------

# Lista con diferente variaciones del Clasificador SGDClassifier
sgd_models = [
    Model(SGDClassifier(), data_container).set_name("Default values"),
    Model(SGDClassifier(n_iter=10), data_container).set_name("n_iter=10"),
    Model(SGDClassifier(fit_intercept=False), data_container).set_name("fit_intercept=False"),
    Model(SGDClassifier(loss='modified_huber'), data_container).set_name("loss='modified_huber'"),
    Model(SGDClassifier(penalty='l1'), data_container).set_name("penalty='l1'"),
    Model(SGDClassifier(penalty='elasticnet'), data_container).set_name("penalty='elasticnet'")]


# Lista con diferentes tipos de clasificadores
models = [
    Model(KNeighborsClassifier(3), data_container).set_name("Nearest Neighbors"),
    Model(SVC(kernel="linear", C=0.025), data_container).set_name("Linear SVM"),
    Model(SVC(gamma=2, C=1), data_container).set_name("RBF SVM"),
    Model(DecisionTreeClassifier(max_depth=5), data_container).set_name("Decision Tree"),
    Model(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1), data_container).set_name("Random Forest"),
    Model(AdaBoostClassifier(), data_container).set_name("AdaBoost"),
    Model(GaussianNB(), data_container).set_name("Naive Bayes"),
    Model(LinearDiscriminantAnalysis(), data_container).set_name("Linear Discrim. Analysis"),
    Model(QuadraticDiscriminantAnalysis(), data_container).set_name("Quadratic Discrim. Analysis"),
    Model(SGDClassifier(), data_container).set_name("Stochastic Gradient Descent")]


# ----------------------------------------------------------------------------------------------------------------------
# 1. Repetir el proceso visto en clase (sin normalizar) para clasificar la especie del iris setosa pero tomado las
#    características longitud y ancho del pétalo. Representar la distribución de las características.
#    Obtener las métricas de precisión, exactitud, ...

Model(SGDClassifier(), data_container)\
    .set_name("Stochastic Gradient Descent")\
    .fit(normalized=False) \
    .print_metrics(show_extended=True)\
    .set_window_title("Distribución de las características")\
    .draw_graph()


# ----------------------------------------------------------------------------------------------------------------------
# 2. Cada vez que lanzamos el entrenamiento, el modelo resultante tendrá un rendimiento diferente. Crear un
#    algoritmo que entrene el modelo 10000 veces motrando por consola cada vez que se obtiene un modelo mejor
#    que el mejor obtenido hasta el momento. Usad el método score de SGDClassifier para obtener la medida de
#    rendimiento. Mostrar por consola el rendimiento sobre los datos de entrenamiento y sobre los datos de test.
#   ¿ Que implica que el rendimiento sobre el test sean mejor que sobre los de entrenamiento ? ¿ y al contrario ?
#   De todas las salidas obtenidas, ¿ cuál considerarías que es el mejor modelo ?

repeat_fit(Model(SGDClassifier(), data_container).set_name("Stochastic Gradient Descent"),
           normalized_data=False, repetitions=repetitions)

# ----------------------------------------------------------------------------------------------------------------------
# 3. SGDClassifier tiene multitud de parámetros de entrada que pueden influir sobre el rendimiento del modelo.
#    Revisar los parámetros en http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#    e intentar manipular estos parámetros para aumentar el rendimiento del modelo anterior.

best_sgdmodel = get_best_sgdmodel_parametrization(normalized_data=False)

# ----------------------------------------------------------------------------------------------------------------------
# 4. Representar graficamente las líneas de clasificación del mejor clasificador obtenido

best_sgdmodel.set_window_title("Líneas de clasificación del mejor clasificador obtenido. Datos Reales").draw_graph()

# ----------------------------------------------------------------------------------------------------------------------
# 5. En http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html podeis encontrar
# una revisión de varios clasificadores. Probad algunos de ellos...

try_different_model(False)

# ----------------------------------------------------------------------------------------------------------------------
# 6. Repetir los ejercicios anteriores normalizando los datos. ¿Se aprecia alguna mejora?

# 6.1 Probar difentes parametrizacions del clasificador SGDClassifier
best_sgdmodel = get_best_sgdmodel_parametrization(normalized_data=True)

# 6.2 Grafica del mejor score
best_sgdmodel.set_window_title("Líneas de clasificación del mejor clasificador obtenido. Datos Normalizados").draw_graph()

# 6.3 Repetir 10000 veces el entrenamiento
repeat_fit(Model(SGDClassifier(), data_container).set_name("Stochastic Gradient Descent"),
           normalized_data=True, repetitions=repetitions)

# 6.4 Probar diferentes clasificadores
try_different_model(True)
