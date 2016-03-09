import time

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


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
    def __init__(self, classifier, data, name="No name"):
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
            self.logger.write("Using normalized data")
            self.logger.write("Mean: ", self.data.mean)
            self.logger.write("Scale: ", self.data.scale)
        else:
            self.logger.write("Using real data")
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

        # Creating color maps (used with the points and for the backgroud)
        cmap_light = ListedColormap(['#FFEEEE', '#EEFFEE', '#EEEEFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Reset just in case we want to draw the graphic in is own plot
        # that means, we won't to put more than one graphic in a simple plot
        if showplot: plt.clf()

        configure_draw()

        X, y = self.data.get_traindata(self.data_normalized)

        d = 0.5
        xmin, xmax = X[:, 0].min() - d, X[:, 0].max() + d
        ymin, ymax = X[:, 1].min() - d, X[:, 1].max() + d

        # Centramos en cuanto a los eje x e y, para x no es necesario puesto que el rango de valores
        # que puede tomar x está definido entre sus minimos y maximos
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        draw_mesh()

        # Draw the points
        # plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        for tipo, marca, color in zip(range(len(self.data.y_classes)), "soD", "rgb"):
            plt.scatter(X[y == tipo, 0], X[y == tipo, 1], marker=marca, c=color)
        # if showlegend: plt.legend(self.data.y_classes)

        # Draw the lines
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
            results.update({"classification_report": metrics.classification_report(y, y_predict, target_names=self.data.y_classes)})

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
            print(message, end=end)
        else:
            print(message)
