# coding: utf-8
from classifier.model import Model, DataContainer, Logger
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
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
        "Training model {0} times. Using data {1}.\nThe showed results are over Test and Training data.\n".format(
            repetitions, "Normalized" if normalized_data else "Real"))
    best_score = float("-inf")
    best_model = None

    for i in range(1, repetitions):
        score = model.fit(normalized=normalized_data).score()
        if score > best_score:
            best_score = score
            best_model = i
            logger.write("Test : ", end="")
            model.print_metrics(show_extended=False)
            logger.write("Train: ", end="")
            model.print_metrics(show_extended=False, use_train_data=True)
            logger.write("\n")

    return best_model

def get_best_sgdmodel_parametrization(normalized_data):
    logger.write("\n\n-----------------------------------------------------------------------------------")
    logger.write("Analyzing different configurations for the SGDClassifier classifier. Data {}\n".format("Normalized" if normalized_data else "Real"))
    best_score = float("-inf")
    best_model = None
    for i, model in enumerate(sgd_models):
        score = model.fit(normalized=normalized_data).print_metrics().score()
        if score > best_score:
            best_score = score
            best_model = model

    # We show statistics of variation with highest score
    logger.write("\n\nThe best parametrization has been:", best_model.name)
    logger.write("Statistics:\n");
    best_model.print_metrics(show_extended=True)

    return best_model

def try_different_model(normalized_data):
    logger.write("\n\n-----------------------------------------------------------------------------------")
    logger.write("Results of different types of classifiers. Data {}\n".format("Normalized" if normalized_data else "Real"))

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
