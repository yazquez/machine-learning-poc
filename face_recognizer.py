from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_olivetti_faces
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# Plot the completed faces
def plot_images(images, title=None, image_shape=None, fig_size=None):
    import math
    if not image_shape:
        pixels = images.shape[1]
        image_shape = (int(math.sqrt(pixels)), int(math.sqrt(pixels)))
        print(image_shape)

    n_faces = images.shape[0]
    n_cols = 10
    if not fig_size:
        fig_size = (2. * n_cols, 0.2 * n_faces)

    plt.figure(figsize=fig_size)

    if title:
        plt.suptitle(title, size=16)

    for i in range(n_faces):
        sub = plt.subplot(n_faces / n_cols, n_cols, i + 1)

        sub.axis("off")
        sub.imshow(images[i].reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")
    plt.show()


faces = fetch_olivetti_faces()
X = faces.data
y = faces.target

plot_images(X[50:100])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

Xn_train = X_train
Xn_test = X_test

normalizador = StandardScaler().fit(X_train)
Xn_train = normalizador.transform(X_train)
Xn_test = normalizador.transform(X_test)

clasificador = SGDClassifier(average=True)
clasificador.fit(Xn_train, y_train)
y_predict = clasificador.predict(Xn_test)

print(metrics.confusion_matrix(y_test, y_predict))
print(metrics.classification_report(y_test, y_predict))
print(metrics.accuracy_score(y_test, y_predict))

