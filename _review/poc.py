from matplotlib import pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

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

# Show some information about the data
print(X.shape)
print(y.shape)
y[:10]
faces.DESCR

# Show some images
plot_images(X[250:300])


def predict(x, y, pipeline, cv=5, classifier=True):
    from sklearn.pipeline import Pipeline
    from sklearn import cross_validation
    from sklearn import metrics

    model = Pipeline(pipeline)
    predicted = cross_validation.cross_val_predict(model, x, y, cv=cv)
    if classifier:
        print(metrics.accuracy_score(y, predicted))
    else:
        print(metrics.explained_variance_score(y, predicted))
    return model


predict(X, y, [('modelo', SGDClassifier())], cv=5)
predict(X, y, [('modelo', SGDClassifier())], cv=5)
predict(X, y, [('normalizador', StandardScaler()),
               ('modelo', SGDClassifier())], cv=5)
predict(X, y, [('normalizador', StandardScaler()),
               ('modelo', SGDClassifier(average=True))], cv=5)
