

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


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


digits = load_digits()
images = digits.images
images.shape
images = np.asarray([im.flatten() for im in images])

# faces = fetch_olivetti_faces()
# images = faces.data
# images.shape

# plot_images(images[:100], fig_size=(10, 4))

npixels = images.shape[1]

Xc = images[:, :npixels / 2]
yc = images[:, npixels / 2:]

# plot_images(Xc[:100], image_shape=(4, 8), fig_size=(5, 4))

X_train, X_test, y_train, y_test = train_test_split(Xc, yc, test_size=0.25, random_state=666)

model = LinearRegression().fit(X_train, y_train)

predictions = model.predict(X_test)

result = []
for x, p in zip(X_test, predictions):
    r = np.hstack((x, p))
    result.append(r)

result = np.asarray(result)

result.shape

plot_images(result[:100], fig_size=(10, 4))