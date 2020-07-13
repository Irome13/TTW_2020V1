#SOURCE: https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html
print(__doc__)

from sklearn.datasets import make_circles, make_moons, make_biclusters, make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)

    return out


def plotModels(models, titles, name):
    plt.rcParams["figure.figsize"] = (20, 3)
    fig, sub = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.9, hspace=0.4)


    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)


    plt.show()

titles = ('Perceptron Classifier',
          'MLP Classifier',
          'SVM Classifier',
          )

#Perceptron level:****************************************
X, y = make_blobs(n_samples=[5, 5], centers=None, n_features=2,random_state=0)
C = 1.0  # SVM regularization parameter
models = (Perceptron(max_iter=1000, tol=1e-3),
          MLPClassifier( alpha=0.01, hidden_layer_sizes=(10,9,5, 2), random_state=1, max_iter=1000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          )

models = (clf.fit(X, y) for clf in models)
plotModels(models, titles,'blobs')

print('Perceptron')
for clf in models:
    clf.fit(X,y)
    xemp = np.array(clf.predict(X))
    NumberOfWrongClassification = np.sum((np.abs(y.transpose() - xemp)))
    print(NumberOfWrongClassification)



#MLP:**************************************************
X, y = make_circles(n_samples=100, factor=.3, noise=0.2)

C = 1.0  # SVM regularization parameter
models = (Perceptron(max_iter=1000,tol=1e-3),
          MLPClassifier(alpha=0.01, hidden_layer_sizes=(10, 9, 5, 2), random_state=1, max_iter=1000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),

          )
models = (clf.fit(X, y) for clf in models)
plotModels(models, titles,'circles')
print('MLP')
for clf in models:
    clf.fit(X,y)
    xemp = np.array(clf.predict(X))
    NumberOfWrongClassification = np.sum((np.abs(y.transpose() - xemp)))
    print(NumberOfWrongClassification)


#SVC:****************************************************************
X, y = make_moons(noise=0.3, random_state=0)
C = 1.0  # SVM regularization parameter
models = (Perceptron(max_iter=1000,tol=1e-3),
          MLPClassifier(alpha=0.01, hidden_layer_sizes=(10, 9, 5, 2), random_state=1, max_iter=1000),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          )
models = (clf.fit(X, y) for clf in models)
plotModels(models, titles,'moons')
print('SVM')
for clf in models:
    clf.fit(X,y)
    xemp = np.array(clf.predict(X))
    NumberOfWrongClassification = np.sum((np.abs(y.transpose() - xemp)))
    print(NumberOfWrongClassification)

