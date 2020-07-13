import csv
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import time

# 1. get dataset
# https://www.kaggle.com/mlg-ulb/creditcardfraud

# 2. read dataset
data_path = 'Datasets/creditcard.csv'
with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    #remove headers:
    headers = next(reader)
    #get all data in the dataset as float numbers:
    data = np.array(list(reader)).astype(float)

print('Complete csv dataset size: ',data.shape)
#get train and test data:
X = data[:, :data.shape[1] - 1]
print('Train and test dataset size: ',X.shape)
#get classification for dataset:
y = data[:, data.shape[1] - 1]
print('Classes for dataset size: ',y.shape)

#Perceptron classifier: train and test
#get initial time for performance counter:
tic = time.perf_counter()
#define classifier:
classifier = Perceptron(class_weight='balanced', max_iter=1000, tol=1e-5)
#train classifier:
classifier.fit(X, y)
#print result of test:
print(classifier.predict(X))
#compute success percentage manually:
print('Score Perceptron: ', 1 - np.sum(np.abs(y - classifier.predict(X)) / y.shape))
#compute success percentage automatically:
print('Automated computed score:', classifier.score(X, y))
#compute success percentage using a cross validation technique: https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(classifier, X, y, cv=5)
print('Cross validation: ', np.mean(scores))
#compute time elapsed for current classifier:
toc = time.perf_counter()
print('Time elapsed: ',toc-tic)

#MultiLayer Perceptron: train and test
#get initial time for performance counter:
tic = time.perf_counter()
#define classifier:
classifier = MLPClassifier()
#train classifier:
classifier.fit(X, y)
#print result of test:
print(classifier.predict(X))
#compute success percentage manually:
print('Score MultiLayer Perceptron: ', 1 - np.sum(np.abs(y - classifier.predict(X)) / y.shape))
#compute success percentage automatically:
print('Automated computed score:', classifier.score(X, y))
#compute success percentage using a cross validation technique: https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(classifier, X, y, cv=5)
print('Cross validation: ', np.mean(scores))
#compute time elapsed for current classifier:
toc = time.perf_counter()
print('Time elapsed: ',toc-tic)

#Support Vector Machine Classifier: train and test
#get initial time for performance counter:
tic = time.perf_counter()
#define classifier:
classifier = SVC(gamma='auto')
#train classifier:
classifier.fit(X, y)
#print result of test:
print(classifier.predict(X))
#compute success percentage manually:
print('Score SVM classifier: ', 1 - np.sum(np.abs(y - classifier.predict(X)) / y.shape))
#compute success percentage automatically:
print('Automated computed score:', classifier.score(X, y))
#compute success percentage using a cross validation technique: https://scikit-learn.org/stable/modules/cross_validation.html
scores = cross_val_score(classifier, X, y, cv=5)
print('Cross validation: ', np.mean(scores))
#compute time elapsed for current classifier:
toc = time.perf_counter()
print('Time elapsed: ',toc-tic)