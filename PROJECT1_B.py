# -*- coding: utf-8 -*-
"""
Created on Sunday March 1, 2019

@author: Sampriti
"""

import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import Perceptron            # the algorithm
from sklearn.metrics import accuracy_score             # grade the results
from sklearn.linear_model import LogisticRegression    # the algorithm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier         # the algorithm                            # the algorithm
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.neighbors import KNeighborsClassifier     # the algorithm

# Load the data into dataframe
dataset=np.loadtxt("data_banknote_authentication.csv",delimiter=",",skiprows=1)
print(dataset.shape)
X=dataset[:,0:4]
y=dataset[:,4]
y=y.transpose()

# Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)



sc = StandardScaler()                      # create the standard scalar
sc.fit(X_train)                            # compute the required transformation
X_train_std = sc.transform(X_train)        # apply to the training data
X_test_std = sc.transform(X_test)          # and SAME transformation of test data!!!
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Perceptron Training
ppn = Perceptron(max_iter=10, tol=1e-3, eta0=0.01, fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, y_train)              # do the training

y_combined_pred = ppn.predict(X_combined_std)
print('Misclassified combined samples For Perceptron: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy of perceptron: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Logistic Regression
lr = LogisticRegression(C=100, solver='liblinear', multi_class='ovr', random_state=0)
lr.fit(X_train_std, y_train)                # apply the algorithm to training data
y_combined_pred = lr.predict(X_combined_std)
print('Misclassified combined samples in Logistic Regression: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy in Logistic Regression: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Support Vector machine
svm = SVC(kernel='rbf', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)   # train using train data and test using test data and find accuracy
y_combined_pred = svm.predict(X_combined_std)
print('Misclassified combined samples in support vector machine: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy in support vector machine: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Decision Tree
tree = DecisionTreeClassifier(criterion='entropy',max_depth=100 ,random_state=1,splitter='random')
tree.fit(X_train,y_train)  # train using train data and test using test data and find accuracy
X_combined = np.vstack((X_train, X_test))
y_combined_pred = tree.predict(X_combined)
print('Misclassified combined samples for Decision tree: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy for decision tree: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# Random forest
forest = RandomForestClassifier(criterion='entropy', n_estimators=10,random_state=0,max_depth=10)
forest.fit(X_train,y_train)  # train using train data and test using test data and find accuracy
y_combined_pred = forest.predict(X_combined)
print('Misclassified combined samples for Random forest: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy for random forest: %.2f' % accuracy_score(y_combined, y_combined_pred))
print('\n')

# K nearest neighbours
knn = KNeighborsClassifier(n_neighbors=200,p=2,metric='minkowski',weights='distance')
knn.fit(X_train_std,y_train)   # train using train data and test using test data and find accuracy
y_combined_pred = knn.predict(X_combined_std)
print('Misclassified combined samples for K nearest neighbours: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy for k nearest neighbours is : %.2f' % accuracy_score(y_combined, y_combined_pred))


