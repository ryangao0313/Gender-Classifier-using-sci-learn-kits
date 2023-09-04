#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 18:12:42 2023

Predict if a data set is male or female

@author: ryangao
"""

#Import libraries
from sklearn import tree
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import neural_network

#Create classifiers
classifier_DT = tree.DecisionTreeClassifier()
classifier_NN = neighbors.KNeighborsClassifier()
classifier_GP = gaussian_process.GaussianProcessClassifier()
classifier_NeuN = neural_network.MLPClassifier()

#Data set [height, weight, shoe_size
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

#Fit data (training)
classifier_DT = classifier_DT.fit(X,Y)
classifier_NN = classifier_NN.fit(X,Y)
classifier_GP = classifier_GP.fit(X,Y)
classifier_NeuN = classifier_NeuN.fit(X,Y)

#Predict
prediction_DT = classifier_DT.predict([[190, 70, 43]])
prediction_NN = classifier_NN.predict([[190, 70, 43]])
prediction_GP = classifier_GP.predict([[190, 70, 43]])
prediction_NeuN = classifier_NeuN.predict([[190, 70, 43]])

#Output results
print('Decision Tree',prediction_DT)
print('Nearest Neighbors',prediction_NN)
print('Gaussian Process',prediction_GP)
print('Neural Net',prediction_NeuN)

#Compare results





