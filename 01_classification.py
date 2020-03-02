from sklearn.datasets import fetch_openml, datasets, svm, metrics

import matplotlib as matplotlib
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version =1)
mnist.keys()

print(mnist.keys())

X, y = mnist["data"], mnist["target"]
#X.shape
#Y.shape

#Labels are strings, make these integers

#Create 4 variables: X_train, X_test, Y_train, Y_test
#The MNIST dataset exist out of (first) 60 000 train images, and 10 000 test images

#Create a straightforward classifier (support vector classifier)
# https://scikit-learn.org/stable/modules/svm.html

#Use the classifier to predict to predict the Y-values for X-test

#Check the accuracy using a classification report and confusion matric