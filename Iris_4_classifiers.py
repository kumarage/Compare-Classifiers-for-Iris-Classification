# 4 classifiers are compred over the Iris dataset classification
# Support vector machine, Neural Network 
# Decsion tree and Naive Bayes classifiers are used for this study

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#import the set of models
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier



#import the Iris dataset
iris = datasets.load_iris()
print(iris.keys())
sepal=iris.data[:,:2] #sepal features
type_iris=iris.target
train_NN = MinMaxScaler().fit_transform(sepal)

names = ['Nueral Network Classifier', 'Support Vector Classifier', 'Desicion Tree Classifier', 'Naive Bayes Classifier']
C = 1
classifiers = [MLPClassifier(hidden_layer_sizes=100,activation='relu',solver='lbfgs',alpha=0.001), 
				SVC(kernel='rbf', gamma=0.5, C=C),
				DecisionTreeClassifier(),
				MultinomialNB()]

h=0.05 #size of the cell in mesh
x_min, x_max = sepal[:,0].min() - 0.5, sepal[:,0].max() + 0.5
y_min, y_max = sepal[:,1].min() - 0.5, sepal[:,1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

i=0
for name, clf in zip(names, classifiers):
	print (name)
	plt.subplot(2,2,i+1)
	plt.subplots_adjust(wspace=0.4, hspace=0.5)
	clf.fit(sepal,type_iris)
	score = clf.score(sepal,type_iris)
	
	Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.autumn, alpha=0.8)
	plt.scatter(sepal[:,0], sepal[:,1], c=type_iris, cmap=plt.cm.magma)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.title(name)
	plt.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
	i+=1

plt.show()