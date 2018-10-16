import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydot	# It requires installing pydot and graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import random

#read file, initial
iris = pd.read_csv ('iris.csv')     #read csv
df = pd.DataFrame (iris)            #build dataframe
data = list(df.columns.values)[0:4]
target = list(df.columns.values)[4]
iris_target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
num_of_trees = 5
num_of_datas = df.count()[0]
bag_size = num_of_datas / num_of_trees

#preprocessing
#df.fillna(df.mean()['B':'C']) fill in nan with average (for some columns)   preprocess

#draw
#df[data].plot.hist(alpha = 0.5, normed = True)                 #histogram
#group = df[data].groupby(df[target], as_index = False).mean()
#group.index = iris_target_names
#group.plot.bar()                                               #bar
#plt.show()                                                     #show all plots


training_data = df.copy()            #set training data
testing_data  = df.copy()            #set testing data
trees = []                           #[[tree, columns used in training], [], ......]
result = []
res = []
training_data = shuffle(training_data).reset_index(drop = True)              #random

train = training_data.copy()
test  = training_data.loc[100:150]
train.drop(train.index[100:150], inplace = True)        #drop some rows

#random forest
forest = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest = forest.fit(train[data], train[target])

res = forest.predict(test[data])

matrix = confusion_matrix(test[target], res)
print(matrix)

accu = forest.score(test[data], test[target])
print(accu)


