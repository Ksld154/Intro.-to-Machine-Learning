import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydot	# It requires installing pydot and graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
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

#random forest
training_data = df.copy()            #set training data
testing_data  = df.copy()            #set testing data
trees = []                           #[[tree, columns used in training], [], ......]
result = []

for i in range(num_of_trees):                                   #building trees
    #train = training_data.copy()
    training_data = shuffle(training_data).reset_index(drop = True)              #random
    random_col = shuffle(data)[0:random.randint(2, 4)]                                  #get random columns
    #print(random_col)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(training_data.loc[0:30, random_col], training_data.loc[0:30, target])
    trees.append([clf, random_col])

#feeding testing data (RESUBSTITUTION)
for i in range(testing_data.count()[0]):                        
    vote = [0, 0, 0]
    for j in range(num_of_trees):
        r = trees[j][0].predict([testing_data.loc[i,trees[j][1]]])
        index = iris_target_names.index(r)
        vote[index] = vote[index] + 1
    result.append(iris_target_names[vote.index(max(vote))])

#output result
matrix = confusion_matrix(testing_data[target], result)
print(matrix)

#accuracy
accu = accuracy_score(testing_data[target], result)
print(accu)


