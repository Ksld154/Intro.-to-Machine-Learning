import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
import pydot	# It requires installing pydot and graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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
#trees = []                           #[[tree, columns used in training], [], ......]

accu_vec = []
precision_vec = []
recall_vec = []
matrix_vec = []
test = []

training_data = shuffle(training_data).reset_index(drop = True)              #random

K = 10
#　k-fold vallidation
for z in range(K):
    test = []
    trees  = []
    result = []
    
    
    n_test = num_of_datas / K
    train = training_data.copy()
    test  = train.loc[(z*n_test) : ((z+1)*n_test - 1)].reset_index(drop=True)
    train.drop(train.index[(z*n_test) : ((z+1)*n_test - 1)], inplace = True)        #drop some rows

    for i in range(num_of_trees):                                   #building trees
        #train = training_data.copy()
        train = shuffle(train).reset_index(drop = True)              #random
        random_col = shuffle(data)[0:random.randint(2, 4)]                                  #get random columns
        #print(random_col)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train.loc[0:30, random_col], train.loc[0:30, target])
        trees.append([clf, random_col])

    #print(trees)

    #feeding testing data 
    for i in range(int(n_test)):                        
        vote = [0, 0, 0]
        for j in range(num_of_trees):
            #print(test.loc[i,trees[j][1]])
            
            r = trees[j][0].predict([ test.loc[i,trees[j][1]] ])   #trees[j][1] == random_col
            index = iris_target_names.index(r)
            vote[index] = vote[index] + 1
        result.append(iris_target_names[vote.index(max(vote))])

    #output result
    matrix_vec.append(confusion_matrix(test[target], result, labels=iris_target_names))
    #print(matrix)

    statistic = classification_report(test[target], result, target_names=iris_target_names, output_dict=True)
    tmp_precision = []
    tmp_recall = []
    for species in iris_target_names:
        tmp_precision.append(statistic[species]['precision'])
        tmp_recall.append(statistic[species]['recall'])
    precision_vec.append(tmp_precision)
    recall_vec.append(tmp_recall)

    #accuracy
    accu_vec.append(accuracy_score(test[target], result))
    #print(accu_vec)


ConfusionMatrix = sum(matrix_vec)
print(ConfusionMatrix)


sum_precision = [sum(i) for i in zip(*precision_vec)]
precision_array = np.array(sum_precision)
precision_array = precision_array / K
print(precision_array)

sum_recall = [sum(i) for i in zip(*recall_vec)]
recall_array = np.array(sum_recall)
recall_array = recall_array / K
print(recall_array)

#total_precision = sum(precision_vec) / len(precision_vec)
#print(total_precision)


accu_mean = sum(accu_vec) / len(accu_vec)
print(accu_mean)