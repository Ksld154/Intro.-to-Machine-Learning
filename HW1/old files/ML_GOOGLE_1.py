import numpy as np
from sklearn import tree
import pydot	# It requires installing pydot and graphviz
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import random

#read file, initial
google = pd.read_csv ('googleplaystore.csv')     #read csv
df = pd.DataFrame (google)            #build dataframe
df  = df.drop(['App','Size','Genres','Last Updated','Current Ver','Android Ver'],axis=1) #drop features
df.dropna(inplace = True)           #drop NaN 
df.drop(df.loc[df['Type']==0].index, inplace=True) #drop bad info
df = df.replace(['Free','Paid'],[0,1])
data = list(df.columns.values)[0:7] #name
del(data[3])
without_cate = data
del(without_cate[0])
#installs = list(df.columns.values)[3]
##print(df['Installs'].value_counts()) Frequency of the Intalls


count_ = df['Content Rating'].value_counts().index.tolist()
df = df.replace(count_,[0,2,3,1,4,0])

labels =  df['Rating'].value_counts().index.tolist()
size =  df['Rating'].value_counts().tolist()
plt.plot( labels,size)
#plt.pie(size, labels = labels,autopct='%1.1f%%')
#plt.axis('equal')
plt.show()



count_b = df['Category'].value_counts().index.tolist()
df = df.replace(count_b,range(0,33))

df['Price'] = df['Price'].str.replace('$','0')

tmp_arr = [0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,4,4,4]

df = df.replace(['1,000+','500+','100+','50+','10+','5+','1+','5,000+','10,000+',
                '50,000+','100,000+','500,000+','1,000,000+','5,000,000+','10,000,000+','50,000,000+',
                    '1,000,000,000+','500,000,000+','100,000,000+'],tmp_arr)

target = list(df.columns.values)[3]
google_draw_target_names = ['<=10,000','10,001~100,000','100,001~1,000,000','1,000,001~10,000,000'
                ,'>10,000,000']
google_target_names = [0,1,2,3,4]
#print(df['Installs'].value_counts().sort_index())

# 1-5000 (-6)// 5000w+(-3)
num_of_trees = 20
num_of_datas = df.count()[0]
bag_size = num_of_datas / num_of_trees

#preprocessing
#df.fillna(df.mean()['B':'C']) fill in nan with average (for some columns)   preprocess

#draw
'''
df[data].plot.hist(alpha = 0.5, normed = True)                 #histogram
group = df[installs].groupby(df[target], as_index = False).mean()
group.index = google_draw_target_names
group.plot.bar()                                               #bar
plt.show()                                                     #show all plots
'''
#random forest
accu_vec = []
precision_vec = []
recall_vec = []
matrix_vec = []
test = []
Result = []

 
'''
K = 10
#  k-fold vallidation
df_shuffle = df.copy()
df_shuffle = shuffle(df_shuffle).reset_index(drop = True)
for z in range(K):
    test = []
    trees  = []
    result = []
    
    
    n_test = int(num_of_datas / K)
    train = df_shuffle
    Max_tmp = (z+1)*n_test - 1
    if z == K - 1 :
        Max_tmp = num_of_datas
    test  = train.loc[(z*n_test) : Max_tmp].reset_index(drop=True)
    if K != 1 : 
        train.drop(train.index[(z*n_test) : Max_tmp], inplace = True)        #drop some rows
    for i in range(num_of_trees):                                   #building trees
        train = shuffle(train).reset_index(drop = True)              #random
        random_col = shuffle(data)[0:random.randint(3, 6)]                                  #get random columns
        #print(random_col)

        clf = tree.DecisionTreeClassifier(max_depth = 6)
        clf = clf.fit(train.loc[0:100, random_col], train.loc[0:100, target])
        trees.append([clf, random_col])

    #feeding testing data 
    for i in range(int(test.count()[0])):                        
        vote = [0, 0, 0, 0, 0]
        for j in range(num_of_trees):
            r = trees[j][0].predict([ test.loc[i,trees[j][1]] ])   #trees[j][1] == random_col
            index = google_target_names.index(r)
            vote[index] = vote[index] + 1
        result.append(google_target_names[vote.index(max(vote))])
        Result.append(google_target_names[vote.index(max(vote))])

    #output result
    matrix_vec.append(confusion_matrix(test[target], result, labels=google_target_names))
    #print(matrix)

    


ConfusionMatrix = sum(matrix_vec)
print(ConfusionMatrix)

statistic = classification_report(df_shuffle[target], Result, target_names=google_target_names, output_dict=True)
output_df = pd.DataFrame(index = ['<=10,000','10,001~100,000','100,001~1,000,000','1,000,001~10,000,000','>10,000,000'])
for species in google_target_names:
    precision_vec.append(statistic[species]['precision'])
    recall_vec.append(statistic[species]['recall'])
output_df['precision'] = precision_vec
output_df['recall'] = recall_vec
print(output_df)

accu_mean = accuracy_score(df_shuffle[target], Result)
print(accu_mean)
'''