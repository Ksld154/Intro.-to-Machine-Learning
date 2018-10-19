import csv
import pandas as pd
import numpy  as np
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


df = pd.read_csv('Iris.csv')
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df = df.replace({'species': mapping})

iris_attr   = df[ ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] ]
iris_target = df['species']
#print(df)



#Dataset:    Iris
#Vaildation: Resubstitution validation

res = []
final_ans = []
DecisionTree = []

test_X = iris_attr
test_Y = iris_target
test_Y_list = test_Y.tolist()


clf = tree.DecisionTreeClassifier()
clf = clf.fit(test_X, test_Y)




#res.append(Y_predicted)




# accuracy
#accu = dtree.score(test_X, test_Y_list)
#print(accu)





"""
for i in range(1, 11):

    clf = tree.DecisionTreeClassifier()
    DecisionTree.append(clf.fit(iris_attr, iris_target))




for i in range(1, 151):
    cnt = np.zeros(3)
    Y_predicted = []
    for j in range(1, 11):
        prediction = DecisionTree[j].predict([test_X.loc[i, :].reshape(-1, 1)])
        if prediction == 0:
            cnt[0] = cnt[0] + 1
        elif prediction == 1:
            cnt[1] = cnt[1] + 1
        elif prediction == 2:
            cnt[2] = cnt[2] + 1
    
    freq = np.bincount(cnt)
    entry_result = np.argmax(freq)

    if entry_result == 0:
        res.append(0)
    elif entry_result == 1:
        res.append(1)
    elif entry_result == 2:
        res.append(2)
print(res)
"""
    # detemine which kind of flower this data is
    
    
    
    
    
    
    
    
    
    

    #test_Y_list = test_Y.tolist()


    
    #res.append(Y_predicted)




    # accuracy
    #accu = dtree.score(test_X, test_Y_list)
    #print(accu)





































"""
#train_X, test_X, train_y, test_y = train_test_split(iris_attr, iris_target, test_size = 0.3)

clf = tree.DecisionTreeClassifier() 
#clf = clf.fit(train_X, train_y)




#kf = KFold(n_splits=10, shuffle=False)

# 預測
test_y_predicted = clf.predict(test_X)
#print(test_y_predicted)

# 標準答案
test_y_list = test_y.tolist()

# accuracy
#accu = clf.score(test_X, test_y_list)
#print(accu)

#print(iris_target)






#scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5) # no use
print(scores)
"""