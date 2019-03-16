import imp
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('dataset_preprocessing.csv')
df = pd.DataFrame (dataset)
target = list(df.columns.values)[8]
attribute = list(df.columns.values)[3:12]
del(attribute[5:7])
#print(target)
#print(attribute)
Max = 0

kf = KFold(n_splits=10,shuffle=True)
for train_index, test_index in kf.split(df):
    df_train, df_test = df.loc[train_index], df.loc[test_index]
    #print("TRAIN:", train_index, "TEST:", test_index)
    clf = RandomForestClassifier(n_estimators=10, max_depth=6, random_state=0) # 10 randomforest tree depth 6
    clf = clf.fit(df_train.loc[:,attribute],df_train.loc[:,target])            # put training data into tree
    tmp = df_test.loc[:,attribute]
    tmp = clf.predict(tmp)
    count = 0
    for i in range (int(df_test.count()[0])):
        temp = tmp[i] - df_test.loc[:,target].values[i]
        if abs(temp) <= 1:
            count = count + 1
    tmp_acc = count/int(df_test.count()[0])
    if tmp_acc > Max:
        Max = tmp_acc

print("Accuracy:",Max)
    