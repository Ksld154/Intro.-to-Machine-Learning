import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset_preprocessing1.csv")  #read .csv file
df_all = pd.DataFrame(dataset)
df_all = df_all.drop((df_all[df_all.scored_int==0]).index, axis = 0)
df_all = df_all.drop((df_all[df_all.scored_int==10]).index, axis = 0)
df_all = df_all.drop((df_all[df_all.scoredBy <= 100]).index, axis = 0)
print(df_all.shape)

df = df_all.drop(df_all.columns[range(0,12)], axis = 1)
# df = df_all[ ['year', 'scoredBy'] ]
df_type = df_all[['scored_int']]

df = df.values
df_type = df_type.values

train_data , test_data , train_label , test_label = train_test_split(df, df_type, test_size=0.2)

train_label = train_label.ravel()
knn = KNeighborsClassifier(n_neighbors=100, metric='hamming')
knn.fit(train_data, train_label)

predict_result = knn.predict(test_data)

correct = 0
for i in range(len(test_label)):
    if test_label[i] == predict_result[i]:
        correct = correct+1
print('Accuracy: %.5f' %float(correct/len(test_label)) )
# print(float(correct/len(test_label)))
