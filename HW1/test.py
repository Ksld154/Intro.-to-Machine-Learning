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
##print(df['Installs'].value_counts()) Frequency of the Intalls


#preprocessing
count_ = df['Content Rating'].value_counts().index.tolist()
df = df.replace(count_,[0,1,2,3,4,5])

count_b = df['Category'].value_counts().index.tolist()
df = df.replace(count_b,range(0,33))

df['Price'] = df['Price'].str.replace('$','0')

tmp_arr = [0,0,0,0,0,0,0,0,1,1,2,2,3,3,4,4,4,4,4]

df = df.replace(['1,000+','500+','100+','50+','10+','5+','1+','5,000+','10,000+',
                '50,000+','100,000+','500,000+','1,000,000+','5,000,000+','10,000,000+','50,000,000+',
                    '1,000,000,000+','500,000,000+','100,000,000+'],tmp_arr)

target = list(df.columns.values)[3]
google_target_names = [0,1,2,3,4]
print(df['Installs'].value_counts())