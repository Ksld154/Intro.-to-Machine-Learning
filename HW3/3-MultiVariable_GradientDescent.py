import numpy  as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def ComputeCost(X, y, theta):
    m = y.shape[0]
    C = X.dot(theta) - y
    J = (C.T.dot(C)) / (2 * m)
    return J

def GradientDescent(X, y, theta, alpha, num_iters):
    m = y.shape[0]
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - (alpha/m) * (X.T.dot(X.dot(theta) - y))
        J_history[iter] = ComputeCost(X, y, theta)
    return J_history,theta

dataset = pd.read_csv('Concrete_Data.csv')
dataset_df = pd.DataFrame(dataset)

n_data = 1030
# attrs = ['Cement (component 1)(kg in a m^3 mixture)', 'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
#          'Fly Ash (component 3)(kg in a m^3 mixture)', 'Water  (component 4)(kg in a m^3 mixture)', 
#          'Superplasticizer (component 5)(kg in a m^3 mixture)','Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
#          'Fine Aggregate (component 7)(kg in a m^3 mixture)', 'Age (day)'
# ]
attrs = dataset.columns
attrs = attrs[:len(attrs)-1]

dataset_X = dataset_df.drop(['Concrete compressive strength(MPa, megapascals) '], axis = 1).values
# dataset_X = preprocessing.scale(dataset_X)
# print(dataset_X)

dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
dataset_Y = np.reshape(dataset_Y, (-1, 1))
# print(dataset_Y)

dataset_X_train, dataset_X_test, dataset_Y_train, dataset_Y_test = train_test_split(dataset_X, dataset_Y, test_size = 0.2)

theta = np.zeros((dataset_X_train.shape[1]+1, 1))
dataset_X_train = np.hstack([dataset_X_train, np.ones((dataset_X_train.shape[0], 1))])


iterations = 10000
alpha = 0.01
J = ComputeCost(dataset_X_train, dataset_Y_train, theta)
J_history,theta = GradientDescent(dataset_X_train, dataset_Y_train, theta, alpha, iterations)