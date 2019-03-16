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

def GradientDescent(X, y, theta, alpha, max_itrs):
    m = y.shape[0]

    for itr in range(max_itrs):
        theta = theta - (alpha / m) * (X.T.dot(X.dot(theta) - y))
    return theta

def GradientDescent_wj(X, y, theta, alpha, max_itrs):
    m = y.shape[0]
    n = theta.shape[0]

    for itr in range(max_itrs):
        for i in range(n):
            Xj = X[:, i]
            theta[i] = theta[i] - (alpha / m) * (Xj.T.dot(X.dot(theta) - y))
    return theta

dataset = pd.read_csv('Concrete_Data.csv')
dataset_df = pd.DataFrame(dataset)

n_data = 1030

attrs = dataset.columns
attrs = attrs[:len(attrs)-1]

dataset_X = dataset_df.drop(['Concrete compressive strength(MPa, megapascals) '], axis = 1).values
dataset_X = preprocessing.scale(dataset_X)
# print(dataset_X)

dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
dataset_Y = np.reshape(dataset_Y, (-1, 1))
# print(dataset_Y)

dataset_X_train, dataset_X_test, dataset_Y_train, dataset_Y_test = train_test_split(dataset_X, dataset_Y, test_size = 0.2)

dataset_X_train = np.hstack([dataset_X_train, np.ones((dataset_X_train.shape[0], 1))])
dataset_X_test = np.hstack([dataset_X_test, np.ones((dataset_X_test.shape[0], 1))])

iterations = 1000
alpha = 0.1

w = np.zeros((dataset_X_train.shape[1], 1))
J = ComputeCost(dataset_X_train, dataset_Y_train, w)
print(J)

print('update w')
print('w1~w8 + Bias:')
w = GradientDescent(dataset_X_train, dataset_Y_train, w, alpha, iterations)
print(w)
print('MSE:')
J = ComputeCost(dataset_X_test, dataset_Y_test, w)
print(J[0][0])
dataset_Y_pred = dataset_X_test.dot(w)
print('R2 score:')
print(r2_score(dataset_Y_test, dataset_Y_pred))

print('-----------------------')

w = np.zeros((dataset_X_train.shape[1], 1))
print('update wj')
print('w1~w8 + Bias:')
w = GradientDescent_wj(dataset_X_train, dataset_Y_train, w, alpha, iterations)
print(w)
print('MSE:')
J = ComputeCost(dataset_X_test, dataset_Y_test, w)
print(J[0][0])
dataset_Y_pred = dataset_X_test.dot(w)
print('R2 score:')
print(r2_score(dataset_Y_test, dataset_Y_pred))