import operator
import numpy  as np
import pandas as pd
from sklearn import datasets, linear_model, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def CostFunc(X, Y, w):
    # theta == w
    m = len(Y)
    #return 0.5 * (1.0/m) * sum((X*w - Y) ** 2)
    return (1.0/2.0) * (1.0/m) * sum((np.dot(X, w) - Y) ** 2)  

def GradientDescent(X, Y, w, learning_rate, itr_limit, w_degree):
    m = len(Y)
    n = len(w)
    # tmp = np.zeros((n,1))
    tmp = w[:]
    
    
    for itr in range(itr_limit):
        for i in range(n):

            tmp[i] = tmp[i] - learning_rate * (1/m) * sum( np.dot(np.transpose((np.dot(X, w) - Y)), X[:, i]) )    
        w = tmp[:]
    return w

if __name__ == '__main__':
    dataset = pd.read_csv('Concrete_Data.csv')

    dataset_df = pd.DataFrame(dataset)

    attrs = dataset.columns
    attrs = attrs[:len(attrs)-1]

    n_data = 1030
    n_features = len(attrs)
    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)

    poly_degree = 3

    X_padded = np.ones((n_data, 1))
    # print(np.shape(X_padded))

    for column in attrs:
        # pass

        # dataset_X = dataset_df['Cement (component 1)(kg in a m^3 mixture)'].values    
        dataset_X = dataset_df[column].values 
        dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
        dataset_X = dataset_X.astype(float)
        dataset_X = preprocessing.scale(dataset_X)


        poly = preprocessing.PolynomialFeatures(degree=poly_degree)
        poly_x = poly.fit_transform(dataset_X)
        poly_x = poly_x[:, 1:poly_degree+1]
        # print(np.shape(poly_x))

        X_padded = np.concatenate((X_padded, poly_x), axis=1)
        # X_padded.append(poly_x)
        # np.append(X_padded, poly_x, axis=1)
        
        # print(np.shape(X_padded))

    # print(np.shape(X_padded))
    # print(X_padded)

    dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    # for poly_degree in range(1,11):
    # poly_degree = 10
    # poly = preprocessing.PolynomialFeatures(degree=poly_degree)
    # poly_x = poly.fit_transform(dataset_X)

    X_test_original  = dataset_X[int(-n_data*0.2) : ]
    X_test  = X_padded[int(-n_data*0.2) : ]
    X_train = X_padded[ : int(-n_data*0.2)]
    Y_test  = dataset_Y[int(-n_data*0.2) : ]
    Y_train = dataset_Y[ : int(-n_data*0.2)]



    w = np.array([0.5] * (poly_degree * n_features + 1))
    w = w.reshape(poly_degree * n_features + 1, 1)
    #print(CostFunc(X_train_padded, Y_train, w))
    # print(column)

    # do Gradient Descent
    iterations = 120000
    alpha = 0.0007

    w = GradientDescent(X_train, Y_train, w, alpha, iterations, poly_degree)
    # print(w)
    print("Bias:   %f" %w[0])
    print("Weight: ", end='')
    print(w[1:].T)


    print("Mean squared error: %f" %CostFunc(X_test, Y_test, w))

    Y_predict = np.dot(X_test, w)
    print('R2 score: %.4f\n' % r2_score(Y_test, Y_predict))
