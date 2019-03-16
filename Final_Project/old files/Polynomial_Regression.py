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

def GradientDescent(X, Y, w, learning_rate, itr_limit):
    m = len(Y)
    n = len(w)
    
    for itr in range(itr_limit):
        w = w - learning_rate * (1/m) * np.dot(X.T, (np.dot(X, w) - Y))    
    return w

if __name__ == '__main__':
    # dataset = pd.read_csv('dataset_preprocessing_old.csv')
    dataset = pd.read_csv('dataset_preprocessing.csv')
    dataset_df = pd.DataFrame(dataset)
    attrs = dataset.columns
    n_data = len(dataset_df)

    available_feature_index = []                         # 13: members
    # available_feature_index = [13]                       # 13: members
    genres_index = list(range(15, len(attrs)))         # genres
    available_feature_index.extend(genres_index)       # genres
    available_feature = [attrs[i] for i in available_feature_index]
    n_features = len(available_feature)
    print(available_feature)
    
    
    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)
    dataset_df = dataset_df.drop(dataset_df[dataset_df.scoredBy <= 1000].index)
    n_data = len(dataset_df)

    X_padded = np.ones((n_data, 1))
   
    poly_degree = 1
    iterations = 500000
    alpha = 0.001

    for column, col_index in zip(available_feature, available_feature_index):
        dataset_X = dataset_df[column].values 
        dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
        dataset_X = dataset_X.astype(float)

        if(col_index < 15):
            dataset_X = preprocessing.scale(dataset_X)
            poly = preprocessing.PolynomialFeatures(degree=poly_degree)
            poly_x = poly.fit_transform(dataset_X)
            poly_x = poly_x[:, 1 : poly_degree+1]
            X_padded = np.concatenate((X_padded, poly_x), axis=1)
        else:
            poly = preprocessing.PolynomialFeatures(degree=1)
            poly_x = poly.fit_transform(dataset_X)
            poly_x = poly_x[:, 1 : 1+1]
            X_padded = np.concatenate((X_padded, poly_x), axis=1)

    print(np.shape(X_padded))
    print(np.shape(X_padded)[1])
    X_padded_col = np.shape(X_padded)[1]

    dataset_Y = dataset_df['score'].values
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    X_test_original  = dataset_X[int(-n_data*0.2) : ]
    X_test  = X_padded[int(-n_data*0.2) : ]
    X_train = X_padded[ : int(-n_data*0.2)]
    Y_test  = dataset_Y[int(-n_data*0.2) : ]
    Y_train = dataset_Y[ : int(-n_data*0.2)]

    w = np.array([0.5] * X_padded_col)
    w = w.reshape(X_padded_col, 1)

    # do Gradient Descent
    print('poly_degree: %d' %poly_degree)
    print('alpha:       %g' %alpha)
    print('iterations:  %d' %iterations)

    w = GradientDescent(X_train, Y_train, w, alpha, iterations)
    print("Bias:   %f" %w[0])
    print("Weight: ", end='')
    print(w[1:].T)


    print("Mean squared error: %f" %CostFunc(X_test, Y_test, w))

    Y_predict = np.dot(X_test, w)
    r_squared = r2_score(Y_test, Y_predict)
    n_data_test = n_data*0.2
    r_squared_adj = 1 - ((1-r_squared)*(n_data_test-1)/(n_data_test-len(available_feature)-1))

    print('R2 score: %.4f' % r_squared)
    print('R2 score adjusted: %.4f\n' % r_squared_adj)
