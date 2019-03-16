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
    
    # Xi_power = [[0.0]*n for i in range(m)]
    # for i in range(n):
    #     Xi_power[:, i] = X[:, i] ** i
    #     #Xi_power = Xi_power.reshape(m, 1)
    
    
    for itr in range(itr_limit):
        for i in range(n):
            # Xi_power = X[:, i] ** i
            # Xi_power = Xi_power.reshape(m, 1)
            # tmp[i] = tmp[i] - learning_rate * (1/m) * sum( np.dot(np.transpose((np.dot(X, w) - Y)), Xi_power) )    
            
            tmp[i] = tmp[i] - learning_rate * (1/m) * sum( np.dot(np.transpose((np.dot(X, w) - Y)), X[:, i]) )    
        w = tmp[:]
    return w

if __name__ == '__main__':
    dataset = pd.read_csv('Concrete_Data.csv')

    dataset_df = pd.DataFrame(dataset)

    attrs = dataset.columns
    attrs = attrs[:len(attrs)-1]

    n_data = 1030
    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)

    for column in attrs:
        pass

    dataset_X = dataset_df['Cement (component 1)(kg in a m^3 mixture)'].values    
    # dataset_X = dataset_df[column].values 
    dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
    dataset_X = dataset_X.astype(float)
    dataset_X = preprocessing.scale(dataset_X)

    dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    # for poly_degree in range(1,11):
    poly_degree = 10
    poly = preprocessing.PolynomialFeatures(degree=poly_degree)
    poly_x = poly.fit_transform(dataset_X)

    X_test_original  = dataset_X[int(-n_data*0.2) : ]
    X_test  = poly_x[int(-n_data*0.2) : ]
    X_train = poly_x[ : int(-n_data*0.2)]
    Y_test  = dataset_Y[int(-n_data*0.2) : ]
    Y_train = dataset_Y[ : int(-n_data*0.2)]

    # regr = linear_model.LinearRegression()
    # regr.fit(X_train, Y_train)
    # Y_pred = regr.predict(X_test)

    # one_train = np.ones((len(X_train), 1))
    # X_train_padded = np.concatenate((one_train, X_train), axis=1)

    w = np.array([0.5]*(poly_degree+1))
    w = w.reshape(poly_degree+1, 1)
    #print(CostFunc(X_train_padded, Y_train, w))
    # print(column)

    # do Gradient Descent
    iterations = 100000
    alpha = 0.0001

    w = GradientDescent(X_train, Y_train, w, alpha, iterations, poly_degree)
    # print(w)
    print("Bias:   %f" %w[0])
    print("Weight: ", end='')
    print(w[1:].T)


    # one_test = np.ones((len(X_test), 1))
    # X_test_padded = np.concatenate((one_test, X_test), axis=1)
    print("Mean squared error: %f" %CostFunc(X_test, Y_test, w))

    Y_predict = np.dot(X_test, w)
    print('R2 score: %.2f\n' % r2_score(Y_test, Y_predict))


    plt.plot(X_test_original, Y_test, 'rx')
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X_test_original, Y_predict), key=sort_axis)
    X_test_original, Y_predict = zip(*sorted_zip)
    plt.plot(X_test_original, Y_predict, color='m')
    
    
    plt.xlabel('Cement (component 1)(kg in a m^3 mixture)')
    plt.ylabel('Concrete compressive strength(MPa, megapascals) ')
    
    plt.show()