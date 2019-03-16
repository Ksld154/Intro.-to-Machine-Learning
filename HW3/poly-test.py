import operator
import numpy  as np
import pandas as pd 
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('Concrete_Data.csv')
dataset_df = pd.DataFrame(dataset)

n_data = 1030

attrs = dataset.columns
attrs = attrs[:len(attrs)-1]

# dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)

# for column in attrs:
    
#     print(column)
for poly_degree in range(1, 6):

    dataset_X = dataset_df['Cement (component 1)(kg in a m^3 mixture)'].values
    # X_1d = dataset_X[:]
    # dataset_X = dataset_df[column].values
    dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
    dataset_X = dataset_X.astype(float)    
    dataset_X = preprocessing.scale(dataset_X)
    # print(dataset_X)

    dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
    Y_1d = dataset_Y[:]
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    poly = preprocessing.PolynomialFeatures(degree=poly_degree)
    poly_x = poly.fit_transform(dataset_X)

    X_test_original  = dataset_X[int(-n_data*0.2) : ]
    X_test  = poly_x[int(-n_data*0.2) : ]
    X_train = poly_x[ : int(-n_data*0.2)]
    Y_test  = dataset_Y[int(-n_data*0.2) : ]
    Y_train = dataset_Y[ : int(-n_data*0.2)]

    regr = linear_model.LinearRegression()
    regr.fit(X_train, Y_train)
    Y_pred = regr.predict(X_test)
    # print(Y_pred)

    # X_1d = X_1d[ : int(-n_data*0.2)]
    # Y_1d = Y_1d[ : int(-n_data*0.2)]
    # res = np.polyfit(X_1d, Y_1d, 4)
    # print(res)



    print('Weight: ', regr.coef_)
    print('Bias: ', regr.intercept_)
    print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f\n' % r2_score(Y_test, Y_pred))





    # # plt.figure(num = column)
    # plt.scatter(X_test_original, Y_test, color='black', marker = '.', s = 1)

    # # sort the values of x before line plot
    # sort_axis = operator.itemgetter(0)
    # sorted_zip = sorted(zip(X_test_original, Y_pred), key=sort_axis)
    # X_test_original, Y_pred = zip(*sorted_zip)
    # plt.plot(X_test_original, Y_pred, color='m')

    # # plt.plot(X_test_original, Y_pred, color='blue', linewidth = 1)

    # plt.xlabel('Cement (component 1)(kg in a m^3 mixture)')
    # # plt.xlabel(column)    
    # plt.ylabel('Concrete compressive strength(MPa, megapascals)')
    # plt.show()