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

dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)

for column in attrs:
    print(column)

    dataset_X = dataset_df[column].values
    dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
    dataset_X = dataset_X.astype(float)    
    dataset_X = preprocessing.scale(dataset_X)
    # print(dataset_X)

    dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    dataset_X_test  = dataset_X[int(-n_data*0.2) : ]
    dataset_X_train = dataset_X[ : int(-n_data*0.2)]
    dataset_Y_test  = dataset_Y[int(-n_data*0.2) : ]
    dataset_Y_train = dataset_Y[ : int(-n_data*0.2)]

    regr = linear_model.LinearRegression()
    regr.fit(dataset_X_train, dataset_Y_train)

    dataset_Y_pred = regr.predict(dataset_X_test)

    print('Weight: ', regr.coef_)
    print('Bias: ', regr.intercept_)
    print("Mean squared error: %.2f" % mean_squared_error(dataset_Y_test, dataset_Y_pred))
    # Explained variance score: 1 is perfect prediction
    print('R2 score: %.2f\n' % r2_score(dataset_Y_test, dataset_Y_pred))

    plt.figure(num = column)
    # plt.subplot(2, 4, i+1)
    plt.scatter(dataset_X_test, dataset_Y_test, color='black', marker = '.', s = 1)
    plt.plot(dataset_X_test, dataset_Y_pred, color='blue', linewidth = 1)

    plt.xlabel(column)
    plt.ylabel('Concrete compressive strength(MPa, megapascals)')

    # plt.show()