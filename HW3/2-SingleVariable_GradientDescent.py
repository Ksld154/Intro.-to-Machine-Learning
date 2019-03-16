import numpy  as np
import pandas as pd 
from sklearn import datasets, linear_model, preprocessing
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
    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)


    for column in attrs:

        dataset_X = dataset_df[column].values 
        dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
        dataset_X = dataset_X.astype(float)
        dataset_X = preprocessing.scale(dataset_X)

        dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
        dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

        X_test  = dataset_X[int(-n_data*0.2) : ]
        X_train = dataset_X[ : int(-n_data*0.2)]
        Y_test  = dataset_Y[int(-n_data*0.2) : ]
        Y_train = dataset_Y[ : int(-n_data*0.2)]

        one_train = np.ones((len(X_train), 1))
        X_train_padded = np.concatenate((one_train, X_train), axis=1)

        w = np.array([[0.2], [0.2]])

        #print(CostFunc(X_train_padded, Y_train, w))
        print(column)

        # do Gradient Descent
        iterations = 3000000
        alpha = 0.00001

        w = GradientDescent(X_train_padded, Y_train, w, alpha, iterations)
        print("Bias:   %f" %w[0])
        print("Weight: %f" %w[1])
        
        one_test = np.ones((len(X_test), 1))
        X_test_padded = np.concatenate((one_test, X_test), axis=1)
        print("Mean squared error: %f" %CostFunc(X_test_padded, Y_test, w))

        Y_predict = np.dot(X_test_padded, w)
        print('R2 score: %.2f\n' % r2_score(Y_test, Y_predict))


        plt.plot(X_test, Y_test, 'rx')
        plt.plot(X_test_padded[:,1], np.dot(X_test_padded, w), '-')
        
        plt.xlabel(column)
        plt.ylabel('Concrete compressive strength(MPa, megapascals) ')
        
        #plt.show()