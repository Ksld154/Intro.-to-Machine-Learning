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
    tmp = w

    for itr in range(itr_limit):
        for i in range(n):
            tmp[i] = tmp[i] - learning_rate * (1/m) * sum( np.dot(np.transpose((np.dot(X, w) - Y)), X[:, i]) ) 
        # tmp[0] = tmp[0] - learning_rate * (1/m) * sum( (np.dot(X, w) - Y) ) 
        # tmp[1] = tmp[1] - learning_rate * (1/m) * sum( np.dot(np.transpose((np.dot(X, w) - Y)), X[:, 1]) )         
        w = tmp
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
        # print(column)

    dataset_X = dataset_df['Cement (component 1)(kg in a m^3 mixture)'].values 
    dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
    dataset_X = preprocessing.scale(dataset_X)
    # print(dataset_X)
    # dataset_X = preprocessing.normalize(dataset_X, norm='l2')

    # options = ['l1', 'l2', 'max']
    # norm_x = Normalizer(norm=opt).fit_transform(x)
    # print("After %s normalization: " % opt.capitalize(), norm_x)
    



    dataset_Y = dataset_df['Concrete compressive strength(MPa, megapascals) '].values
    dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

    X_test  = dataset_X[int(-n_data*0.2) : ]
    X_train = dataset_X[ : int(-n_data*0.2)]
    Y_test  = dataset_Y[int(-n_data*0.2) : ]
    Y_train = dataset_Y[ : int(-n_data*0.2)]


    one_train = np.ones((len(X_train), 1))
    X_train_padded = np.concatenate((one_train, X_train), axis=1)

    w = np.array([[0.2], [0.2]])

    print(CostFunc(X_train_padded, Y_train, w))

    
    # do Gradient Descent
    iterations = 300000
    alpha = 0.0001

    w = GradientDescent(X_train_padded, Y_train, w, alpha, iterations)
    print("Bias:   %f" %w[0])
    print("Weight: %f" %w[1])
    #print(w)
    
    one_test = np.ones((len(X_test), 1))
    X_test_padded = np.concatenate((one_test, X_test), axis=1)
    print("Mean squared error: %f" %CostFunc(X_test_padded, Y_test, w))

    plt.plot(X_test, Y_test, 'rx')
    plt.plot(X_train_padded[:,1], np.dot(X_train_padded, w), '-')
    

    plt.xlabel('Cement (component 1)(kg in a m^3 mixture)')
    plt.ylabel('Concrete compressive strength(MPa, megapascals) ')
    
    plt.show()