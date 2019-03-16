import operator
import numpy  as np
import pandas as pd 
from sklearn import datasets, linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def Accuracy(error):
    n_correct = 0
    for pred,test in zip(Y_pred, Y_test):
        if abs(pred-test) <= error:
            n_correct += 1
    accu = float(n_correct / (n_data*0.2))
    print('Accuracy (error= +- %.1f): %.2f\n' %(error, accu))


if __name__ == "__main__":
    
    dataset = pd.read_csv('dataset_preprocessing.csv')
    dataset_df = pd.DataFrame(dataset)
    attrs = dataset.columns
    n_data = len(dataset_df)
    

    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)
    dataset_df = dataset_df.drop(dataset_df[dataset_df.scoredBy <= 1000].index)
    # dataset_df = dataset_df.drop(dataset_df[dataset_df.members <= 10000].index)
    n_data = len(dataset_df)
    print(n_data)

    poly_deg = 1
    # for poly_deg in [1,2,3,4]:
    print('degree:{}'.format(poly_deg))

    dataset_df[['favorites', 'members']] = dataset_df[['favorites', 'members']].astype(float)
    dataset_df['fav/mem'] = dataset_df['favorites'] / dataset_df['members']
    
    # dataset_df.rename(index=str, columns={"popularity": "popularity_rank"})
    attrs = ['members', 'favorites', 'fav/mem', 'studio_num', 'popularity', 'episodes']
    # attrs = ['members', 'favorites', 'fav/mem','popularity',]


    for column in attrs:
        
        dataset_X = dataset_df[column].values    
        dataset_X = np.reshape(dataset_X, (len(dataset_X), 1))
        dataset_X = dataset_X.astype(float)    
        dataset_X = preprocessing.scale(dataset_X)
        dataset_Y = dataset_df['score'].values
        dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

        X_test  = dataset_X[int(-n_data*0.2) : ]
        X_train = dataset_X[ : int(-n_data*0.2)]
        Y_test  = dataset_Y[int(-n_data*0.2) : ]
        Y_train = dataset_Y[ : int(-n_data*0.2)]

        poly = preprocessing.PolynomialFeatures(degree=poly_deg)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly  = poly.fit_transform(X_test)    
        
        regr = linear_model.LinearRegression()
        regr.fit(X_train_poly, Y_train)
        Y_pred = regr.predict(X_test_poly)

        # print('Weight: ', regr.coef_)
        # print('Bias: ', regr.intercept_)
        # print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
        if column == 'popularity':
            column += '_rank'
        print('features: {}'.format(column))
        r_squared = r2_score(Y_test, Y_pred)
        print('R2 score: %.2f' % r_squared)
        Accuracy(0.2)

        plt.plot(X_test, Y_test,'.', markersize=2)
        # plt.scatter(X_test, Y_test, color='black', marker = '.', s = 2)
        sort_axis = operator.itemgetter(0)
        sorted_zip = sorted(zip(X_test, Y_pred), key=sort_axis)
        X_test, Y_pred = zip(*sorted_zip)
        plt.plot(X_test, Y_pred, color='m')

        plt.title('R2 score: {0:0.4f}'.format(r_squared))
        plt.xlim((-1, 2.5))
        plt.ylim((5, 10))
        plt.xlabel(column+' (Scaled)')
        plt.ylabel('Score')
        plt.show()
