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
    print('Accuracy (error= +- %.2f): %.2f\n' %(error, accu))


if __name__ == "__main__":
    
    dataset = pd.read_csv('dataset_preprocessing.csv')
    dataset_df = pd.DataFrame(dataset)
    attrs = dataset.columns
    n_data = len(dataset_df)
    

    dataset_df = dataset_df.sample(n = n_data).reset_index(drop = True)
    dataset_df = dataset_df.drop(dataset_df[dataset_df.scoredBy <= 1000].index)
    # dataset_df = dataset_df.drop(dataset_df[dataset_df.members <= 10000].index)
    n_data = len(dataset_df)
    # print(n_data)

    poly_deg = 4

    for poly_deg in [1,2,3,4]:
        print('degree:{}'.format(poly_deg))

                            
        numeric_feature_index = [12,13,14]                       # 10:studio_num 12: popularity 13: members 14:favorite

        numeric_feature = [attrs[i] for i in numeric_feature_index]
        n_features = len(numeric_feature)
        print("features: {} + all genres".format(numeric_feature))


        dataset_X = dataset_df[numeric_feature].values    
        dataset_X = dataset_X.astype(float)    
        dataset_X = preprocessing.scale(dataset_X)
        dataset_Y = dataset_df['score'].values
        dataset_Y = np.reshape(dataset_Y, (len(dataset_Y), 1))

        genres_index = list(range(15, len(attrs)))             # genres
        numeric_feature_index.extend(genres_index)           # genres
        genres_feature = [attrs[i] for i in numeric_feature_index]
        genres_X = dataset_df[genres_feature].values    
        n_features = len(numeric_feature)+len(genres_feature)
        
        poly = preprocessing.PolynomialFeatures(degree=poly_deg)
        X_poly = poly.fit_transform(dataset_X)
        X_poly = np.concatenate((X_poly, genres_X), axis=1)

        X_test  = dataset_X[int(-n_data*0.2) : ]
        X_train = dataset_X[ : int(-n_data*0.2)]
        Y_test  = dataset_Y[int(-n_data*0.2) : ]
        Y_train = dataset_Y[ : int(-n_data*0.2)]
        X_test_poly  = X_poly[int(-n_data*0.2) : ] 
        X_train_poly = X_poly[ : int(-n_data*0.2)]
        # print(X_train_poly.shape)
        
        regr = linear_model.LinearRegression()
        regr.fit(X_train_poly, Y_train)
        Y_pred = regr.predict(X_test_poly)

        # print('Weight: ', regr.coef_)
        # print('Bias: ', regr.intercept_)
        # print("Mean squared error: %.2f" % mean_squared_error(Y_test, Y_pred))
        # print('R2 score: %.3f' % r2_score(Y_test, Y_pred))
        
        r_squared = r2_score(Y_test, Y_pred)
        n_data_test = n_data*0.2
        r_squared_adj = 1 - ((1-r_squared)*(n_data_test-1)/(n_data_test-n_features-1))

        print('R2 score: %.3f' % r_squared)
        print('R2 score adjusted: %.3f' % r_squared_adj)
        Accuracy(0.25)

