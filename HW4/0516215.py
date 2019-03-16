import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])
    
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    X = np.matrix(X)

    #Write codes here 
    a1 = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1)              # 5000x401
    z2 = np.dot(a1, theta1.T)                                                   # 5000x401 * 401*10 = 5000*10
    a2 = np.concatenate((np.ones((m, 1), dtype=float), sigmoid(z2)), axis=1)    # 5000x11
    z3 = np.dot(a2, theta2.T)                                                   # 5000x11  * 11x10  = 5000x10
    h  = sigmoid(z3)                                                            # 5000x10

    return a1, z2, a2, z3, h



def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.array(X)
    y = np.array(y)
    
    # reshape the parameter array into parameter matrices for each layer
    # theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    # theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # print(params.shape)
    # print(hidden_size * (input_size+1))
    theta1 = params[0, 0 : (hidden_size * (input_size+1)) ]
    theta2 = params[0,     (hidden_size * (input_size+1)) : ]
    # print(theta1.shape)
    # print(theta2.shape)
    theta1 = theta1.reshape( hidden_size, (input_size+1) )
    theta2 = theta2.reshape( num_labels,  (hidden_size+1))




    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))
    
    return J





# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1

# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])

# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)




def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]  #5000 testcases
   
    #Write codes here
    
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    nabla1 = np.zeros((hidden_size, input_size+1))  # 10x401
    nabla2 = np.zeros((num_labels, hidden_size+1))  # 10x11

    for i in range(0, m):

        # STEP1: Forward Propagation
        # a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
        tmp = X[i,:]
        tmp.shape
        a1 = np.c_[np.ones((1,)), X[i, :]]                  # 1x401
        z2 = np.dot(a1, theta1.T)                           # 1x401 * 401x10 = 1x10
        a2 = np.c_[np.ones((1,)), sigmoid(z2)]              # 1x11  
        z3 = np.dot(a2, theta2.T)                           # 1x11  *  11x10 = 1x10
        h  = sigmoid(z3)                                    # 1x10
        
        # STEP2: Calculate delta3
        delta3 = h - y[i, :]                          # 1x10
        delta3 = delta3.T                             # 10x1
        delta3.shape

        #STEP3: Calculate delta2
        # delta2 = np.dot((theta2[:,1:]).T, delta3) * sigmoid_gradient(z2).T   # 10x10 * 10x1 = 10x1
        delta2 = np.multiply(np.dot((theta2[:,1:]).T, delta3) , sigmoid_gradient(z2).T)
        # delta2 = np.dot((theta2[:,1:]).T, delta3)   # 10x10 * 10x1 = 10x1


        #STEP4: Accumulate the gradient
        nabla1 = nabla1 + np.dot(delta2, a1)   # 10x1 * 1x401 = 10x401
        nabla2 = nabla2 + np.dot(delta3, a2)   # 10x1 * 1x11  = 10x11


    # STEP5: Obtain the gradient
    grad1 = (1.0/m) * nabla1   # 10x401
    grad2 = (1.0/m) * nabla2   # 10x11

    lambda_ = 1
    grad1[:, 1:] = grad1[:, 1:] + (lambda_/m) * theta1[:, 1:]  # do not regularize bias
    grad2[:, 1:] = grad2[:, 1:] + (lambda_/m) * theta2[:, 1:]

 
    grad = np.hstack((grad1.ravel(), grad2.ravel()))

    params_trained = np.hstack((theta1.flatten(), theta2.flatten()))
    J = cost(params_trained, input_size, hidden_size, num_labels, X, y, learning_rate)

    return J, grad





from scipy.optimize import minimize

# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})
      
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))