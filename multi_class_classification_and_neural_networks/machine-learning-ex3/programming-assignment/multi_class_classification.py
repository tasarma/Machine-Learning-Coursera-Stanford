# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all


import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat 



# =========== Part 1: Loading and Visualizing Data =============

input_layer_size  = 400 # 20x20 Input Images of Digits
num_labels = 10 # 10 labels, from 1 to 10

data = loadmat(os.path.join('ex3data1.mat'))

X = data['X']
y = data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# Because this dataset was used in MATLAB where there is no index 0
y[y == 10] = 0

m = y.shape[0]


# Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100, replace=False)
sel = X[rand_indices, :]

#utils.displayData(sel) # Display the data



# ============ Part 2a: Vectorize Logistic Regression ============


# test values for the parameters theta
theta_t = np.array([-2, -1, 1, 2], dtype=float )

# test values for the inputs
X_t = np.concatenate([np.ones((5, 1)), 
    np.arange(1, 16).reshape(5, 3, order='F')/10.0], axis=1)

# test values for the labels
y_t = np.array([1,0,1,0,1])

# test value for the regularization parameter
lambda_t = 3


def sigmoid(z):
    #Computes the sigmoid of z.
    
    return 1.0 / (1.0 + np.exp(-z))



def lrCostFunction(theta, X, y, Lambda):
    """
    Computes the cost of using theta as the parameter for regularized
    logistic regression and the gradient of the cost to the parameters.
    """

    m = y.size

    # convert labels to ints if their type is bool
    if y.dtype == bool:
        y = y.astype(int)

    J = 0
    grad = np.zeros(theta.shape)

    h_theta = sigmoid(X @ theta)

    J += (1/m) * ((-y @ np.log(h_theta)) - ((1-y) @ np.log(1 - h_theta))) + (Lambda/(2*m)) * (sum(pow(theta[1:],2)))

    temp = theta #np.concatenate(theta[:,None], axis = 1)
    temp[0] = 0
    grad = (1/m) * (X.T @ (h_theta - y)) + (Lambda/m) * temp

    return J, grad
    
"""
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
print('Cost     : {:.6f}'.format(J))
print('Gradients: [{:.6f}, {:.6f}, {:.6f}, {:.6f}]'.format(*grad))
"""



# ============ Part 2b: One-vs-All Training ============


def oneVsAll(X, y, num_labels, Lambda):
    """
    Trains num_labels logistic regression classifiers and returns
    each of these classifiers in a matrix all_theta, where the i-th
    row of all_theta corresponds to the classifier for label i.
    """

    # Some useful variables
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))
        
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    # Set Initial theta
    initial_theta = np.zeros(n + 1)
    
    # Set options for minimize
    options = {'maxiter': 50}
    
    # Optimize the cost function
    for c in range(num_labels):
        res = optimize.minimize(lrCostFunction,
                                initial_theta,
                                (X, (y == c), Lambda),
                                jac=True,
                                method='CG',
                                options=options)
        theta = res.x
        cost = res.fun
        
        #theta = np.insert(theta,0,cost)
        
        all_theta[c,:] = theta

    return all_theta
     
Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)



# ================ Part 3: Predict for One-Vs-All ================

def predictOneVsAll(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier. The labels 
    are in the range 1..K, where K = size(all_theta, 1). 
    """

    m = X.shape[0];
    num_labels = all_theta.shape[0]
    
    p = np.zeros(m)

    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    predict = sigmoid(X @ all_theta.T)
    p = np.argmax(predict, axis=1)
    
    return p

pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred == y) * 100))



