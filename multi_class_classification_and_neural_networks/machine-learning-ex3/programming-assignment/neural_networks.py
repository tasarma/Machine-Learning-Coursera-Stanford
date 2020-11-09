# Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks


import os
import numpy as np
from matplotlib import pyplot
from scipy import optimize
import utils

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat


# Setup the parameters for this exercise
input_layer_size  = 400;  # 20x20 Input Images of Digits
hidden_layer_size = 25;   # 25 hidden units
num_labels = 10;          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)



# =========== Part 1: Loading and Visualizing Data =============


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

# Display the data
#utils.displayData(sel)



# ================ Part 2: Loading Pameters ================


# Load the weights into variables Theta1 and Theta2
weights = loadmat(os.path.join('ex3weights.mat'))

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)


# ================= Part 3: Implement Predict =================


def sigmoid(z):
    #Computes the sigmoid of z.

    return 1.0 / (1.0 + np.exp(-z))


def predict(Theta1, Theta2, X):
    # Predict the label of an input given a trained neural network

    # Make sure the input has two dimensions
    if X.ndim == 1:
        X = X[None]  # promote to 2-dimensions
    
    m = X.shape[0]
    num_labels = Theta2.shape[0]

    p = np.zeros(X.shape[0])
    
    # Add ones to the X data matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    predict1 = sigmoid(X @ Theta1.T)
    predict1 = np.concatenate([np.ones((m, 1)), predict1], axis=1)

    predict2 = sigmoid(predict1 @ Theta2.T)

    p = np.argmax(predict2, axis=1)

    return p

"""
pred = predict(Theta1, Theta2, X)
print('Training Set Accuracy: {:.1f}%'.format(np.mean(pred == y) * 100))
"""

# randomly permute examples, to be used for visualizing one 
# picture at a time
indices = np.random.permutation(m)

for i in range(m):
    print('Displaying Example Image')
    utils.displayData(X[indices[i],:], figsize=(4, 4))

    pred = predict(Theta1, Theta2, X[indices[i],:])
    print('Neural Network Prediction: {} )'.format(*pred))

    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit: ');
    if s == 'q':
        break


