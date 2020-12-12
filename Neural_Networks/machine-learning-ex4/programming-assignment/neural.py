import os
import utils
import numpy as np
from scipy import optimize
from scipy.io import loadmat
from matplotlib import pyplot as plt

input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9



# =========== Part 1: Loading and Visualizing Data =============

data = loadmat(os.path.join('ex4data1.mat'))
X, y = data['X'], data['y'].ravel()

# Set the zero digit to 0, rather than its mapped 10 in this dataset
y[y == 10] = 0

# Number of training examples
m = y.size

# Randomly select 100 data points to display
sel = np.random.choice(m, 100, replace=False)
sel = X[sel, :]

#utils.displayData(sel)



# ================ Part 2: Loading Parameters ================

weights = loadmat(os.path.join('ex4weights.mat'))

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters 
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])



# ================ Part 3: Sigmoid Gradient  ================
def sigmoidGradient(z):
    """
    Compute the gradient of the sigmoid function evaluated at
    each value of z (z can be a matrix, vector or scalar)
    """
    g = np.zeros(z.shape)
    g = utils.sigmoid(z) * (1 - utils.sigmoid(z))

    return g




# ================ Part 4: Compute Cost (Feedforward) ================

def nnCostFunction(nn_params,
                   input_layer_size,
                   hidden_layer_size,
                   num_labels,
                   X, y, lambda_=0.0):
    """
    Implements the neural network cost function and gradient for a two layer neural 
    network which performs classification
    """
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))

    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size

    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    #Part 1: Feedforward the neural network and return the cost in the variable J

    # Forward propagation
    X = np.concatenate([np.ones((m,1)), X], axis=1)
    z2 = Theta1 @ X.T
    a2 = utils.sigmoid(z2)

    a2 = np.concatenate([np.ones((m,1)), a2.T], axis=1)
    z3 = Theta2 @ a2.T
    h_theta = utils.sigmoid(z3)  # h_theta equals to a3

    # y(k) - the great trick - we need to recode the labels as vectors
    # containing only values 0 or 1
    y_new = np.zeros((num_labels, m))
    for i in range(m):
        y_new[y[i], i] = 1

    J = (1/m) * sum(sum((-y_new) * np.log(h_theta) - (1 - y_new) * np.log(1 - h_theta)))

    # Note we should not regularize the terms that correspond to the bias. 
    # For the matrices Theta1 and Theta2, this corresponds to the first
    # column of each matrix.
    t1 = Theta1[:, 1:]
    t2 = Theta2[:, 1:]

    # Regularization
    Reg = (lambda_ / (2*m)) * (np.sum(np.sum(t1**2)) + np.sum(np.sum(t2**2)))

    # Regularized cost function
    J = J + Reg

    # Back propagation
    for i in range(m):
        # Step 1
        a1 = X[i,:] # X already have a bias Line 44 (1*401)
        a1 = a1.T # (401*1)
        z2 = Theta1 @ a1    # (25*401)*(401*1)
        a2 = utils.sigmoid(z2)    # (25*1)
        
        a2 = np.concatenate([np.ones((1, 1)), a2[:,None]], axis=0) # adding a bias (26*1)
        z3 = Theta2 @ a2    # (10*26)*(26*1)
        a3 = utils.sigmoid(z3)    # final activation layer a3 == h(theta) (10*1)

        # Step 2
        delta_3 = a3 - y_new[:, i][:,None]  # (10*1)
        z2 = np.concatenate([np.ones((1, 1)), z2[:, None]], axis=0) # adding a bias (26*1)

        # Step 3
        delta_2 = (Theta2.T @ delta_3) * sigmoidGradient(z2) # (26*10)*(10*1)

        # Step 4
        delta_2 = delta_2[1:]   # skipping sigma2(0) (25*1)

        Theta2_grad = Theta2_grad + delta_3 * a2.T
        Theta1_grad = Theta1_grad + delta_2 * a1.T

    # Step 5
    Theta2_grad = (1/m) * Theta2_grad # (10*26)
    Theta1_grad = (1/m) * Theta1_grad # (25*401)

    # Regularization
    Theta1_grad[:, 1] = Theta1_grad[:, 1] + ((lambda_/m) * Theta1[:, 1])
    Theta2_grad[:, 1] = Theta2_grad[:, 1] + ((lambda_/m) * Theta2[:, 1])

    # Unroll gradients
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J,grad



# ================ Part 5: Initializing Pameters ================
def randInitializeWeights(L_in, L_out, epsilon_init=0.12):
    """
    Randomly initialize the weights of a layer with L_in
    incoming connections and L_out outgoing connections
    """
    W = np.zeros((L_out, 1 + L_in))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

    return W


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()], axis=0)
    


# =================== Part 6: Training NN ===================
#  After you have completed the assignment, change the maxiter to a larger
#  value to see how more training helps.
options= {'maxiter': 100}

lambda_ = 1

# Create "short hand" for the cost function to be minimized
costFunction = lambda p: nnCostFunction(p, input_layer_size,
                                        hidden_layer_size,
                                        num_labels, X, y, lambda_)

# Now, costFunction is a function that takes in only one argument
# (the neural network parameters)
res = optimize.minimize(costFunction,
                        initial_nn_params,
                        jac=True,
                        method='TNC',
                        options=options)

# Get the solution of the optimization
nn_params = res.x
        
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))


pred = utils.predict(Theta1, Theta2, X)
print('Training Set Accuracy: %f' % (np.mean(pred == y) * 100))

    
utils.displayData(Theta1[:, 1:])


    
