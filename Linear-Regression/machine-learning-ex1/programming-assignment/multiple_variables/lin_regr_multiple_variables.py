import matplotlib.pyplot as plt
import numpy as np
import os


class multiple_variables():

    def __init__(self):
        self.x = None           #Size of the house (in square feet) The number of bedrooms
        self.y = None           #The price of the house
        self.m = None           #Length of price
        self.__reading_file()



    #=== Read the Data From File ===
    def __reading_file(self):
        
        data = np.loadtxt(os.path.join('ex1data2.txt'), delimiter=',')

        self.x = data[:,:2]
        self.y = data[:,2]
        self.m = len(self.y)


    
    # === Feature Normalization ===
    def  featureNormalize(self,x):

        self.x_norm = x.copy()
        self.mu = np.zeros((1,x.shape[1]))      #Storing the mean value in mu
        self.sigma = np.zeros((1,x.shape[1]))   #storing the standard deviation in sigma

        for i in range(x.shape[1]):
            self.mu[0][i] = np.mean(x[:,i])
            self.sigma[0][i] = np.std(x[:,i])

        self.x_norm = (x - self.mu) / (self.sigma)

        def add_ones(a):
            #Add a column of ones to self.x
            a = np.concatenate([np.ones((self.m,1)),a], axis=1)
            return a

        self.x_norm = add_ones(self.x_norm)
        
        return self.x_norm, self.mu, self.sigma


    
    #=== compute the cost of linear regression ===
    def computeCostMulti1(self,theta, x):

        self.gd_x = x
        self.theta = theta

        self.multi = np.dot(self.gd_x, self.theta) - self.y
        self.J = np.dot(self.multi[None,:] , self.multi)
        self.J *= (1 / (2*self.m))

        return self.J



    def computeCostMulti2(self,theta,x):

        self.gd_x = x
        self.J = 0

        self.J = np.sum((np.dot(self.gd_x, theta) - self.y)**2)
        self.J *= (1 / (2*self.m))

        return self.J



    #=== Performs Gradient descent with multiple variables to learn theta ===
    def gradientDescentMulti(self,theta, alpha=0.1, num_iters=50):

        self.fn_x,self.mu, self.sigma = self.featureNormalize(self.x)
        self.theta = theta
        self.J_history = []
        
        a = 1
        while (a <= num_iters):
            for i in range(len(theta)):
                self.multi = (np.dot(( np.dot(self.fn_x, theta) - self.y)[None,:], self.fn_x[:,i]))
                self.theta[i] = theta[i] - (alpha / self.m) * self.multi
            self.J_history.append(self.computeCostMulti1(self.theta,self.fn_x))
            a += 1

        print('\nTheta computed from the gradient Descent : {:s}\n'.format(str(self.theta)))

        return self.theta, self.J_history



    #=== Plot the convergence graph with  alpha=0.000001, num_iters=50 === 
    def plotting(self, J_history):

        plt.plot(np.arange(len(J_history)), J_history, lw=2)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.show()



    #=== Computes the closed-form solution to linear regression using the normal equations ===

    def normalEqn(self, x, y):
        
        self.X = x
        self.Y = y
        self.M = len(self.Y)

        self.X = np.concatenate([np.ones((self.M,1)), self.X], axis = 1)
        self.theta_normal = np.zeros(self.X.shape[1])

        #equation
        self.step1 = (self.X.T@self.X) # @ means np.dot()
        self.step2 = np.linalg.pinv(self.step1)
        self.step3 = (self.X.T@self.Y)
        self.theta_normal = (self.step2@self.step3)

        # Display normal equation's result
        print('\nTheta computed from the normal equations: {:s}\n'.format(str(self.theta_normal)))

        return self.theta_normal
