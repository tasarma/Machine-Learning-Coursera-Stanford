import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class regularized_log_reg():
     
    def __init__(self):
        #=== Read data from file ===

        data = np.loadtxt(os.path.join('ex2data2.txt'), delimiter=',')
        self.x = data[:, :2]        #Exam scores
        self.y = data[:, 2]         #Labels
        self.m = self.y.shape[0]



    def plotData(self, x, y):
        #=== Display the data on a 2-dimensional plot ===
        
        self.X = x
        self.Y = y
        # Find Indices of Positive and Negative Examples
        self.pos = self.Y == 1
        self.neg = self.Y == 0

        plt.plot(self.X[self.pos, 0], self.X[self.pos, 1] ,'k*', lw=2, ms=10)
        plt.plot(self.X[self.neg, 0], self.X[self.neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)


        # Labels and Legend
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')

        # Specified in plot order
        plt.legend(['y = 1', 'y = 0'], loc='upper right')
        
        plt.show()



    def add_ones(self, added):
        #=== Add intercept term to x ===

        self.a, self.b = self.x.shape
        self.added = np.concatenate([np.ones((self.a, 1)), added], axis = 1)

        return self.added



    def mapFeature(self, x1, x2, degree = 6):

        self.x1 = x1
        self.x2 = x2

        if (self.x1.ndim > 0):
            self.out = [np.ones(self.x1.shape[0])]
        else:
            self.out = [np.ones(1)]

        for i in range(1, degree + 1):
            for j in range( i + 1):
                self.out.append((self.x1 ** (i - j)) * (self.x2 ** j))

        if self.x1.ndim > 0:
            return np.stack(self.out, axis=1)
        else:
            return np.array(self.out)



    def sigmoid(self,z):
        #=== Compute sigmoid function given the input z ===     
       
        #if z is a scalar
        if (type(z) == int or type(z) == float):
            self.g = (1 / (1 + math.exp(-z)))
            return self.g

        elif (type(z) == np.int64 or type(z) == np.float64):
            self.g = (1 / (1 + math.exp(-int(z))))
            return self.g

        #if z is a matrix
        elif (type(z) == np.ndarray and z.ndim > 1):
            self.z = np.array(z)
            self.g = np.zeros(self.z.shape)
            for i in range(self.z.shape[1]):
                for j in range(self.z.shape[0]):
                    self.g[j][i] = (1 / (1 + math.exp(-z[j][i])))

            return self.g

        #if z is a vector
        else:
            self.z = np.array(z)
            self.g = np.zeros(self.z.shape)
            
            for i in range(self.z.shape[0]):
                self.g[i] = (1 / (1 + math.exp(-z[i])))
            
            return self.g



    def costFunctionReg(self, theta, Lambda):
        #=== Compute cost and gradient for logistic regression with regularization ===

        self.theta = theta
        self.lambda_ = Lambda
        self.new_x = self.mapFeature(self.x[:, 0], self.x[:, 1])

        self.J = 0
        self.grad = np.zeros(self.theta.shape)
        
        self.h_theta = self.sigmoid(self.new_x.dot(self.theta))
        
        for i in range(self.m):
            self.J += ((-self.y[i] * math.log(self.h_theta[i])) - ((1-self.y[i]) * math.log(1 - self.h_theta[i])))

        self.J *= (1 / self.m)
        

        self.J += (self.lambda_ / (2*self.m)) * np.dot((self.theta[1:len(self.theta)]).T, (self.theta[1:len(self.theta)]))

        
        self.grad = (1 / self.m) * ((self.h_theta - self.y)[None,:].dot(self.new_x)) + (self.lambda_ / self.m) * self.theta.T

        return self.J, self.grad




