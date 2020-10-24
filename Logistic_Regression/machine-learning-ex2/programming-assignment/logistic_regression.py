import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


class logistic_regression():
     
    def __init__(self):
        #=== Read data from file ===

        data = np.loadtxt(os.path.join('ex2data1.txt'), delimiter=',')
        self.x = data[:, 0:2]       #Exam scores
        self.y = data[:, 2]         #Labels
        self.m = self.y.size             #Number of training examples



    def plotData(self):
        #=== Display the data on a 2-dimensional plot ===

        # Find Indices of Positive and Negative Examples
        self.pos = self.y == 1
        self.neg = self.y == 0

        plt.plot(self.x[self.pos, 0], self.x[self.pos, 1] ,'k*', lw=2, ms=10)
        plt.plot(self.x[self.neg, 0], self.x[self.neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)

        # add axes labels
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.legend(['Admitted', 'Not admitted'])
        plt.show()



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



    def add_ones(self, added):
        #=== Add intercept term to x ===
        
        self.a, self.b = self.x.shape
        self.added = np.concatenate([np.ones((self.a, 1)), added], axis = 1)

        return self.added



    def costFunction(self, theta):
        #=== Compute cost and gradient for logistic regression ===

        self.new_x = self.add_ones(self.x)
        self.J = 0
        self.grad = np.zeros(self.y.size)

        self.h_theta = self.sigmoid(self.new_x.dot(theta) )
        
        for i in range(self.y.shape[0]):
            self.J += ((-self.y[i] * math.log(self.h_theta[i])) - ((1-self.y[i]) * math.log(1 - self.h_theta[i])))

        self.J *= (1 / self.m)
        
        self.grad = (1 / self.m) * ((self.h_theta - self.y)[None,:].dot(self.new_x))

        return self.J, self.grad



    def learning_parameters(self, theta):
        #=== Learning parameters using scipy.optimize ===
        
        self.initial_theta = theta

        # Use truncated Newton algorithm for optimization which is 
        #equivalent to MATLAB's fminunc
        self.res = minimize(self.costFunction, 
                self.initial_theta,
                jac = True,
                method = 'TNC', 
                options = {'maxiter': 400})
        
        #The fun property of `OptimizeResult` object returns
        #The value of costFunction at optimized theta
        self.cost = self.res.fun

        #The optimized theta is in the x property
        self.theta = self.res.x
        
        #print(self.cost,self.theta)
        return self.theta



    def mapFeature(self, x1, x2, degree = 6):
                
        self.x1 = x1
        self.x2 = x2
        
        if (self.x1.ndim > 0):
            self.out = [np.ones(self.x1.shape[0])]
        else:
            self.out = [np.ones(1)]

        for i in range(1, self.degree + 1):
            for j in range( i + 1):
                self.out.append((self.x1 ** (i - j)) * (self.x2 ** j))
        
        if X1.ndim > 0:
            return np.stack(self.out, axis=1)
        else:
            return np.array(self.out)



    def plotDecisionBoundary(self, theta):
        #=== Plot Boundary ===
         
        self.xx = self.add_ones(self.x)
        self.db_theta = theta
        if (self.xx.shape[1] <= 3):

            #=== Only need 2 points to define a line, so choose two endpoints ===
            self.plot_x = [self.x[:, 0].min() - 1, self.x[:, 0].max() + 1]
            self.plot_x = np.array(self.plot_x)

            self.plot_y = (-1/(self.db_theta[2])) * (self.db_theta[1] * self.plot_x + self.db_theta[0])
            self.plot_y = np.array(self.plot_y)

            fig  = plt.figure(figsize = (8,8))
            ax = fig.add_subplot(1,1,1)
            ax.plot(self.plot_x, self.plot_y)

            ax.legend(['Decision Boundary'])
            plt.connect(self.plotData(), ax) 
        
            #Legend, specific for the exercise
            ax.legend(['Decision Boundary','Admitted', 'Not admitted'])

        
        else:
            #=== The grid range ===
            self.u = np.linspace(-1, 1.5, 50)
            self.v = np.linspace(-1, 1.5, 50)

            self.z = np.zeros((len(self.u), len(self.v)))
            
            
            # Evaluate z = theta*x over the grid
            for i, ui in enumerate(self.u):
                for j, vj in enumerate(self.v):
                    self.z[i,j] = np.dot(self.mapFeature(ui, vj),theta)
            
            #important to transpose z before calling contour
            self.z = self.z.T
            
            # Plot z = 0
            plt.contour(u, v, z, levels=[0], linewidths=2, colors='g') 
            plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], 
                    cmap='Greens', alpha=0.4)


    def predict(self, theta, x):
        #=== Predict whether the label is 0 or 1 using learned logisti regression parameters theta ===

        # Number of training examples
        self.p_theta = theta
        self.p_x = x
        self.M = self.p_x.shape[0]
        self.P = np.zeros(self.M)

        self.P = self.sigmoid(self.p_x * self.p_theta)
        
        return self.P

