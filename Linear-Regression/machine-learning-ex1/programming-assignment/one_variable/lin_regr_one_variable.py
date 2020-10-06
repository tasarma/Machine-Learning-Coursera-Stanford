import matplotlib.pyplot as plt
import numpy as np


class one_variable():
    def __init__(self):
        self.x = [] #populations of city in 10,000s 
        self.y = [] #profits in $10,000s
        self.m = 0

        self.__reading_file()


    def __reading_file(self):

        # read data from file
        
        with open('ex1data1.txt','r') as data1:
            self.lines = data1.readlines()
            self.s = len(self.lines)

            for i in range(self.s):
                self.lines[i] = self.lines[i].strip('\n')
    
            for i in range(self.s):
                self.data = self.lines[i].split(',')
                self.x.append(float(self.data[0]))
                self.y.append(float(self.data[1]))
        self.m = len(self.y)
        self.add_ones()

    
    def plotting(self):
        
        #=== Plotting the Data ===

        plt.scatter(self.x[:,1],self.y)
        plt.ylabel('Profit in $10,000s')
        plt.xlabel('Population of City in 10,000s')

        plt.show()


    def add_ones(self):
        #Add a column of ones to self.x
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        self.x = np.stack([np.ones((self.m)),self.x], axis=1) 



    def compute_cost(self, theta):

        #=== compute the cost of linear regression ===
        
        self.J = 0
        for i in range(self.m):
            self.J += ((theta[0] + theta[1] * self.x[i][1]) - self.y[i])**2

        self.J *= (1 / (2*self.m))

        return self.J



    def gradient_descent(self,theta, alpha=0.01, iterations=1500):

        #=== Performs Gradient descent to learn theta ===

        self.theta = theta
        self.J_history = []
        
        self.t_0 = 0
        self.t_1 = 0
        a = 1
        while (a <= iterations):
            for i in range(self.m):
                self.t_0 =  (theta[0] + theta[1] * self.x[i][1] - self.y[i])
                self.t_1 =  (theta[0] + theta[1] * self.x[i][1] - self.y[i]) * (self.x[i][1])

                self.theta[0] -= (alpha / self.m) * self.t_0
                self.theta[1] -= (alpha / self.m) * self.t_1
                self.J_history.append(self.compute_cost(self.theta))
                #print('theta : ',theta,'  cost : ',self.compute_cost(theta))
            a += 1

        return self.theta, self.J_history



    def plotting2(self, theta):
        
        #plot data and linear fit
        
        plt.scatter(self.x[:,1], self.y)
        plt.plot(self.x[:,1], np.dot(self.x, theta), color = 'r')
        plt.legend(['Training data', 'Linear Regression'])

        plt.show()

   

    def visualizing(self, theta):
        #=== Visualizing J(theta_0, theta_1) ===

        # grid over which we will calculate J
        self.theta0_vals = np.linspace(-10, 10, 100)
        self.theta1_vals = np.linspace(-1, 4, 100)

        # initialize J_vals to a matrix of 0's
        self.J_vals = np.zeros((self.theta0_vals.shape[0], self.theta1_vals.shape[0]))

        # Fill out J_vals
        for i, theta0 in enumerate(self.theta0_vals):
            for j, theta1 in enumerate(self.theta1_vals):
                self.J_vals[i, j] = self.compute_cost([theta0, theta1])

        # Because of the way meshgrids work in the surf command, we need to
        # transpose J_vals before calling surf, or else the axes will be flipped
        self.J_vals = self.J_vals.T
        
        
        # surface plot
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_surface(self.theta0_vals, self.theta1_vals, self.J_vals, cmap='viridis')
        plt.xlabel('theta0')
        plt.ylabel('theta1')
        plt.title('Surface')
        

        # contour plot
        # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
        ax = plt.subplot(122)
        plt.contour(self.theta0_vals, self.theta1_vals, self.J_vals, linewidths=2, cmap='viridis', levels=np.logspace(-2, 3, 20))
        plt.xlabel('theta0')
        plt.ylabel('theta1')
        plt.plot(theta[0], theta[1], 'ro', ms=10, lw=2)
        plt.title('Contour, showing minimum')

        plt.show()

