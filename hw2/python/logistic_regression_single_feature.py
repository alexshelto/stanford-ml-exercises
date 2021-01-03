
import numpy as np
import matplotlib.pyplot as plt
'''optimization algo'''
import scipy.optimize


def plotData(data):
    positives = data[data[:,2] == 1]
    negatives = data[data[:,2] == 0]

    plt.scatter(negatives[:,0], negatives[:,1], c='y', marker='o',label='Not Admitted')
    plt.scatter(positives[:,0], positives[:,1], c='b', marker='+', label='Admitted')
    plt.legend()
    return


def sigmoid(x):
    '''x can be a vector or single num
    sigmoid = g(x) = 1 / 1+e^x'''
    return  ( (1/ (1+np.exp(-x))) )


def costFunction(theta, X, y):
    '''returns: (1) the cost -loss- on the training set for the given thetas
                (2) gradiant
    cost = -(1/m * summation(y^i log(h(x^i)) + (1-y^i)*log(1-h(x^i)))
    h(x) = sigmoid(x*theta) 
    gradiant = 1/m * sum (h(x^i) - y) * x^i '''
    J = 0 #cost
    m = np.size(y) # number of training examples
    gradiant = np.zeros((np.size(y), 1)) #initialized vector 
    h_x = sigmoid(X @ theta)             # h(x), sigmoid of X*theta
    J = (-1 / m) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1-h_x))
    #gradiant = (1/m) * np.sum(np.multiply((h_x - y), X))
    gradiant = (X.T @ (h_x - y)) / m
    return J, gradiant



def main():
    try:
        data = np.genfromtxt('ex2data1.txt', delimiter=',')
    except:
        print("error loading file")
        return -1
    
    X = data[:, (0,1)] # first two columns are input data
    y = data[:, 2]     # output is the last column
    m = np.size(y)     # number of training examples
    y = y.reshape(m,1) # giving y a dimension
    
    print(f'Sizes of vectors: X: {np.shape(X)},  y: {np.shape(y)}')
    # Plotting the data:
    #plotData(data)
    #plt.draw()
    
    X = np.c_[np.ones((m,1)), X] # add bias (theta 0)
    theta = np.zeros((np.shape(X)[1], 1)) # theta matrix size of num of features
    # Computing cost and gradient:
    # should result in cost: 0.693, grad = [-0.1, -12.0092, -11.2628]
    
    cost, gradiant = costFunction(theta, X, y)
    print(f'cost at initial theta: {cost}')
    print(f'Gradiants for inintial theta(zeros): \n{gradiant}') 

    #plt.show() # keeping plotted window open 
    return 0


if __name__ == '__main__':
    exit(main())
