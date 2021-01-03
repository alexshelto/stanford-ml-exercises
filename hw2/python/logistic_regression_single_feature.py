import numpy as np
import matplotlib.pyplot as plt
'''optimization algo'''
from scipy import optimize

def plotDecisionBoundary(plotData, theta, X, y):
    """
    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for  the negative examples.
    Parameters
    ----------
    plotData : func
        A function reference for plotting the X, y data.
    theta : array_like
        Parameters for logistic regression. A vector of shape (n+1, ).
    X : array_like
        The input dataset. X is assumed to be  a either:
            1) Mx3 matrix, where the first column is an all ones column for the intercept.
            2) MxN, N>3 matrix, where the first column is all ones.
    y : array_like
        Vector of data labels of shape (m, ).
    """
    # make sure theta is a numpy array
    theta = np.array(theta)

    # Plot Data (remember first column in X is the intercept)
    plotData(X[:, 1:3], y)

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2])

        # Calculate the decision boundary line
        plot_y = (-1. / theta[2]) * (theta[1] * plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.xlim([30, 100])
        plt.ylim([30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i, ui in enumerate(u):
            for j, vj in enumerate(v):
                z[i, j] = np.dot(mapFeature(ui, vj), theta)

        z = z.T  # important to transpose z before calling contour
        # print(z)

        # Plot z = 0
        plt.contour(u, v, z, levels=[0], linewidths=2, colors='g')
        plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap='Greens', alpha=0.4)


def plotData(X,y):
    data = np.c_[X,y]
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
    

def computeCost(theta, X, y):
    '''used for fmin optimization, same as costFunction but only returns cost'''
    J = 0 # initialize costt
    m = np.size(y) #
    h = sigmoid(X @ theta)
    J = (1 / m) * np.sum(-y * (np.log(h)) - (1 - y) * (np.log(1 - h)))
    return J


def costFunction(theta, X, y):
    '''returns: (1) the cost -loss- on the training set for the given thetas
                (2) gradiant
    cost = -(1/m * summation(y^i log(h(x^i)) + (1-y^i)*log(1-h(x^i)))
    h(x) = sigmoid(x*theta) 
    gradiant = 1/m * sum (h(x^i) - y) * x^i '''
    theta = theta.reshape(np.size(theta),1)
    J = 0 #cost
    m = np.size(y) # number of training examples
    h_x = sigmoid(X @ theta)
    gradiant = np.zeros((np.size(y), 1)) #initialized vector 
    J = computeCost(theta, X, y)
    gradiant = (X.T @ (h_x - y)) / m
    return J, gradiant
    


# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
def optimizeTheta(theta,X,y):
    '''MATLABS equiv of fminunc, runs a library optimization algorithm'''
    print(f'shape of X: {np.shape(X)}\nshape of y: {np.shape(y)}\nshape of theta: {np.shape(theta)}')
    options = {'maxiter': 450}
    result = optimize.minimize(costFunction,
                        theta,
                        (X, y),
                        jac=True,
                        method='TNC',
                        options=options)
    print(f'Cost at theta found by optimize.minimize: {result.fun}')
    return ( result['x'], result.fun )

def predict(theta, X):
    '''predicts whether the label is 0 or 1 based on threshold'''
    # theta: (3,1). X:(100,3)
    print(f'inside of predict\ntheta: {theta}')
    p = np.round(sigmoid(X @ theta))
    print(p)
    return p

def main():
    try:
        data = np.genfromtxt('ex2data1.txt', delimiter=',')
    except:
        print("error loading file")
        return -1
    
    X = data[:, (0,1)] # first two columns are input data
    y = data[:, 2].reshape(-1,1)     # output is the last column
    m = np.size(y)     # number of training examples
    #y = y.reshape(m,1) # giving y a dimension
    
    print(f'Sizes of vectors: X: {np.shape(X)},  y: {np.shape(y)}')
    # Plotting the data:
    plotData(X,y)
    plt.draw()
    
    X = np.c_[np.ones((m,1)), X] # add bias (theta 0)   X = |1's|feature 1| feature 2|
    n = np.shape(X)[1]           # number of features in X
    theta = np.zeros((n))        # (features,) dimention vector 
    
    # Computing cost and gradient:
    # should result in cost: 0.693, grad = [-0.1, -12.0092, -11.2628]
    cost, gradiant = costFunction(theta, X, y)
    print(f'cost at initial theta: {cost}')
    print(f'Gradiants for inintial theta(zeros): \n{gradiant}') 
    
    theta, cost = optimizeTheta(theta, X, y)
    print(f'Optimized theta values:\n {theta}')
    plotDecisionBoundary(plotData, theta, X, y)
    
    # predict prob that a student with a score of 45 on exam1 and 85 on exam 2
    prediction_vector = np.array([ 1, 45, 85])
    prob = sigmoid(prediction_vector @ theta.T)
    print(f'For a student with scores 45 and 85, we predict an admission probability of {prob}')
    p = predict(theta, X) #compute accuracy on training set

    #optimize this
    yes,no = 0,0
    for i in range(0, len(p)):
        if p[i] == y[i]:
            yes += 1
        else: 
            no += 1
    print(f'Train accuracy: {(yes/len(p))*100}')

    plt.show() # keeping plotted window open 
    return 0


if __name__ == '__main__':
    exit(main())
