

import numpy as np



def featureNormalization(X):
    '''Normalizes the features, each mean is 9 and std deviation is 1
       returns normalized X matrix, mean, & std deviation'''
    # initializing a np array of zeros to the # of features
    mu    = np.zeros((1,np.shape(X)[1])) # shape(rows, cols), cols = features
    sigma = np.zeros((1,np.shape(X)[1]))
    # Calculating the mean and standard deviation
    mu    = np.mean(X, axis=0) # Take the mean of each column. 2 cols, returns 1x2
    sigma = np.std(X, axis=0, ddof=1) # standard deviation with 1 degree of freedom
    # Normalizing X array
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)

def computeCost(X,y,theta):
    '''computes the cost on given theta inputs'''
    # cost = J = 1/2m * summation(guess - correct)^2
    m = np.size(y) # number of training examples
    # h = X_norm.dot(theta) # vector of guesses: (m,1) vec
    return ( (1/2/m) * np.sum((X_norm.dot(theta) - y)**2) )

def gradientDescent(X,y,theta,alpha,iters):
    '''reduces cost function by altering theta'''
    # theta(i) = theta(i) - alpha* 1/m*sum(guess - answer)
    m = np.size(y)
    J_history = np.zeros((iters,1))
    for i in range(0,iters):
        differences = (X.dot(theta) - y) # difference in value
        theta_change = (alpha / m) * (X.T(differences))
        theta = theta - theta_change
        J_history[i] = computeCost(X,y,theta)
        print(J_history[i]) #debugging

    return (J_histoy, theta)



def main() -> int:
    try:
        data = np.genfromtxt('ex1data2.txt', delimiter=',')
    except:
        print("error parsing data")
        return 1

    X = data[:, (0,1)] # parameters are the first two columns of the data
    y = data[:, 2]     # output is the last column of data
    m = np.size(y)     # number of training examples
    y = y.reshape(m,1) # (n, ) => (n,1) np array, -giving dimension-
    
    # Feature Normalization
    X, mu, sigma = FeatureNormalization(X)
    X = np.c_[np.ones((m,1)), X] # column extend X, add 1's for theta0
    
    # Gradient Descent
    # initializing values
    alpha = 0.1
    num_iters = 450
    theta = np.zeros((np.shape(X)[1],1)) #initializing theta to 0

    _, theta = gradientDescent(X,y,theta,alpha, num_iters)
    print(f'Theta computed from gradient descent: {theta}')
    

    return 0


if __name__ == '__main__':
    exit(main())
