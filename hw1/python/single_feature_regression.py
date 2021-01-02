
'''scientific'''
import numpy as np

'''plotting'''
import matplotlib.pyplot as plt




def plotData(X, y):
    '''Plotting the input and output data of training set'''
    plt.plot(X, y, 'rx', 10) # plotting points, red x's
    plt.ylabel('Profit in $10,000s')      # setting y-axis label
    plt.xlabel('Population of cits in 10,000s')
    return


def computeCost(X, y, theta):
    '''cost function'''
    # Cost = 1/2m * summation(squared error)
    m = len(y)
    h = X.dot(theta) # matrix holding the guessed values
    return ( (1/2/m) * np.sum((h-y)**2) )


def gradientDescent(X, y, theta, alpha, iters):
    '''Minimizing cost function with parameters theta'''
    #theta(i) = theta(i) - (alpha * 1/m summation(guess - answer))
    m = np.size(y) #number of training examples
    J_history = np.zeros((iters,1)) # creating an array to hold old values of J
    for i in range(0, iters):
        differences = (X.dot(theta)) - y # difference in prediction vs actual result
        change_theta = (alpha / m) * (X.T.dot(differences))
        theta = theta - change_theta
        J_history[i] = computeCost(X,y,theta)
        print(J_history[i]) 
    return (J_history, theta)


def showLinearRegression(X, theta):
    plt.plot(X[:,1], X.dot(theta), '-', color='b') #plotting data and theta guess. index 1 since index 0 is all 1's
    plt.legend(['training data', 'Linear regression'])



def main() -> int:
    # loading file data:
    try:
        data = np.genfromtxt('ex1data1.txt', delimiter=',')
    except:
        print("failure to load input data")
        return 1
    
    # Loading data into X and y vectors
    X = data[:,0] # pulling all of the features(column 1 (index 0)) into a 
    y = data[:,1]
    
    m = np.shape(y)[0] # m = number of exercises
    y = y.reshape(m, 1) # y(91,) => y(97,1), turining to cols
    # Displaying training set 2d graph
    plotData(X,y) #creating graph
    plt.draw()    #showing graph

    # Gradient descent 
    #
    X = np.c_[np.ones((m,1)), X] # Adding column of 1's, bias (theta 0)
    theta = np.zeros((2,1))
    alpha = 0.01
    iterations = 1500
   
    # Testing cost function on two known values 
    # print( computeCost(X,y,theta) )
    # print( computeCost(X,y, np.array([ [-1], [2] ])) ) 
    
    theta = gradientDescent(X, y, theta, alpha, iterations)[1]
    print(f'Found a theta value of: {theta}')
    showLinearRegression(X,theta)

    
    plt.show() #ensuring window doesnt close 
    

    return 0



if __name__ == '__main__':
  exit(main())

