function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% cost = 1/2m * sum (square difference of guess)
% cost = 1/2m * sum(h(theta(x^i)) - y)^2

prediction_matrix =  X*theta; % function h(theta(x)) 

%for each value in guess matrix, subtract its value from actual and square
square_difference = (prediction_matrix - y).^2;

% implementing the whole formula with solved square difference
J = 1/(2*m)*sum(square_difference);



% =========================================================================

end
