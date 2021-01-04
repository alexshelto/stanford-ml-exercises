function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% J(theta) = - [1/m * sum(y^i * log(h(x^i)) + (1-y^i) * log(1-h(x)) ] + lam/2m * sum(theta_sub_j ^2)
% Gradiant = {
%.            1/m * sum(h(x^i) - y^i)* x_sub_j^i , for j = 0
%.            (1/m * sum(h(x^i) - y^i)* x_sub_j^i) * lambda/m * theta_sub_j

h_x = sigmoid(X * theta);
J = ((1/m) * (-y' * log(h_x) - (1 - y)' * log(1 - h_x))) + ((lambda/2/m) * sum(theta.^2))

theta_0 = (1/m) * X'*(h_x - y);
%theta_rest = (1/m * sum((h_x - y) .* X)) * ((lambda/m) .* theta([2:size(theta)], 1));
theta_rest = (( 1/m) * X'*(h_x - y) ) + ((lambda/m) .* theta);
grad = [theta_0(1); theta_rest([2:size(theta)], 1)];



% =============================================================

end

