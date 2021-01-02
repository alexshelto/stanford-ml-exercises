function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


   % hypothesis_values = X * theta;
   % error = hypothesis_values - y; % the error
    % theta0 = theta0 - 1/alpha * summation(error)*ith x
    % theta1 = theta1 - 1/alpha * summation(error)*ith x
    % Using 1 index matrices so 0 index => 1 index, 1 index => 2 index ...
   % temp_theta1 = theta(1) - ( (alpha / m) * sum(error .*X(:,1)));
   % temp_theta2 = theta(2) - ( (alpha / m) * sum(error .*X(:,2)));

    % Assigning all values of theta at the same time
   % theta = [temp_theta1; temp_theta2];
    error = (X*theta) - y;
    theta_change = (alpha/m) * (X'*error);
    theta = theta - theta_change;
    computeCost(X,y,theta)
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
