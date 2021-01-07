function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%


% adding a column of ones (bias) to the X matrix
X = [ones(m,1) X];


% ----Hidden layer || a2-----
% z^2 = X * theta1.  where X==a1
% a^2 = g(z^2)

% ----Output layer || a3------
% z^3 = a2 * theta2
% a^3 = g(z^3)

% size(X) 5000 x 401
% size(Theta1) 25 x 401
% size(Theta2) 10 x 26
% size(a_2) 5000 x 25

% calculating hidden layer 
a_2 = sigmoid(X * Theta1'); % X * Theta1' == z^2
a_2 = [ones(5000,1), a_2]; % adding a bias term or a row of 0's

% calculating output layer
% z_3 = a_2 * Theta2'; % 5000x26 * 26x10 == (5000x10)
a_3 = sigmoid(a_2 * Theta2');

[value,p] = max(a_3, [], 2); % take the max from all rows, index of vector == class 


% =========================================================================


end

