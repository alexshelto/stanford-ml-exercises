
% Loading in the data
data = load('ex1data2.txt'); % House size | # rooms | price
% Data is 3 columns, features x0, x1, and output y
X = data(:, 1:2); % load every row of columns 1 and 2 for inputs
y = data(:, 3);   % load the 3rd and final column of data for outputs
m = length(y);

%
% Feature Normalization
%

% Outputting the first 10 data elements
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');
% note that House size is roughly 1000x size of room # so we feature scale
% Scale features and set them to zero mean
[X, mu, sigma] = featureNormalize(X);



%
% Gradient Descent
%

X = [ones(m, 1) X]; % Adding theta0 column of 1's

% Run gradient descent
% Choose some alpha value
alpha = 0.1;
num_iters = 400;
% Init Theta and Run Gradient Descent
theta = zeros(3, 1);
[theta, ~] = gradientDescentMulti(X, y, theta, alpha, num_iters);
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f\n%f\n%f\n',theta(1),theta(2),theta(3))




%
% Predition on new input data
%

% predict the price of 1650 sq-ft house with 3 bedrooms
prediction_set = [1650, 3];
normalized_prediction_set = (prediction_set - mu) ./ sigma; % normalize new input data to guess on
% Add bias: (theta 0)
normalized_prediction_set = [1, normalized_prediction_set];

prediction = normalized_prediction_set * theta;
fprintf('For a square footage of 1650 and 3 bedrooms, the estimated cost: %f\n',prediction);


