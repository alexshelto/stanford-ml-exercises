% Load saved matrices from file
load('ex3data1.mat');
% The matrices X and y will now be in your MATLAB environment
m = size(X, 1); %5000

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);
displayData(sel);

% Cost function
theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10]; % (5,5) matrix
y_t = ([1;0;1;0;1] >= 0.5); % (5,1) logicall vector: true, false, false ,etc
lambda_t = 3;

% Calling cost function
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('Cost: %f | Expected cost: 2.534819\n',J);
fprintf('Gradients:\n'); fprintf('%f\n',grad);
fprintf('Expected gradients:\n 0.146561\n -0.548558\n 0.724722\n 1.398003\n');