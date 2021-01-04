% Load data
% The first two columns contain the exam scores and the third column contains the label.
data = load('ex2data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

% Plot the data indicating labels
plotData(X,y);
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')

% Initializing data for cost function and gradient
[m, n] = size(X);  % m: rows, n: cols 
X = [ones(m,1) X]; % add column of 1's to X for theta(0)
initial_theta = zeros(n + 1, 1); % theta matrix. # of rows = cols of X

% Checking cost function on known values
[cost, grad] = costFunction(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradients at initial theta (zeros):\n%f\n%f\n%f\n', grad(1), grad(2), grad(3));

test_theta = [24; -0.2; -0.2];
[c,g] = costFunction(test_theta, X, y);
fprintf('Cost at initial theta: %f\n', c);
fprintf('Gradients at initial theta:\n%f\n%f\n%f\n', g(1), g(2), g(3));

% Opimizing theta with fmin function
%  Set options for fminunc
%options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
[theta, cost] = fminunc(@(t)(costFunction(t, X, y)), initial_theta, options)

% Print theta
fprintf('Cost at theta found by fminunc: %f\n', cost);
disp('theta:');disp(theta);

% Plot Boundary
hold on;
plotDecisionBoundary(theta, X, y);
hold off;

% Predict probability for student with ex1: 45 and ex2: 85 scores
prob = sigmoid([1 45 85] * theta);
fprintf('For a student with scores 45 and 85, we predict an admission probability of %f\n\n', prob);
% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
