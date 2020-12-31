% Alexander Shelton
% Week 2 hw 1 

data = load('ex1data1.txt'); % loading the data
X = data(:, 1); % loading the input data
y = data(:, 2); % the correct answers for y value

% Plotting data
plotData(X,y); % using the plot data function

% Gradient Descent to reduce cost function 
m = length(X) % Grabbing the number of training examples
X = [ones(m,1), X]; % appending a row of 1's to X for ( theta 0 )
theta = zeros(2,1); % initialize theta to 0's
iterations = 1500; % initializing iters
alpha = 0.01; % learning rate

% calculating cost
computeCost(X,y,theta)
computeCost(X,y,[-1;2]) % compute with non-zero theta

% Run gradient decent:
% Compute theta
theta = gradientDescent(X,y,theta, alpha, iterations);

% Printing theta to the screen
fprintf('theta computed from gradient descent: \n%f , %f\n', theta(1), theta(2))

% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,2), X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
% Applying a new vector, index 1 = 1 for theta0 * x1, index 2 is the x value
prediction_set_1 = [1, 3.5] * theta;
prediction_set_2 = [1, 7] * theta;
% outputting the prediction values
fprintf('for population size = 35,000, prediction: %f\n', prediction_set_1 * 10000);
fprintf('for population size = 70,000, prediction: %f\n', prediction_set_2 * 10000);







% Visualizing
% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];
	  J_vals(i,j) = computeCost(X, y, t);
    end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;

