% Approximation using multilayer perceptron
clc
clear all
close all
% Vector with 20 values from 0 to 1
X = linspace(0, 1, 20)';

% Weight and Biases
w1 = rand(1, 5);   
w2 = rand(5, 1);   
b1 = rand(1, 5);  
b2 = rand(1, 1);   

% Training parameters
eta = 0.1;         % Training rate
max_epochs = 2000; % Maximum number of training "rounds"
epoch = 0;         % Initial epoch
error_threshold = 1e-8;    % Treshold to stop training
prev_avg_error = 1;      % Initialize previous error 

% Expected Output
Y = (1 + 0.6 * sin(2 * pi * X / 0.7) + 0.3 * sin(2 * pi * X)) / 2;

% Activation functions
Hyp_tan_fct = @(x) tanh(x);          % Hyperbolic tangent for hidden layer
Hyp_tan_derivative = @(x) 1 - tanh(x).^2; % Derivative of tanh for backpropagation
linear_fct = @(x) x;                 % Linear function for output layer

% Training algorithm
while epoch < max_epochs
    e_total = 0; % Resets total error to 0 for each epoch
    for i = 1:length(X)
        % Forward pass
        x = X(i);                  % Current input
        target = Y(i);             % Target output for the current input
        
        % Hidden layer computation
        hidlay_input = x * w1 + b1;             % Input to hidden layer
        hidlay_output = Hyp_tan_fct(hidlay_input); % Activation of hidden layer
        
        % Output layer computation
        input = hidlay_output * w2 + b2;        % Input to output layer
        output = linear_fct(input);             % Output layer activation
        
        % Calculate error
        error = target - output;
        e_total = e_total + error^2;           % Accumulate total error

        % Backpropagation
        % Calculate local gradient for output layer
        local_grad_output = error;             % Since the output layer is linear
        
        % Calculate local gradient for hidden layer
        local_grad_hidlay = (local_grad_output * w2') .* Hyp_tan_derivative(hidlay_input);

        % Weight updates for hidden and output layers
        w2 = w2 + eta * (hidlay_output' * local_grad_output);
        b2 = b2 + eta * local_grad_output;

        w1 = w1 + eta * (x * local_grad_hidlay);
        b1 = b1 + eta * local_grad_hidlay;
    end
    
   % Calculate average squared error 
    avg_error = e_total / length(X);
    % Check if average squared error<error treshold
    if abs(prev_avg_error - avg_error) < error_threshold
        break;
    end
    % Update previous average error and increment epoch
    prev_avg_error = avg_error;
    epoch = epoch + 1; 
    fprintf('Epoch %d, Average Error: %f, Error Change: %f\n', epoch, avg_error, abs(prev_avg_error - avg_error));
end

% Final output after training
output = zeros(size(X));
for i = 1:length(X)
    hidlay_output = Hyp_tan_fct(X(i) * w1 + b1);
    output(i) = linear_fct(hidlay_output * w2 + b2);
end

% Plot the results
plot(X, Y, 'b-', X, output, 'r--');
legend('Target Output', 'MLP Output');
title('MLP Approximation of Target Function');
xlabel('Input');
ylabel('Output');
