function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and should be converted back into the weight matrix.
% 
%   The returned parameter grad is an "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%Feedfoward propagation
X = [ones(m,1), X];
a1 = X';
a2 = sigmoid(Theta1 * a1);
a2 = [ones(1,m); a2];
a3 = sigmoid(Theta2 * a2); 
anstrans = a3';
newymatrix = zeros(num_labels,m);
for i = 1:m
    newymatrix(y(i),i) = 1;
end
%sigmoid cost function
neglog = (-1) * log(a3);
neglogoneminus = (-1) * log(1 - a3);
J = (1/m) * sum(sum((neglog .* newymatrix) + (neglogoneminus .* (1 - newymatrix)),2),1);

%Calculating regularization for cost function
Theta1b = Theta1;
Theta1b(:,1) = []; % Removing first column
Theta2b = Theta2;
Theta2b(:,1) = []; % Removing first column
Theta1s = Theta1b .^ 2;
Theta2s = Theta2b .^ 2;
regterm = (((sum(sum(Theta1s,1),2) + sum(sum(Theta2s,1),2))) * (lambda/(2 * m)));
J = J + regterm;

%Backpropagation
delta3 = a3 - newymatrix;
delta2 = ((Theta2)' * delta3) .* (a2 .* (1 - a2));
delta2(1, :) = [];
Delta1 = delta2 * a1';
Delta2 = delta3 * a2';
Theta1_grad = Delta1 * (1/m);
Theta2_grad = Delta2 * (1/m);

%Regularization for gradient
Theta1wozeros = Theta1(:,(2:end));
Theta2wozeros = Theta2(:,(2:end));
Theta1_grad(:,(2:end)) = Theta1_grad(:, (2:end)) + ((lambda/m) * Theta1wozeros);
Theta2_grad(:,(2:end)) = Theta2_grad(:, (2:end)) + ((lambda/m) * Theta2wozeros);

    















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
