function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

error = X*theta-y;
error2 = error.*error;
sumerror = sum(error2);
sumerror = sumerror/(2*m);

thetal = theta(2:end,:).* theta(2:end,:);
re = (sum(thetal)*lambda)/(2*m);
J = sumerror + re;

n = size(theta,1);
for j = 1:n,
    for i = 1:m,
        grad(j) = grad(j) + (theta'* X(i,:)'-y(i))*X(i,j);
    end
    grad(j) = grad(j)/m;
end

grad(2:end) = grad(2:end) + (lambda*theta(2:end))/m;
% =========================================================================

grad = grad(:);

end
