function [f,g] = softmax_regression_vec(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%

a = theta'*X;
exp_a = exp(a);
exp_a = [exp_a;ones(1,size(exp_a,2))];%ones(1,size(exp_a,2))
exp_a_de = sum(exp_a,1);
% exp_a_num = exp_a(y);
% f = -(sum(log2(exp_a_num./exp_a_de)));

for i=1:m,
    for j=1:num_classes,
        if y(i) ~= j
            continue;
        end
        f = f+ log2(exp_a(j,i)/exp_a_de(i));
    end
end
f = -f;

flag =0;
for j=1:num_classes-1,
    for i=1:m,
        if(y(i)==j),
            flag=1;
        else
            flag=0;
        end
        g(:,j) = g(:,j)+ X(:,i)*(exp_a(j,i)/exp_a_de(i)-flag);
    end
end

  
  g=g(:); % make gradient a vector for minFunc

