%classifier :
%takes as parameter data_train and data_test
%returns the classes for data_test (and the parameters of the model if need
%be)
%and parameters for the boundary : p(y=1|x)=beta0+beta*x'+x*quadratic*x'
function [classes,beta0,beta,quadratic] = logis_classifier(data_train,data_test,eps)

if nargin == 2
	eps = 0.01;
end
converged = false;
[X, Y] = model(data_train);
theta = zeros(size(X, 2), 1);


%i = 0;
while ~converged,
	eta = sigma(X * theta);
	W = diag(eta .* (1 - eta));
	update = - (X'*W*X)\(X'*(eta - Y));
	theta = theta + update;
	converged = (norm(update) < eps);
	%i = i+1;
end

quadratic = 0;
beta0 = theta(end,1);
beta = theta(1:end-1,1);

classes=(data_test(:,1:2)*beta+beta0>0);

end

function [y] = sigma(x)
	y = 1.0 ./(1 + exp(-x));
end


function [X, Y] = model(data)

 X = [data(:,1:2), ones(size(data, 1), 1)];
 Y = data(:,3);

end
