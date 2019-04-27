%classifier :
%takes as parameter data_train and data_test
%returns the classes for data_test (and the parameters of the model if need
%be)
%and parameters for the boundary : p(y=1|x)=beta0+beta*x'+x*quadratic*x'
function [classes,beta0,beta,quadratic] = lin_classifier(data_train,data_test,eps)

[X, Y] = model(data_train)

theta =(X'*X)\(X'*Y)
%sigma2 = sum((Y - theta' * X).^2)/size(Y, 1);

quadratic = 0;
beta0 = theta(end,1) - 0.5;
beta = theta(1:end-1,1);


classes=(data_test(:,1:2)*beta+beta0>0);

end


function [X, Y] = model(data)

 X = [data(:,1:2), ones(size(data, 1), 1)];
 Y = data(:,3);

end
