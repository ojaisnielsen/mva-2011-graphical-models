%classifier :
%takes as parameter data_train and data_test (data_test must not contain
%the labels)
%returns the classes for data_test (and the parameters of the model if need
%be)
%and parameters for the boundary : p(y=1|x)=beta0+beta*x'+x*quadratic*x'
function [classes,beta0,beta,quadratic] = qda_classifier(data_train,data_test)
[sigma0,sigma1,mu0,mu1,pi1] = model(data_train);

delta0=inv(sigma0);
delta1=inv(sigma1);

beta0=log(pi1/(1-pi1)*sqrt(det(sigma0)/det(sigma1)))+(-mu1*delta1*mu1'+mu0*delta0*mu0')/2;
beta=delta1*mu1'-delta0*mu0';
quadratic=(delta0-delta1)/2;
quadra=sum(data_test'.*(quadratic*data_test'))';
betaX=quadra+data_test*beta;


classes=(betaX+beta0>0);%classe = 1
end


%computes the parameters of the data (sigma, mu, pi)
function [sigma0,sigma1,mu0,mu1,p1] = model(data)

%classe 0
classe=(data(:,3)==0);
L=sum(classe);
mu0=mean(data(classe,1:end-1));
dx=data(classe,1:2)-ones(L,1)*mu0;
sigma0=dx'*dx/L;

%classe 1
classe=(data(:,3)==1);
p1=mean(classe);
L=sum(classe);
mu1=mean(data(classe,1:end-1));
dx=data(classe,1:2)-ones(L,1)*mu1;
sigma1=dx'*dx/L;

end
