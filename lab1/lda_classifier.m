%classifier :
%takes as parameter data_train and data_test (data_test must not contain
%the labels)
%returns the classes for data_test (and the parameters of the model if need
%be)
%and parameters for the boundary : p(y=1|x)=beta0+beta*x'+x*quadratic*x'
function [classes,beta0,beta,quadratic] = lda_classifier(data_train,data_test)
[sigma,mu0,mu1,pi1]=model(data_train);
delta=inv(sigma);
beta0=(-mu1*delta*mu1'+mu0*delta*mu0')/2+log(pi1/(1-pi1));
beta=delta*(mu1'-mu0');
quadratic=0;
classes=(data_test(:,1:2)*beta+beta0>0);

end




%computes the parameters of the data (sigma, mu, pi)
function [sigma,mu0,mu1,pi1] = model(data)


L=length(data(:,1));
m=mean(data(:,1:end-1));

dx=data(:,1:2)-ones(L,1)*m;
sigma=1/L*dx'*dx;

%classe 0
classe0=(data(:,3)==0);
pi1=1-mean(classe0);

%means
mu0=mean(data(classe0,1:end-1));
mu1=mean(data(classe0==0,1:end-1));
end
