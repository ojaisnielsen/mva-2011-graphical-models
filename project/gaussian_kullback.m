%distance de kullback symétrisée
function distance = gaussian_kullback(mu0,sigma0,mu1,sigma1 )
    N=length(mu0);
    distance= 0.5*(log(det(sigma1)/det(sigma0))+trace(inv(sigma1)*sigma0)+(mu1-mu0)'*inv(sigma1)*(mu1-mu0)-N);
    distance= distance + 0.5*(log(det(sigma0)/det(sigma1))+trace(inv(sigma0)*sigma1)+(mu0-mu1)'*inv(sigma0)*(mu0-mu1)-N);
end