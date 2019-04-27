%% Load data
clear all
u = textread('EMGaussienne.dat');
u = u';
[d, T] = size(u);
K = 4;

%% Parameters estimation

% Initialization
load('initparams.mat');
mu = mu';
sigma = reshape(cell2mat(sigmas),d,d,K);
logpi = log(pis);
loga = log(ones(K)/K);

loglikold = -Inf;
max_iterations = 1000;
tolerance = 1e-10;

% Iterations
for i=1:max_iterations
    
    logemit = loggaussian(u, mu, sigma);
    
    % Alpha - Beta recursion

    logalpha = zeros(K,T);
    logbeta = zeros(K,T);

    logalpha(:,1) = logemit(:,1) + logpi;
    logbeta(:,T) = zeros(1,K);

    for t=1:T-1
        for qtp1=1:K              
            logalpha(qtp1,t+1) = logemit(qtp1,t+1) + logsumexp(loga(:,qtp1) + logalpha(:,t));        
        end

        s = T-t;
        for qt=1:K                
            logbeta(qt, s) = logsumexp(loga(qt,:)' + logbeta(:,s+1) + logemit(:, s+1));
        end    
    end  
    
    loggamma = logalpha + logbeta;
    for t=1:T       
        loggamma(:,t) = loggamma(:,t) - logsumexp(loggamma(:,t));
    end
    gamma = exp(loggamma);    
    
    logxi = zeros(K,K,T-1);    
    for t=1:T-1
        for qt=1:K
            for qtp1=1:K
                logxi(qt,qtp1,t) = loga(qt,qtp1) + logalpha(qt,t) + logbeta(qtp1,t+1) + logemit(qtp1,t+1);
            end            
        end        
        logxi(:,:,t) = logxi(:,:,t) - logsumexp(logxi(:,:,t));
    end    
    xi = exp(logxi);
    
    logpi = loggamma(:,1);
    pi = exp(logpi);
    
    for qt=1:K
        for qtp1=1:K
            loga(qt, qtp1) = logsumexp(logxi(qt,qtp1,:));
        end
        loga(qt,:) = loga(qt,:) - logsumexp(loga(qt,:));  
    end     
    a = exp(loga);

    % compute log-likelihood
    
    
    loglik = sum(gamma(:,1) .* loggamma(:,1));       
    for t=1:t-1
        loglik = loglik + sum(sum(xi(:,:,t) .* loga));
    end    
    loglik = loglik + sum(gamma(:) .* logemit(:));
    
    fprintf('i = %d - loglik = %e\n',i,loglik);
    % check changes in log likelihood
    if loglik < loglikold - tolerance, error('the distortion is going up!'); end % important for debugging
    if loglik < loglikold + tolerance, break; end
    loglikold = loglik;
    

    % M-step:
    
    mu = zeros(d,K);
    sigma = zeros(d,d,K);
    
    
    
    for qt=1:K     
        for t=1:T
            mu(:,qt) = mu(:,qt) + xi(qt,t) * u(:,t);
            mu(:,qt) = mu(:,qt) + xi(qt,t) * u(:,t);
            sigma(:,:,qt) = sigma(:,:,qt) + xi(qt,t) * ((u(:,t) - mu(:,qt)) * (u(:,t) - mu(:,qt))');
        end
        mu(:,qt) = mu(:,qt) / sum(xi(qt,:));
        sigma(:,:,qt) = sigma(:,:,qt) / sum(xi(qt,:));                     
    end

end




