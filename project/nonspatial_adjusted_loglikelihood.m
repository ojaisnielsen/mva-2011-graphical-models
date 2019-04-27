function loglikelihood = nonspatial_adjusted_loglikelihood(EM_model,data)
 



K=length(EM_model.model);
   [N,d]=size(data);
   d=d-2;
   data=data(:,1:d);

   penality = (K-1+K*d+K*d*(d+1)/2)/2*log(N);
   
   %remove spacial dependance
   for q=1:K
       model{q}.pi=EM_model.model{q}.pi;
       model{q}.sigma=EM_model.model{q}.sigma(1:d,1:d);
       model{q}.mu=EM_model.model{q}.mu(1:d,1);
   end

    %compute probabilities
    tau=zeros(N,K);
    for q=1:K
        tau(:,q)=EM_model.proba(data,model(q));
    end
    
    normalisation=sum(tau');
    tau=tau./(normalisation'*ones(1,K));
    
    loglikelihood=sum(log(max(tau')))-penality;
    
    
end
