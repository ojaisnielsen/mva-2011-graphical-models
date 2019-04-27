function EM_model = EM(data,labels,height_or_picture,width)

   n=size(data,1);
   d=size(data,2);
   n_classes=max(labels);
   
   
    
    for k = 1:n_classes
        %indexes of the elements of class k
        i=(labels==k);
        %mean
        model{k}.mu=mean(data(i,:))';
        %covariance matrix
        diff =data(i,:)-ones(sum(i),1)*model{k}.mu';
        model{k}.sigma=1/sum(i)*diff'*diff;
        %probability for each class
        model{k}.pi=sum(i)/n;
    end
    
    %loglikelihood= sum(log(proba(data,model)));
    loglikelihood= -10000000000;
    prev_loglikelihood= 2*loglikelihood;
    
    
    %while we can increase likelihood
    while abs((loglikelihood-prev_loglikelihood)/loglikelihood)>(10^-9)
    %while loglikelihood~=prev_loglikelihood
%         if loglikelihood-prev_loglikelihood<0
%             coeff=(prev_loglikelihood-loglikelihood)/4.547474e-013;
%             if min(abs(floor(coeff)-coeff),abs(floor(coeff)+1-coeff))<0.0001
%                 fprintf(1,'LogLikelihood decreased of %2.4f x 4.547474e-013. (discrete error caused by computation limitation)\n',coeff)
%             else
%                 fprintf(1,'Error: logLikelihood decreased of %5.5f x 4.547474e-013.\n',(prev_loglikelihood-loglikelihood)/4.547474e-013)
%             end
%         end
        
        prev_loglikelihood= loglikelihood;
        prev_model=model;
        
        %FIRST STEP : E-step - Expectation computation
        
        %computation of the expectations
        tau=zeros(n,n_classes);
        for q=1:n_classes
            tau(:,q)=proba(data,model(q));
        end
        normalisation=sum(tau');
        tau=tau./(normalisation'*ones(1,n_classes));
        

        %SECOND STEP : M-step - Maximisation with the parameters
        
        
        %mean and covariance for each class
        for k = 1:n_classes
            %sum of the probabilities of belonging to class k
            sum_w=sum(tau(:,k));
            weights=(tau(:,k)*ones(1,d));
            
            %polynomial
            model{k}.pi=sum_w/n;
            
            %weigthed mean
            model{k}.mu=sum(data.*weights)'/sum_w;
            
            %difference to mean
            diff =data-ones(n,1)*model{k}.mu';
            %weighted covariance
            model{k}.sigma=(diff.*weights)'*diff/sum_w;
        end


        %computation of the log-likelihood
        %loglikelihood= sum(log(proba(data,model)));
        loglikelihood= sum(log(max(tau')));
        
        if nargin>2
            labels=sum( ([1:n_classes]'*ones(1,n)) .* (tau'==ones(n_classes,1)*max(tau')) )';
            if nargin ==4
                display_map(labels,height_or_picture,width);
            else
                display_map(labels,height_or_picture);
            end
            pause(1)
        end
        
  
    end
    
    %EM model
    EM_model.model=model;
    EM_model.logLikelihood=loglikelihood;
    EM_model.proba=@proba;
    EM_model.cluster=@cluster;
    EM_model.plot_gauss=@plot_gauss;
    if isnan(loglikelihood)
        EM_model.logLikelihood=prev_loglikelihood;
        EM_model.model=prev_model;
    end
    
    
end



function plot_gauss(mu,sigma)
    [U,S,V]=svd(sigma);
    steps=100;
    points=V'*([2*sqrt(S(1,1))*cos(2*pi*[1:steps]/steps);2*sqrt(S(2,2))*sin(2*pi*[1:steps]/steps)]);
    points=points'+ones(steps,1)*mu';
    plot(points(:,1),points(:,2),'color','black')
end



%x : n x d
%model.mu : d x 1 x c (clusters)
function p=proba(x,model)
    d=length(model{1}.mu);
    p=zeros(length(x),1);
    
    
    %for each part of the mixture of gaussian
    for k=1:length(model)
        
        model{k}.sigma;
        model{k}.mu;
        denum=(2*pi)^(d/2)*sqrt(det(model{k}.sigma));
        
        %differences to mean
        temp1=x-ones(length(x),1)*model{k}.mu';
        %product with the covariance
        temp2=sum(temp1'.*(inv(model{k}.sigma)*temp1'))';
        %add to the probability
        p=p+model{k}.pi/denum*exp(-1/2*temp2);
    end
end



function labels=cluster(EM_model,data,tag)
    n=size(data,1);
    d=size(data,2);
    model=EM_model.model;
    n_classes=length(model);
    
    %computation of tau
    tau=zeros(n,n_classes);
    for q=1:n_classes
        tau(:,q)=EM_model.proba(data,model(q));
    end
    normalisation=sum(tau');
    tau=tau./(normalisation'*ones(1,n_classes));
    
    %affectation to a cluster
    tau=tau';
    labels=sum( ([1:n_classes]'*ones(1,n)) .* (tau==ones(n_classes,1)*max(tau)) )';

end