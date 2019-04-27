function EM_model = EM_id(data,labels)

   n=size(data,1);
   d=size(data,2);
   n_classes=max(labels);
   
    
    for k = 1:n_classes
        %indexes of the elements of class k
        i=(labels==k);
        %mean
        mu_{k}=mean(data(i,:))';
        %covariance matrix
        diff =data(i,:)-ones(sum(i),1)*mu_{k}';
        sigma_{k}=1/2*(1/sum(i)*sum(sum(diff.*diff)))*[1,0;0,1];
        %probability for each class
        pi_{k}=sum(i)/n;
    end
    
    loglikelihood= sum(log(proba(data,mu_,sigma_,pi_)));
    prev_loglikelihood= loglikelihood-1;
    
    
    %while we can increase likelihood
    while abs((loglikelihood-prev_loglikelihood)/loglikelihood)>(10^-12)
        if loglikelihood-prev_loglikelihood<0
            coeff=(prev_loglikelihood-loglikelihood)/4.547474e-013;
            if min(abs(floor(coeff)-coeff),abs(floor(coeff)+1-coeff))<0.0001
                fprintf(1,'LogLikelihood decreased of %2.4f x 4.547474e-013. (discrete error caused by computation limitation)\n',coeff)
            else
                fprintf(1,'Error: logLikelihood decreased of %5.5f x 4.547474e-013.\n',(prev_loglikelihood-loglikelihood)/4.547474e-013)
            end
        end
        
        prev_loglikelihood= loglikelihood;
        
        %FIRST STEP : E-step - Maximisation along pi
        
        %computation of the expectations
        tau=zeros(n,n_classes);
        for q=1:n_classes
            tau(:,q)=proba(data,mu_(q),sigma_(q),pi_(q));
        end
        normalisation=sum(tau');
        tau=tau./(normalisation'*ones(1,n_classes));
        

        %computation of the optimal pi
        sum_pi_n=sum(tau)/n;
        for q=1:n_classes
            pi_{q}=sum_pi_n(q);
        end
        
        
         %SECOND STEP : M-step -Maximisation along mu and covariance
        
        %computation of tau
        tau=zeros(n,n_classes);
        for q=1:n_classes
            tau(:,q)=proba(data,mu_(q),sigma_(q),pi_(q));
        end
        normalisation=sum(tau');
        tau=tau./(normalisation'*ones(1,n_classes));

        %mean and covariance for each class
        for k = 1:n_classes
            %sum of the probabilities of belonging to class k
            sum_w=sum(tau(:,k));
            weights=(tau(:,k)*ones(1,d));
            
            %weigthed mean
            mu_{k}=sum(data.*weights)'/sum_w;
            
            %difference to mean
            diff =data-ones(n,1)*mu_{k}';
            %weighted covariance
            sigma_{k}=1/2*(sum(sum(diff.*(diff.*weights)))/sum_w)*[1,0;0,1];
        end


        %computation of the log-likelihood
        loglikelihood= sum(log(proba(data,mu_,sigma_,pi_)));
        
        
  
    end
    
    %EM model
    EM_model.mu=mu_;
    EM_model.cov=sigma_;
    EM_model.pi=pi_;
    EM_model.proba=@proba;
    EM_model.cluster=@cluster;
    EM_model.plot_gauss=@plot_gauss;
    EM_model.logLikelihood=loglikelihood;
    
end



function plot_gauss(mu,sigma)
    [U,S,V]=svd(sigma);
    steps=100;
    points=V'*([2*sqrt(S(1,1))*cos(2*pi*[1:steps]/steps);2*sqrt(S(2,2))*sin(2*pi*[1:steps]/steps)]);
    points=points'+ones(steps,1)*mu';
    plot(points(:,1),points(:,2),'color','black')
end



%x : n x d
%mu_ : d x 1 x c (clusters)
function p=proba(x,mu_,sigma_,pi_)
    d=length(mu_{1});
    p=zeros(length(x),1);
    
    
    %for each part of the mixture of gaussian
    for k=1:length(pi_)
        denum=(2*pi)^(d/2)*sqrt(det(sigma_{k}));
        
        %differences to mean
        temp1=x-ones(length(x),1)*mu_{k}';
        %product with the covariance
        temp2=sum(temp1'.*(inv(sigma_{k})*temp1'))';
        %add to the probability
        p=p+pi_{k}/denum*exp(-1/2*temp2);
    end
end



function labels=cluster(model,data,tag)
    n=size(data,1);
    d=size(data,2);
    n_classes=length(model.pi);
    mu_=model.mu;
    sigma_=model.cov;
    pi_=model.pi;
    
    %computation of tau
    tau=zeros(n,n_classes);
    for q=1:n_classes
        tau(:,q)=model.proba(data,mu_(q),sigma_(q),pi_(q));
    end
    normalisation=sum(tau');
    tau=tau./(normalisation'*ones(1,n_classes));
    
    %affectation to a cluster
    tau=tau';
    labels=sum( ([1:n_classes]'*ones(1,n)) .* (tau==ones(n_classes,1)*max(tau)) )';

    if nargin>2
        %drawing
        f=figure('Name','Clustering with isotropic E-M','NumberTitle','off');
        hold on
        %plot the samples
        colors={'red','blue','yellow','green','orange','magenta','cyan','black'};
        for k=1:n_classes
            plot(data(labels==k,1),data(labels==k,2),'.','color',colors{mod(k-1,8)+1},'MarkerSize',10)
        end
        %things to write
        xlabel('x')
        ylabel('y')
        title(tag,'FontSize',12)

        %show likelihood
        loglikelihood= sum(log(model.proba(data,mu_,sigma_,pi_)));
        v=axis;
        text(v(1)+0.05*(v(2)-v(1)),v(3)+0.05*(v(4)-v(3)),['Loglikelihood: ',num2str(loglikelihood)])

        %show mean and covariances
        for k=1:n_classes
            plot(mu_{k}(1),mu_{k}(2),'.','color','black','MarkerSize',14)
            text(mu_{k}(1),mu_{k}(2),[' ',num2str(k)],'FontSize',14,'HorizontalAlignment','left')
            plot_gauss(mu_{k},sigma_{k});
        end

        hold off
        %saveas(f,[tag,'.eps'], 'psc2');
    end

end