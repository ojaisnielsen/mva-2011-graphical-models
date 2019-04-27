function KM_model = KMeans(data,n_classes,old_labels)
    n=size(data,1);
    d=size(data,2);
    changes=1;
    %if initial labels are not specified, use random ones
    if (nargin<3)
        labels =floor(rand(n,1)*n_classes)+1;
        old_labels=labels;
    else
        labels=old_labels
    end

    while (changes>0)
        %FIND CENTROIDS
         centers = zeros(n_classes,d);
         
         for k =1:n_classes
             if sum(labels==k)>0
                centers(k,:)=mean(data(labels==k,:));
             end
         end
         

         %RECLASS
         %compute distances to each centroid
         dist_2=zeros(n_classes,n);
         for k=1:n_classes
             diff=data-ones(n,1)*centers(k,:);
             dist_2(k,:)=sum((diff.*diff)');
         end
         %find the minimum distance to centroid (returns an index in dist_2)
         labels=find(dist_2==ones(n_classes,1)*min(dist_2));
         %find the number of the centroid
         labels=mod(labels-1,n_classes)+1;

         %compute the number of changes
         changes=sum(labels~=old_labels);
         old_labels=labels;
    end
    
    %model
    diff=data-centers(labels,:);
    mean_dist=sum(sum((diff.*diff)'))/n;
    KM_model.centroids=centers;
    KM_model.distortion=mean_dist;
    KM_model.cluster=@cluster;

end


function labels=cluster(model,data,tag)
    centers=model.centroids;
    n=size(data,1);
    d=size(data,2);
    n_classes=size(centers,1);

     %compute distances to each centroid
     dist_2=zeros(n_classes,n);
     for k=1:n_classes
         diff=data-ones(n,1)*centers(k,:);
         dist_2(k,:)=sum((diff.*diff)');
     end
     %find the minimum distance to centroid (returns an index in dist_2)
     labels=find(dist_2==ones(n_classes,1)*min(dist_2));
     %find the number of the centroid
     labels=mod(labels-1,n_classes)+1;


     if nargin>2
             %DRAWING
            f=figure('Name','Clustering with K-Means','NumberTitle','off');
            colors={'red','blue','yellow','green','orange','magenta','cyan','black'};
            %plot the samples
            for k=1:n_classes
                plot(data(labels==k,1),data(labels==k,2),'.','color',colors{mod(k-1,8)+1},'MarkerSize',10)
                hold on
            end
            %texts
            xlabel('x')
            ylabel('y')
            title(tag,'FontSize',12)
            v=axis;

            %distortion
            diff=data-centers(labels,:);
            mean_dist=sum(sum((diff.*diff)'))/n;
            text(v(1)+0.05*(v(2)-v(1)),v(3)+0.05*(v(4)-v(3)),['Mean distortion: ',num2str(mean_dist)])

            %show the means
            for k=1:n_classes
                plot(centers(k,1),centers(k,2),'.','color','black','MarkerSize',14)
                text(centers(k,1),centers(k,2),[' ',num2str(k)],'FontSize',14,'HorizontalAlignment','left')
            end

            hold off
            %saveas(f,[tag,'.eps'], 'psc2');
     end
end

