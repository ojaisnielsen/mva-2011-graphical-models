input = imread('input2.png');
% input = imread('tigre_2.jpg');
[H,W,L] = size(input);

%% L*a*b* colorspace
if (L < 3),
    input = cat(3, input, input, input);
end
lab = lab2double(applycform(input, makecform('srgb2lab')));

%% Gradient
gradX = filter2([-1, 0, 1], lab(:,:,1));
gradY = filter2([-1, 0, 1]', lab(:,:,1));
grad = cat(3, gradY, gradX);

% Second-moment
grad2 = cat(3,grad(:,:,1).^2, grad(:,:,1).*grad(:,:,2), grad(:,:,1).*grad(:,:,2), grad(:,:,2).^2);

%% Polarity

n_sigma = 8;
scales = (0:n_sigma-1)/2;

polarity = zeros([H,W, n_sigma]);
smoothed_polarity = zeros([H,W, n_sigma]);
mean_contrast = zeros(H,W);
M = zeros([n_sigma,H,W,4]);
phi = zeros([n_sigma,H,W,2]);

for sigma_k = 1:n_sigma,
    sigma = scales(sigma_k);
    fprintf(1,'Computation for sigma = %2.3f\n',sigma)
    gauss = gaussian(sigma);
    rad = (size(gauss, 1)-1)/2;
    
    % Second moment matrix approximation
    M(sigma_k,:,:,:) = cat(3,filter2(gauss, grad2(:,:,1)),filter2(gauss, grad2(:,:,2)),filter2(gauss, grad2(:,:,3)),filter2(gauss, grad2(:,:,4)));
    
    for x=1:W;
        for y=1:H;
            
            % Actual area for the convolution (partial at the image
            % boundaries)
            y1 = max(y-rad,1);
            y2 = min(y+rad,H);
            x1 = max(x-rad,1);
            x2 = min(x+rad,W); 
            
            min_l = min(min(lab(y1:y2,x1:x2,1)));
            max_l = max(max(lab(y1:y2,x1:x2,1)));
            contrast = (max_l - min_l) / (max_l + min_l);
            if isnan(contrast),
                contrast = 0;
            end
            mean_contrast(y,x) = mean_contrast(y,x) + contrast;
            
            
            % Second eigen vector: principal direction
            [V,v] = eig(reshape(M(sigma_k,y,x,:),2,2)); 
            phi(sigma_k,y,x,:) = V(:,2);  
            if phi(sigma_k,y,x,1) == phi(sigma_k,y,x,2),
                n = [1; -1];
            else
                n = [-phi(sigma_k,y,x,2); phi(sigma_k,y,x,1)]/(phi(sigma_k,y,x,1) - phi(sigma_k,y,x,2));
            end
            n = n/norm(n);
                                     
            
            local_orient = grad(y1:y2,x1:x2,1)*n(1) + grad(y1:y2,x1:x2,2)*n(2);          
            cropped_gauss = gauss(y1-y+rad+1:y2-y+rad+1,x1-x+rad+1:x2-x+rad+1);
            E_plus = sum(sum(cropped_gauss.*max(local_orient, 0)));
            E_minus = -sum(sum(cropped_gauss.*min(local_orient, 0)));
            
            if E_plus + E_minus == 0,
                polarity(y,x,sigma_k) = 0; 
                continue
            end
            
            polarity(y,x,sigma_k) = abs(E_plus - E_minus)/(E_plus + E_minus);           
                        
        end
    end
    
    % Display principal directions
    y = 1:uint16(H/50):H;
    x = 1:uint16(W/50):W;
    imagesc(lab(:,:,1)), colormap gray, axis equal, axis off
    hold on
    quiver(x, y, reshape(phi(sigma_k,y,x,2),numel(y),numel(x)), reshape(phi(sigma_k,y,x,1),numel(y),numel(x)));
    hold off
    drawnow
    
    smoothed_polarity(:,:,sigma_k) = filter2(gaussian(2*sigma), polarity(:,:,sigma_k));
end

mean_contrast = mean_contrast / n_sigma;

%% Display polarities and smoothed polarities

figure
for sigma_k = 1:n_sigma,
    figure
    imagesc(polarity(:,:,sigma_k)), colormap gray 
    drawnow
    pause(0.5);
end

figure
for sigma_k = 1:n_sigma,
    imagesc(smoothed_polarity(:,:,sigma_k)), colormap gray 
    drawnow
    pause(0.5);
end

  

%% Scale selection

scale_decrease = abs(smoothed_polarity(:,:,2:sigma_k)-smoothed_polarity(:,:,1:sigma_k-1));

plot(reshape(sum(sum(scale_decrease,1),2)/(W*H), n_sigma-1, 1));
hold on
plot(0.02*ones(n_sigma-1,1));

scale_decrease = (scale_decrease <= 0.02);
scale_decrease(:,:,n_sigma - 1) = ones(H,W); % if nothing works, we take the biggest scale

is_uniform = (mean_contrast < 0.1);
scale = zeros(H,W);
for x=1:W;
    for y=1:H;  
        if is_uniform(y,x),
            scale(y,x) = 1;
            continue
        end
        % first scale for which the polarity decreases less than 2% 
        scale_k = find(scale_decrease(y,x,:), 1) + 1; 
        scale(y,x) = scale_k;
    end
end
figure
imagesc(scale), colormap gray;
%% Spacially variant smoothing

smoothed_lab = zeros(H,W,3);
for x=1:W;
    for y=1:H;
        gauss = gaussian(scales(scale(y,x)));
        r = (size(gauss, 1)-1)/2;
        y1 = max(y-r,1);
        y2 = min(y+r,H);
        x1 = max(x-r,1);
        x2 = min(x+r,W);            
        cropped_gauss = gauss(y1-y+r+1:y2-y+r+1,x1-x+r+1:x2-x+r+1);        
        smoothed_lab(y,x,1) = sum(sum(cropped_gauss.*lab(y1:y2,x1:x2,1)));
        smoothed_lab(y,x,2) = sum(sum(cropped_gauss.*lab(y1:y2,x1:x2,2)));
        smoothed_lab(y,x,3) = sum(sum(cropped_gauss.*lab(y1:y2,x1:x2,3)));
    end
end

figure, colormap gray;
imagesc(smoothed_lab(:,:,1));

%% Descriptors

desc = zeros(H, W, 8);

for x=1:W;
    for y=1:H;               
        [V,v] = eig(reshape(M(scale(y,x),y,x,:),2,2));
        if v(2,2) == 0,
            a = 0;
        else
            a = 1-(v(1,1)/v(2,2));
        end
        c = 2*sqrt(v(1,1)+v(2,2));
        p = polarity(y,x,scale(y,x));
        
        desc(y,x,:) = [reshape(smoothed_lab(y,x,:),3,1); a*c; p*c; c; x; y];
        
    end
end

%% Initial clusters
labels = cell(3);

labels{1} = ones(H,W);
labels{1}(round(H/4):round(3*H/4),round(W/4):round(3*W/4)) = 2;

labels{2} = ones(H,W,2);
labels{2}(round(H/2):end,:,1) = 2;
labels{2}(:,round(W/2):end,2) = 2;
labels{2}(round(H/4):round(3*H/4),round(W/4):round(3*W/4),:) = 3;

labels{3} = ones(H,W);
labels{3}(1:round(H/2),round(W/2):end) = 2;
labels{3}(round(H/2):end,1:round(W/2)) = 3;
labels{3}(round(H/2):end,round(W/2):end) = 4;

labels{4}=labels{3};
a=sqrt(5)/10;
labels{4}([round(H*(1/2-a)):round(H*(1/2+a))],[round(W*(1/2-a)):round(W*(1/2+a))]) = 5;
init=labels{4}(:,:);


%% normalisation of the data
data=reshape(desc,W*H,8);
data=data(:,std(data)~=0);
data_n=data;
data_n-ones(W*H,1)*mean(data_n);
data_n=data_n./(ones(W*H,1)*std(data_n));
[N,d]=size(data_n);


%EM_model = EM(data_n,reshape(init, W*H, 1),input);

%2 classes
%K=2;
%penality = (K-1+K*d+K*d*(d+1)/2)/2*log(N);
%EM_model = EM(data_n,reshape(init, W*H, 1),input);
%best_penalised_likelihood = EM_model.likelihood-penality;
%best_model = EM_model



%To find K using the method in the article

% best_penalised_likelihood=-10^10;
% 
% for K=[2:4]
%     penality = (K-1+K*d+K*d*(d+1)/2)/2*log(N);
%     [w,h,T]=size(labels{K-1});
%     for t=1:T
%         init=labels{K-1}(:,:,t);
%         EM_model = EM(data_n,reshape(init, W*H, 1),input);
%         fprintf('Penalised log-likelihood for model with %i classes : %f\n',EM_model.logLikelihood-penality)
%         if best_penalised_likelihood<EM_model.logLikelihood-penality
%             best_penalised_likelihood = EM_model.logLikelihood-penality;
%             best_model = EM_model;
%         end
%     end
% end
% 
% new_labels=best_model.cluster(best_model,data_n);
% display_map(new_labels,input);




%KM
%KM_model = KMeans(data_n,2,reshape(init, W*H, 1),input);
%new_labels_KM=KM_model.cluster(KM_model,data_n);

init = ones(H,W,1);
init(1:round(H/3),round(W/3):round(2*W/3)) = 2;
init(1:round(H/3),round(2*W/3):end) = 3;

init(round(H/3):round(2*H/3),1:round(W/3)) = 4;
init(round(H/3):round(2*H/3),round(W/3):round(2*W/3)) = 5;
init(round(H/3):round(2*H/3),round(2*W/3):end) = 6;

init(round(2*H/3):end,1:round(W/3)) = 7;
init(round(2*H/3):end,round(W/3):round(2*W/3)) = 8;
init(round(2*H/3):end,round(2*W/3):end) = 9;
display_map(init);



init=reshape(init, W*H, 1);
for K = 9:-1:2
    K
    penality = (K-1+K*d+K*d*(d+1)/2)/2*log(N);
    
    EM_model = EM(data_n,init,input);
    
    all_models{10-K}=EM_model;
    all_models{10-K}.labels = EM_model.cluster(EM_model,data_n);
    all_models{10-K}.penalised_loglik=EM_model.logLikelihood-penality;
    all_models{10-K}.non_spa_penalised_loglik=nonspatial_adjusted_loglikelihood(EM_model,data_n);
    
    
    fprintf('Penalised log-likelihood for model with %i classes : %f\n',K,all_models{10-K}.penalised_loglik)
    fprintf('Non-spatial penalised log-likelihood for model with %i classes : %f\n',K,all_models{10-K}.non_spa_penalised_loglik)
    
    
    
    min_dist=100000000000000000000;
    gauss=EM_model.model;
    min_dist_index=[1,2];
    for i =[1:K-1]
        for j= [i+1:K]
            dist=gaussian_kullback(gauss{i}.mu(1:end-2),gauss{i}.sigma(1:end-2,1:end-2),gauss{j}.mu(1:end-2),gauss{j}.sigma(1:end-2,1:end-2));
            if dist<min_dist
                min_dist=dist;
                min_dist_index=[i,j];
            end
        end
    end
    init=all_models{10-K}.labels; 
    max_l=max(init);
    init(init==j)=i;
    if j~=max_l
         init(init==max_l)=j
    end
end



for k = 1:8
    k
    display_map(all_models{k}.labels,input);
    pause(0.5)
end