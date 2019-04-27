train=load('EMGaussienne.data');
test=load('EMGaussienne.test');

%launch several KM with random initialisation, to keep the best one
best_KM_model=KMeans(train,4);
worst_KM_model=best_KM_model;
for k=1:1000
    KM_model=KMeans(train,4);
    if KM_model.distortion<best_KM_model.distortion
        best_KM_model=KM_model;
    end
        if KM_model.distortion>worst_KM_model.distortion
        worst_KM_model=KM_model;
    end
end

%launch KM clustering with best KM model
labels=best_KM_model.cluster(best_KM_model,train,'Best found K-Means on train data');
%launch KM clustering with best KM model
worst_labels=worst_KM_model.cluster(worst_KM_model,train,'Worst found K-Means on train data');

%train EM using data initialised with KM
EM_model=EM(train,worst_labels);
%sort test data using the EM model
EM_model.cluster(EM_model,test,'EM-model on test data (worst initialisation)');

%train EM using data initialised with KM
EM_model_id=EM_id(train,labels);
%sort test data using the EM model
EM_model_id.cluster(EM_model_id,test,'Isotropic EM-model on test data');

%train EM using data initialised with KM
EM_model_id=EM_id(train,labels);
%sort test data using the EM model
EM_model_id.cluster(EM_model_id,train,'Isotropic EM-model on train data');

%train EM using data initialised with KM
EM_model=EM(train,labels);
%sort test data using the EM model
EM_model.cluster(EM_model,train,'EM-model on train data');

%train EM using data initialised with KM
EM_model=EM(train,labels);
%sort test data using the EM model
EM_model.cluster(EM_model,test,'EM-model on test data');

