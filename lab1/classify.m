%plots the data, and shows the result of the classifier (on both train and
%test data)
%param:
%train_data : the data used to build the model
%test_data  : the data on which to test the model
%classifier : a handle to a classifier (must take into parameter the train
%data_name       : name of the data (for the plot)
%classifier_name : name of the classifier (for the plot)
%and test data, and give as a result the classes and parameters for the bondary)
function classify(train_data,test_data,classifier,data_name,classifier_name)

tag=['Data ',data_name,' with classifier ',classifier_name];
fprintf(1,tag)

f=figure('Name',tag,'NumberTitle','off');

i=train_data(:,end)==0;
plot(train_data(i,1),train_data(i,2),'.','color','red','MarkerSize',10)
hold on
plot(train_data(i==0,1),train_data(i==0,2),'.','color','blue','MarkerSize',10)

i=test_data(:,end)==0;
plot(test_data(i,1),test_data(i,2),'.','color','yellow','MarkerSize',10)
plot(test_data(i==0,1),test_data(i==0,2),'.','color','green','MarkerSize',10)


fprintf(1,'\nTest de performance sur les données test :\n')
[classes,beta0,beta,quadratic] = classifier(train_data,test_data(:,1:2));
success_rate = performance(test_data,classes);


%tracer la frontière
v=axis;
mx=min([train_data(:,1);test_data(:,1)]);
Mx=max([train_data(:,1);test_data(:,1)]);
my=min([train_data(:,2);test_data(:,2)]);
My=max([train_data(:,2);test_data(:,2)]);

if (quadratic==0)
    poly=[-beta(1)/beta(2),-beta0/beta(2)];
    x=linspace(mx,Mx,10)';
    y=polyval(poly,x);
    i=((y>my).*(y<My)>0);
    plot(x(i),y(i),'color','black','linewidth',2)
    hold off
else
    PlotConic(quadratic, beta, beta0, 1000, 'color','black','linewidth',2);  
    axis(v)
    hold off
end


xlabel('x')
ylabel('y')
title(tag,'FontSize',12)
legend('classe 0 - train','classe 1 - train','classe 0 - test','classe 1 - test','boundary')
text(mx,my,['Success rate: ',num2str(success_rate,3),'%'])
saveas(f,[data_name,'-',classifier_name,'.eps'], 'psc2')
fprintf(1,'\n')

end

%shows the performances

function success_rate = performance(data,classes)
c = 100*confusion(data(:,3),classes)/length(classes);
success_rate = c(1,1)+c(2,2);
fprintf(1,'%i éléments classifiés avec un taux de succés : %2.1f.\n',length(classes),success_rate)
fprintf(1,'Matrice de confusion (en pour cent) :\n')
fprintf(1,'   datahat :    0      1\n')
fprintf(1,'   data :  O    %3.1f   %3.1f\n',c(1,1),c(1,2))
fprintf(1,'           1    %3.1f   %3.1f\n',c(2,1),c(2,2))



end

%computes the confusion matrix
function confusionmatrix = confusion(data,datahat)
confusionmatrix=zeros(2,2);
confusionmatrix(1,1)=sum((data==0).*(datahat==0));
confusionmatrix(2,2)=sum((data==1).*(datahat==1));
confusionmatrix(1,2)=sum((data==0).*(datahat==1));
confusionmatrix(2,1)=sum((data==1).*(datahat==0));
end