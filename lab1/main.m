classifAtest=load('classificationA.test');
classifAtrain=load('classificationA.train');
classifBtest=load('classificationB.test');
classifBtrain=load('classificationB.train');
classifCtest=load('classificationC.test');
classifCtrain=load('classificationC.train');



classify(classifAtrain,classifAtest,@lda_classifier,'A','LDA')
classify(classifAtrain,classifAtest,@lin_classifier,'A','Linear regression')
classify(classifAtrain,classifAtest,@logis_classifier,'A','Logisitic regression')
classify(classifAtrain,classifAtest,@qda_classifier,'A','QDA')

classify(classifBtrain,classifBtest,@lda_classifier,'B','LDA')
classify(classifBtrain,classifBtest,@lin_classifier,'B','Linear regression')
classify(classifBtrain,classifBtest,@logis_classifier,'B','Logisitic regression')
classify(classifBtrain,classifBtest,@qda_classifier,'B','QDA')

classify(classifCtrain,classifCtest,@lda_classifier,'C','LDA')
classify(classifCtrain,classifCtest,@lin_classifier,'C','Linear regression')
classify(classifCtrain,classifCtest,@logis_classifier,'C','Logisitic regression')
classify(classifCtrain,classifCtest,@qda_classifier,'C','QDA')







