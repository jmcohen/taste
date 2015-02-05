function [testClassPredicted,sparsity,Y,otherOutput]=nnlsClassifier(trainSet,trainClass,testSet,testClass,option)
% NNLS Classifier: testSet=trainSet*Y, s.t. Y>=0.
% Usage:
% [testClassPredicted,sparsity]=nnlsClassifier(trainSet,trainClass,[],testClass)
% [testClassPredicted,sparsity]=nnlsClassifier(trainSet,trainClass,testSet,testClass)
% [testClassPredicted,sparsity]=nnlsClassifier(trainSet,trainClass,testSet,testClass,option)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set. 
% testClass: column vector of numbers or string, the class labels of the
% test/unknown set. It is actually unused in this function, thus, set it [].
% option: struct, the options to configue this function:
% option.SRmethod, string, the optimization algorithm used to solve the NNLS problem. It could be
%     'nnlsAS': use the active-set NNLS algorithm (default);
%     'l1nnlsAS': use the active-set l1NNLS algorithm.
% option.predicter: the method to find the class label of a test sample according to Y. It could be
%     'nn': the same class label with the training sample with the maximum coefficient;
%     'knn': select k training samples with the k largest coefficients, and decide the class labels by majority voting (default).
%     'subspace': nearest subspace rule.
% option.k: scalar, only for option.predicter='kvote'. The default is 1.
% option.kernel, string, specifies the kernel. can be 'linear'(default),'polynomial','rbf','sigmoid','ds'
% option.param, scalar or column vector, the parameters for kernels, the default is [].
% testClassPredicted: column vector, the predicted class labels of the test/unknown samples.
% sparsity: scalar, the sparsity of the coefficient matrix Y.
% Y, matrix, the coefficient vectors.
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 23, 2011

% if tensor data
if size(trainSet,3)>1
    trainSet=matrizicing(trainSet,3);
    testSet=matrizicing(testSet,3);
    trainSet=trainSet';
    testSet=testSet';
end

optionDefault.SRmethod='nnlsAS';
optionDefault.predicter='knn';
optionDefault.knn=numel(trainClass);
optionDefault.kernel='linear';
optionDefault.param=[];
% optionDefault.search=false;
optionDefault.sparsityThreshold=1e-4;
if nargin<5
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% % normalization to length 1
% trainSet=normc(trainSet);
% testSet=normc(testSet);

AtA=computeKernelMatrix(trainSet,trainSet,option);
AtB=computeKernelMatrix(trainSet,testSet,option);
BtB=diag(computeKernelMatrix(testSet,testSet,option));

if strcmp(option.SRmethod,'nnlsAS')
   option.lambda=0;
end
Y=l1NNQPActiveSet(AtA,-AtB,option.lambda);

% Y=KSRSC(AtA,AtB,BtB,option);
% switch option.method
%     case 'nnls'
%         Y=kfcnnls(trainSet,testSet,option);
%     case 'sparsennls'
%         optionDefault.beta=0.1;
%         optionDefault.eta=max(max(trainSet))^2;
%         option=mergeOption(option,optionDefault);
%         outTrain.factors{1}=trainSet;
%         outTrain.option=option;
%         Y=sparsenmfnnlstest(testSet,outTrain);
%     case 'seminmfupdaterule'
% %         optionDefault.iter=1000;
% %         option=mergeOption(option,optionDefault);
%         option.dis=0;
%         outTrain.factors{1}=trainSet;
%         outTrain.option=option;
%         Y=kernelseminmfruletest(testSet,outTrain);
% end

% compute sparsity
sparsity=sum(sum(Y<=option.sparsityThreshold))/(size(Y,1)*size(Y,2));
otherOutput=[];
switch option.predicter
    case  'max'
        [val,ind]=max(Y,[],1);
        testClassPredicted=trainClass(ind);
    case 'knn'
        testClassPredicted=knnrule(Y,trainClass,option.knn);
    case 'subspace'
        [testClassPredicted,residuals]=subspace(Y,testSet,trainSet,trainClass);
        otherOutput=residuals;
end
end

