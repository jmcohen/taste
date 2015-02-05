function testClassPredicted=bootstrapnnlsClassifier(trainSet,trainClass,testSet,testClass,option)
% Bootstrap NNLS Classifier: testSet=trainSet*Y, s.t. Y>=0.
% Usage:
% [testClassPredicted,sparsity]=bootstrapnnlsClassifier(trainSet,trainClass,[],testClass)
% [testClassPredicted,sparsity]=bootstrapnnlsClassifier(trainSet,trainClass,testSet,testClass)
% [testClassPredicted,sparsity]=bootstrapnnlsClassifier(trainSet,trainClass,testSet,testClass,option)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set.
% testClass: column vector of numbers or string, the class labels of the
% test/unknown set. It is actually unused in this function, thus, set it [].
% option: struct, the options to configue this function:
% option.method, string, the optimization algorithm used to solve the NNLS problem. It could be
%     'nnls': used the NNLS algorithm (default);
%     'seminmfupdaterule': use the update rules based algorithm;
%     'sparsennls': used NNLS algorithm with sparse constraint.
% option.predicter: the method to find the class label of a test sample according to Y. It could be
%     'max': the same class label with the training sample with the maximum coefficient (default);
%     'kvote': select k training samples with the k largest coefficients, and decide the class labels by majority voting.
% option.k: scalar, only for option.predicter='kvote'. The default is 1.
% option.numRandom, scalar, the times to use bootstrapping. The default is 99.
% testClassPredicted: column vector, the predicted class labels of the test/unknown samples.
% sparsity: scalar, the sparsity of the coefficient matrix Y.
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 23, 2011



if nargin<5
   option=[]; 
end
optionDefault.method='nnls';
optionDefault.predicter='max';
optionDefault.kernel='linear';
optionDefault.param=[];
optionDefault.kernelParamRandomAssign=false;
optionDefault.k=1;
optionDefault.numRandom=99;
option=mergeOption(option,optionDefault);

trainSetOrigin=trainSet;
trainClassOrigin=trainClass;
testSetOrigin=testSet;

if size(trainSetOrigin,3)>1 % tensor
    trainSetOrigin=matrizicing(trainSetOrigin,3);
    testSet=matrizicing(testSetOrigin,3);
    trainSetOrigin=trainSetOrigin';
    testSet=testSet';
end
testClassPredicteds=nan(numel(testClass),option.numRandom);
for r=1:option.numRandom
    numTrain=size(trainClassOrigin,1);
    trainId=randi([1,numTrain],[numTrain,1]);
    trainId=unique(trainId);
    % get data
    %     if size(trainSetOrigin,3)>1 % tensor
    %         trainSet=trainSetOrigin(:,:,trainId);
    %         trainClass=trainClassOrigin(trainId);
    %         testSet=testSetOrigin;
    %     else
    %         trainSet=trainSetOrigin(:,trainId);
    %         trainClass=trainClassOrigin(trainId);
    %     end
    trainSet=trainSetOrigin(:,trainId);
    trainClass=trainClassOrigin(trainId);
    % preprocess
    %     if size(trainSet,3)>1&& option.normalization % centerize and scale
    %         cent=[0 0 1];
    %         scale=[0 0 0];
    %         [trainSet,Means,Scales]=nprocess(trainSet,cent,scale);
    %         testSet=nprocess(testSet,cent,scale,Means,Scales);
    %     end
    %     % matricize
    %     if size(trainSetOrigin,3)>1 % tensor
    %         trainSet=matrizicing(trainSet,3);
    %         testSet=matrizicing(testSet,3);
    %         trainSet=trainSet';
    %         testSet=testSet';
    %     end
    if strcmp(option.kernel,'rbf') && option.kernelParamRandomAssign
        option.param=2^(randi([0,8]));
    end
    if strcmp(option.kernel,'ds')
            rank=(randi([1]));
            lambda=(randi([4,6]));
           option.param([3;4])=[rank;lambda]; 
    end
    testClassPredicteds(:,r)=nnlsClassifier(trainSet,trainClass,testSet,testClass,option);
    
    % switch option.method
    %     case 'nnls'
    %         Y=fcnnls(trainSet,testSet);
    %     case 'sparsennls'
    %         optionDefault.beta=0.01;
    %         option=mergeOption(option,optionDefault);
    %         Y=sparseNMFtest(testSet,trainSet,option.beta);
    %     case 'seminmfupdaterule'
    %         optionDefault.iter=3000;
    %         option=mergeOption(option,optionDefault);
    %         Y=semiNMFTest(testSet,trainSet,option.iter);
    % end
    %
    % switch option.predicter
    %     case  'max'
    %         [val,ind]=max(Y,[],1);
    %         testClassPredicteds(:,r)=trainClass(ind);
    %     case 'kvote'
    %         for s=1:size(Y,2)
    %             [sortedCoeff,ind] = getBestScores(Y(:,s),option.k);
    %             predicted(s,:)=trainClass(ind);
    %         end
    %         testClassPredicteds(:,r)=vote(predicted);
    % end
end
testClassPredicted=vote(testClassPredicteds);
end