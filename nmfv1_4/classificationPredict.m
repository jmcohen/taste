function [testClassPredicted,OtherOutput]=classificationPredict(model,testSet,testClass)
% predict the test samples by the model learned by "classificationTrain"
% Usage:
% [testClassPredicted,OtherOutput]=classificationPredict(model,testSet,testClass)
% testSet: matrix, the test set.
% testClass: column vector of numbers or string, the class labels of the
%       test/unknown set. It is actually unused in this function, thus, set it [].
% testClassPredicted: column vector, the predicted class labels of the test/unknown samples.
% OtherOutput: cell or row vector of cell.
%     If method=
%     'bayesian': OtherOutput{1}=tElapsed; OtherOutput{2}=trainError; OtherOutput{3}=testPosterior;
%     'knn': OtherOutput{1}=tElapsed;
%     'svmLi': OtherOutput{1}=tElapsed;
%     'lsvm': OtherOutput{1}=tElapsed;
%     'hdlm': OtherOutput{1}=tElapsed; OtherOutput{2}=beta;
%     'lrc': OtherOutput{1}=tElapsed;
%     'nnls': OtherOutput{1}=tElapsed; OtherOutput{2}=sparsity;
%     'mnnls': OtherOutput{1}=tElapsed; OtherOutput{2}=sparsity;
%     'nmf': OtherOutput{1}=tElapsed;
%     'rnmf': OtherOutput{1}=tElapsed;
%     'logistic': OtherOutput{1}=tElapsed;
%%%%
% Copyright (C) <2012>  <Yifeng Li>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 18, 2011
%%%%


option=model.trainOpt;

% downsample face images
if isfield(option,'ifDownSample')&& option.ifDownSample
    testSet=downsample(testSet,option.p);
end
% if tensor data
if option.ifTensor % only 3-order
    option.ifTensor=true;
    trainSet=matrizicing(testSet,3);
    testSet=testSet';
end

% handle missing value, imputation
if option.ifMissValueImpute
    tfTest=isnan(testSet);
    ifMissValTestSet=any(any(tfTest));
    if ifMissValTestSet
        switch option.imputeMethod
            case 'knn'
                numTrain=size(model.trainSet,2);
                testSet=knnimpute([model.trainSet,testSet]);
                testSet=testSet(:,numTrain+1:end);
            case 'zero'
                testSet(tfTest)=0;
        end
    end
end

% normalization
if logical(option.normalization)
    switch option.normMethod
        case 'mean0std1'
            testSet=normmean0std1(testSet',option.trainSetMean,option.trainSetSTD);
            testSet=testSet';
        case 'unitl2norm'
            testSet=normc(testSet);
    end
end

% classification
switch model.method
    case 'knn'
        tic;
        testClassPredicted=knnclassify(testSet',model.trainSet',model.trainClass,option.k);%KNN Classification
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'svmLi'
        tic;
        [testClassPredicted,vals]=softSVMPredict2(model,testSet);
        tElapsed=toc;
%         perSVM=perform(testClassPredicted,testClass,model.numCl);
        OtherOutput{1}=tElapsed;
    case 'lsvm' % local svm
        tic;
        testClassPredicted=localSVM(model.trainSet,model.trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
%     case 'bayesian'
%         optionDefault.type='diaglinear'; % 'linear','diaglinear','quadratic','diagquadratic','mahalanobis'
%         option=mergeOption(option,optionDefault);
%         tic;
%         [testClassPredicted,trainError,testPosterior]=classify(testSet',trainSet',trainClass,option.type);
%         tElapsed=toc;
%         OtherOutput{1}=tElapsed;
%         OtherOutput{2}=trainError;
%         OtherOutput{3}=testPosterior;
    case 'nnls'
        tic;
        [testClassPredicted,sparsity]=nnlsClassifier(model.trainSet,model.trainClass,testSet,[],option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
        OtherOutput{2}=sparsity;
    case 'nmf'
        tic;
        [testClassPredicted]=nmfClassifier(model.trainSet,model.trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'rnmf'
        tic;
        [testClassPredicted]=repetitivenmfClassifier(model.trainSet,model.trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'mnnls'
        tic;
        [testClassPredicted,sparsity]=nnlsClassifier(model.trainSet,model.trainClass,testSet,testClass,option);
%         methodCl='svm'; % SVM does not work
%         optionsvm.normalization=0;
%         optionsvm.trainSetting='-t 2 -c 1 -b 1';
%         optionsvm.testSetting='-b 1';
%         optionsvm.search=false;
%         [testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,methodCl,optionsvm);
%         sparsity=[];
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
        OtherOutput{2}=sparsity;
    case 'lrc'
        tic;
        [testClassPredicted]=lrc(model.trainSet,model.trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'logistic'
        tic;
        testClassPredicted=logisticRegressPredict(model,testSet);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    otherwise
        error('Please a correct method parameter for a classifier.');
end
end