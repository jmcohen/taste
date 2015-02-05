function [model,OtherOutput]=classificationTrain(trainSet,trainClass,method,option)
% Learn a classification model specified by "method"
% Usage:
% [model,OtherOutput]=classificationTrain(trainSet,trainClass,method)
% [model,OtherOutput]=classificationTrain(trainSet,trainClass,method,option)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% method: string, designate which classifier to use. It could be
%     'bayesian': Bayesian classifier;
%     'knn': K-NN classifier;
%     'svmLi': my implementation of SVM classifier;
%     'lsvm': local SVM classifier;
%     'hdlm': high dimensional linear machine;
%     'lrc': linear regression classifier;
%     'nnls': the NNLS classifier;
%     'mnnls': metasample NNLS classifier;
%     'nmf': the NMF classifier;
%     'rnmf': the RNMF classifier;
%     'logistic': logistic regression;
% option: struct: the options to configue specific classifier:
% option.normalization: scalar, 0: no normaization (default), 1: normalize
% the data under each feature to have mean 1 and std 1;
% for the rest fields of option, please see the corresponding classifier.
% if method =='knn', option.k: the number of neares neighbors. The default is 1.
% if method=='bayesian',  option.type: string, specify the type of discriminant function. It could be 'linear','diaglinear','quadratic','diagquadratic','mahalanobis'. Type "help classify" for more information.
% model: struct, the specific model learned.
% OtherOutput: cell or row vector of cell.
%     If method=
%     'bayesian': OtherOutput{1}=tElapsed; OtherOutput{2}=trainError; OtherOutput{3}=testPosterior;
%     'knn': OtherOutput{1}=tElapsed;
%     'svmLi': OtherOutput{1}=tElapsed;
%     'lsvm': OtherOutput{1}=tElapsed;
%     'hdlm': OtherOutput{1}=tElapsed; OtherOutput{2}=beta;
%     'lrc': OtherOutput{1}=tElapsed;
%     'nnls': OtherOutput{1}=tElapsed; 
%     'mnnls': OtherOutput{1}=tElapsed;
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


if nargin<4
    option=[];
end
OtherOutput=[];
optionDefault.normalization=0;
optionDefault.normMethod='mean0std1';
optionDefault.ifDownSample=false;
optionDefault.p=16; % downsample rate
optionDefault.ifMissValueImpute=true;
optionDefault.imputeMethod='knn';
optionDefault.search=false;
option=mergeOption(option,optionDefault);

% downsample face images
if isfield(option,'ifDownSample')&& option.ifDownSample
    trainSet=downsample(trainSet,option.p);
end
% if tensor data
option.ifTensor=false;
if size(trainSet,3)>1
    option.ifTensor=true;
    option.numR=size(trainSet,1);
    option.numC=size(trainSet,2);
    trainSet=matrizicing(trainSet,3);
    trainSet=trainSet';
end

% handle missing value, imputation
if option.ifMissValueImpute
    tfTrain=isnan(trainSet);
    ifMissValTrainSet=any(any(tfTrain));
    if ifMissValTrainSet
        switch option.imputeMethod
            case 'knn'
                trainSet=knnimpute(trainSet);
                model.trainSet=trainSet;
            case 'zero'
                trainSet(tfTrain)=0;
        end
    end
end

% normalization
if logical(option.normalization)
    switch option.normMethod
        case 'mean0std1'
            [trainSet,trainSetMean,trainSetSTD]=normmean0std1(trainSet');
            trainSet=trainSet';
            option.trainSetMean=trainSetMean;
            option.trainSetSTD=trainSetSTD;
        case 'unitl2norm'
            trainSet=normc(trainSet);
    end
end

% classification
switch method
    case 'knn'
        optionDefault.k=1;
        option=mergeOption(option,optionDefault);
        tic;
        model.method='knn';
        model.trainOpt=option;
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'svmLi'
        tic;
        % model selection
        if option.search
            optGS.classifier='svmLi';
            [alphaOptimal,betaOptimal,accMax]=gridSearchUniverse(trainSet,trainClass,optGS);
            option.C=alphaOptimal;
            option.param=betaOptimal;
        end
        model=softSVMTrain2(trainSet,trainClass,option);
       model.trainOpt=option;
       model.method='svmLi';
        tElapsed=toc;
%         perSVM=perform(testClassPredicted,testClass,model.numCl);
        OtherOutput{1}=tElapsed;
    case 'lsvm' % local svm
        tic;
        model.method='lsvm';
        model.trainOpt=option;
        model.trainSet=trainSet;
        model.trainClass=trainClass;
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
        if isfield(option,'kernel') && strcmp(option.kernel,'ds')
            rank=1;
            lambda=5;
            option.param=[numR;numC;rank;lambda];
        end
        % model selection
        if option.search
            optGS.classifier='nnls';
            optGS.alphaRange=0:0;
            optGS.betaRange=3:8;
            [alphaOptimal,betaOptimal,accMax]=gridSearchUniverse(trainSet,trainClass,optGS);
        option.param=betaOptimal;
        end
        model.trainOpt=option;
        model.method='nnls';
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'nmf'
        optionDefault.facts=numel(unique(trainClass));
        option=mergeOption(option,optionDefault);
        tic;
        model.trainOpt=option;
        model.method='nmf';
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'rnmf'
        optionDefault.facts=7;%numel(unique(trainClass));
        option=mergeOption(option,optionDefault);
        tic;
        model.trainOpt=option;
        model.method='rnmf';
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'mnnls'
        tic;
        optionDefault.ifModelSelection=false;
        optionDefault.metaSampleMethod='nmf';
        option=mergeOption(option,optionDefault);
        [trainSet,trainClass]=computeMetaSample(trainSet,trainClass,option);
        model.trainOpt=option;
        model.method='mnnls';
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'lrc'
        tic;
        model.trainOpt=option;
        model.method='lrc';
        model.trainSet=trainSet;
        model.trainClass=trainClass;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'logistic'
        tic;
        optionDefault=[];
        option=mergeOption(option,optionDefault);
        model=logisticRegressTrain(trainSet,trainClass,option);
        model.trainOpt=option;
        model.method='logistic';
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    otherwise
        error('Please a correct method parameter for a classifier.');
end
end