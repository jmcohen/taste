function [testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option)
% Classification using NNLS, BNNLS, NMF, RNMF, KNN, SVM, Bayesian, Kernel-FDA (for future extension), or Optimal-Kernel-FDA (for future extension) classifiers.
% Usage:
% [testClassPredicted,classPerform,OtherOutput]=classification(trainSet,trainClass,[],testClass)
% [testClassPredicted,sparsity]=classification(trainSet,trainClass,testSet,testClass)
% [testClassPredicted,sparsity]=classification(trainSet,trainClass,testSet,testClass,option)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set.
% testClass: column vector of numbers or string, the class labels of the
%       test/unknown set. It is actually unused in this function, thus, set it [].
% method: string, designate which classifier to use. It could be
%     'nnls': the NNLS classifier;
%     'bootstrapnnls': the Bootstrap based NNLS classifier;
%     'nmf': the NMF classifier;
%     'rnmf': the RNMF classifier;
%     'knn': the KNN classifier;
%     'svm': the SVM classifier;
%     'bayesian': the Bayesian classifier.
%%%     'kernelfda': (for future extension) the Kernel Fisher Discriminative Analysis classifier;
%%%    'psdkernelfda': (for future extension) the Optimal Kernel Fisher Discriminative Analysis classifier;
% option: struct: the options to configue specific classifier:
%     If method=
%     'nnls':  
%            option.normalization, scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            The rest fields of option is the same as the input option for "nnlsClassifier" function. Type "help nnlsClassifier" for more information.
%     'bootstrapnnls': 
%            option.normalization, scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            The rest fields of option is the same as the input option for "bootstrapnnlsClassifier" function. Type "help bootstrapnnlsClassifier" for more information.
%     'nmf': 
%            option.normalization, scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            The rest fields of option is the same as the input option for "nmfClassifier" function. Type "help nmfClassifier" for more information.
%     'rnmf': 
%            option.normalization, scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            The rest fields of option is the same as the input option for "repetitivenmfClassifier" function. Type "help repetitivenmfClassifier" for more information.
%     'knn': 
%            option.normalization: scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            option.k: the number of neares neighbors. The default is 1.
%     'svm': 
%            option.normalization: scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            option.trainSetting: string, the options for the "svmtrain" function. The default is '-s 0 -t 2 -b 1'. See the LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm;
%            option.testSetting: string, the options for the "svmpredict" function. The default is '-b 1'. See the LIBSVM: http://www.csie.ntu.edu.tw/~cjlin/libsvm.
%     'bayesian': 
%            option.normalization: scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            option.type: string, specify the type of discriminant function. 
%                 It could be 'linear','diaglinear','quadratic','diagquadratic','mahalanobis'. Type "help classify" for more information.
%%%     'kernelfda': (for future extension)
%            option.normalization: scalar, 0: no normaization (default), 1: normalize the data under each feature to have mean 1 and std 1;
%            option.kernel: string, the kernel function. Type "kfda" for more information. It could be
%                  'linear':  option.param=[];
%                  'polynomial': option.param is [Gamma;Coefficient;Degree], the default is [1;0;2];
%                  'rbf': option.param is sigma, the default is 1/#features;
%                  'sigmoid': option.param is [alpha;beta], the default is [1;0];
%                  you own kernel function name.
%%%     'psdkernelfda': (for future extension) the Optimal Kernel Fisher Discriminative Analysis classifier;
% testClassPredicted: column vector, the predicted class labels of the test/unknown samples.
% classPerform: row vector, the classification performance. 
%     If binary-class problem, classPerform includes PPV, NPV, Specificity, Sensitivity, Accuracy, and Balanced Accuracy.
%     If multi-class problem, classPerform includes accuracy for class 1, accuracy of class 2, ..., accuracy, balanced accuracy.
% OtherOutput: cell or row vector of cell. 
%     If method=
%     'nnls': OtherOutput{1}=tElapsed; OtherOutput{2}=sparsity;
%     'bootstrapnnls': OtherOutput{1}=tElapsed;
%     'nmf': OtherOutput{1}=tElapsed;
%     'rnmf': OtherOutput{1}=tElapsed;
%     'knn': OtherOutput{1}=tElapsed;
%     'svm': OtherOutput{1}=tElapsed; OtherOutput{2}= prob_estimates;
%     'bayesian': OtherOutput{1}=tElapsed; OtherOutput{2}=trainError; OtherOutput{3}=testPosterior;
% Contact Information:
% Yifeng Li
% University of Windsor
% li11112c@uwindsor.ca; yifeng.li.cn@gmail.com
% May 18, 2011

if nargin<6
   option=[];
end
OtherOutput=[];
optionsDefault.normalization=0;
option=mergeOption(option,optionsDefault);
% normalization
if logical(option.normalization)
    [trainSet,trainSetMean,trainSetSTD]=normmean0std1(trainSet');
    trainSet=trainSet';
    testSet=normmean0std1(testSet',trainSetMean,trainSetSTD);
    testSet=testSet';
end

% classification
switch method
    case 'knn'
        optionDefault.k=1;
        option=mergeOption(option,optionDefault);
        tic;
        testClassPredicted=knnclassify(testSet',trainSet',trainClass,option.k);%KNN Classification
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'svm' % libsvm matlab toolbox is required
        optionDefault.trainSetting='-s 0 -t 2 -b 1';
        optionDefault.testSetting='-b 1';
        option=mergeOption(option,optionDefault);
        tic;
        model=svmtrain(trainClass,trainSet',option.trainSetting);
        [predicted_label, accuracy, prob_estimates] = svmpredict(testClass, testSet', model,option.testSetting);
        testClassPredicted=predicted_label;
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
        OtherOutput{2}= prob_estimates;
%     case 'kernelfda'
%         %kernel FDA
%         optionDefault.kernel='rbf';
%         optionDefault.kernelParameterValue=1/size(trainSet,1);
%         option=mergeOption(option,optionDefault);
%         tic;
%         testClassPredicted=kfda(trainSet, testSet, trainClass, option.kernel,option.kernelParameterValue);
%         tElapsed=toc;
%         OtherOutput{1}=tElapsed;
%     case 'psdkernelfda'
%         optionDefault.numKernel=10;
%         optionDefault.lambda=10.^(-8);
%         log10Sigma=rand(optionDefault.numKernel,1)*3-1;
%         optionDefault.sigma=10.^(log10Sigma);
%         option=mergeOption(option,optionDefault);
%         tic;
%         [testClassPredicted,testValues,Theta]=OKernelFDA(trainSet,testSet,trainClass,option.lambda,option.sigma,option.numKernel,[]);
%         tElapsed=toc;
%         OtherOutput{1}=tElapsed;
%          OtherOutput{2}=Theta;
    case 'bayesian'
        optionDefault.type='linear'; % 'linear','diaglinear','quadratic','diagquadratic','mahalanobis'
        option=mergeOption(option,optionDefault);
        tic;
        [testClassPredicted,trainError,testPosterior]=classify(testSet',trainSet',trainClass,option.type);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
        OtherOutput{2}=trainError;
        OtherOutput{3}=testPosterior;
    case 'nnls'
        optionDefault.method='nnls';%seminmfupdaterule
        optionDefault.predicter='max';
        optionDefault.k=1;
        option=mergeOption(option,optionDefault);
        tic;
        [testClassPredicted,sparsity]=nnlsClassifier(trainSet,trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
        OtherOutput{2}=sparsity;
   case 'bootstrapnnls'
%        if size(trainSet,3)>1&& option.normalization
%             cent=[0 0 1];
%             scale=[0 0 0];
%             [trainSet,Means,Scales]=nprocess(trainSet,cent,scale);
%             testSet=nprocess(testSet,cent,scale,Means,Scales,1);
%         end
        optionDefault.method='nnls';%
        optionDefault.predicter='max';
        optionDefault.k=1;
        optionDefault.numRandom=99;
        option=mergeOption(option,optionDefault);
        tic;
        testClassPredicted=bootstrapnnlsClassifier(trainSet,trainClass,testSet,testClass,option);
        tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'nmf'
        optionDefault.facts=numel(unique([trainClass;testClass]));
        option=mergeOption(option,optionDefault);
        tic;
        [testClassPredicted]=nmfClassifier(trainSet,trainClass,testSet,testClass,option);
         tElapsed=toc;
        OtherOutput{1}=tElapsed;
    case 'rnmf'
        optionDefault.facts=7;%numel(unique([trainClass;testClass]));
        option=mergeOption(option,optionDefault);
        tic;
        [testClassPredicted]=repetitivenmfClassifier(trainSet,trainClass,testSet,testClass,option);
         tElapsed=toc;
        OtherOutput{1}=tElapsed;
    otherwise
        error('Please a correct method parameter for a classifier.');   
end
% calculate the performance
classPerform=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));
end