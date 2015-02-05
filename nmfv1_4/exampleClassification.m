% This is an example of how to use the SVM, NNLS, NMF, and RNMF classifiers.
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
% Aug. 10, 2012
%%%%

clear

% load data
% suppose the current folder is the one containing the NMF toolbox
load('.\data\ALLAML.mat','classes012','D');
classes=classes012;
clear('classes012');
kfold=3;
ind=crossvalind('Kfold',classes,kfold);
indTest=(ind==1);
trainSet=D(:,~indTest);
testSet=D(:,indTest);
trainClass=classes(~indTest);
testClass=classes(indTest);

% classification
disp('NNLS classifier...');
tic;
[testClassPredicted,sparsity]=nnlsClassifier(trainSet,trainClass,testSet,testClass);
tElapsedNNLS=toc;
% classPerformNNLS includes the accuracies of class 0, class 1, class 2, and total accuracy, and balanced accuracy
classPerformNNLS=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));

disp('NMF classifier...');
optionnmf.facts=5; % number of clusters
[A,Y]=nmfnnls(D,optionnmf.facts); % clustering
optionnmf.Y=Y;
clear('A');
% reorder Y
Yreordered=[optionnmf.Y(:,~indTest),optionnmf.Y(:,indTest)];
optionnmf.Y=Yreordered;
tic;
[testClassPredicted]=nmfClassifier(trainSet,trainClass,testSet,testClass,optionnmf);
tElapsedNMF=toc;
classPerformNMF=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));

disp('RNMF classifier...');
optionrnmf.facts=5;
optionrnmf.repetitive=9;
Ys=cell(optionrnmf.repetitive,1);
for r=1:optionrnmf.repetitive
    if any(any(D<0))
        [A,Yr]=seminmfnnls(D,optionrnmf.facts);
    else
        [A,Yr]=nmfnnls(D,optionrnmf.facts);
    end
    Ys{r}=Yr;
end
optionrnmf.Ys=Ys;
clear('A');
% reorder Ys
for p=1:optionrnmf.repetitive
    Yreordered=[optionrnmf.Ys{p}(:,~indTest),optionrnmf.Ys{p}(:,indTest)];
    optionrnmf.Ys{p}=Yreordered;
end
tic;
[testClassPredicted]=repetitivenmfClassifier(trainSet,trainClass,testSet,testClass,optionrnmf);
tElapsedRNMF=toc;
classPerformRNMF=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));

disp('NNLS classifier using ''classificationTrain'' and ''classificationPredict'' functions...');
method='nnls';
optNNLS.kernel='linear';
optNNLS.param=[];
optNNLS.predicter='knn'; % can be 'max', 'knn', or 'subspace'
[model,otherOutput]=classificationTrain(trainSet,trainClass,method,optNNLS);
[testClassPredicted,otherOutput]=classificationPredict(model,testSet,testClass);
tElapsedNNLS2=otherOutput{1};
classPerformNNLS2=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));


disp('SVM classifier using ''classification'' function...');
method='svmLi';
optionSVM.normalization=1; % normalization
optionSVM.kernel='linear';
optionSVM.C=2^0;
optionSVM.param=[];
[model,otherOutput]=classificationTrain(trainSet,trainClass,method,optionSVM);
[testClassPredicted,otherOutput]=classificationPredict(model,testSet,testClass);
tElapsedSVM=otherOutput{1};
classPerformSVM=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));

disp('KNN, SVM, NNLS, NMF, and RNMF classifiers using ''multiClassifiers'' function...');
methods={'knn';'svmLi';'nnls';'nmf';'rnmf'};
optionKNN.k=1;
optionSVM.normalization=1; % normalization
optionSVM.kernel='linear'; % normalization
optionNNLS.kernel='linear';
optionNMF.Y=Y;
optionNMF.facts=5;
optionRNMF.Ys=Ys;
optionRNMF.facts=5;
options={optionKNN;optionSVM;optionNNLS;optionNMF;optionRNMF};
[testClassPredicteds,classPerforms,tElapseds,OtherOutputs]=multiClassifiers(trainSet,trainClass,testSet,testClass,methods,options);
tElapsedM=tElapseds;
classPerformM=classPerforms;

disp('KNN, SVM, NNLS, NMF, and RNMF classifiers using ''cvExperiment'' function...');
kfold=3;
rerun=5;
fsMethod='none';
fsOption=[];
feMethod='none';
feOption=[];
methods={'knn';'svmLi';'nnls';'nmf';'rnmf'};
optionKNN.k=1;
optionSVM.normalization=1; % normalization
optionSVM.kernel='linear'; % normalization
optionNNLS.kernel='linear';
optionNMF.Y=[];
optionNMF.facts=5;
optionRNMF.Ys=[];
optionRNMF.facts=5;
options={optionKNN;optionSVM;optionNNLS;optionNMF;optionRNMF};
[meanMeanPerformsAllRun,stdSTDAllRun,meanconMatAllRun,meantElapsedsAllRun]=cvExperiment(D,classes,kfold,rerun,fsMethod,fsOption,feMethod,feOption,methods,options);
tElapsedM=tElapseds;
classPerformCV=meanMeanPerformsAllRun;


fprintf('The classification performance of NNLS classifier is: \n\r');
classPerformNNLS
fprintf('The classification performance of NMF classifier is: \n\r');
classPerformNMF
fprintf('The classification performance of RNMF classifier is: \n\r');
classPerformRNMF
fprintf('The classification performance of NNLS classifier is: \n\r');
classPerformNNLS2
fprintf('The classification performance of SVM classifier is: \n\r');
classPerformSVM
fprintf('The classification performance of KNN, SVM, NNLS, NMF, and RNMF classifiers are: \n\r');
classPerformM
fprintf('The CV classification performance of KNN, SVM, NNLS, NMF, and RNMF classifiers are: \n\r');
classPerformCV
