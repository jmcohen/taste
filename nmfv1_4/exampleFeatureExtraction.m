% This is an example of how to extract features using NMF
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
% % suppose the current folder is the one containing the NMF toolbox
load('.\data\ALLAML.mat','classes012','D');
classes=classes012;
clear('classes012');
kfold=4;
ind=crossvalind('Kfold',classes,kfold);
indTest=(ind==1);
trainSet=D(:,~indTest);
testSet=D(:,indTest);
trainClass=classes(~indTest);
testClass=classes(indTest);
% feature extraction
feMethod='nmf';
optionFE.facts=3;
[trainExtr,outTrain]=featureExtractionTrain(trainSet,[],trainClass,feMethod,optionFE);
[testExtr,outTest]=featureExtrationTest(trainSet,testSet,outTrain);

trainSet=trainExtr;
testSet=testExtr;
% classification
disp('KNN classifier:');
tic;
testClassPredicted=knnclassify(testSet',trainSet',trainClass,1);
tElapsedKNN=toc;
% classPerformKNN includes the accuracies of class 0, class 1, class 2, and total accuracy, and balanced accuracy
classPerformKNN=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));

disp('SVM classifier:');
% normalization
[trainSet,trainSetMean,trainSetSTD]=normmean0std1(trainSet');
trainSet=trainSet';
testSet=normmean0std1(testSet',trainSetMean,trainSetSTD);
testSet=testSet';
tic;
optionSVM.C=2^0;
optionSVM.kernel='linear';
model=softSVMTrain2(trainSet,trainClass,optionSVM);
[testClassPredicted,vals]=softSVMPredict2(model,testSet);
tElapsedSVM=toc;
% classPerformSVM includes the accuracies of class 0, class 1, class 2, and total accuracy, and balanced accuracy
classPerformSVM=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])))







