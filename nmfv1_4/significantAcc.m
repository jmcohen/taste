function [en,tn,en25,en75,significant]=significantAcc(D,classes,numTrain,classifier,optCl,option) 
% determine if the accuracy is significant
% D: matrix, whole data set, feature X sample.
% classes: vector, the class labels: 0,1,2,...
% numTrain: number of training samples to be seperated from D
% classifier: string, can be 'nnls' (default), 'svm', and others, please see function classificationTrain
% optCl: the option of classifier, see function classificationTrain
% option: struct, including fields: T1 (number of repetitions, default is 50), T2 (the
% number of permutation, default is 50), and alpha (significant level, default is 0.05)
% en: mean error rate
% tn: p value
% en25: error rate at 25th quantile
% en75: error rate at 75th quantile
% significant: logical, if the accuracy is significant
%
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
% Nov. 29, 2012

if nargin<6
    option=[];
end
if nargin<5
    optCl=[];
end
if nargin<4
    classifier='nnls';
end

% suppose the default classifer is NNLS
optClDefault.normalization=1;
optClDefault.normMethod='unitl2norm';
optClDefault.method='nnls';%'seminmfupdaterule';
optClDefault.predicter='subspace';
optClDefault.kernel='linear';
optClDefault.param=2^(0);
optClDefault.search=false;
optClDefault.ifMissValueImpute=false;
optCl=mergeOption(optCl,optClDefault);

optionDefault.T1=50;
optionDefault.T2=50;
optionDefault.alpha=0.05;

option=mergeOption(option,optionDefault);
numD=size(D,2);
numCl=numel(unique(classes));
if numTrain>=numD
    error('Number of training samples is larger than the number of whole samples!');
end

% get average accuracy
ens=zeros(option.T1,1);
errorRatesRand=zeros(option.T1,option.T2);
for i=1:option.T1
    i
%     [testInd,trainInd] = crossvalind('LeaveMOut', numD, numTrain);
    trainInd=leaveMOut(numTrain,classes);
    testInd=~trainInd;
    trainSet=D(:,trainInd);
    testSet=D(:,testInd);
    trainClass=classes(trainInd);
    testClass=classes(testInd);
    
    [model,OtherOutputTr]=classificationTrain(trainSet,trainClass,classifier,optCl);
    % predict
    [testClassPredicted,OtherOutput]=classificationPredict(model,testSet,testClass);
    % calculate the performance
    [classPerform,conMat]=perform(testClassPredicted,testClass,numCl);
    ens(i)=1-classPerform(numCl+1);
    % permutation test
    fprintf('Starting Permutation Test...\n');
    for j=1:option.T2
        numInd=randperm(sum(trainInd));
        trainClassRand=trainClass(numInd);
        [model,OtherOutputTr]=classificationTrain(trainSet,trainClassRand,classifier,optCl);
        % predict
        [testClassPredicted,OtherOutput]=classificationPredict(model,testSet,testClass);
        % calculate the performance
        [classPerformRand,conMatRand]=perform(testClassPredicted,testClass,numCl);
        errorRatesRand(i,j)=1-classPerformRand(numCl+1);
    end
end
en=mean(ens);
tn=(sum(sum(errorRatesRand<en)))/(option.T1*option.T2);

% determine if significant
if tn<optionDefault.alpha
   significant=true;
else
    significant=false;
end
ens=sort(ens);
en25=ens(round(option.T1/4));
en75=ens(option.T1 - round(option.T1/4));
end