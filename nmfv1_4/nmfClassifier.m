function [testClassPredicted]=nmfClassifier(trainSet,trainClass,testSet,testClass,option)
% NMF classifier.
% Usage:
% [testClassPredicted]=nmfClassifier(trainSet,trainClass,[],testClass)
% [testClassPredicted]=nmfClassifier(trainSet,trainClass,testSet,testClass)
% [testClassPredicted]=nmfClassifier(trainSet,trainClass,testSet,testClass,option)
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set. Can be [].
% testClass: column vector of numbers or string, the class labels of the
% test/unknown set. It is actually unused in this function, thus, set it [].
% option: struct, the options to configue this function:
% option.facts: scalar, the number of clusters. The default is the number of classes;
% option.Y: matrix, the coefficient matrix Y produced outside this function. The default is [];
% option.optionCl, struct, the options to configue specific classifier. 
%     The default is []. Type "help nnlsClassifier" for more information.
% testClassPredicted: column vector, the predicted class labels of the test/unknown samples.
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
% May 23, 2011
%%%%

if nargin<5
    option=[];
end
optionDefault.facts=numel(unique(trainClass));
optionDefault.Y=[];
optionDefault.optionCl.predicter='subspace';
option=mergeOption(option,optionDefault);

% if tensor data
if size(trainSet,3)>1
    trainSet=matrizicing(trainSet,3);
    testSet=matrizicing(testSet,3);
    trainSet=trainSet';
    testSet=testSet';
end
% clustering based sample selection
if isempty(option.Y)
    if any(any([trainSet,testSet]<0))
        [A,Y]=seminmfnnls([trainSet,testSet],option.facts);
    else
        [A,Y]=nmfnnls([trainSet,testSet],option.facts);
    end
    clear('A');
else
    Y=option.Y;    
end

if iscellstr(trainClass) 
   testClassPredicted=num2cellstr(nan(size(testSet,2),1));
else
    testClassPredicted=nan(size(testSet,2),1);
end
%classesTrue=[trainClass;testClass];
classes=[trainClass;testClassPredicted];
[ind,clusters,clusterClasses,order]=getClusters(Y,[trainSet,testSet],classes);
numCluster=numel(clusters);
% classify
for i=1:numCluster
    % no test samples in the cluster
    if isempty(clusters{i})||~any(isnan(clusterClasses{i}))
        continue;
    end
    curClasses=clusterClasses{i}; % class labels for all sample in the cluster
    curClassesTrue=[];%classesTrue(order{i});
    % all samples in the cluster are test samples
    if ~any(~isnan(curClasses))
        curTestClassPredicted=nnlsClassifier(trainSet,trainClass,clusters{i},curClassesTrue,option.optionCl);
        classes(order{i})=curTestClassPredicted;
        continue;
    end
    curTestSampleInd=isnan(curClasses);% logical
    curClassesUnique=unique(curClasses(~isnan(curClasses))); % unique class labels for the training samples in the cluster
    if numel(curClassesUnique)==1 % only one class in the training set of the cluster
        curTestClassPredicted=curClasses(curTestSampleInd);
        for tp=1:numel(curTestClassPredicted)
            curTestClassPredicted(tp)=curClassesUnique;
        end
    else
        %curTestClassPredicted=nnlsClassifier(clusters{i}(:,~curTestSampleInd),curClasses(~curTestSampleInd),clusters{i}(:,curTestSampleInd),curClassesTrue(curTestSampleInd),option.optionCl);
        curTestClassPredicted=nnlsClassifier(clusters{i}(:,~curTestSampleInd),curClasses(~curTestSampleInd),clusters{i}(:,curTestSampleInd),[],option.optionCl);
    end
    classes(order{i}(curTestSampleInd))=curTestClassPredicted;
end
testClassPredicted=classes(size(trainSet,2)+1:end);

end