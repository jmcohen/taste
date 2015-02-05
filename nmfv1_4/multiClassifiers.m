function [testClassPredicteds,classPerforms,conMats,tElapseds,OtherOutputs]=multiClassifiers(trainSet,trainClass,testSet,testClass,methods,options)
% trainSet, matrix, the training set with samples in columns and features in rows.
% Usuage:
% [testClassPredicteds,classPerforms,tElapseds,OtherOutputs]=multiClassifiers(trainSet,trainClass,testSet,testClass,methods)
% [testClassPredicteds,classPerforms,tElapseds,OtherOutputs]=multiClassifiers(trainSet,trainClass,testSet,testClass,methods,options)
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set.
% testClass: column vector of numbers or string, the class labels of the
%       test/unknown set. It is actually unused in this function, thus, set it [].
% methods: column vector of cell strings, designate which one or more classifier to use. It could be
%     'nnls': the NNLS classifier;
%     'bootstrapnnls': the Bootstrap based NNLS classifier;
%     'nmf': the NMF classifier;
%     'rnmf': the RNMF classifier;
%     'knn': the KNN classifier;
%     'svm': the SVM classifier;
%     'bayesian': the Bayesian classifier;
%%%     'kernelfda': (for future extension);
%%%    'psdkernelfda': (for future extension).
%      For example, methods={nnls;nmf;svm};
% options: column vector of cells. options{i} is the option for methods{i}. Type "help classification" for more information.
% testClassPredicteds: matrix, testClassPredicteds(:,i) are the predicted class labels by methods{i}.
% classPerforms: matrix, classPerforms(i,:) is the class performance by methods{i}.
% tElapseds: column vector; tElapseds(i) is the computing time of methods{i}.
% OtherOutputs: column vector of cell. OtherOutputs{i} is the other otherOutput of methods{i}.
% See also "classfication".
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

% % if tensor data
% if size(trainSet,3)>1
%     trainSet=matrizicing(trainSet,3);
%     testSet=matrizicing(testSet,3);
%     trainSet=trainSet';
%     testSet=testSet';
% end
numClasses=numel(unique([trainClass;testClass]));
numberPerfMeasure=2+numClasses;
% if numClasses<=4
%     numberPerfMeasure=6;
% else
%     numberPerfMeasure=2+numClasses;
% end
testClassPredicteds=nan(numel(testClass),numel(methods));
classPerforms=nan(numel(methods),numberPerfMeasure);
conMats=nan(numClasses,numClasses,numel(methods));
OtherOutputs=cell(numel(methods),1);
tElapseds=nan(numel(methods),1);
for m=1:numel(methods)
    method=methods{m};
    if isempty(options{m})
        option=[];
    else
        option=options{m};
    end
    [model,~]=classificationTrain(trainSet,trainClass,method,option);
    % predict
    [testClassPredicted,OtherOutput]=classificationPredict(model,testSet,testClass);
    % calculate the performance
    [classPerform,conMat]=perform(testClassPredicted,testClass,numel(unique([trainClass;testClass])));
    %     [testClassPredicted,classPerform,conMat,OtherOutput]=classification(trainSet,trainClass,testSet,testClass,method,option);
    testClassPredicteds(:,m)=testClassPredicted;
    classPerforms(m,:)=classPerform;
    conMats(:,:,m)=conMat;
    tElapseds(m,1)=OtherOutput{1};
    OtherOutputs{m}=OtherOutput;
end
end
