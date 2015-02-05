function testClassPredicted=localSVM(trainSet,trainClass,testSet,testClass,option)
% local SVM
% trainSet, matrix, the training set with samples in columns and features in rows.
% trainClass: column vector of numbers or string, the class labels of the traning set.
% testSet: matrix, the test set.
% testClass: column vector of numbers or string, the class labels of the
%       test/unknown set. It is actually unused in this function, thus, set it [].
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
% May 20, 2012
%%%%

optionDefault.similarity='rbf';
optionDefault.simParam=1;
optionDefault.kernel='linear';
optionDefault.param=[];
optionDefault.C=1;
optionDefault.tol=2^(-12);
if nargin<5
   option=[]; 
end
option=mergeOption(option,optionDefault);

numTr=size(trainSet,2);
numTe=size(testSet,2);
optK.kernel=option.similarity;
optK.param=option.simParam;
S=computeKernelMatrix(trainSet,testSet,optK);
testClassPredicted=zeros(numTe,1);
for i=1:numTe
    optSVM.kernel=option.kernel;
    optSVM.param=option.param;
    optSVM.C=option.C.*S(:,i);
    model=softSVMTrain2(trainSet,trainClass,optSVM);
    testClassPredicted(i)=softSVMPredict2(model,testSet(:,i));
end
end