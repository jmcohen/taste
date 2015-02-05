function [testClassPredicted,otherOutput]=lrc(trainSet,trainClass,testSet,testClass,option)
% linear regression classifier
% trainSet, matrix, each column is a training sample
% trainClass: column vector, the class labels for the training samples
% testSet: matrix, each column is a new or testing sample
% testClass; column vector, the class labels for the testing samples, can
% be [], if unknown.
% option, struct, reserved for future use.
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
% August 04, 2010
%%%%

% optionDefault.p=16;
% option=mergeOption(option,optionDefault);
% % trainSet=downsample(trainSet,option.p);
% testSet=downsample(testSet,option.p);

utrCl=unique(trainClass);
numCl=numel(utrCl);
numTest=size(testSet,2);
testClassPredicted=zeros(numTest,1);
residuals=[];
for s=1:numTest
    testSets=testSet(:,s);
    residual=zeros(numCl,1);
    for i=1:numCl
        ind=(trainClass==utrCl(i));
        trainSeti=trainSet(:,ind);
        beta=pinv(trainSeti)*testSets;
        residual(i)=matrixNorm(testSets-trainSeti*beta);
    end
    residuals=[residuals;residual];
    [val,resInd]=min(residual);
    testClassPredicted(s)=utrCl(resInd);
end
otherOutput=residuals;
end