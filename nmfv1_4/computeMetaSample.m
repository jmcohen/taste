function [metaSample,metaClass]=computeMetaSample(trainSet,trainClass,option)
% Compute metasamples for each class using SVD. This function is used by
% MSRC.
% trainSet: matrix, each column is a training sample
% trainClass: numeric column vector, the class labels of the training samples
% option: struct, with fields:
% option.ks: column vector, the number metasamples for each class, i.e.
% [8;5;6], the default is 8 for each classes
% option.labmda: scalar, the weight on the l_1 norm, the default is 0.1
% option.ifModelSelection: logical, indicates if model selection is
% conducted to select lambda, the default is false
% option.metaSampleMethod: string, indicates the methods to generate
% metasamples. This only value is 'svd' for the current version
% metaSample: matrix, each column is a metasample
% metaClass: column vector, the class labels for the metasamples 
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
% September 13, 2011
%%%%


% trainSet=normc(trainSet);

unikClass=unique(trainClass);
numUnikClass=numel(unikClass);
optionDefault.ks=5*ones(numUnikClass,1);
optionDefault.lambda=0.1;
optionDefault.ifModelSelection=false;
optionDefault.metaSampleMethod='svd';
% optionDefault.normalization=0;
if nargin<3
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% if logical(option.normalization)
%     [trainSet,trainSetMean,trainSetSTD]=normmean0std1(trainSet');
%     trainSet=trainSet';
% end

metaSample=[];
metaClass=[];
for i=1:numUnikClass
    curInd=(trainClass==unikClass(i));
    curTrainSet=trainSet(:,curInd);
    numCurTrain=size(curTrainSet,2);
    if numCurTrain<=option.ks(i)
        option.ks(i)=numCurTrain;
    end
    switch option.metaSampleMethod
        case 'svd'
            [U,S,V]=svd(curTrainSet,0);
            metaSample=[metaSample,U(:,1:option.ks(i))];
        case 'nmf'
            if option.ks(i)==numCurTrain
               A=curTrainSet;
            else
                optionnmf.algorithm='nmfnnls';%'sparsenmfnnls';
              [A,Y]=nmf(curTrainSet,option.ks(i),optionnmf);
            end
            metaSample=[metaSample,A];
    end
    curTrainClass=trainClass(curInd);
    curTrainClass=curTrainClass(1:option.ks(i));
    metaClass=[metaClass;curTrainClass(:)];
end
end