function [alphaOptimal,betaOptimal,accMax]=gridSearchUniverse(trainSet,trainClass,option)
% Line or Grid Search
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
% July 04, 2012
%%%%

optionDefault.classifier='svmLi';
optionDefault.alphaRange=-2:5; % exponent of 2
optionDefault.betaRange=3:8; % exponent of 2
optionDefault.kfold=3;
optionDefault.rerun=1;
optionDefault.normalization=0;
optionDefault.optCl=[];

if nargin==2
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end
numClass=numel(unique(trainClass));
numAlphaRange=numel(option.alphaRange);
numBetaRange=numel(option.betaRange);
AllAccs=zeros(numAlphaRange,numBetaRange);
for c=1:numAlphaRange
    for i=1:numBetaRange
        alpha=2^(option.alphaRange(c));
        beta=2^(option.betaRange(i));
        switch option.classifier
            case {'svm','svmLi','lsvmLi'}
                option.optCl.C=alpha;
                option.optCl.param=beta;
            case {'nnls'}
                option.optCl.param=beta;
            otherwise
                error('Please select a correct classifier!');
        end
        perf=cvExperiment(trainSet,trainClass,option.kfold,option.rerun,'none',[],'none',[],{option.classifier},{option.optCl}); 
        AllAccs(c,i)=perf(end-1);
    end
end
[AllAccsRow,indsRow]=max(AllAccs,[],1);
[accMax,indCol]=max(AllAccsRow);
indRow=indsRow(indCol);
betaOptimal=2^(option.betaRange(indCol));
alphaOptimal=2^(option.alphaRange(indRow));
end