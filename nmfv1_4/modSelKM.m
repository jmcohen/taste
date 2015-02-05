function [param,val]=modSelKM(trainSet,trainClass,option)
% model selection
% this function is not currently useful

if nargin<3
   option=[]; 
end
optionDefault.range=[0:30];
optionDefault.epsilon=1e-4;
option=mergeOption(option,optionDefault);
numTr=size(trainSet,2);
numP=numel(optionDefault.range);
M=nan(numP,1);
for i=1:numP
    option.param=2^optionDefault.range(i);
    K=computeKernelMatrix(trainSet,trainSet,option);
    sumK=sum(sum(K));
    if sumK<=(1+option.epsilon)*numTr || sumK>=(1-option.epsilon)*numTr*numTr
        continue;
    end
    M(i)=JSM(K,trainClass);
end
[val,ind]=nanmin(M);
param=2^optionDefault.range(ind);
end