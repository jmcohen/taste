function [predicted,vals]=softSVMPredict2(model,testSet)
% soft margin svm, express bias explictly
%model.mu: % Lagrance multiplier
%model.sv: support vectors
% example:
% load('C:\YifengLi\Reseach Program\dataset\Colon\ColonCancer.mat');
% dataStr='Colon';
% kfold=3;
% ind=crossvalind('Kfold',classes,kfold);
% indTest=(ind==1);
% trainSet=D(:,~indTest);
% testSet=D(:,indTest);
% trainClass=classes(~indTest);
% testClass=classes(indTest);
% % normalization
% trainSet=normc(trainSet);
% testSet=normc(testSet);
% model=softSVMTrain(trainSet,trainClass,option);
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
% Yifeng Li
% May 28, 2012
%%%%

 numTe=size(testSet,2);
%  if model.numCl==2
%      vals=zeros(numTe,model.numCl-1);
%  else
%      vals=zeros(numTe,model.numCl);
%  end
% the predicted target using one vs all strategy
vals=zeros(numTe,model.numCl);
for i=0:model.numCl-1
    if model.numCl==2 && i==1 % two classes
        vals(:,i+1)=-vals(:,i);
        break;
    end
    
    % kernel matrix
    modelThis=model.binary{i+1};
    S=computeKernelMatrix(modelThis.sv,testSet,model.option);
    % change sign of each row
    for r=1:modelThis.numsv
        S(r,:)=modelThis.svClM(r).*S(r,:);
    end
    
    % predict
    vals(:,i+1)=modelThis.musv'*S + modelThis.b; 
end

% softmax
[mVal,predicted]=max(vals,[],2);
predicted=predicted-1;
end