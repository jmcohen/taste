function model=softSVMTrain2(trainSet,trainClass,option)
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

optionDefault.C=1;
optionDefault.kernel='linear';
optionDefault.param=[];
optionDefault.tol=2^(-12);
if nargin<3
   option=[]; 
end
option=mergeOption(option,optionDefault);

[numFe,numTr]=size(trainSet);
% transform class labels to -1,1 if binary, or 0,1,2,3,... if multi-class
unikCl=unique(trainClass);
numCl=numel(unikCl);
for i=0:numCl-1
    if numCl==2 && i==1
        break;
    end
    ind=(trainClass==i);
    trainClassM=trainClass;
    trainClassM(~ind)=-1;
    trainClassM(ind)=1;
    % compute kernel matrix
    K=computeKernelMatrix(trainSet,trainSet,option);

%     % change signs of training Set
%     
%     trainSetSign=trainSet;
%     for t=1:numTr
%         trainSetSign(:,t)=trainClassM(t).*trainSetSign(:,t);
%     end
    % multiply signs of each element
    for r=1:numTr
       for c=1:numTr 
        K(r,c)=trainClassM(r)*trainClassM(c)*K(r,c);
       end
    end
    
    if isscalar(option.C)
        option.C=option.C .* ones(numTr,1);
    end
    
    % apply quadratic programming solver, SMO is future work
    Aeq=trainClassM';
    beq=0;
    lb=zeros(numTr,1);
    ub=option.C;
    f=-ones(numTr,1);
%     x0=(K+option.tol*eye(numTr))\trainClassMAbs; % initialized by HDLM Largrane multiplier
%     x0(x0>option.C(1))=option.C(1); % let the initial point feasible
    %optQP
    optQP=optimset('Display','off','LargeScale','off');
    mu= quadprog(K,f,[],[],Aeq,beq,lb,ub,[],optQP);
    modelThis.mu=mu;
    maxmu=max(mu);
    nonmuInd=mu>maxmu*option.tol;
    modelThis.musv=mu(nonmuInd);
    modelThis.numsv=sum(nonmuInd);
    modelThis.sv=nan(numFe,modelThis.numsv);
    modelThis.svCl=nan(modelThis.numsv,1);
    modelThis.svClM=nan(modelThis.numsv,1);
    modelThis.svLogical=false(numTr,1);
    count=1;
    for t=1:numTr
        if mu(t)>maxmu*option.tol
            modelThis.sv(:,count)=trainSet(:,t);
            modelThis.svCl(count)=trainClass(t);
            modelThis.svClM(count)=trainClassM(t);
            modelThis.svLogical(t)=true;
            count=count+1;
        end
    end
    % kernel matrix using only support vectors
    Ksv=K(modelThis.svLogical,modelThis.svLogical);
    % change sign of each column
    for c=1:modelThis.numsv
        Ksv(:,c)=modelThis.svClM(c).*Ksv(:,c);
    end
    % compute bias using support vectors
    % b=1/|S| * sum(trainClass - supportVectors'*(trainClass.*supportVectors')*multiplier)
    modelThis.b=mean(modelThis.svClM-Ksv*modelThis.musv);
    model.binary{i+1}=modelThis;
end
model.option=option;
model.numCl=numCl;
end

