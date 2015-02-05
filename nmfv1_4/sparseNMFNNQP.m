function [A,Y,numIter,tElapsed,finalResidual]=sparseNMFNNQP(X,k,option)
% linear sparse NMF based on NNQP: X=AY, s.t. A,Y>0.
% Definition:
%     [Ak,A,Y,numIter,tElapsed,finalResidual]=kernelnmfrule(X,k)
%     [Ak,A,Y,numIter,tElapsed,finalResidual]=kernelnmfrule(X,k,option)
% X: matrix of mixed signs, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option.lambda: scalar, the trade-off parameter
% option.iter: max number of interations. The default is 200.
% option.dis: boolen scalar, It could be
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate.
%     If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% AtA: matrix, the kernel matrix, AtA=kernel(A,A).
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
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
% Dec 30, 2011
%%%%

tStart=tic;
optionDefault.lambda=2^0;
optionDefault.iter=200;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
    option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

% initialize
[r,c]=size(X); % c is # of samples, r is # of features
A=rand(r,k);
A=normc(A);
AtA=A'*A;
XtX=X'*X;
AtX=A'*X;

% Y=rand(k,c);
prevRes=Inf;
% iterative updating
for i=1:option.iter
    % update Y
    [zeroA,tf0,num0]=iszero(sum(AtA));
    AtA(zeroA,:)=[];
    AtA(:,zeroA)=[];
    AtX(zeroA,:)=[];
    if num0>0 && k>1
        k=k-num0;
    end
    if k<1
       error('k<1, impossible!'); 
    end
    Y=l1NNQPActiveSet(AtA,-AtX,option.lambda);
    % update kernel matrices
    YYt=Y*Y';
    YXt=Y*X';
    A=l1NNQPActiveSet(YYt,-YXt,0);
    A=A';
    % normalize AtA and AtX
    A=normc(A);
    AtA=A'*A;
    XtX=X'*X;
    AtX=A'*X;   
    if mod(i,20)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        curRes=trace(XtX-Y'*AtX-AtX'*Y+Y'*AtA*Y);
        fitRes=prevRes-curRes;
        prevRes=curRes;
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            disp(['kernelSparseNMFNNQP successes!, # of iterations is ',num2str(i),'. The final residual is ',num2str(curRes)]);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
