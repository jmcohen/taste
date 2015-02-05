function [Y,numIter,tElapsed,finalResidual]=vsmftest(testSet,outTrain)
% Versatile NMF to test samples
% Definition:
%     [A,Y,AtA,numIter,tElapsed,finalResidual]=vsmf(X,k)
%     [A,Y,AtA,numIter,tElapsed,finalResidual]=vsmf(X,k,option)
% X: matrix, dataset to factorize, each column is a sample, and each row is a feature.
% k: number of clusters.
% option: struct:
% option.alpha2: non-negative scalar, control the smoothness and scale of
% the basis vectors. The default is 0.
% option.alpha1: non-negative scalar, control the sparsity of the basis
% vectors. The default is 0.
% option.lambda2: non-negative scalar, control the smoothness and scale of
% the coefficient vectors. The default is 0.
% option.lambda1: non-negative scalar, control the sparsity of the
% coefficient vectors. The default is 0.
% option.t1: logical, if A should be constrained by non-negativity. Default
% is true.
% option.t2: logical, if Y should be constrained by non-negativity. Default
% is true.
% option.kernelizeAY: 0: do not kernelize any matrix (default), 1:
% kernelize columns of X and A, 2: kernelize rows of X and Y.
% option.kernel, string, the kernel function. The default is 'linear'. See function
% computeKernelMatrix for more information. 
% option.param, column array, the parameter of a kernel function.
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%    If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% References:
%  []
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
% Feb 26, 2013
%%%%

tStart=tic;
optionDefault.alpha2=0;
optionDefault.alpha1=0;
optionDefault.lambda2=0;
optionDefault.lambda1=0;
optionDefault.t1=true;
optionDefault.t2=true;
optionDefault.kernel='linear';
optionDefault.kernelizeAY=0;
optionDefault.param=[];
optionDefault.iter=100;
optionDefault.dis=true;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

[numFeat,numS]=size(X);

if option.kernelizeAY~=0 && option.kernelizeAY~=1 && option.kernelizeAY~=2 % not 0, 1, 2, then set to 0
   option.kernelizeAY=0; 
end

if option.kernelizeAY==1 || (~option.t1 && option.alpha1==0)
   XtX=computeKernelMatrix(X,X,option); 
end

if option.kernelizeAY==2 || (~option.t2 && option.lambda1==0)
   XXt=computeKernelMatrix(X',X',option); 
end

% iter: number of iterations
[r,c]=size(X); % c is # of samples, r is # of features
Y=rand(k,c);
% Y(Y<eps)=0;
Y=max(Y,eps);
YYt=Y*Y';
YXt=Y*X';
A=X/Y;
% A(A<eps)=0;
A=max(A,eps);
AtA=A'*A;
AtX=A'*X;
XfitPrevious=Inf;
% E1=option.alpha1.*ones(numFeat,k);
% E2=option.lambda1.*ones(k,numS);
ep=0;
for i=1:option.iter
    % update A
    I1=option.alpha2.*eye(k);
    H1=YYt+I1;
    if option.kernelizeAY~=1 && option.t1
        G1=option.alpha1-YXt;
        A=NNQPActiveSet(H1,G1);
        A=max(A,ep);
        A=A';
        AtA=A'*A;
        AtX=A'*X;
    end
    if option.kernelizeAY~=1 && ~option.t1 && option.alpha1>0
        G1=-YXt;
        A=l1QPActiveSet(H1,G1,option.alpha1);
        A=A';
        AtA=A'*A;
        AtX=A'*X;
    end
    if option.kernelizeAY==1 || (~option.t1 && option.alpha1==0)
        Ydagger=Y'/H1;
        A=X*Ydagger;
        AtA=Ydagger'*XtX*Ydagger;
        AtX=Ydagger'*XtX;
    end
    
    Asum=sum(AtA,1);
    indRm=(Asum==0);
    A(:,indRm)=[];
    AtA(indRm,:)=[];
    AtA(:,indRm)=[];
    AtX(indRm,:)=[];
    Y(indRm,:)=[];
    YYt(indRm,:)=[];
    YYt(:,indRm)=[];
    YXt(indRm,:)=[];
    if any(indRm)
        k=k-sum(indRm);
    end
    
    %             A(A<eps)=0;
    % update Y
    I2=option.lambda2.*eye(k);
    H2=AtA+I2;   
    if option.kernelizeAY~=2 && option.t2
        G2=option.lambda1-AtX;
        Y=NNQPActiveSet(H2,G2);
        Y=max(Y,ep);
        YYt=Y*Y';
        YXt=Y*X';
    end
    if option.kernelizeAY~=2 && ~option.t2 && option.lambda1>0
        G2=-AtX;
        Y=l1QPActiveSet(H2,G2,option.lambda1);
        YYt=Y*Y';
        YXt=Y*X';
    end
    if option.kernelizeAY==2 || (~option.t2 && option.lambda1==0)
        Adagger=H2\A';
        Y=Adagger*X;
        YYt=Adagger*XXt*Adagger';
        YXt=Adagger*XXt;
    end
    
    Ysum=sum(YYt,2);
    indRm=(Ysum==0);
    YYt(indRm,:)=[];
    YYt(:,indRm)=[];
    YXt(indRm,:)=[];
    A(:,indRm)=[];
    AtA(indRm,:)=[];
    AtA(:,indRm)=[];
    AtX(indRm,:)=[];
    if any(indRm)
        k=k-sum(indRm);
    end
    
    if mod(i,10)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            s=sprintf('VSMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
            disp(s);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
