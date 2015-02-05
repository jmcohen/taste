function [AtA,Y,numIter,tElapsed,finalResidual,A]=kernelSparseNMFNNQP(X,k,option)
% Kernel semi-NMF based on NNQP
% Usages:
%     [Ak,Y,numIter,tElapsed,finalResidual]=kernelSparseNMFNNQP(X,k)
%     [Ak,Y,numIter,tElapsed,finalResidual]=kernelSparseNMFNNQP(X,k,option)
% X: matrix of mixed signs, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option.kernel: see function computeKernelMatrix.
% option.param: see function computeKernelMatrix.
% option.lambda: scalar, the trade-off parameter
% option.iter: max number of interations. The default is 1000.
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
% A: the basis matrix, if linear, other A=[];
% References:
%  [1]\bibitem{cibcb2012}
%     Y. Li and A. Ngom,
%     ``A New Kernel Non-Negative Matrix Factorization and Its Application in Microarray Data Analysis,''
%     {\it CIBCB},
%     pp. 371-378, 2012.
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
optionDefault.kernel='rbf';
optionDefault.param=[];
optionDefault.lambda=2^-6;
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
XtX=computeKernelMatrix(X,X,option);

if size(X,3)>1
    [r1,r2,c]=size(X); % c is # of samples, r is # of features
    r=r1*r2;
else
    [r,c]=size(X); % c is # of samples, r is # of features
end
% inx=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
% H=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
% Y=H'+0.2;
% use NMF to initialize Y
% if ~strcmp(option.kernel,'linear')
%     optionIni=option;
%     optionIni.kernel='linear';
%     [~,Y]=kernelSparseNMFNNQP(X,k,optionIni);
% else
%     Y=rand(k,c);
% end
Y=rand(k,c);
% initialize AtA
if r==k
    AtA=Y'\XtX/Y;
    AtX=Y'\XtX;
else
    pinvY=pinv(Y);
    AtA=pinvY'*XtX*pinvY;
    AtX=pinvY'*XtX;
end
% normalize AtA and AtX
[AtA,~,XtX,~,AtX]=normalizeKernelMatrix(AtA,XtX,AtX);

% Y=rand(k,c);
prevRes=Inf;

% iterative updating
for i=1:option.iter
    % update Y
    [zeroA,tf0]=iszero(sum(AtA));
    AtA(zeroA,:)=[];
    AtA(:,zeroA)=[];
    AtX(zeroA,:)=[];
    if tf0 && k>1
        k=k-1;
    end
    if k<1
       error('k<1, impossible!'); 
    end
    Y=l1NNQPActiveSet(AtA,-AtX,option.lambda);
    % update kernel matrices
    if r==k
        AtA=Y'\XtX/Y;
        AtX=Y'\XtX;
    else
        pinvY=pinv(Y);
        AtA=pinvY'*XtX*pinvY;
        AtX=pinvY'*XtX;
    end
    % normalize AtA and AtX
    [AtA,~,XtX,~,AtX]=normalizeKernelMatrix(AtA,XtX,AtX);
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
if strcmp(option.kernel,'linear')
   A=X*pinv(Y);
   A=normc(A);
else
    A=[];
end
tElapsed=toc(tStart);
end
