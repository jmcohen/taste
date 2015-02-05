function [A,Y,numIter,tElapsed,finalResidual]=convexnmfrule(X,k,option)
% Convex-NMF based on multiplicative update rules: X=XAY, s.t. A,Y>0.
% Definition:
%     [A,Y,numIter,tElapsed,finalResidual]=convexnmfrule(X,k)
%     [A,Y,numIter,tElapsed,finalResidual]=convexnmfrule(X,k,option)
% X: matrix of mixed signs, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%     If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the weighting matrix.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% References:
% [1]\bibitem{Ding2010}
%    C. Ding, T. Li, and M.I. Jordan,
%    ``Convex and semi-nonnegative matrix factorizations,''
%    {\it IEEE Transations on Pattern Analysis and Machine Intelligence},
%    vol. 32, no. 1, pp. 45-55, 2010.
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
% May 01, 2011
%%%%


tStart=tic;
optionDefault.iter=1000;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end
Ak=X'*X;
Ap=(abs(Ak)+Ak)./2;
An=(abs(Ak)-Ak)./2;
[r,c]=size(X); % c is # of samples, r is # of features
% inx=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
% H=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
% G=H+0.2;
% D=diag(1./sum(H));
% W=G*D;
G=rand(c,k);
W=G*diag(1./sum(G));
XfitPrevious=Inf;

for i=1:option.iter
    ApW=Ap*W;
    AnW=An*W;
    GWt=G*W';
    G=G.*sqrt((ApW + GWt*AnW)./(AnW + GWt*ApW));
%     G(G<eps)=0;
    G=max(G,eps);
    GtG=G'*G;
    W=W.*sqrt((Ap*G + AnW*GtG)./(An*G + ApW*GtG)); 
%     W(W<eps)=0;
    W=max(W,eps);
    A=W;
    Y=G';
    if mod(i,100)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=X*A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            s=sprintf('Mutiple update rules based ConvexNMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
            disp(s);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
