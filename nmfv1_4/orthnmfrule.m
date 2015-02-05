function [A,S,Y,numIter,tElapsed,finalResidual]=orthnmfrule(X,k,option)
% Orthogonal-NMF based on NNLS: X=ASY, s.t. A'*A=I, Y*Y'=I, A,S,Y>0.
% Definition:
%     [A,S,Y,numIter,tElapsed,finalResidual]=orthnmfrule(X,k)
%     [A,S,Y,numIter,tElapsed,finalResidual]=orthnmfrule(X,k,option)
% X: non-negative matrix, dataset to factorize, each column is a sample, and each row is a feature.
% k: scalar, number of clusters.
% option: struct:
% option.orthogonal: row vector of length 2. 
%      option.orthogonal(1)=1: enforce orthogonal constraint on A; 0: unconstrained. 
%      option.orthogonal(2)=1: enforce orthogonal constraint on Y; 0: unconstrained. 
% option.iter: max number of interations. The default is 1000.
% option.dis: boolen scalar, It could be 
%     false: not display information,
%     true: display (default).
% option.residual: the threshold of the fitting residual to terminate. 
%     If the ||X-XfitThis||<=option.residual, then halt. The default is 1e-4.
% option.tof: if ||XfitPrevious-XfitThis||<=option.tof, then halt. The default is 1e-4.
% A: matrix, the basis matrix. 
% S: matrix, absorb the values due to orthnormality of A and Y.
% Y: matrix, the coefficient matrix.
% numIter: scalar, the number of iterations.
% tElapsed: scalar, the computing time used.
% finalResidual: scalar, the fitting residual.
% References:
%  [1]Chris Ding, Tao Li, Wei Peng, and Haesun Park. 2006. 
%     Orthogonal nonnegative matrix t-factorizations for clustering. 
%     In KDD '06. ACM, New York, NY, USA, 126-135.
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
optionDefault.orthogonal=[1,1];
optionDefault.iter=200;
optionDefault.dis=1;
optionDefault.residual=1e-4;
optionDefault.tof=1e-4;
if nargin<3
   option=optionDefault;
else
    option=mergeOption(option,optionDefault);
end

[r,c]=size(X); % c is # of samples, r is # of features
[inx,C]=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
Y=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
Y=Y'+0.2;
A=C';
if sum(option.orthogonal)==2
    S=A'*X*Y';
else
    S=eye(k,k);
end

XfitPrevious=Inf;
for i=1:option.iter
    if option.orthogonal(1)==1
        A=A.*((X*Y'*S')./(A*A'*X*Y'*S'));
    else
        A=A.*((X*Y')./(A*(Y*Y')));
    end
    A=max(A,eps);
    if option.orthogonal(2)==1
        Y=Y.*((S'*A'*X)./(S'*A'*X*(Y'*Y)));
    else
        Y=Y.*((A'*X)./(A'*A*Y));
    end
    Y=max(Y,eps);
    if sum(option.orthogonal)==2
        S=S.*(A'*X*Y')./(A'*A*S*(Y*Y'));
        S=max(S,eps);
    end
    
    if mod(i,10)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*S*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            s=sprintf('Multiple rule based orthNMF successes! \n # of iterations is %0.0d. \n The final residual is %0.4d.',i,curRes);
            disp(s);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
