function [A,Y,numIter,tElapsed,finalResidual]=seminmfrule(X,k,option)
% SemiNMF based on update rules: X=AY, s.t. Y>0;
% X, dataset, matrix, each column is a sample, each row is a feature, each column of X is a sample
% k, number of clusters
% option: struct:
% option.iter, max number of interations
% option.dis, boolen scalar, false: (not display information) or true
% (display).
% option.residual, if the ||X-XfitThis||<=option.residual, halt.
% option.tof, if ||XfitPrevious-XfitThis||<=option.tof, halt.
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

% initialize X by k-mean
[r,c]=size(X); % c is # of samples, r is # of features
% inx=kmeans(X',k); % k-mean clustering, get idx=[1,1,2,2,1,3,3,1,2,3]
% H=(inx*ones(1,k)-ones(c,1)*cumsum(ones(1,k)))==0; % obtain logical matrix [1,0,0;1,0,0;0,1,0;0,1,0;1,0,0;0,0,1;...]
% G=H+0.2;
G=rand(c,k);
XfitPrevious=Inf;
% iter: number of iterations
for i=1:option.iter
    F=X/G'; % pseudoinverse
    A=X'*F;
    Ap=(abs(A)+A)./2;
    An=(abs(A)-A)./2;
    B=F'*F;
    Bp=(abs(B)+B)./2;
    Bn=(abs(B)-B)./2;
    G=G.*sqrt((Ap + G*Bn)./(An + G*Bp));
    A=F;
    Y=G';
    if mod(i,50)==0 || i==option.iter
        if option.dis
            disp(['Iterating >>>>>> ', num2str(i),'th']);
        end
        XfitThis=A*Y;
        fitRes=matrixNorm(XfitPrevious-XfitThis);
        XfitPrevious=XfitThis;
        curRes=norm(X-XfitThis,'fro');
        if option.tof>=fitRes || option.residual>=curRes || i==option.iter
            disp(['Mutiple update rules based on SemiNMF successes!, # of iterations is ',num2str(i),'. The final residual is ',num2str(curRes)]);
            numIter=i;
            finalResidual=curRes;
            break;
        end
    end
end
tElapsed=toc(tStart);
end
