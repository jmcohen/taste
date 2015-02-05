function [indCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=NMFCluster(X,k,option)
% NMF based clustering
% Usage:
% [indCluster,numIter,tElapsed,finalResidual]=NMFCluster(X) % in this case, X is coefficient matrix obtained by a NMF outside this function
% [indCluster,numIter,tElapsed,finalResidual]=NMFCluster(X,k)
% [indCluster,numIter,tElapsed,finalResidual]=NMFCluster(X,k,option)
% X: matrix, the data to cluster, each column is a sample/data point
% k: the number of clusters
% option: struct, 
% option.reorder, logical scalar, if to reorder the data points based on
% clustering result, the default is true
% the rest fields of option is the same as the option in the nmf function, type "help nmf" for more informtion
% indCluster: the cluster label of each sample, i.e. indCluster=[2;2;1;1;3;2;1;1;2;3];
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
% May 21, 2011
%%%%

callNMF=true;
if nargin==2
    option=[];
    callNMF=true;
end
if nargin==1
    Y=X; % X is coefficient matrix
    A=[];
    callNMF=false;
    option=[];
end
optionDefault.reorder=true;
optionDefault.algorithm='nmfnnls';
optionDefault.optionnmf=[];
option=mergeOption(option,optionDefault);

if nargin==1
    Y=X; % X is coefficient matrix
    callNMF=false;
end
if callNMF
   [A,Y,numIter,tElapsed,finalResidual]=nmf(X,k,option); 
end
[C,indCluster]=max(Y,[],1);
indCluster=indCluster';
Xout=[];
Aout=[];
Yout=[];
if option.reorder
    [indClusterSorted,ind]=sort(indCluster);
    Xout=X(:,ind);
    Yout=Y(:,ind);
    Aout=A;
end
end