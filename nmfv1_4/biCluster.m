function [ACluster,YCluster,indACluster,indYCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=biCluster(X,k,option)
% The biclustering method using one of the NMF algorithms
% Usage:
% [ACluster,YCluster,indACluster,indYCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=biCluster(X,k) % in this case, X and k are basis matrix and coefficient matrix obtained by a NMF outside this function.
% [ACluster,YCluster,indACluster,indYCluster,Xout,Aout,Yout,numIter,tElapsed,finalResidual]=biCluster(X,k,option)
% X: matrix, the data to cluster, each column is a sample/data point
%     if the number of input arguments is 2, X is basis matrix obtained by
%     a NMF outside this function.
% k: the number of clusters
%     if the number of input arguments is 2, k is coefficient matrix
%     obtained by a NMF outside this function.
% option: struct:
% option.propertyName and option.propertyValue: type "help featureFilterNMF" for more information.
%      The rest of the fields are the same as the option in the "nmf"
%      function, type "help nmf" for more informtion.
% ACluster: column vector, the ordered cluster labels of features, i.e. [1;1;1;1;2;2;2;3;3;3;4;4;4;4;...].
% YCluser: column vector, the ordered cluster labels of the samples, i.e. [1;1;1;2;2;3;3;3;3;...].
% indAcluster: column vector, the position indices of the features, i.e.[342;699;2;45;100;...].
% indAcluster: column vector, the position indices of the samples, i.e.[12;19;2;7;8;...].
% numIter: scalar, the number of iterations in NMF.
% tElapsed: scalar, the computing time by biCluster.
% finalResidual: scalar, the fitting residual of NMF.
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
% May 20, 2011
%%%%


tStart=tic;
callNMF=true;
if nargin==2
    if size(X,2)==size(k,1)&& size(k,1)~=1&& size(k,2)~=1
        A=X; % the first input is the basis matrix obtain outside this function
        Y=k; % the second input is the coefficient matrix obtain outside this function
        callNMF=false;
    else
    option.optionnmf=[];
    callNMF=true;
    end
end
if nargin==1
    error('Insufficient input arguments!');% X is coefficient matrix
end
if callNMF
   [A,Y,numIter,tElapsed,finalResidual]=nmf(X,k,option); 
end

[m,n]=size(X);
% A, Y
[valA,indA]=max(A,[],2);
A01=zeros(m,k);
for i=1:m
    A01(i,indA(i))=1;
end
[valY,indY]=max(Y,[],1);
Y01=zeros(n,k);
for i=1:n
    Y01(indY(i),i)=1;
end
% arrange A and Y
indA=(1:m)';
indY=(1:n)';

optff=[]; % the option input of featureFilterNMF function
optff.isBasis=true;
if isfield(option,'propertyName')
    optff.propertyName=option.propertyName;
    optff.propertyValue=option.propertyValue;
end
mask=featureFilterNMF(A,[],optff);
A=A(mask,:);
A01=A01(mask,:);
indA=indA(mask);
ACluster=[];
Aout=[];
indACluster=[]; 
YCluster=[];
Yout=[];
indYCluster=[];
for i=1:k
    ACluster=[ACluster;i*ones(sum(A01(:,i)==1),1)];
    Aout=[Aout;A(A01(:,i)==1,:)];
    indACluster=[indACluster;indA(A01(:,i)==1)];
    YCluster=[YCluster;i*ones(sum(Y01(i,:)==1),1)];
    Yout=[Yout,Y(:,Y01(i,:)==1)];
    indYCluster=[indYCluster;indY(Y01(i,:)==1)];
end
Xout=X(indACluster,:);
Xout=Xout(:,indYCluster);
tElapsed=toc(tStart);
end