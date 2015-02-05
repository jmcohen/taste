function [ind,clusters,clusterClasses,order]=getClusters(Y,X,classes)
% X, matrix, data, each column is a sample (data point)
% Y, matrix, coefficient matrix, obtained through NMF: X=AY
% classes: ,column vector, class labels of the samples in X
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
% May 03, 2011
%%%%

if nargin<3
    classes=[];
end
if nargin<2
   X=[]; 
end

ind=[];
clusters=[];
clusterClasses=[];
order=[];

[numCluster,numSample]=size(Y);
[C,in]=max(Y,[],1);
ind=false(numCluster,numSample);
% indicator matrix
for i=1:numSample
    ind(in(i),i)=true;
end
if isempty(X)&& isempty(classes)
    return;
end
% for each cluster, obtain the samples and class labels
if ~isempty(X)
    clusters=cell(1,numCluster);
    for j=1:numCluster
        clusters{j}=X(:,ind(j,:));
    end
end
if ~isempty(classes)
    clusterClasses=cell(1,numCluster);
    for j=1:numCluster
        clusterClasses{j}=classes(ind(j,:));
    end
end
numorder=(1:numSample)';
order=cell(1,numCluster);
for j=1:numCluster
   order{j}=numorder(ind(j,:));
end
end