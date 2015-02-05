function [en,dist]=entropyCluster(indCluster,classes)
% compute the entropy of the clustering result
% indCluster: column vector, the cluster index of each data point
% classes: column vector, the actual class label of each data point
% en: scalar, the entropy
% dist: matrix of size #clusters*#classes, the sample distribution
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
% May 26, 2011
%%%%

numSample=numel(classes);
uniClass=unique(classes);
numClass=numel(uniClass);
uniCluster=unique(indCluster);
numCluster=numel(uniCluster);
dist=zeros(numCluster,numClass);
logn=zeros(numCluster,numClass);
for i=1:numCluster
    curClu=(indCluster==(uniCluster(i)));
    numCurClu=sum(curClu);
    for j=1:numClass
        curCla=(classes==uniClass(j));
        curCluCla=curClu&curCla;
        numCurCluCla=sum(curCluCla);
        if numCurCluCla==0
            continue;
        end
        dist(i,j)=numCurCluCla;
        logn(i,j)=numCurCluCla*log2(numCurCluCla/numCurClu);
    end
end
en=-1/(numSample*log2(numClass))*sum(sum(logn));
end