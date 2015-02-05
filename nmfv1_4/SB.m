function b=SB(K, trainClass)
% compte the averaged distance between all class centroids
% K: kernel matrix
% trainClass: column vector, the class labels of the samples
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
%%%%

% centroid
uniCls=unique(trainClass);
numCl=numel(uniCls);
overAll=sum(sum(K));
b=0;
for i=1:numCl-1
    indi=(trainClass==uniCls(i));
    numi=sum(indi);
   for j=i+1:numCl
       indj=(trainClass==uniCls(j));
       numj=sum(indj);
    b=b+1/(numi^2) * sum(sum(K(indi,indi))) - 1/(numi*numj) * 2*sum(sum(K(indi,indj))) + 1/(numj^2) * sum(sum(K(indj,indj)));   
   end
end
b=2/(numCl^2-numCl);
end