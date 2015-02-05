function range=searchkernelRange(trainSet,trainClass,option)
% search the feasible range of gamma of rbf kernel
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
% Nov 01, 2011
%%%%

if nargin<3
   option=[]; 
end
optionDefault.range=5:30;
optionDefault.epsilon=1e-4;
option=mergeOption(option,optionDefault);
numTr=size(trainSet,2);
numP=numel(option.range);
M=nan(numP,1);
for i=1:numP
    option.param=2^option.range(i);
    K=computeKernelMatrix(trainSet,trainSet,option);
    sumK=sum(sum(K));
    if sumK<=(1+option.epsilon)*numTr || sumK>=(1-option.epsilon)*numTr*numTr
        continue;
    end
    M(i)=1;
end
range=option.range(~isnan(M));
end
