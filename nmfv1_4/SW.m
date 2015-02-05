function w=SW(K,trainClass)
% compute the expected distance of an image from the mass center of a class
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

uniCls=unique(trainClass);
numCl=numel(uniCls);
w=0;
for i=1:numCl
   ind=(trainClass==uniCls(i));
   numCurCl=sum(ind);
   w=w + (1/numCurCl * sum(diag(K(ind,ind)))) - (1/numCurCl^2 * sum(sum(K(ind,ind))));
end
w=w/numCl;
end
